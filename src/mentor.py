"""
mentor.py
=========
Mentor Agent for the Onboarding Copilot pipeline.

Implements a lightweight Retrieval-Augmented Generation (RAG) system that
answers developers' questions about an unfamiliar codebase using only in-memory
TF-IDF vectors — no vector database, no neural embedding model.

Pipeline position:
    Mapper → **Mentor** → Coach

Public API
----------
- :func:`build_knowledge_base` — ingest, chunk, and vectorise all sources.
- :func:`retrieve`            — cosine-similarity retrieval over in-memory chunks.
- :func:`ask`                 — retrieve + generate a grounded answer.
- :func:`save_session`        — persist Q&A history to ``mentor_output.json``.
- :func:`run_session`         — interactive REPL: build KB, loop, log, save.

Usage (standalone)::

    python mentor.py

Or programmatically::

    from mentor import build_knowledge_base, ask, save_session
    chunks = build_knowledge_base("mapper_output.json", "/path/to/repo")
    answer = ask("What does the AuthService do?", chunks)
"""

from __future__ import annotations

import ast
import os
import textwrap
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mapper_schema import MapperOutput
from mentor_schema import Confidence, MentorOutput, SessionEntry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Chunk size (characters) used when a file cannot be parsed structurally.
FIXED_CHUNK_SIZE: int = 1_500

#: Number of characters to overlap between successive fixed-size chunks.
FIXED_CHUNK_OVERLAP: int = 200

#: Minimum cosine similarity for a retrieved chunk to be considered relevant.
RELEVANCE_THRESHOLD: float = 0.10

#: Response returned when the knowledge base cannot answer confidently.
INSUFFICIENT_INFO_RESPONSE: str = (
    "I don't have enough information to answer that."
)

# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------


def _fixed_chunks(
    text: str,
    source_path: str,
    source_type: str,
    chunk_size: int = FIXED_CHUNK_SIZE,
    overlap: int = FIXED_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split *text* into fixed-size overlapping chunks.

    Used as a fallback for non-Python files or files that cannot be parsed
    via the AST.

    Args:
        text: Raw file content to chunk.
        source_path: Relative path of the originating file (for metadata).
        source_type: ``"code"`` or ``"mapper_summary"``.
        chunk_size: Maximum character length of each chunk.
        overlap: Number of characters of overlap between successive chunks.

    Returns:
        A list of chunk dicts, each with keys ``text``, ``source_path``,
        and ``source_type``.
    """
    chunks: list[dict[str, Any]] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "source_path": source_path,
                    "source_type": source_type,
                }
            )
        start += step
    return chunks


def _python_chunks(text: str, source_path: str) -> list[dict[str, Any]]:
    """Parse a Python source file and emit one chunk per top-level definition.

    Each function and class (including methods) becomes its own chunk,
    preserving its docstring and body. The leading module-level code
    (imports, constants, etc.) is collected as a single preamble chunk.

    Falls back to :func:`_fixed_chunks` if the file cannot be parsed.

    Args:
        text: Python source code.
        source_path: Relative path used for chunk metadata.

    Returns:
        A list of chunk dicts (``text``, ``source_path``, ``source_type``).
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _fixed_chunks(text, source_path, "code")

    lines = text.splitlines(keepends=True)
    chunks: list[dict[str, Any]] = []
    covered_lines: set[int] = set()  # 0-based line indices

    # One chunk per top-level function / class
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Only emit top-level and class-level definitions (depth ≤ 1)
            start_line = node.lineno - 1  # convert to 0-based
            end_line = node.end_lineno     # inclusive, 1-based → exclusive 0-based

            node_src = "".join(lines[start_line:end_line]).strip()
            if node_src:
                name = getattr(node, "name", "<anonymous>")
                chunks.append(
                    {
                        "text": f"# {source_path} — {name}\n{node_src}",
                        "source_path": source_path,
                        "source_type": "code",
                    }
                )
            for i in range(start_line, end_line):
                covered_lines.add(i)

    # Preamble: lines not covered by any function/class
    preamble_lines = [
        line for idx, line in enumerate(lines) if idx not in covered_lines
    ]
    preamble = "".join(preamble_lines).strip()
    if preamble:
        chunks.insert(
            0,
            {
                "text": f"# {source_path} — module preamble\n{preamble}",
                "source_path": source_path,
                "source_type": "code",
            },
        )

    return chunks if chunks else _fixed_chunks(text, source_path, "code")


def _chunk_source_file(
    content: str, relative_path: str
) -> list[dict[str, Any]]:
    """Chunk a single source file intelligently.

    For ``.py`` files the AST-based chunker is used; all other files fall back
    to fixed-size overlapping chunks.

    Args:
        content: Raw file content.
        relative_path: Path relative to *repo_root*, used only for metadata.

    Returns:
        A list of chunk dicts (``text``, ``source_path``, ``source_type``).
    """
    if relative_path.endswith(".py"):
        return _python_chunks(content, relative_path)
    return _fixed_chunks(content, relative_path, "code")


def _chunk_mapper_output(mapper_output: MapperOutput) -> list[dict[str, Any]]:
    """Convert a :class:`~mapper_schema.MapperOutput` into text chunks.

    Produces:
    - One chunk per :class:`~mapper_schema.EntryPoint`.
    - One chunk summarising the ``classification`` map.
    - One chunk for :class:`~mapper_schema.SetupInfo`.

    Args:
        mapper_output: Validated mapper output object.

    Returns:
        A list of chunk dicts (``text``, ``source_path``, ``source_type``).
    """
    chunks: list[dict[str, Any]] = []

    # --- Entry-point chunks -------------------------------------------------
    for ep in mapper_output.entry_points:
        text = (
            f"Entry-point: {ep.file}\n"
            f"Import count: {ep.import_count}\n"
            f"Confidence: {ep.confidence}"
        )
        chunks.append(
            {
                "text": text,
                "source_path": ep.file,
                "source_type": "mapper_summary",
            }
        )

    # --- Classification chunk -----------------------------------------------
    if mapper_output.classification:
        lines = ["File classification map:"]
        for file_path, entry in mapper_output.classification.items():
            lines.append(f"  {file_path}: {entry.category} — {entry.reason}")
        chunks.append(
            {
                "text": "\n".join(lines),
                "source_path": "mapper_output.json",
                "source_type": "mapper_summary",
            }
        )

    # --- Setup-info chunk ---------------------------------------------------
    si = mapper_output.setup
    if si:
        run_steps_str = "\n".join(f"    - {s}" for s in si.run_steps) or "    (none)"
        text = (
            f"Project setup:\n"
            f"  Language: {si.language}\n"
            f"  Dependencies file: {si.dependencies_file or '(none)'}\n"
            f"  Run steps:\n{run_steps_str}"
        )
        chunks.append(
            {
                "text": text,
                "source_path": "mapper_output.json",
                "source_type": "mapper_summary",
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# Knowledge-base construction
# ---------------------------------------------------------------------------


def build_knowledge_base(
    mapper_output_path: str, repo_root: str
) -> list[dict[str, Any]]:
    """Ingest, chunk, and TF-IDF-vectorise all knowledge sources.

    Steps:
    1. Load and validate ``mapper_output.json`` via
       :meth:`~mapper_schema.MapperOutput.from_json_file`.
    2. Read every source file listed in ``classification`` from *repo_root*.
    3. Chunk raw source files (AST-based for Python, fixed-size otherwise).
    4. Chunk the Mapper output into logical summary units.
    5. Fit a :class:`~sklearn.feature_extraction.text.TfidfVectorizer` over
       all chunk texts and attach the dense TF-IDF vector to each chunk dict.

    Args:
        mapper_output_path: Absolute or relative path to ``mapper_output.json``.
        repo_root: Root directory of the repository being analysed.

    Returns:
        A list of chunk dicts.  Each dict has the following keys:

        - ``text`` (*str*): The chunk's textual content.
        - ``source_path`` (*str*): Relative path of the originating file.
        - ``source_type`` (*str*): ``"code"`` or ``"mapper_summary"``.
        - ``vector`` (*numpy.ndarray*): Dense TF-IDF feature vector (1-D).

    Raises:
        FileNotFoundError: If *mapper_output_path* or any listed source file
            cannot be found.
    """
    # 1. Load mapper output
    mapper_output = MapperOutput.from_json_file(mapper_output_path)

    # 2. Read & chunk source files
    chunks: list[dict[str, Any]] = []

    for rel_path in mapper_output.classification:
        abs_path = os.path.join(repo_root, rel_path)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except FileNotFoundError:
            # Gracefully skip files that have been removed since mapping
            print(
                f"[mentor] Warning: source file not found, skipping: {abs_path}"
            )
            continue
        chunks.extend(_chunk_source_file(content, rel_path))

    # 3. Chunk mapper output
    chunks.extend(_chunk_mapper_output(mapper_output))

    if not chunks:
        raise ValueError(
            "No chunks could be built — knowledge base is empty. "
            "Check that repo_root and mapper_output_path are correct."
        )

    # 4. TF-IDF vectorise
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)  # sparse (n_chunks × n_terms)

    # Attach dense vectors to each chunk (convert sparse row → 1-D ndarray)
    for idx, chunk in enumerate(chunks):
        chunk["vector"] = np.asarray(tfidf_matrix[idx].todense()).flatten()

    # Store the fitted vectorizer in a special sentinel chunk for later use
    # (avoids passing it as a separate argument through the public API).
    chunks.append({"__vectorizer__": vectorizer})

    return chunks


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _get_vectorizer(chunks: list[dict[str, Any]]) -> TfidfVectorizer:
    """Extract the fitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`
    stored in the sentinel entry at the end of *chunks*.

    Args:
        chunks: The list returned by :func:`build_knowledge_base`.

    Returns:
        The fitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.

    Raises:
        ValueError: If the sentinel entry is missing (i.e. *chunks* was not
            produced by :func:`build_knowledge_base`).
    """
    if chunks and "__vectorizer__" in chunks[-1]:
        return chunks[-1]["__vectorizer__"]
    raise ValueError(
        "No fitted vectorizer found in chunks. "
        "Pass the list returned by build_knowledge_base()."
    )


def _content_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the content chunks, excluding the internal sentinel entry.

    Args:
        chunks: Full list as returned by :func:`build_knowledge_base`.

    Returns:
        List of chunk dicts that have a ``text`` key.
    """
    return [c for c in chunks if "text" in c]


def retrieve(
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve the *top_k* most relevant chunks for *question*.

    Vectorises *question* with the TF-IDF vocabulary fitted during
    :func:`build_knowledge_base`, computes cosine similarity against every
    stored chunk vector, and returns the top results sorted by descending
    score.  A ``score`` key is added to each returned chunk dict.

    Args:
        question: The natural-language question to answer.
        chunks: In-memory knowledge base produced by :func:`build_knowledge_base`.
        top_k: Maximum number of chunks to return.

    Returns:
        A list of up to *top_k* chunk dicts, each augmented with a ``score``
        key (float, 0–1) indicating cosine similarity to *question*.

    Raises:
        ValueError: If *chunks* does not contain a fitted vectorizer sentinel.
    """
    vectorizer = _get_vectorizer(chunks)
    content = _content_chunks(chunks)

    if not content:
        return []

    q_vec = vectorizer.transform([question])  # sparse (1 × n_terms)
    q_dense = np.asarray(q_vec.todense()).flatten()

    # Stack all chunk vectors into a matrix for batch cosine similarity
    chunk_matrix = np.vstack([c["vector"] for c in content])  # (n × n_terms)
    scores = cosine_similarity(q_dense.reshape(1, -1), chunk_matrix)[0]

    # Pair each content chunk with its score
    scored = sorted(
        zip(scores, content), key=lambda t: t[0], reverse=True
    )[:top_k]

    results = []
    for score, chunk in scored:
        result = {**chunk, "score": float(score)}
        result.pop("vector", None)  # exclude raw vector from returned dicts
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are Mentor, an onboarding assistant for software developers.
    You answer questions about a codebase using only the context provided below.
    Be concise, precise, and developer-friendly.
    If the context does not contain enough information, say so explicitly.
    Do not invent details that are not in the context.
    """
)


def _build_context_block(retrieved: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt.

    Args:
        retrieved: List of chunk dicts as returned by :func:`retrieve`.

    Returns:
        A formatted multi-line string ready for inclusion in an LLM prompt.
    """
    lines: list[str] = []
    for i, chunk in enumerate(retrieved, start=1):
        source = chunk.get("source_path", "unknown")
        score = chunk.get("score", 0.0)
        lines.append(f"[{i}] (source: {source}, relevance: {score:.3f})")
        lines.append(chunk["text"])
        lines.append("")
    return "\n".join(lines)


def _generate_answer(question: str, context_block: str) -> str:
    """Generate a plain-text answer from *question* and *context_block*.

    This implementation uses a **deterministic template-based synthesis**
    rather than an LLM call, keeping the module dependency-free beyond
    scikit-learn.  The context chunks are presented verbatim with headings;
    a brief introductory sentence names the question being answered.

    To integrate a real LLM (e.g. watsonx.ai or OpenAI), replace the body
    of this function with an API call that passes ``_SYSTEM_PROMPT``,
    ``context_block``, and ``question`` as the user message.

    Args:
        question: The developer's original question.
        context_block: Formatted context retrieved by :func:`retrieve`.

    Returns:
        A synthesised answer string.
    """
    intro = f'Answering: "{question}"\n\nRelevant context from the codebase:\n'
    separator = "─" * 60
    return f"{intro}{separator}\n{context_block}{separator}"


def ask(
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> str:
    """Retrieve relevant chunks and generate a grounded answer.

    Retrieves the *top_k* most similar chunks for *question* and generates an
    answer from them.  If the highest-scoring chunk's similarity falls below
    :data:`RELEVANCE_THRESHOLD`, returns :data:`INSUFFICIENT_INFO_RESPONSE`
    instead of fabricating an answer.

    Args:
        question: The natural-language question to answer.
        chunks: In-memory knowledge base from :func:`build_knowledge_base`.
        top_k: Number of chunks to retrieve and present as context.

    Returns:
        A grounded answer string, or :data:`INSUFFICIENT_INFO_RESPONSE` when
        the knowledge base does not contain sufficient relevant information.
    """
    retrieved = retrieve(question, chunks, top_k=top_k)

    if not retrieved or retrieved[0].get("score", 0.0) < RELEVANCE_THRESHOLD:
        return INSUFFICIENT_INFO_RESPONSE

    context_block = _build_context_block(retrieved)
    return _generate_answer(question, context_block)


# ---------------------------------------------------------------------------
# Session logging
# ---------------------------------------------------------------------------


def _prompt_confidence() -> Confidence:
    """Interactively prompt the developer to self-rate their confidence.

    Loops until a valid value (``high``, ``medium``, or ``low``) is entered.

    Returns:
        The selected :class:`~mentor_schema.Confidence` enum value.
    """
    valid = {c.value for c in Confidence}
    while True:
        raw = input("Rate your confidence (high / medium / low): ").strip().lower()
        if raw in valid:
            return Confidence(raw)
        print(f"  Please enter one of: {', '.join(sorted(valid))}")


def save_session(
    entries: list[SessionEntry],
    output_path: str = "mentor_output.json",
) -> None:
    """Persist Q&A session entries to a JSON file.

    Wraps *entries* in a :class:`~mentor_schema.MentorOutput` and calls
    :meth:`~mentor_schema.MentorOutput.to_json_file`.

    Args:
        entries: Ordered list of :class:`~mentor_schema.SessionEntry` objects
            accumulated during the session.
        output_path: Destination path for the JSON file.  Defaults to
            ``"mentor_output.json"`` in the current working directory.
    """
    output = MentorOutput(session=entries)
    output.to_json_file(output_path)
    print(f"[mentor] Session saved -> {output_path} ({len(entries)} entries)")


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------


def run_session(
    mapper_output_path: str,
    repo_root: str,
    output_path: str = "mentor_output.json",
) -> None:
    """Run an interactive Mentor session on the command line.

    Steps:
    1. Build the in-memory knowledge base (once).
    2. Loop: accept developer questions, print answers.
    3. After each answer, ask the developer to self-rate confidence.
    4. Log each exchange as a :class:`~mentor_schema.SessionEntry`.
    5. On ``exit`` or ``quit``, call :func:`save_session` and return.

    Args:
        mapper_output_path: Path to ``mapper_output.json``.
        repo_root: Root directory of the repository being analysed.
        output_path: Where to write the session JSON on exit.
    """
    print("[mentor] Building knowledge base … (this may take a moment)")
    chunks = build_knowledge_base(mapper_output_path, repo_root)
    n_content = len(_content_chunks(chunks))
    print(f"[mentor] Knowledge base ready — {n_content} chunks indexed.\n")
    print("  Ask me anything about the codebase.")
    print("  Type 'exit' or 'quit' to end the session.\n")

    entries: list[SessionEntry] = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        answer = ask(question, chunks)
        print(f"\nMentor:\n{answer}\n")

        confidence = _prompt_confidence()
        timestamp = datetime.now(timezone.utc)

        entries.append(
            SessionEntry(
                question=question,
                answer=answer,
                developer_confidence=confidence,
                timestamp=timestamp,
            )
        )
        print()

    if entries:
        save_session(entries, output_path=output_path)
    else:
        print("[mentor] No questions answered — session not saved.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        _mapper_path = sys.argv[1]
        _repo_root = sys.argv[2]
    else:
        _mapper_path = input(
            "Path to mapper_output.json [mapper_output.json]: "
        ).strip() or "mapper_output.json"
        _repo_root = input("Repo root directory [.]: ").strip() or "."

    run_session(_mapper_path, _repo_root)