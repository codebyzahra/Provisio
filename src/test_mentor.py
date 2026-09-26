"""
test_mentor.py
==============
Test suite for mentor.py — the Mentor Agent (RAG-based Q&A over Mapper output).

Run from the repo root:
    python -m pytest src/test_mentor.py -v

Or from src/:
    python -m pytest test_mentor.py -v

UPDATE (LLM integration): _generate_answer() now calls a live LLM (via Groq,
through llm_client.chat_completion()) when available, falling back to the
original deterministic string-template behavior if the LLM call fails or is
unavailable (e.g. no LLM_API_KEY set, network error). Tests in this file that
exercise ask() may therefore hit either path depending on environment and
network availability. All other components — the TF-IDF vectoriser, chunking,
and retrieval — remain 100% local, offline, and unchanged.

Mocking notes
-------------
- TF-IDF vectoriser  → NOT mocked. It is a local sklearn operation; instant.
- Answer generator   → NOT mocked. _generate_answer() may call a live LLM via
                       llm_client.chat_completion() if LLM_API_KEY is set and
                       reachable; otherwise it falls back to the original
                       deterministic string-template behavior. Tests assert on
                       grounding/fallback behavior (e.g. the exact
                       INSUFFICIENT_INFO_RESPONSE string), which holds under
                       either path, rather than asserting on exact answer text.
- File I/O           → Uses tempfile.TemporaryDirectory / NamedTemporaryFile;
                       no permanent files are written to the repo.
- mapper_output.json → A synthetic MINIMAL_MAPPER_JSON fixture (3 entry_points,
                       3 classification entries) is written to a temp dir so
                       tests are self-contained and never depend on the repo's
                       own mapper_output.json being present or up-to-date.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import textwrap
import unittest
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Ensure src/ is importable when the test is invoked from any working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from mentor import (
    INSUFFICIENT_INFO_RESPONSE,
    RELEVANCE_THRESHOLD,
    _chunk_mapper_output,
    _chunk_source_file,
    _content_chunks,
    _fixed_chunks,
    _python_chunks,
    ask,
    build_knowledge_base,
    retrieve,
    save_session,
)
from mentor_schema import Confidence, MentorOutput, SessionEntry
from mapper_schema import (
    Category,
    ClassificationEntry,
    EntryPoint,
    MapperOutput,
    SetupInfo,
)

# ===========================================================================
# Shared fixtures
# ===========================================================================

# ---------------------------------------------------------------------------
# MINIMAL_MAPPER_JSON
# A self-contained synthetic mapper_output.json compatible with the real
# mapper_schema.MapperOutput Pydantic model.
#
# Three entry-points and three classification entries are enough to exercise
# all code paths without needing a real repository checkout.
#
# Key known tokens that test_ask uses for its "answerable" questions:
#   - "auth_service" / "AuthService"  (entry-point file name)
#   - "core_logic"                    (classification category)
#   - "requirements.txt"             (setup dependencies_file)
# ---------------------------------------------------------------------------
MINIMAL_MAPPER_JSON: dict = {
    "repo_root": "/fake/repo",
    "generated_at": "2024-01-15T10:00:00",
    "entry_points": [
        {
            "file": "src/auth_service.py",
            "import_count": 8,
            "confidence": "high",
        },
        {
            "file": "src/user_model.py",
            "import_count": 5,
            "confidence": "medium",
        },
        {
            "file": "src/config.py",
            "import_count": 3,
            "confidence": "low",
        },
    ],
    "classification": {
        "src/auth_service.py": {
            "category": "core_logic",
            "reason": "folder 'src' → core_logic",
        },
        "src/user_model.py": {
            "category": "core_logic",
            "reason": "folder 'src' → core_logic",
        },
        "requirements.txt": {
            "category": "configuration",
            "reason": "filename matches 'requirements.txt' → configuration",
        },
    },
    "setup": {
        "language": "python",
        "dependencies_file": "requirements.txt",
        "run_steps": [
            "python -m venv venv",
            "pip install -r requirements.txt",
            "python src/auth_service.py",
        ],
    },
    "unclear_items": [],
}

# Minimal Python source that will be written alongside the mapper fixture so
# build_knowledge_base() can read the classified source files.
AUTH_SERVICE_SRC = textwrap.dedent(
    """\
    \"\"\"auth_service.py — handles user authentication.\"\"\"

    class AuthService:
        \"\"\"Validates credentials and issues session tokens.\"\"\"

        def login(self, username: str, password: str) -> str:
            \"\"\"Return a session token for valid credentials.\"\"\"
            return f"token-{username}"

        def logout(self, token: str) -> None:
            \"\"\"Invalidate an existing session token.\"\"\"
            pass
    """
)

USER_MODEL_SRC = textwrap.dedent(
    """\
    \"\"\"user_model.py — data model for application users.\"\"\"

    class User:
        def __init__(self, user_id: int, email: str) -> None:
            self.user_id = user_id
            self.email = email
    """
)

REQUIREMENTS_TXT = "fastapi\npydantic\nscikit-learn\nnumpy\n"


# ---------------------------------------------------------------------------
# Helper: build a self-contained temp directory with the fixture files
# ---------------------------------------------------------------------------

def _build_fixture_dir(tmp_dir: str) -> tuple[str, str]:
    """Write MINIMAL_MAPPER_JSON and matching source files into *tmp_dir*.

    Returns:
        (mapper_output_path, repo_root) — both suitable for passing to
        build_knowledge_base().
    """
    # Write mapper JSON
    mapper_path = os.path.join(tmp_dir, "mapper_output.json")
    with open(mapper_path, "w", encoding="utf-8") as fh:
        json.dump(MINIMAL_MAPPER_JSON, fh, indent=2)

    # Write source files so build_knowledge_base() can ingest them
    src_dir = os.path.join(tmp_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    with open(os.path.join(src_dir, "auth_service.py"), "w", encoding="utf-8") as fh:
        fh.write(AUTH_SERVICE_SRC)
    with open(os.path.join(src_dir, "user_model.py"), "w", encoding="utf-8") as fh:
        fh.write(USER_MODEL_SRC)
    with open(os.path.join(tmp_dir, "requirements.txt"), "w", encoding="utf-8") as fh:
        fh.write(REQUIREMENTS_TXT)

    return mapper_path, tmp_dir


# ===========================================================================
# Low-level chunking helpers
# ===========================================================================

SAMPLE_PYTHON = textwrap.dedent(
    """\
    import os

    CONSTANT = 42

    def greet(name: str) -> str:
        \"\"\"Return a greeting.\"\"\"
        return f"Hello, {name}"

    class Greeter:
        def say_hi(self, name: str) -> str:
            return greet(name)
    """
)

SAMPLE_TEXT = "This is a plain text file with no Python syntax.\n" * 40


class TestFixedChunks(unittest.TestCase):
    """_fixed_chunks() — overlapping fixed-size text splitter."""

    def test_produces_chunks(self):
        chunks = _fixed_chunks(SAMPLE_TEXT, "readme.txt", "code")
        self.assertGreater(len(chunks), 0)

    def test_chunk_fields_present(self):
        chunks = _fixed_chunks(
            "hello world " * 100, "x.txt", "code", chunk_size=50, overlap=10
        )
        for c in chunks:
            self.assertIn("text", c)
            self.assertIn("source_path", c)
            self.assertIn("source_type", c)
            self.assertEqual(c["source_type"], "code")
            self.assertEqual(c["source_path"], "x.txt")

    def test_overlap_produces_correct_chunk_count(self):
        # step = chunk_size - overlap = 100 - 50 = 50 → 4 chunks for 200 chars
        chunks = _fixed_chunks("A" * 200, "f.txt", "code", chunk_size=100, overlap=50)
        self.assertEqual(len(chunks), 4)

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(_fixed_chunks("", "empty.txt", "code"), [])


class TestPythonChunks(unittest.TestCase):
    """_python_chunks() — AST-based per-definition chunker."""

    def test_function_becomes_chunk(self):
        chunks = _python_chunks(SAMPLE_PYTHON, "sample.py")
        self.assertTrue(any("greet" in c["text"] for c in chunks))

    def test_class_becomes_chunk(self):
        chunks = _python_chunks(SAMPLE_PYTHON, "sample.py")
        self.assertTrue(any("Greeter" in c["text"] for c in chunks))

    def test_preamble_chunk_present(self):
        chunks = _python_chunks(SAMPLE_PYTHON, "sample.py")
        self.assertTrue(any("preamble" in c["text"] for c in chunks))

    def test_syntax_error_falls_back_to_fixed(self):
        bad = "def broken(\n  pass"
        chunks = _python_chunks(bad, "bad.py")
        self.assertGreater(len(chunks), 0)

    def test_all_chunks_have_required_keys(self):
        chunks = _python_chunks(SAMPLE_PYTHON, "sample.py")
        for c in chunks:
            self.assertIn("text", c)
            self.assertIn("source_path", c)
            self.assertIn("source_type", c)
            self.assertEqual(c["source_type"], "code")


class TestChunkSourceFile(unittest.TestCase):
    """_chunk_source_file() — dispatcher between AST and fixed chunkers."""

    def test_py_extension_uses_ast_chunker(self):
        chunks = _chunk_source_file(SAMPLE_PYTHON, "s.py")
        self.assertTrue(any("greet" in c["text"] for c in chunks))

    def test_non_py_uses_fixed_chunker(self):
        chunks = _chunk_source_file(SAMPLE_TEXT, "readme.md")
        for c in chunks:
            self.assertEqual(c["source_type"], "code")


# ===========================================================================
# Mapper-output chunker
# ===========================================================================

class TestChunkMapperOutput(unittest.TestCase):
    """_chunk_mapper_output() — converts MapperOutput to text chunks.

    Uses MapperOutput.model_validate() (Pydantic v2) instead of from_dict().
    """

    def setUp(self):
        # Build the MapperOutput object directly from MINIMAL_MAPPER_JSON
        # using Pydantic's model_validate — no from_dict() needed.
        self.mo = MapperOutput.model_validate(MINIMAL_MAPPER_JSON)

    def test_entry_point_chunks_produced(self):
        chunks = _chunk_mapper_output(self.mo)
        texts = [c["text"] for c in chunks]
        # Each entry-point file name should appear in at least one chunk
        self.assertTrue(any("auth_service" in t for t in texts))
        self.assertTrue(any("Entry-point" in t for t in texts))

    def test_classification_chunk_produced(self):
        chunks = _chunk_mapper_output(self.mo)
        texts = [c["text"] for c in chunks]
        self.assertTrue(any("classification" in t.lower() for t in texts))

    def test_classification_chunk_contains_known_entry(self):
        chunks = _chunk_mapper_output(self.mo)
        texts = [c["text"] for c in chunks]
        self.assertTrue(any("auth_service.py" in t for t in texts))
        self.assertTrue(any("core_logic" in t for t in texts))

    def test_setup_info_chunk_produced(self):
        chunks = _chunk_mapper_output(self.mo)
        texts = [c["text"] for c in chunks]
        self.assertTrue(any("requirements.txt" in t for t in texts))
        self.assertTrue(any("python" in t.lower() for t in texts))

    def test_all_chunks_have_mapper_summary_type(self):
        chunks = _chunk_mapper_output(self.mo)
        for c in chunks:
            self.assertEqual(c["source_type"], "mapper_summary")

    def test_number_of_chunks_matches_structure(self):
        # 3 entry-point chunks + 1 classification chunk + 1 setup chunk = 5
        chunks = _chunk_mapper_output(self.mo)
        self.assertEqual(len(chunks), 5)


# ===========================================================================
# Knowledge-base construction
# ===========================================================================

class TestBuildKnowledgeBase(unittest.TestCase):
    """build_knowledge_base() — end-to-end ingestion + TF-IDF vectorisation.

    No LLM or neural embeddings involved — this stage (chunking + TF-IDF
    vectorisation) is unchanged by the LLM integration and remains pure
    sklearn TF-IDF. Tests run offline, instantly, at zero cost.
    """

    def setUp(self):
        self._tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmp = self._tmpdir_obj.name
        self.mapper_path, self.repo_root = _build_fixture_dir(self.tmp)

    def tearDown(self):
        self._tmpdir_obj.cleanup()

    def test_returns_non_empty_list(self):
        """Knowledge base must be a non-empty list (the 'vector store')."""
        chunks = build_knowledge_base(self.mapper_path, self.repo_root)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

    def test_content_chunks_non_empty(self):
        """After filtering the internal sentinel, content chunks must exist."""
        chunks = build_knowledge_base(self.mapper_path, self.repo_root)
        content = _content_chunks(chunks)
        self.assertGreater(len(content), 0)

    def test_every_content_chunk_has_tfidf_vector(self):
        """Every content chunk must carry a numpy TF-IDF vector."""
        import numpy as np

        chunks = build_knowledge_base(self.mapper_path, self.repo_root)
        for c in _content_chunks(chunks):
            self.assertIn("vector", c, msg=f"chunk missing 'vector': {c}")
            self.assertIsInstance(c["vector"], np.ndarray)
            self.assertGreater(c["vector"].shape[0], 0)

    def test_vectorizer_sentinel_appended(self):
        """Last element must be the internal vectorizer sentinel."""
        chunks = build_knowledge_base(self.mapper_path, self.repo_root)
        self.assertIn("__vectorizer__", chunks[-1])

    def test_mapper_summary_chunks_present(self):
        """At least one chunk must originate from the mapper summary."""
        chunks = build_knowledge_base(self.mapper_path, self.repo_root)
        types = [c.get("source_type") for c in _content_chunks(chunks)]
        self.assertIn("mapper_summary", types)

    def test_code_chunks_present(self):
        """Source files listed in classification must produce 'code' chunks."""
        chunks = build_knowledge_base(self.mapper_path, self.repo_root)
        types = [c.get("source_type") for c in _content_chunks(chunks)]
        self.assertIn("code", types)

    def test_missing_source_file_is_skipped_gracefully(self):
        """A classification entry whose file doesn't exist must not raise."""
        data = dict(MINIMAL_MAPPER_JSON)
        data = json.loads(json.dumps(data))  # deep copy via JSON
        data["classification"]["src/ghost_file.py"] = {
            "category": "core_logic",
            "reason": "folder 'src' → core_logic",
        }
        ghost_mapper = os.path.join(self.tmp, "mapper_ghost.json")
        with open(ghost_mapper, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        # Should NOT raise FileNotFoundError, just print a warning and continue
        chunks = build_knowledge_base(ghost_mapper, self.repo_root)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)


# ===========================================================================
# Retrieval
# ===========================================================================

class TestRetrieve(unittest.TestCase):
    """retrieve() — TF-IDF cosine-similarity retrieval (offline, free)."""

    def setUp(self):
        self._tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmp = self._tmpdir_obj.name
        mapper_path, repo_root = _build_fixture_dir(self.tmp)
        self.chunks = build_knowledge_base(mapper_path, repo_root)

    def tearDown(self):
        self._tmpdir_obj.cleanup()

    def test_returns_list(self):
        results = retrieve("AuthService login token", self.chunks)
        self.assertIsInstance(results, list)

    def test_top_k_upper_bound(self):
        results = retrieve("auth service", self.chunks, top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_every_result_has_score(self):
        results = retrieve("authentication", self.chunks)
        for r in results:
            self.assertIn("score", r)
            self.assertGreaterEqual(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.0)

    def test_results_sorted_descending_by_score(self):
        results = retrieve("auth_service login session token", self.chunks, top_k=5)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_raw_vector_stripped_from_results(self):
        """Vectors must not leak into retrieve() return values."""
        results = retrieve("auth", self.chunks, top_k=1)
        for r in results:
            self.assertNotIn("vector", r)


# ===========================================================================
# ask() — grounded Q&A
# ===========================================================================

class TestAsk(unittest.TestCase):
    """ask() — RAG answer generation.

    _generate_answer() now calls a live LLM (via Groq, through
    llm_client.chat_completion()) when LLM_API_KEY is set and the call
    succeeds, falling back to the original deterministic string-template
    behavior if the LLM call fails or is unavailable. These tests assert on
    grounding/fallback behavior — the exact INSUFFICIENT_INFO_RESPONSE
    string, and the presence of known KB tokens in a grounded answer — both
    of which hold true regardless of which generation path is taken. Tests
    do NOT assert on exact answer wording, since that varies with the LLM.
    """

    def setUp(self):
        self._tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmp = self._tmpdir_obj.name
        mapper_path, repo_root = _build_fixture_dir(self.tmp)
        self.chunks = build_knowledge_base(mapper_path, repo_root)

    def tearDown(self):
        self._tmpdir_obj.cleanup()

    # --- answerable questions -----------------------------------------------

    def test_answerable_question_returns_non_empty_string(self):
        """A question with tokens present in the KB must return a non-empty answer."""
        answer = ask("What is auth_service.py classified as?", self.chunks)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    def test_answerable_question_is_grounded(self):
        """Grounded answer must mention the queried file or its category."""
        answer = ask("What is auth_service.py classified as?", self.chunks)
        # The answer must NOT be the fallback string
        self.assertNotEqual(
            answer,
            INSUFFICIENT_INFO_RESPONSE,
            msg="Expected a grounded answer but got the fallback response.",
        )
        # The answer should reference at least one known term from the KB
        known_tokens = ("auth_service", "core_logic", "AuthService")
        self.assertTrue(
            any(tok in answer for tok in known_tokens),
            msg=f"Answer did not contain any known KB token.\nAnswer: {answer}",
        )

    def test_answerable_question_about_entry_point(self):
        """Asking about a known entry-point file produces a grounded answer."""
        answer = ask("What are the entry points in this project?", self.chunks)
        self.assertNotEqual(answer, INSUFFICIENT_INFO_RESPONSE)
        self.assertGreater(len(answer), 0)

    def test_answerable_question_about_setup(self):
        """Asking about project setup produces a grounded answer."""
        answer = ask("How do I install dependencies using requirements.txt?", self.chunks)
        self.assertNotEqual(answer, INSUFFICIENT_INFO_RESPONSE)
        self.assertIn("requirements.txt", answer)

    # --- unanswerable questions ---------------------------------------------

    def test_unanswerable_question_returns_exact_fallback(self):
        """A question with zero KB overlap must return the exact fallback string.

        This check happens in ask() BEFORE any LLM call is made (the
        RELEVANCE_THRESHOLD gate short-circuits generation entirely), so this
        assertion holds regardless of LLM availability.
        """
        # Uses tokens guaranteed to be absent from the fixture KB
        answer = ask(
            "xyzzy frobnicate the quantum blockchain microservice",
            self.chunks,
        )
        self.assertEqual(
            answer,
            INSUFFICIENT_INFO_RESPONSE,
            msg=(
                "Expected the exact fallback string for an out-of-KB question.\n"
                f"Got: {answer!r}"
            ),
        )

    def test_unanswerable_question_about_nonexistent_file(self):
        """Asking about a file that is NOT in the sample data returns the fallback."""
        answer = ask(
            "What does the PaymentGatewayController do?",
            self.chunks,
        )
        # PaymentGatewayController appears nowhere in MINIMAL_MAPPER_JSON
        self.assertEqual(answer, INSUFFICIENT_INFO_RESPONSE)

    # --- parameter forwarding -----------------------------------------------

    def test_top_k_one_does_not_raise(self):
        answer = ask("authentication login", self.chunks, top_k=1)
        self.assertIsInstance(answer, str)


# ===========================================================================
# SessionEntry / MentorOutput schema validation
# ===========================================================================

class TestSessionEntrySchema(unittest.TestCase):
    """Validate SessionEntry field names and Confidence enum against mentor_schema.py."""

    def test_field_names_on_session_entry(self):
        """SessionEntry must have exactly: question, answer, developer_confidence, timestamp."""
        entry = SessionEntry(
            question="What does AuthService do?",
            answer="It handles login and logout.",
            developer_confidence=Confidence.HIGH,
            timestamp="2024-06-01T12:00:00",
        )
        self.assertEqual(entry.question, "What does AuthService do?")
        self.assertEqual(entry.answer, "It handles login and logout.")
        self.assertEqual(entry.developer_confidence, Confidence.HIGH)
        self.assertEqual(entry.timestamp, "2024-06-01T12:00:00")

    def test_confidence_enum_values(self):
        """Confidence enum must expose 'high', 'medium', 'low' string values."""
        self.assertEqual(Confidence.HIGH.value, "high")
        self.assertEqual(Confidence.MEDIUM.value, "medium")
        self.assertEqual(Confidence.LOW.value, "low")

    def test_confidence_all_valid_values_constructible(self):
        for raw in ("high", "medium", "low"):
            c = Confidence(raw)
            self.assertIsInstance(c, Confidence)

    def test_timestamp_is_iso8601_string(self):
        """Timestamps stored on SessionEntry must be parseable as ISO-8601."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        entry = SessionEntry(
            question="q", answer="a",
            developer_confidence=Confidence.MEDIUM,
            timestamp=ts,
        )
        # Must parse without raising ValueError
        parsed = datetime.fromisoformat(entry.timestamp)
        self.assertIsNotNone(parsed)

    def test_session_entry_to_dict_and_back(self):
        entry = SessionEntry(
            question="q",
            answer="a",
            developer_confidence=Confidence.LOW,
            timestamp="2024-01-01T00:00:00",
        )
        d = entry.to_dict()
        restored = SessionEntry.from_dict(d)
        self.assertEqual(restored.question, entry.question)
        self.assertEqual(restored.answer, entry.answer)
        self.assertEqual(restored.developer_confidence, Confidence.LOW)
        self.assertEqual(restored.timestamp, entry.timestamp)


class TestMentorOutputSchema(unittest.TestCase):
    """Validate MentorOutput field names: 'entries' and 'session_metadata'."""

    def _make_entries(self) -> list[SessionEntry]:
        return [
            SessionEntry(
                question="What is auth_service.py?",
                answer="It handles authentication.",
                developer_confidence=Confidence.HIGH,
                timestamp="2024-06-01T09:00:00",
            ),
            SessionEntry(
                question="What language is used?",
                answer="Python.",
                developer_confidence=Confidence.MEDIUM,
                timestamp="2024-06-01T09:05:00",
            ),
        ]

    def test_field_name_entries(self):
        """MentorOutput must expose an 'entries' attribute (list of SessionEntry)."""
        entries = self._make_entries()
        mo = MentorOutput(entries=entries)
        self.assertTrue(hasattr(mo, "entries"))
        self.assertEqual(len(mo.entries), 2)

    def test_field_name_session_metadata(self):
        """MentorOutput must expose a 'session_metadata' attribute (dict)."""
        mo = MentorOutput(entries=[], session_metadata={"model": "tfidf"})
        self.assertTrue(hasattr(mo, "session_metadata"))
        self.assertEqual(mo.session_metadata["model"], "tfidf")

    def test_to_dict_has_correct_top_level_keys(self):
        mo = MentorOutput(
            entries=self._make_entries(),
            session_metadata={"total_questions": 2},
        )
        d = mo.to_dict()
        self.assertIn("entries", d)
        self.assertIn("session_metadata", d)

    def test_from_dict_roundtrip(self):
        entries = self._make_entries()
        mo = MentorOutput(
            entries=entries,
            session_metadata={"total_questions": len(entries)},
        )
        d = mo.to_dict()
        restored = MentorOutput.from_dict(d)
        self.assertEqual(len(restored.entries), len(entries))
        self.assertEqual(restored.entries[0].question, "What is auth_service.py?")
        self.assertEqual(
            restored.entries[1].developer_confidence, Confidence.MEDIUM
        )
        self.assertEqual(
            restored.session_metadata["total_questions"],
            len(entries),
        )


# ===========================================================================
# save_session() → mentor_output.json round-trip
# ===========================================================================

class TestSaveSessionRoundTrip(unittest.TestCase):
    """save_session() must produce a JSON file that round-trips through
    MentorOutput.from_json_file() with no data loss."""

    def _make_entries(self) -> list[SessionEntry]:
        return [
            SessionEntry(
                question="What does AuthService do?",
                answer="Handles user authentication.",
                developer_confidence=Confidence.HIGH,
                timestamp="2024-06-01T10:00:00",
            ),
            SessionEntry(
                question="Which files are entry points?",
                answer="src/auth_service.py is an entry point.",
                developer_confidence=Confidence.MEDIUM,
                timestamp="2024-06-01T10:05:00",
            ),
        ]

    def test_save_then_load_preserves_entries(self):
        entries = self._make_entries()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            out_path = fh.name
        try:
            save_session(entries, output_path=out_path)
            loaded = MentorOutput.from_json_file(out_path)

            self.assertEqual(
                len(loaded.entries),
                len(entries),
                msg="Round-tripped MentorOutput has wrong number of entries.",
            )
            self.assertEqual(
                loaded.entries[0].question, entries[0].question
            )
            self.assertEqual(
                loaded.entries[0].developer_confidence, Confidence.HIGH
            )
            self.assertEqual(
                loaded.entries[1].developer_confidence, Confidence.MEDIUM
            )
        finally:
            os.unlink(out_path)

    def test_save_adds_session_metadata(self):
        """save_session() must attach 'total_questions' in session_metadata."""
        entries = self._make_entries()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            out_path = fh.name
        try:
            save_session(entries, output_path=out_path)
            loaded = MentorOutput.from_json_file(out_path)
            self.assertIn("total_questions", loaded.session_metadata)
            self.assertEqual(
                loaded.session_metadata["total_questions"], len(entries)
            )
        finally:
            os.unlink(out_path)

    def test_save_empty_session_produces_valid_json(self):
        """An empty session must produce a valid, loadable JSON file."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            out_path = fh.name
        try:
            save_session([], output_path=out_path)
            loaded = MentorOutput.from_json_file(out_path)
            self.assertEqual(loaded.entries, [])
            self.assertIn("total_questions", loaded.session_metadata)
            self.assertEqual(loaded.session_metadata["total_questions"], 0)
        finally:
            os.unlink(out_path)

    def test_output_file_is_valid_json(self):
        """The written file must be parseable as raw JSON."""
        entries = self._make_entries()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            out_path = fh.name
        try:
            save_session(entries, output_path=out_path)
            with open(out_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.assertIn("entries", raw)
            self.assertIn("session_metadata", raw)
        finally:
            os.unlink(out_path)

    def test_confidence_values_serialised_as_strings(self):
        """developer_confidence must be stored as a plain string in JSON."""
        entries = [
            SessionEntry(
                question="q",
                answer="a",
                developer_confidence=Confidence.LOW,
                timestamp="2024-01-01T00:00:00",
            )
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            out_path = fh.name
        try:
            save_session(entries, output_path=out_path)
            with open(out_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            conf_val = raw["entries"][0]["developer_confidence"]
            self.assertIsInstance(conf_val, str)
            self.assertEqual(conf_val, "low")
        finally:
            os.unlink(out_path)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)