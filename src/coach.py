"""
Coach Agent for the Onboarding Copilot pipeline.
Owner: Zahra (Coach Agent)

Loads the outputs of the Mapper Agent (MapperOutput) and the Mentor Agent
(MentorOutput), generates a 3-question quiz testing concepts actually covered
in the session log, scores the developer's answers, and recommends one real
starter task from the Mapper's entry_points list.

Typical usage (CLI)::

    python coach.py \\
        --mapper-output mapper_output.json \\
        --mentor-output mentor_output.json \\
        --answers answers.json

Or from Python / an orchestrator::

    from coach import run_coach_session
    output = run_coach_session(mapper_output, mentor_output, answers)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from coach_schema import CoachOutput, QuizItem, TaskRecommendation, TopicVerdict
from mapper_schema import Confidence, EntryPoint, MapperOutput
from mentor_schema import MentorOutput, SessionEntry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_FILE: str = "coach_output.json"

# Minimum keyword hit-rate required to award "understood".
# A developer must match at least this fraction of expected keywords.
_UNDERSTOOD_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Quiz generation
# ---------------------------------------------------------------------------

def generate_quiz(session: list[SessionEntry], n: int = 3) -> list[QuizItem]:
    """Generate *n* quiz questions from the topics covered in *session*.

    Each quiz question is derived from one :class:`~mentor_schema.SessionEntry`
    in the session log.  The question is phrased to probe whether the developer
    understood the concept, and the expected_keywords are extracted from the
    Mentor's original answer so that scoring is grounded in real content.

    If the session contains fewer than *n* entries, all available entries are
    used and the returned list will be shorter than *n*.

    Args:
        session: Ordered list of :class:`~mentor_schema.SessionEntry` objects
                 from the Mentor session log.
        n:       Number of quiz questions to generate.  Defaults to 3.

    Returns:
        A list of :class:`~coach_schema.QuizItem` objects with empty
        ``developer_answer`` and a default ``needs_review`` verdict (both are
        filled in later by :func:`score_quiz`).
    """
    items: list[QuizItem] = []
    # Take at most n entries, spread across the session if possible.
    selected = _select_entries(session, n)

    for entry in selected:
        topic = _extract_topic(entry.question)
        quiz_question = _rephrase_as_quiz_question(entry.question, entry.answer)
        keywords = _extract_keywords(entry.answer)
        items.append(
            QuizItem(
                topic=topic,
                question=quiz_question,
                expected_keywords=keywords,
                developer_answer="",
                verdict=TopicVerdict.NEEDS_REVIEW,
            )
        )

    return items


def _select_entries(session: list[SessionEntry], n: int) -> list[SessionEntry]:
    """Return up to *n* entries spread evenly across *session*.

    Prioritises entries where the developer reported lower confidence, so the
    quiz targets weaker areas first.

    Args:
        session: Full session log.
        n:       Maximum number of entries to return.

    Returns:
        A sub-list of up to *n* :class:`~mentor_schema.SessionEntry` objects.
    """
    if not session:
        return []

    # Confidence priority: low > medium > high (we quiz weaker areas first).
    _confidence_rank = {
        Confidence.LOW: 0,
        Confidence.MEDIUM: 1,
        Confidence.HIGH: 2,
    }

    sorted_by_confidence = sorted(
        session,
        key=lambda e: _confidence_rank.get(e.developer_confidence, 1),
    )
    return sorted_by_confidence[:n]


def _extract_topic(question: str) -> str:
    """Derive a short topic label from a Mentor question string.

    Strips common filler phrases and returns the first 60 characters so that
    topic labels are concise enough to use as dict keys.

    Args:
        question: Full question text from the session log.

    Returns:
        A short, lowercased topic label string.
    """
    fillers = (
        "what is ", "what are ", "can you explain ", "how does ", "how do ",
        "describe ", "explain ", "tell me about ",
    )
    clean = question.strip().rstrip("?").lower()
    for filler in fillers:
        if clean.startswith(filler):
            clean = clean[len(filler):]
            break
    # Truncate to keep keys readable.
    return clean[:60].strip()


def _rephrase_as_quiz_question(question: str, answer: str) -> str:
    """Construct a targeted quiz question from the original Mentor Q&A pair.

    The quiz question asks the developer to demonstrate understanding of the
    concept rather than simply repeating the Mentor's words.

    Args:
        question: The Mentor's original question.
        answer:   The developer's original answer (used to anchor the concept).

    Returns:
        A quiz question string.
    """
    topic = _extract_topic(question)
    # Use the first sentence of the answer as a hint anchor if it is short.
    first_sentence = answer.split(".")[0].strip()
    hint = f" (hint: think about \"{first_sentence}\")" if len(first_sentence) < 80 else ""
    return f"In your own words, explain {topic}{hint}."


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from *text* for answer scoring.

    Filters out common English stop words and short tokens, returning
    lower-cased unique tokens that represent the core concepts.

    Args:
        text: Free-form text (typically a Mentor answer).

    Returns:
        A deduplicated list of lowercase keyword strings.
    """
    _STOP_WORDS: frozenset[str] = frozenset({
        "a", "an", "the", "is", "it", "in", "on", "of", "to", "and", "or",
        "for", "with", "that", "this", "are", "be", "by", "at", "as", "so",
        "we", "you", "can", "has", "have", "had", "not", "but", "its",
        "which", "was", "were", "they", "them", "their", "from", "will",
        "when", "what", "how", "does", "do", "if", "any", "all",
    })
    tokens = text.lower().split()
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        # Strip punctuation from both ends.
        clean = token.strip(".,;:!?\"'()")
        if len(clean) > 3 and clean not in _STOP_WORDS and clean not in seen:
            seen.add(clean)
            keywords.append(clean)
    return keywords


# ---------------------------------------------------------------------------
# Answer scoring
# ---------------------------------------------------------------------------

def score_quiz(
    quiz: list[QuizItem],
    answers: list[str],
) -> list[QuizItem]:
    """Fill in developer answers and assign a verdict to each :class:`~coach_schema.QuizItem`.

    Scoring is keyword-based: a developer's answer is compared against the
    ``expected_keywords`` list for that question.  If the fraction of matched
    keywords meets or exceeds :data:`_UNDERSTOOD_THRESHOLD`, the verdict is
    ``"understood"``; otherwise it is ``"needs_review"``.

    If *answers* has fewer entries than *quiz*, unanswered questions receive
    an empty string and default to ``"needs_review"``.  Extra answers beyond
    the quiz length are silently ignored.

    Args:
        quiz:    List of :class:`~coach_schema.QuizItem` objects as returned
                 by :func:`generate_quiz`.
        answers: Developer's answers in the same order as *quiz*.

    Returns:
        A new list of :class:`~coach_schema.QuizItem` objects with
        ``developer_answer`` and ``verdict`` populated.
    """
    scored: list[QuizItem] = []

    for idx, item in enumerate(quiz):
        raw_answer = answers[idx] if idx < len(answers) else ""
        verdict = _score_single_answer(raw_answer, item.expected_keywords)
        # Build a new QuizItem with the answer and verdict filled in.
        scored.append(
            QuizItem(
                topic=item.topic,
                question=item.question,
                expected_keywords=item.expected_keywords,
                developer_answer=raw_answer,
                verdict=verdict,
            )
        )

    return scored


def _score_single_answer(answer: str, expected_keywords: list[str]) -> TopicVerdict:
    """Assign a :class:`~coach_schema.TopicVerdict` to a single answer.

    Uses a simple keyword-overlap heuristic:

    - If *expected_keywords* is empty, any non-empty answer is ``"understood"``.
    - If *answer* is empty, the verdict is always ``"needs_review"``.
    - Otherwise, the fraction of matched keywords is compared against
      :data:`_UNDERSTOOD_THRESHOLD`.

    Args:
        answer:            The developer's answer string.
        expected_keywords: Keywords extracted from the original Mentor answer.

    Returns:
        :attr:`~coach_schema.TopicVerdict.UNDERSTOOD` or
        :attr:`~coach_schema.TopicVerdict.NEEDS_REVIEW`.
    """
    if not answer.strip():
        return TopicVerdict.NEEDS_REVIEW

    if not expected_keywords:
        # No keywords to check against — a non-empty answer is good enough.
        return TopicVerdict.UNDERSTOOD

    answer_tokens = set(answer.lower().split())
    matched = sum(
        1 for kw in expected_keywords if kw in answer_tokens
    )
    hit_rate = matched / len(expected_keywords)
    return (
        TopicVerdict.UNDERSTOOD
        if hit_rate >= _UNDERSTOOD_THRESHOLD
        else TopicVerdict.NEEDS_REVIEW
    )


# ---------------------------------------------------------------------------
# Task recommendation
# ---------------------------------------------------------------------------

def recommend_task(
    mapper_output: MapperOutput,
    scored_quiz: list[QuizItem],
) -> TaskRecommendation | None:
    """Select one starter task from ``mapper_output.entry_points``.

    Strategy:
    1. Collect topics the developer ``"understood"``.
    2. Walk ``entry_points`` in importance order (highest ``import_count``
       first, as Mapper already sorts them).
    3. For each entry point, look up its ``classification.reason`` in
       ``mapper_output.classification``.
    4. Prefer a file whose classification reason contains a word that overlaps
       with the understood topics.
    5. If no topic-matched file is found, fall back to the highest-ranked
       ``core_logic`` entry point; if none, use the overall top entry point.
    6. Return ``None`` if ``entry_points`` is empty.

    Args:
        mapper_output: The :class:`~mapper_schema.MapperOutput` from the Mapper
                       Agent, used for ``entry_points`` and ``classification``.
        scored_quiz:   The scored quiz items from :func:`score_quiz`.

    Returns:
        A :class:`~coach_schema.TaskRecommendation` or ``None``.
    """
    if not mapper_output.entry_points:
        return None

    understood_topics: list[str] = [
        item.topic
        for item in scored_quiz
        if item.verdict == TopicVerdict.UNDERSTOOD
    ]

    # Try to find a file whose classification reason overlaps with understood topics.
    candidate = _find_topic_matched_entry(
        mapper_output.entry_points,
        mapper_output.classification,
        understood_topics,
    )

    # Fallback 1: highest-ranked core_logic file.
    if candidate is None:
        candidate = _find_core_logic_entry(
            mapper_output.entry_points,
            mapper_output.classification,
        )

    # Fallback 2: whatever is at the top of the ranked list.
    if candidate is None:
        candidate = mapper_output.entry_points[0]

    reason = _build_reason(candidate, mapper_output, understood_topics)
    return TaskRecommendation(file=candidate.file, reason=reason)


def _find_topic_matched_entry(
    entry_points: list[EntryPoint],
    classification: dict,
    understood_topics: list[str],
) -> EntryPoint | None:
    """Return the first entry point whose classification reason overlaps with understood topics.

    Args:
        entry_points:      Ranked list of :class:`~mapper_schema.EntryPoint` objects.
        classification:    Path → :class:`~mapper_schema.ClassificationEntry` mapping.
        understood_topics: Topics the developer demonstrated understanding of.

    Returns:
        The first matching :class:`~mapper_schema.EntryPoint`, or ``None``.
    """
    if not understood_topics:
        return None

    topic_words: set[str] = set()
    for topic in understood_topics:
        topic_words.update(topic.lower().split())

    for ep in entry_points:
        cls_entry = classification.get(ep.file)
        if cls_entry is None:
            # Try prefix match — Mapper may classify the folder, not the file.
            for key, val in classification.items():
                if ep.file.startswith(key.rstrip("/")):
                    cls_entry = val
                    break

        if cls_entry is not None:
            reason_words = set(cls_entry.reason.lower().split())
            if topic_words & reason_words:
                return ep

    return None


def _find_core_logic_entry(
    entry_points: list[EntryPoint],
    classification: dict,
) -> EntryPoint | None:
    """Return the highest-ranked ``core_logic`` entry point.

    Args:
        entry_points:  Ranked list of :class:`~mapper_schema.EntryPoint` objects.
        classification: Path → :class:`~mapper_schema.ClassificationEntry` mapping.

    Returns:
        The first core_logic :class:`~mapper_schema.EntryPoint`, or ``None``.
    """
    for ep in entry_points:
        cls_entry = classification.get(ep.file)
        if cls_entry is None:
            for key, val in classification.items():
                if ep.file.startswith(key.rstrip("/")):
                    cls_entry = val
                    break
        if cls_entry is not None and cls_entry.category == "core_logic":
            return ep
    return None


def _build_reason(
    entry: EntryPoint,
    mapper_output: MapperOutput,
    understood_topics: list[str],
) -> str:
    """Compose a one-sentence explanation for recommending *entry*.

    Args:
        entry:             The recommended :class:`~mapper_schema.EntryPoint`.
        mapper_output:     Full :class:`~mapper_schema.MapperOutput` for context.
        understood_topics: Topics the developer understood (may be empty).

    Returns:
        A single-sentence string explaining the recommendation.
    """
    cls_entry = mapper_output.classification.get(entry.file)
    if cls_entry is None:
        for key, val in mapper_output.classification.items():
            if entry.file.startswith(key.rstrip("/")):
                cls_entry = val
                break

    category_label = cls_entry.category if cls_entry else "key"
    mapper_reason = cls_entry.reason if cls_entry else "it is a highly imported file"

    if understood_topics:
        topics_str = ", ".join(f'"{t}"' for t in understood_topics[:2])
        return (
            f"Start with `{entry.file}` — it is a {category_label} file "
            f"({mapper_reason}) and your quiz results show you already understand "
            f"{topics_str}, which maps directly to this file's responsibilities."
        )

    return (
        f"Start with `{entry.file}` — it is the most-imported {category_label} "
        f"file in the repository ({mapper_reason}), making it the best "
        f"entry point for a new contributor."
    )


# ---------------------------------------------------------------------------
# Top-level session runner
# ---------------------------------------------------------------------------

def run_coach_session(
    mapper_output: MapperOutput,
    mentor_output: MentorOutput,
    answers: list[str],
) -> CoachOutput:
    """Run a complete Coach session and return a :class:`~coach_schema.CoachOutput`.

    This is the main entry point for orchestrator scripts.  All logic is
    encapsulated here so the CLI and any future API wrapper can call it
    identically.

    Steps:
    1. Validate that the session log is non-empty.
    2. Generate 3 quiz questions from the session log.
    3. Score the developer's answers.
    4. Build the ``topic_scores`` mapping.
    5. Recommend one starter task from ``entry_points``.
    6. Return a fully populated :class:`~coach_schema.CoachOutput`.

    Args:
        mapper_output:  Loaded :class:`~mapper_schema.MapperOutput` from the
                        Mapper Agent.
        mentor_output:  Loaded :class:`~mentor_schema.MentorOutput` from the
                        Mentor Agent.
        answers:        Developer's quiz answers as an ordered list of strings.
                        May be shorter than 3 (unanswered items → needs_review)
                        or empty (all items → needs_review).

    Returns:
        A :class:`~coach_schema.CoachOutput` object.  On error, ``error`` is
        set and all other meaningful fields are empty/None.
    """
    # --- Guard: empty session log ---------------------------------------- #
    if not mentor_output.entries:
        return CoachOutput(
            repo_root=mapper_output.repo_root,
            error=(
                "Session log is empty — the Mentor Agent produced no "
                "question-answer entries.  Run the Mentor Agent first."
            ),
        )

    # --- Step 1: generate quiz ------------------------------------------- #
    quiz = generate_quiz(mentor_output.entries, n=3)

    # --- Step 2: score answers ------------------------------------------- #
    scored_quiz = score_quiz(quiz, answers)

    # --- Step 3: build topic_scores dict ---------------------------------- #
    topic_scores: dict[str, str] = {
        item.topic: item.verdict for item in scored_quiz
    }

    # --- Step 4: recommend one task --------------------------------------- #
    recommendation = recommend_task(mapper_output, scored_quiz)

    return CoachOutput(
        repo_root=mapper_output.repo_root,
        quiz=scored_quiz,
        topic_scores=topic_scores,
        recommended_task=recommendation,
        error="",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Coach Agent.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).

    Returns:
        Parsed :class:`argparse.Namespace` with attributes:
        ``mapper_output``, ``mentor_output``, ``answers``, ``output``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Coach Agent — generates a quiz from a Mentor session log, "
            "scores the developer's answers, and recommends a starter task."
        ),
    )
    parser.add_argument(
        "--mapper-output",
        metavar="FILE",
        default="mapper_output.json",
        help="Path to the Mapper Agent output JSON (default: mapper_output.json)",
    )
    parser.add_argument(
        "--mentor-output",
        metavar="FILE",
        default="mentor_output.json",
        help="Path to the Mentor Agent output JSON (default: mentor_output.json)",
    )
    parser.add_argument(
        "--answers",
        metavar="FILE",
        default=None,
        help=(
            "Path to a JSON file containing a list of answer strings, one per "
            "quiz question.  Omit (or leave empty) to score all questions as "
            "'needs_review'."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=OUTPUT_FILE,
        help=f"Where to write coach_output.json (default: {OUTPUT_FILE})",
    )
    return parser.parse_args(argv)


def _load_answers(path: str | None) -> list[str]:
    """Load developer answers from a JSON file.

    The file must contain a JSON array of strings.  Returns an empty list if
    *path* is ``None``, the file does not exist, or parsing fails.

    Args:
        path: File path to the answers JSON, or ``None``.

    Returns:
        A list of answer strings (possibly empty).
    """
    if path is None:
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return [str(item) for item in data]
        print(
            f"warning: answers file '{path}' is not a JSON array — "
            "treating all answers as empty.",
            file=sys.stderr,
        )
        return []
    except FileNotFoundError:
        print(
            f"warning: answers file '{path}' not found — "
            "treating all answers as empty.",
            file=sys.stderr,
        )
        return []
    except json.JSONDecodeError as exc:
        print(
            f"warning: could not parse answers file '{path}': {exc} — "
            "treating all answers as empty.",
            file=sys.stderr,
        )
        return []


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the Coach Agent.

    Loads Mapper and Mentor outputs, runs the coaching session, writes
    the result to disk, and prints the JSON to stdout.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).
    """
    args = parse_args(argv)

    # --- Load Mapper output ---------------------------------------------- #
    try:
        mapper_output = MapperOutput.from_json_file(args.mapper_output)
    except FileNotFoundError:
        error_output = CoachOutput(
            error=f"Mapper output file not found: '{args.mapper_output}'. "
                  "Run mapper.py first."
        )
        error_output.to_json_file(args.output)
        print(error_output.model_dump_json(indent=2))
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        error_output = CoachOutput(
            error=f"Failed to load Mapper output from '{args.mapper_output}': {exc}"
        )
        error_output.to_json_file(args.output)
        print(error_output.model_dump_json(indent=2))
        sys.exit(1)

    # --- Load Mentor output ---------------------------------------------- #
    try:
        mentor_output = MentorOutput.from_json_file(args.mentor_output)
    except FileNotFoundError:
        error_output = CoachOutput(
            repo_root=mapper_output.repo_root,
            error=f"Mentor output file not found: '{args.mentor_output}'. "
                  "Run the Mentor Agent first.",
        )
        error_output.to_json_file(args.output)
        print(error_output.model_dump_json(indent=2))
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        error_output = CoachOutput(
            repo_root=mapper_output.repo_root,
            error=f"Failed to load Mentor output from '{args.mentor_output}': {exc}",
        )
        error_output.to_json_file(args.output)
        print(error_output.model_dump_json(indent=2))
        sys.exit(1)

    # --- Load answers ---------------------------------------------------- #
    answers = _load_answers(args.answers)

    # --- Run session ----------------------------------------------------- #
    coach_output = run_coach_session(mapper_output, mentor_output, answers)

    # --- Write and print output ------------------------------------------ #
    coach_output.to_json_file(args.output)
    print(coach_output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
