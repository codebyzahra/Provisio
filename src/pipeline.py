"""
pipeline.py
===========
Orchestrator for the Onboarding Copilot pipeline.

Runs Mapper → Mentor → Coach as direct in-process function calls (no subprocess,
no stdin injection).  Each stage's output is validated against its schema before
being passed to the next stage.  On validation failure, one clear error line is
printed and the process exits cleanly.

Usage::

    python src/pipeline.py --path /path/to/repo
    python src/pipeline.py --url https://github.com/org/repo
    python src/pipeline.py --path /path/to/repo --output-dir /tmp/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import NoReturn

from pydantic import ValidationError

import coach
import mapper
import mentor
from coach_schema import CoachOutput
from llm_client import chat_completion
from mapper_schema import MapperOutput
from mentor_schema import Confidence, MentorOutput, SessionEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_onboarding_questions(mapper_output: MapperOutput) -> list[str]:
    """Generate 3–5 repo-specific onboarding questions via the LLM.

    Builds a prompt from the mapper output — including the repo root, detected
    language, top entry points, a sample of the file classification map, and
    the project run steps — and asks the LLM to return a JSON array of
    question strings tailored to a new developer on this specific repository.

    **Fallback behaviour:** if the LLM call raises any exception, or if JSON
    parsing fails after a cleanup retry, a small hardcoded list of generic
    onboarding questions is returned so the pipeline continues uninterrupted.

    Args:
        mapper_output: Validated :class:`~mapper_schema.MapperOutput` produced
            by Stage 1 of the pipeline.

    Returns:
        A list of 3–5 question strings.
    """
    _FALLBACK_QUESTIONS: list[str] = [
        "What are the main entry points of this project and which file should I open first?",
        "How do I install the dependencies and run this project locally?",
        "How is the test suite organised and how do I run the tests?",
        "What is the overall purpose and architecture of this codebase?",
        "Which directories contain the core business logic versus configuration or tooling?",
    ]

    # Build a concise representation of the top 5 entry points.
    top_entry_points = mapper_output.entry_points[:5]
    ep_lines = "\n".join(
        f"  - {ep.file} (import_count: {ep.import_count})"
        for ep in top_entry_points
    )

    # Sample up to 10 classification entries.
    classification_sample = list(mapper_output.classification.items())[:10]
    cls_lines = "\n".join(
        f"  - {path}: [{entry.category}] {entry.reason}"
        for path, entry in classification_sample
    )

    run_steps = mapper_output.setup.run_steps if mapper_output.setup else []
    run_steps_str = "\n".join(f"  - {s}" for s in run_steps) or "  (none)"
    language = mapper_output.setup.language if mapper_output.setup else "unknown"

    system_message: str = (
        "You are an onboarding assistant for software developers. "
        "When asked, reply ONLY with a valid JSON array of question strings — "
        "no prose, no markdown code fences, no extra keys. "
        "Example of a valid reply: "
        '[\"Question one?\", \"Question two?\", \"Question three?\"]'
    )
    user_message: str = (
        f"Repository root: {mapper_output.repo_root}\n"
        f"Primary language: {language}\n\n"
        f"Top entry points:\n{ep_lines}\n\n"
        f"File classification sample (up to 10 files):\n{cls_lines}\n\n"
        f"Run steps:\n{run_steps_str}\n\n"
        "Generate 3 to 5 specific onboarding questions that a new developer "
        "on THIS repository should be able to answer after their first day. "
        "Reply with a JSON array of question strings only."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    try:
        raw = chat_completion(messages)
    except Exception:  # noqa: BLE001
        return _FALLBACK_QUESTIONS

    # First parse attempt.
    try:
        questions: list[str] = json.loads(raw)
        return questions
    except (json.JSONDecodeError, ValueError):
        pass

    # Cleanup retry: strip whitespace and markdown code fences, then retry.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        questions = json.loads(cleaned)
        return questions
    except (json.JSONDecodeError, ValueError):
        return _FALLBACK_QUESTIONS


def _fail(stage: str, reason: str) -> NoReturn:
    """Print one clear error line and exit with code 1.

    Never raises; always terminates the process so no raw traceback leaks out.
    """
    print(f"Pipeline failed at {stage}: {reason}", file=sys.stderr)
    sys.exit(1)


def _output_path(output_dir: str, filename: str) -> str:
    """Join *output_dir* and *filename*, creating the directory if needed."""
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _run_mapper(args: argparse.Namespace, output_dir: str) -> MapperOutput:
    """Stage 1: run Mapper and return a validated MapperOutput."""
    mapper_argv = ["--path", args.path] if args.path else ["--url", args.url]
    mapper_argv += ["--output-dir", output_dir]

    try:
        mapper_output = mapper.main(mapper_argv)
    except SystemExit as exc:
        _fail("Mapper", f"mapper.main() exited with code {exc.code}")
    except Exception as exc:  # noqa: BLE001
        _fail("Mapper", str(exc))

    dest = _output_path(output_dir, "mapper_output.json")

    # Validate by round-tripping through the file (per plan requirement).
    try:
        validated = MapperOutput.from_json_file(dest)
    except (ValidationError, Exception) as exc:  # noqa: BLE001
        _fail("Mapper (validation)", str(exc))

    n_files = len(validated.classification)
    n_entries = len(validated.entry_points)
    print(f"Step 1: Mapper analysed {n_files} files, found {n_entries} entry points")

    return validated


def _run_mentor(
    mapper_output: MapperOutput,
    output_dir: str,
) -> MentorOutput:
    """Stage 2: build knowledge base, ask fixed questions, return MentorOutput."""
    mapper_json = _output_path(output_dir, "mapper_output.json")

    try:
        chunks = mentor.build_knowledge_base(mapper_json, mapper_output.repo_root)
    except Exception as exc:  # noqa: BLE001
        _fail("Mentor (build_knowledge_base)", str(exc))

    entries: list[SessionEntry] = []
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    for question in generate_onboarding_questions(mapper_output):
        try:
            answer = mentor.ask(question, chunks)
        except Exception as exc:  # noqa: BLE001
            _fail("Mentor (ask)", str(exc))

        confidence = (
            Confidence.LOW
            if answer == mentor.INSUFFICIENT_INFO_RESPONSE
            else Confidence.MEDIUM
        )
        entries.append(
            SessionEntry(
                question=question,
                answer=answer,
                developer_confidence=confidence,
                timestamp=now_str,
            )
        )

    # Validate: assert structural integrity of the dataclass output.
    try:
        mentor_output = MentorOutput(entries=entries)
        assert isinstance(mentor_output.entries, list), "entries must be a list"
        for e in mentor_output.entries:
            _ = e.question, e.answer, e.developer_confidence, e.timestamp
    except (AttributeError, TypeError, AssertionError) as exc:
        _fail("Mentor (validation)", str(exc))

    dest = _output_path(output_dir, "mentor_output.json")
    try:
        mentor.save_session(entries, dest)
    except Exception as exc:  # noqa: BLE001
        _fail("Mentor (save_session)", str(exc))

    sufficient = sum(
        1 for e in entries if e.answer != mentor.INSUFFICIENT_INFO_RESPONSE
    )
    print(
        f"Step 2: Mentor answered {len(entries)} questions "
        f"({sufficient} with sufficient context)"
    )

    return mentor_output


def _run_coach(
    mapper_output: MapperOutput,
    mentor_output: MentorOutput,
    output_dir: str,
) -> CoachOutput:
    """Stage 3: run Coach session and return a validated CoachOutput."""
    try:
        coach_output = coach.run_coach_session(
            mapper_output=mapper_output,
            mentor_output=mentor_output,
            answers=[],
        )
    except Exception as exc:  # noqa: BLE001
        _fail("Coach", str(exc))

    if coach_output.error:
        _fail("Coach", coach_output.error)

    try:
        CoachOutput.model_validate(coach_output.model_dump())
    except ValidationError as exc:
        _fail("Coach (validation)", str(exc))

    dest = _output_path(output_dir, "coach_output.json")
    try:
        coach_output.to_json_file(dest)
    except Exception as exc:  # noqa: BLE001
        _fail("Coach (save)", str(exc))

    n_quiz = len(coach_output.quiz)
    rec_file = (
        coach_output.recommended_task.file
        if coach_output.recommended_task
        else "none"
    )
    verdicts = ", ".join(
        f"{t}:{v}" for t, v in coach_output.topic_scores.items()
    ) or "no scores"
    print(
        f"Step 3: Coach generated {n_quiz} quiz items; "
        f"recommended task: {rec_file} ({verdicts})"
    )

    return coach_output


# ---------------------------------------------------------------------------
# Programmatic entry point (used by app.py)
# ---------------------------------------------------------------------------


def run_pipeline(target: str, output_dir: str = ".") -> None:
    """Run the full Mapper → Mentor → Coach pipeline from Python.

    This is the callable entry point for the Streamlit dashboard.  It mirrors
    what ``main()`` does via the CLI but accepts plain strings rather than an
    ``argparse.Namespace``, so callers do not need to touch ``sys.argv``.

    Args:
        target: A local repository path or a remote git URL.
        output_dir: Directory where JSON output files will be written.
                    Defaults to the current working directory.

    Raises:
        SystemExit: Propagated from ``_fail()`` on any stage error so the
                    caller can catch it and inspect ``exc.code`` / stderr.
    """
    is_url = target.startswith(("http://", "https://", "git@", "git://"))
    args = argparse.Namespace(
        path=None if is_url else target,
        url=target if is_url else None,
        output_dir=output_dir,
    )

    mapper_output = _run_mapper(args, output_dir)
    mentor_output = _run_mentor(mapper_output, output_dir)
    _run_coach(mapper_output, mentor_output, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse pipeline CLI arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with ``.path``, ``.url``, and ``.output_dir``.
    """
    parser = argparse.ArgumentParser(
        description="Provisio onboarding pipeline: Mapper → Mentor → Coach.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", metavar="DIR", help="Local repository directory.")
    source.add_argument("--url", metavar="URL", help="Remote git URL to clone.")
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=".",
        help="Directory for JSON output files (default: current directory).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the full Mapper → Mentor → Coach pipeline."""
    args = parse_args(argv)
    output_dir = args.output_dir

    mapper_output = _run_mapper(args, output_dir)
    mentor_output = _run_mentor(mapper_output, output_dir)
    _run_coach(mapper_output, mentor_output, output_dir)


if __name__ == "__main__":
    main()
