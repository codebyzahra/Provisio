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
import os
import sys
from datetime import datetime, timezone
from typing import NoReturn

from pydantic import ValidationError

import coach
import mapper
import mentor
from coach_schema import CoachOutput
from mapper_schema import MapperOutput
from mentor_schema import Confidence, MentorOutput, SessionEntry

# ---------------------------------------------------------------------------
# Fixed sample questions — generic, repo-agnostic onboarding questions
# ---------------------------------------------------------------------------

_SAMPLE_QUESTIONS: list[str] = [
    "What are the main entry points of this project and which file should I open first?",
    "How do I install the dependencies and run this project locally?",
    "How is the test suite organised and how do I run the tests?",
    "What is the overall purpose and architecture of this codebase?",
    "Which directories contain the core business logic versus configuration or tooling?",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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

    try:
        mapper_output = mapper.main(mapper_argv)
    except SystemExit as exc:
        _fail("Mapper", f"mapper.main() exited with code {exc.code}")
    except Exception as exc:  # noqa: BLE001
        _fail("Mapper", str(exc))

    # mapper.main() saves to mapper_output.json in the CWD; if a different
    # output_dir was requested, write there as well.
    dest = _output_path(output_dir, "mapper_output.json")
    if os.path.abspath(dest) != os.path.abspath("mapper_output.json"):
        mapper_output.to_json_file(dest)

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

    for question in _SAMPLE_QUESTIONS:
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
