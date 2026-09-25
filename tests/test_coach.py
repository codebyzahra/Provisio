"""
test_coach.py — manual integration tests for the Coach Agent.

Run from the repo root:
    python tests/test_coach.py

Tests
-----
1. happy_path          — loads sample_mentor_output.json + a synthetic
                         MapperOutput, calls run_coach_session(), and prints
                         the resulting coach_output.json.
2. empty_session_log   — passes an empty MentorOutput; expects an error JSON
                         instead of a crash.
3. corrupted_mapper    — passes a missing / corrupted mapper_output.json file
                         path to main(); expects an error JSON instead of a
                         crash.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Make src/ importable regardless of where the script is invoked from.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from coach import run_coach_session          # noqa: E402
from coach_schema import CoachOutput         # noqa: E402
from mapper_schema import (                  # noqa: E402
    Category,
    ClassificationEntry,
    Confidence,
    EntryPoint,
    MapperOutput,
    SetupInfo,
)
from mentor_schema import MentorOutput       # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TESTS_DIR = Path(__file__).resolve().parent
SAMPLE_MENTOR_OUTPUT = TESTS_DIR / "sample_mentor_output.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEPARATOR = "-" * 60


def _print_section(title: str) -> None:
    print(f"\n{_SEPARATOR}")
    print(f"  {title}")
    print(_SEPARATOR)


def _build_synthetic_mapper() -> MapperOutput:
    """Return a realistic MapperOutput for use in tests."""
    return MapperOutput(
        repo_root=str(REPO_ROOT),
        generated_at=datetime(2024, 11, 15, 8, 0, 0, tzinfo=timezone.utc),
        entry_points=[
            EntryPoint(
                file="src/coach.py",
                import_count=5,
                confidence=Confidence.HIGH,
            ),
            EntryPoint(
                file="src/mapper.py",
                import_count=3,
                confidence=Confidence.MEDIUM,
            ),
            EntryPoint(
                file="src/mentor_schema.py",
                import_count=2,
                confidence=Confidence.LOW,
            ),
        ],
        classification={
            "src/": ClassificationEntry(
                category=Category.CORE_LOGIC,
                reason="matched pattern 'src'",
            ),
            "tests/": ClassificationEntry(
                category=Category.TESTING,
                reason="matched pattern 'tests'",
            ),
        },
        setup=SetupInfo(
            language="python",
            dependencies_file="requirements.txt",
            run_steps=[
                "python -m venv venv",
                "venv/Scripts/activate  # Windows",
                "pip install -r requirements.txt",
                "python src/coach.py --help",
            ],
        ),
        unclear_items=[],
    )


def _print_result(label: str, output: CoachOutput, coach_output_path: str | None = None) -> None:
    """Print a human-readable summary of a CoachOutput."""
    success = output.error == ""
    status = "SUCCESS ✓" if success else "EXPECTED FAILURE ✓"
    print(f"\nStatus : {status}")

    if coach_output_path and Path(coach_output_path).exists():
        print(f"Written : {coach_output_path}")

    print("\n--- coach_output.json contents ---")
    print(json.dumps(json.loads(output.model_dump_json()), indent=2))


# ---------------------------------------------------------------------------
# Test 1 — happy path
# ---------------------------------------------------------------------------

def test_happy_path() -> bool:
    """Load real sample_mentor_output.json + synthetic mapper, run session."""
    _print_section("TEST 1 — Happy Path")

    # Load the sample mentor output shipped alongside this script.
    mentor_output = MentorOutput.from_json_file(str(SAMPLE_MENTOR_OUTPUT))
    mapper_output = _build_synthetic_mapper()

    # Provide realistic answers that hit the expected keywords.
    answers = [
        "The Mapper Agent scans the repo and ranks files by import count to find entry points.",
        "The Mentor picks files from entry_points and classification to ask the developer questions.",
        "developer_confidence is self-reported knowledge; mapper confidence is based on import data.",
    ]

    output = run_coach_session(mapper_output, mentor_output, answers)

    # Persist coach_output.json to the tests/ directory for inspection.
    out_path = str(TESTS_DIR / "coach_output.json")
    output.to_json_file(out_path)

    _print_result("happy_path", output, out_path)

    passed = output.error == ""
    if not passed:
        print(f"\n[FAIL] Expected no error, got: {output.error!r}")
    return passed


# ---------------------------------------------------------------------------
# Test 2 — empty session log
# ---------------------------------------------------------------------------

def test_empty_session_log() -> bool:
    """Passing an empty MentorOutput must return an error JSON, not crash."""
    _print_section("TEST 2 — Empty Session Log (edge case)")

    empty_mentor = MentorOutput(session=[])
    mapper_output = _build_synthetic_mapper()

    output = run_coach_session(mapper_output, empty_mentor, answers=[])

    _print_result("empty_session_log", output)

    passed = (
        output.error != ""
        and output.quiz == []
        and output.recommended_task is None
    )
    if passed:
        print(f"\n[PASS] Got expected error: {output.error!r}")
    else:
        print(
            f"\n[FAIL] Expected non-empty error and empty quiz. "
            f"error={output.error!r}  quiz={output.quiz}"
        )
    return passed


# ---------------------------------------------------------------------------
# Test 3 — missing / corrupted mapper_output.json
# ---------------------------------------------------------------------------

def test_corrupted_mapper_output() -> bool:
    """Feeding main() a corrupted mapper file must produce an error JSON, not crash."""
    _print_section("TEST 3 — Missing/Corrupted mapper_output.json (edge case)")

    # Use a temporary directory so we don't pollute the workspace.
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupted_mapper_path = os.path.join(tmpdir, "mapper_output.json")
        mentor_path = str(SAMPLE_MENTOR_OUTPUT)
        out_path = os.path.join(tmpdir, "coach_output.json")

        # Write deliberately invalid JSON to simulate corruption.
        with open(corrupted_mapper_path, "w", encoding="utf-8") as fh:
            fh.write("{ this is not valid JSON }")

        # Call main() with the corrupted file; it must not raise.
        # Import here to call the CLI entry-point directly.
        from coach import main as coach_main  # noqa: PLC0415

        try:
            coach_main([
                "--mapper-output", corrupted_mapper_path,
                "--mentor-output", mentor_path,
                "--output", out_path,
            ])
            raised = False
        except SystemExit:
            # main() calls sys.exit(1) on error — that's fine, it's not a crash.
            raised = False
        except Exception as exc:  # noqa: BLE001
            print(f"\n[FAIL] Unexpected exception: {type(exc).__name__}: {exc}")
            return False

        # The error output file must still have been written.
        if not Path(out_path).exists():
            print("\n[FAIL] coach_output.json was not written on error path.")
            return False

        result = CoachOutput.from_json_file(out_path)
        _print_result("corrupted_mapper_output", result, out_path)

        passed = result.error != ""
        if passed:
            print(f"\n[PASS] Got expected error: {result.error!r}")
        else:
            print("\n[FAIL] Expected non-empty error field in coach_output.json.")
        return passed


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    results: dict[str, bool] = {}

    results["happy_path"]             = test_happy_path()
    results["empty_session_log"]      = test_empty_session_log()
    results["corrupted_mapper_output"] = test_corrupted_mapper_output()

    _print_section("SUMMARY")
    all_passed = True
    for name, ok in results.items():
        icon = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {icon}  {name}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed.")
        sys.exit(0)
    else:
        print("One or more tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
