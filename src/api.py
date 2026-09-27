"""FastAPI application — exposes the Provisio pipeline as a REST API."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Project root — two levels up from this file (src/api.py → src/ → project/)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent

# pipeline.py and its siblings (coach, mapper, mentor, *_schema) use bare
# module imports and must be importable by name.  Ensure src/ is on sys.path
# before any import of those modules happens.
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_OUTPUT_FILES = {
    "mapper": _ROOT / "mapper_output.json",
    "mentor": _ROOT / "mentor_output.json",
    "coach": _ROOT / "coach_output.json",
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Provisio API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    target: str


class QuizScoreRequest(BaseModel):
    quiz: list[dict]
    answers: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/quiz/score")
def quiz_score(request: QuizScoreRequest) -> list[dict]:
    """Reconstruct QuizItem objects, score the developer's answers, and return results."""
    import coach as _coach  # noqa: PLC0415
    from coach_schema import QuizItem  # noqa: PLC0415

    try:
        quiz_items = [QuizItem(**item) for item in request.quiz]
        scored = _coach.score_quiz(quiz=quiz_items, answers=request.answers)
        return [item.model_dump() for item in scored]
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb  # noqa: PLC0415
        detail = f"{exc}\n{_tb.format_exc()}"
        raise HTTPException(status_code=500, detail=detail)


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    """Run the full Mapper → Mentor → Coach pipeline and return all outputs."""
    target = request.target.strip()
    if not target:
        raise HTTPException(status_code=400, detail="'target' must not be blank.")

    # URL validation — only accept well-formed public GitHub repository URLs.
    if target.startswith("http://") or target.startswith("https://"):
        import re  # noqa: PLC0415
        if not re.fullmatch(r"https://github\.com/[^/]+/[^/]+(/.*)?", target):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid URL. Please provide a valid public GitHub repository URL "
                    "(e.g., https://github.com/owner/repo)."
                ),
            )

    # pipeline and its siblings are resolvable via absolute import because
    # _SRC is already on sys.path (added at module load above).
    import pipeline  # noqa: PLC0415

    request_dir = tempfile.mkdtemp(prefix="provisio_")
    buf = io.StringIO()
    try:
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                pipeline.run_pipeline(target, output_dir=request_dir)
        except SystemExit:
            captured = buf.getvalue()
            err = captured.lower()
            if any(k in err for k in ("exit 128", "fatal", "not found")):
                raise HTTPException(
                    status_code=404,
                    detail="Repository not found. Please ensure the repository exists and is public.",
                )
            raise HTTPException(status_code=500, detail=captured or "Failed to process repository. Please try again.")
        except Exception as exc:  # noqa: BLE001
            captured = buf.getvalue()
            err = f"{exc} {captured}".lower()
            if any(k in err for k in ("exit 128", "fatal", "not found")):
                raise HTTPException(
                    status_code=404,
                    detail="Repository not found. Please ensure the repository exists and is public.",
                )
            import traceback as _tb  # noqa: PLC0415
            detail = f"{exc}\n{_tb.format_exc()}"
            raise HTTPException(status_code=500, detail=detail)

        # Read the three output files from this request's private directory.
        _request_output_files = {
            "mapper": os.path.join(request_dir, "mapper_output.json"),
            "mentor": os.path.join(request_dir, "mentor_output.json"),
            "coach": os.path.join(request_dir, "coach_output.json"),
        }
        result: dict = {}
        for key, path in _request_output_files.items():
            try:
                with open(path, encoding="utf-8") as fh:
                    result[key] = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=500,
                    detail=f"Pipeline completed but could not read {os.path.basename(path)}: {exc}",
                )

        return result
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)
