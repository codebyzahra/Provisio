"""FastAPI application — exposes the Provisio pipeline as a REST API."""

from __future__ import annotations

import io
import json
import sys
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
# Request model
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    target: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
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

    # Lazy import — pipeline and its siblings are resolvable because _SRC is
    # already on sys.path (added at module load above).
    from . import pipeline  # noqa: PLC0415

    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            pipeline.run_pipeline(target)
    except SystemExit:
        err = buf.getvalue().lower()
        if any(k in err for k in ("exit 128", "fatal", "not found")):
            raise HTTPException(
                status_code=404,
                detail="Repository not found. Please ensure the repository exists and is public.",
            )
        raise HTTPException(status_code=500, detail="Failed to process repository. Please try again.")
    except Exception as exc:  # noqa: BLE001
        err = f"{exc}".lower()
        if any(k in err for k in ("exit 128", "fatal", "not found")):
            raise HTTPException(
                status_code=404,
                detail="Repository not found. Please ensure the repository exists and is public.",
            )
        raise HTTPException(status_code=500, detail="Failed to process repository. Please try again.")

    # Read the three output files written to the project root.
    result: dict = {}
    for key, path in _OUTPUT_FILES.items():
        try:
            result[key] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline completed but could not read {path.name}: {exc}",
            )

    return result
