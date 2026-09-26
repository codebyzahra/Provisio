"""
data_loader.py
==============
Isolated JSON I/O for the PROVISIO dashboard.

Each public function resolves its target file relative to the project root
(two directory levels above this file: src/ → project root) so the dashboard
works correctly regardless of the current working directory.

Returns the parsed dict on success, or None if the file does not exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Project root is always two levels up: src/data_loader.py → src/ → root
_ROOT = Path(__file__).parent.parent

_MAPPER_PATH = _ROOT / "mapper_output.json"
_MENTOR_PATH = _ROOT / "mentor_output.json"
_COACH_PATH = _ROOT / "coach_output.json"


def _load(path: Path) -> dict[str, Any] | None:
    """Load a JSON file at *path* and return its parsed content, or None on FileNotFoundError."""
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def load_mapper() -> dict[str, Any] | None:
    """Return the parsed contents of mapper_output.json, or None if missing."""
    return _load(_MAPPER_PATH)


def load_mentor() -> dict[str, Any] | None:
    """Return the parsed contents of mentor_output.json, or None if missing."""
    return _load(_MENTOR_PATH)


def load_coach() -> dict[str, Any] | None:
    """Return the parsed contents of coach_output.json, or None if missing."""
    return _load(_COACH_PATH)
