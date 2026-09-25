"""
Shared data contract for the Onboarding Copilot pipeline.
Owner: Maira (Mapper Agent)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Category(str, Enum):
    CORE_LOGIC = "core_logic"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    UNCLEAR = "unclear"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EntryPoint(BaseModel):
    file: str = Field(..., description="Repo-relative path, forward-slash normalized")
    import_count: int = Field(..., ge=0, description="Number of in-repo files that import this one")
    confidence: Confidence

    @field_validator("file")
    @classmethod
    def normalize_slashes(cls, v: str) -> str:
        return v.replace("\\", "/")


class ClassificationEntry(BaseModel):
    category: Category
    reason: str = Field(..., max_length=200, description="Short deterministic rule that fired")


class SetupInfo(BaseModel):
    language: str = Field(..., description="Primary language, e.g. 'python', 'node'")
    dependencies_file: Optional[str] = Field(None, description="e.g. 'requirements.txt', 'package.json'")
    run_steps: list[str] = Field(default_factory=list, description="Ordered, plain-English shell/setup steps")


class UnclearItem(BaseModel):
    path: str
    reason: str = Field(..., max_length=200)

    @field_validator("path")
    @classmethod
    def normalize_slashes(cls, v: str) -> str:
        return v.replace("\\", "/")


class MapperOutput(BaseModel):
    repo_root: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    entry_points: list[EntryPoint] = Field(
        default_factory=list,
        description="Top N files ranked by inward import count, highest first",
    )
    classification: dict[str, ClassificationEntry] = Field(
        default_factory=dict,
        description="Path -> category mapping for every top-level folder/file scanned",
    )
    setup: SetupInfo
    unclear_items: list[UnclearItem] = Field(default_factory=list)

    model_config = {
        "use_enum_values": True,
    }

    def to_json_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json_file(cls, path: str) -> "MapperOutput":
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())


if __name__ == "__main__":
    example = MapperOutput(
        repo_root="D:/Projects/Provisio",
        entry_points=[
            EntryPoint(file="src/main.py", import_count=12, confidence=Confidence.HIGH),
            EntryPoint(file="src/utils/helpers.py", import_count=7, confidence=Confidence.MEDIUM),
        ],
        classification={
            "src/": ClassificationEntry(category=Category.CORE_LOGIC, reason="matched pattern 'src'"),
            "tests/": ClassificationEntry(category=Category.TESTING, reason="matched pattern 'tests'"),
            "scripts/legacy_thing.py": ClassificationEntry(
                category=Category.UNCLEAR, reason="no rule matched; ambiguous folder name"
            ),
        },
        setup=SetupInfo(
            language="python",
            dependencies_file="requirements.txt",
            run_steps=[
                "python -m venv venv",
                "venv/Scripts/activate",
                "pip install -r requirements.txt",
                "python src/main.py",
            ],
        ),
        unclear_items=[
            UnclearItem(path="scripts/legacy_thing.py", reason="ambiguous folder name, no rule matched")
        ],
    )

    print(example.model_dump_json(indent=2))