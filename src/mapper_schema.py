"""
mapper_schema.py
================
Defines the data model for Mapper Agent output used by downstream pipeline
stages (Mentor, Coach, etc.).

Do NOT modify this file — it is consumed as-is by mentor.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EntryPoint:
    """A single entry-point discovered by the Mapper Agent.

    Attributes:
        name: The function or class name.
        file_path: Relative path of the file that contains this entry-point.
        line_number: Line number where the entry-point begins (1-based).
        description: Free-text description of what this entry-point does.
        metadata: Optional extra key/value data surfaced by the Mapper.
    """

    name: str
    file_path: str
    line_number: int
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntryPoint":
        """Deserialise from a plain dict."""
        return cls(
            name=data["name"],
            file_path=data["file_path"],
            line_number=data["line_number"],
            description=data["description"],
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "name": self.name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class SetupInfo:
    """High-level setup / environment information for the repository.

    Attributes:
        language: Primary programming language.
        framework: Framework or runtime (e.g. "FastAPI", "Django").
        dependencies: List of top-level dependency names.
        notes: Free-text notes about the project setup.
    """

    language: str
    framework: str
    dependencies: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SetupInfo":
        """Deserialise from a plain dict."""
        return cls(
            language=data["language"],
            framework=data["framework"],
            dependencies=data.get("dependencies", []),
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "language": self.language,
            "framework": self.framework,
            "dependencies": self.dependencies,
            "notes": self.notes,
        }


@dataclass
class MapperOutput:
    """Root object for the Mapper Agent's JSON output.

    Attributes:
        entry_points: Discovered entry-points in the repository.
        classification: Mapping of relative file path → module classification
            label (e.g. "service", "model", "util").
        setup_info: Repository setup/environment metadata.
    """

    entry_points: list[EntryPoint]
    classification: dict[str, str]
    setup_info: SetupInfo

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapperOutput":
        """Deserialise from a plain dict."""
        return cls(
            entry_points=[EntryPoint.from_dict(ep) for ep in data["entry_points"]],
            classification=data["classification"],
            setup_info=SetupInfo.from_dict(data["setup_info"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "entry_points": [ep.to_dict() for ep in self.entry_points],
            "classification": self.classification,
            "setup_info": self.setup_info.to_dict(),
        }

    @classmethod
    def from_json_file(cls, path: str) -> "MapperOutput":
        """Load and deserialise a MapperOutput from a JSON file.

        Args:
            path: File-system path to the JSON file.

        Returns:
            A fully populated :class:`MapperOutput` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If required fields are missing from the JSON.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    def to_json_file(self, path: str) -> None:
        """Serialise and write the object to a JSON file.

        Args:
            path: Destination file-system path.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
