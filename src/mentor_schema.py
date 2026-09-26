"""
mentor_schema.py
================
Defines the data model for Mentor Agent session output, consumed by the
Coach Agent in the Onboarding Copilot pipeline.

Do NOT modify this file — it is consumed as-is by mentor.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Confidence(str, Enum):
    """Developer's self-reported confidence after reading the Mentor's answer.

    Values:
        HIGH: The developer feels fully confident about the topic.
        MEDIUM: The developer has partial understanding; may need follow-up.
        LOW: The developer is still unsure; further guidance recommended.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SessionEntry:
    """A single Q&A exchange in a Mentor session.

    Attributes:
        question: The question the developer asked.
        answer: The Mentor's generated answer.
        developer_confidence: The developer's self-reported confidence level.
        timestamp: ISO-8601 UTC timestamp of the exchange.
    """

    question: str
    answer: str
    developer_confidence: Confidence
    timestamp: str  # ISO-8601 UTC string, e.g. "2024-01-15T10:30:00"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionEntry":
        """Deserialise from a plain dict.

        Args:
            data: Dictionary with keys matching the dataclass fields.

        Returns:
            A :class:`SessionEntry` instance.
        """
        return cls(
            question=data["question"],
            answer=data["answer"],
            developer_confidence=Confidence(data["developer_confidence"]),
            timestamp=data["timestamp"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON output.

        Returns:
            A dictionary representation of this entry.
        """
        return {
            "question": self.question,
            "answer": self.answer,
            "developer_confidence": self.developer_confidence.value,
            "timestamp": self.timestamp,
        }


@dataclass
class MentorOutput:
    """Root object for a complete Mentor Agent session.

    Attributes:
        entries: Ordered list of Q&A exchanges from the session.
        session_metadata: Optional key/value metadata (e.g. repo path, model).
    """

    entries: list[SessionEntry] = field(default_factory=list)
    session_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MentorOutput":
        """Deserialise from a plain dict.

        Args:
            data: Dictionary with ``entries`` list and optional metadata.

        Returns:
            A :class:`MentorOutput` instance.
        """
        return cls(
            entries=[SessionEntry.from_dict(e) for e in data.get("entries", [])],
            session_metadata=data.get("session_metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict.

        Returns:
            A dictionary with ``entries`` and ``session_metadata`` keys.
        """
        return {
            "entries": [e.to_dict() for e in self.entries],
            "session_metadata": self.session_metadata,
        }

    @classmethod
    def from_json_file(cls, path: str) -> "MentorOutput":
        """Load and deserialise a :class:`MentorOutput` from a JSON file.

        Args:
            path: File-system path to the JSON file.

        Returns:
            A fully populated :class:`MentorOutput` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    def to_json_file(self, path: str) -> None:
        """Serialise and write the session to a JSON file.

        Args:
            path: Destination file-system path.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
