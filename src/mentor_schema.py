"""
Shared data contract for the Mentor Agent output.
Owner: Mentor Agent

This file defines the session log format that the Mentor Agent MUST produce.
Any implementation of the Mentor Agent must serialise its output as a
``MentorOutput`` object so that the Coach Agent can load it without modification.

CONTRACT NOTICE
---------------
Do **not** change the field names or types in this file without coordinating
with the Coach Agent (coach.py / coach_schema.py).  Both agents share this
schema as their integration boundary.

REAL OUTPUT SHAPE (as produced by the real Mentor Agent)
---------------------------------------------------------
- Top-level list field is ``entries``, not ``session``.
- ``timestamp`` is a plain ISO-8601 string, not a datetime object.
- ``session_metadata`` is an optional dict (not used by Coach).
- Serialised with plain json.dump; Pydantic validates on load here.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from mapper_schema import Confidence


class SessionEntry(BaseModel):
    """A single question-answer exchange recorded during a Mentor session.

    Attributes:
        question:              The question the Mentor posed to the developer.
        answer:                The developer's verbatim answer.
        developer_confidence:  The developer's self-reported confidence level
                               for this answer, using the shared
                               :class:`~mapper_schema.Confidence` enum
                               (``"high"`` / ``"medium"`` / ``"low"``).
                               This field is intentionally named
                               *developer_confidence* to distinguish it from
                               the Mapper's ``EntryPoint.confidence`` field,
                               which reflects *import-count importance*, not
                               developer knowledge.
        timestamp:             ISO-8601 timestamp string of when this exchange
                               occurred (plain string, not a datetime object).
    """

    question: str = Field(..., description="Question posed by the Mentor")
    answer: str = Field(..., description="Developer's verbatim answer")
    developer_confidence: Confidence = Field(
        ...,
        description=(
            "Developer's self-reported confidence: 'high', 'medium', or 'low'. "
            "Must use the shared Confidence enum from mapper_schema."
        ),
    )
    timestamp: str = Field(..., description="ISO-8601 timestamp string of the exchange")

    model_config = {"use_enum_values": True}


class MentorOutput(BaseModel):
    """The complete output of one Mentor session.

    A ``MentorOutput`` is a list of :class:`SessionEntry` objects, one per
    question-answer exchange.  The Mentor Agent writes this to
    ``mentor_output.json`` using plain json.dump; Coach loads it here via
    :meth:`from_json_file`.

    Field name is ``entries`` (not ``session``) — this matches the real
    Mentor Agent output contract.
    """

    entries: list[SessionEntry] = Field(
        default_factory=list,
        description="Ordered list of question-answer exchanges from the session",
    )
    session_metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional metadata dict produced by the Mentor Agent (not used by Coach)",
    )

    model_config = {"use_enum_values": True}

    def to_json_file(self, path: str) -> None:
        """Serialise this object to a JSON file at *path*."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json_file(cls, path: str) -> "MentorOutput":
        """Deserialise a ``MentorOutput`` from a JSON file at *path*."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.model_validate_json(fh.read())
