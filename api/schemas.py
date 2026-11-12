"""Request models of the EvidenceLens API. Validation happens here, at the boundary."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from engine.evidence import CLASSES

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
MAX_TEXT_CHARACTERS = 300


class SearchRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, pattern=SESSION_ID_PATTERN)
    text: str
    evidence_filter: Optional[str] = None
    ranker: Literal["agent", "bm25"] = "agent"

    @field_validator("text")
    @classmethod
    def text_length(cls, value):
        value = value.strip()
        if not 1 <= len(value) <= MAX_TEXT_CHARACTERS:
            raise ValueError(f"text must have 1 to {MAX_TEXT_CHARACTERS} characters after trimming")
        return value

    @field_validator("evidence_filter")
    @classmethod
    def known_class(cls, value):
        if value is not None and value not in CLASSES:
            raise ValueError(f"evidence_filter must be null or one of {CLASSES}")
        return value


class RerankRequest(BaseModel):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    ranker: Literal["agent", "bm25"]


class ClarifyRequest(BaseModel):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    choice_id: int
