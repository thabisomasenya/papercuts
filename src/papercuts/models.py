"""Source-independent data models. Every source adapter normalises into these."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

_DOI_PREFIX = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:)", re.IGNORECASE)


def normalize_doi(doi: str | None) -> str | None:
    """'https://doi.org/10.1000/ABC' -> '10.1000/abc'. DOIs are case-insensitive."""
    if not doi:
        return None
    doi = _DOI_PREFIX.sub("", doi.strip()).lower()
    return doi if doi.startswith("10.") else None


class Author(BaseModel):
    id: str  # source-scoped id, e.g. "openalex:A123"
    name: str
    orcid: str | None = None
    position: int = 0  # 0-based author order
    institutions: list[str] = Field(default_factory=list)


class Topic(BaseModel):
    id: str
    name: str
    score: float | None = None


class Paper(BaseModel):
    id: str  # canonical id, e.g. "openalex:W2741809807"
    source: str  # "openalex", "semantic_scholar", "arxiv", ...
    title: str
    abstract: str | None = None
    doi: str | None = None
    year: int | None = None
    publication_date: str | None = None
    venue: str | None = None
    type: str | None = None
    language: str | None = None
    cited_by_count: int = 0
    is_oa: bool = False
    oa_url: str | None = None
    authors: list[Author] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)  # canonical ids of cited works
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("doi", mode="before")
    @classmethod
    def _norm_doi(cls, v: str | None) -> str | None:
        return normalize_doi(v)

    @field_validator("title", mode="before")
    @classmethod
    def _clean_title(cls, v: str | None) -> str:
        return re.sub(r"\s+", " ", (v or "").strip()) or "(untitled)"


class SearchFilters(BaseModel):
    from_year: int | None = None
    to_year: int | None = None
    types: list[str] = Field(default_factory=list)  # e.g. ["article", "review"]
    min_citations: int | None = None
    open_access_only: bool = False
    language: str | None = None  # ISO 639-1, e.g. "en"
