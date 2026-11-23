"""Typed data shapes shared by the vault layer, the MCP tools and the agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["To apply", "Applied", "Interview", "Offer", "Rejected", "Withdrawn", "Skip"]
STATUSES: tuple[str, ...] = ("To apply", "Applied", "Interview", "Offer", "Rejected", "Withdrawn", "Skip")
PROGRESSION: tuple[str, ...] = ("To apply", "Applied", "Interview", "Offer")
TERMINAL: tuple[str, ...] = ("Rejected", "Withdrawn", "Skip")
# The only statuses a triage run may ever propose. Everything else needs a human to edit the note.
TRIAGE_STATUSES: tuple[str, ...] = ("To apply", "Skip")


class PostingSummary(BaseModel):
    file: str
    title: str
    company: str
    role: str
    link: str = ""
    location: str = ""


class Posting(PostingSummary):
    body: str
    truncated: bool = False
    untrusted: bool = Field(True, description="Always true: posting text is third-party content.")
    injection_flags: list[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    file: str
    title: str
    snippet: str


class Application(BaseModel):
    file: str
    company: str = ""
    role: str = ""
    track: str = ""
    status: str = ""
    fit: str = ""
    applied: str = ""
    follow_up: str = ""
    location: str = ""
    link: str = ""
    resume: str = ""


class UpdateResult(BaseModel):
    file: str
    previous_status: str
    changed: dict[str, str]
    message: str


class SeenJobResult(BaseModel):
    added: bool
    line: str
    message: str


class CreateResult(BaseModel):
    file: str
    created: bool
    message: str


class FollowUp(BaseModel):
    file: str
    company: str
    role: str
    status: str
    follow_up: str
    days_overdue: int


class MatchResult(BaseModel):
    posting: str
    track: str
    score: int = Field(ge=0, le=100)
    fit: Literal["good", "bad"]
    hard_fail: bool
    reasons: list[str]
    matched_required: list[str]
    matched_preferred: list[str]
    missing_required: list[str]
    missing_preferred: list[str]
    years_required: int | None = None
    location_status: Literal["ok", "rejected", "unknown"]
    method: Literal["baseline", "llm", "hybrid"] = "baseline"


class Suggestion(BaseModel):
    skill: str
    kind: Literal["learn", "build"]
    suggestion: str


class GapAnalysis(BaseModel):
    posting: str
    track: str
    matched: list[str]
    missing_required: list[str]
    missing_preferred: list[str]
    blockers: list[str]
    suggestions: list[Suggestion]
