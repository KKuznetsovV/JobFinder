"""Dataclasses for the two core records: JobPosting and Application.

These mirror the DB schema (see schema.sql) but stay decoupled from sqlite3
Row objects so the rest of the codebase can pass plain, typed objects around.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApplyMethod(str, Enum):
    EMAIL = "email"
    COMPANY_SITE = "company_site"
    LINKEDIN_EASY_APPLY = "linkedin_easy_apply"


class ResumeVariant(str, Enum):
    FULLSTACK = "fullstack"
    PROJECT_MANAGER = "project_manager"


class CoverLetterRequirement(str, Enum):
    """What a given application's target form actually needs, decided before
    Stage B (cover-letter generation) runs so postings that don't want a
    full letter don't waste a generation + fact-check Claude call on one."""

    NONE = "none"  # no cover-letter field at all - skip generation entirely
    SHORT_NOTE = "short_note"  # a small char/word-limited text box
    FULL_LETTER = "full_letter"  # a full letter (file upload or unrestricted text field)


class ApplicationStatus(str, Enum):
    FOUND = "found"
    NOTIFIED = "notified"  # Stage B (cover letter) approval pending
    PENDING_STAGE_A_APPROVAL = "pending_stage_a_approval"  # Stage A (job+resume) approval pending
    APPROVED = "approved"
    REJECTED = "rejected"  # legacy pre-two-stage rejection, no longer written
    REJECTED_TIER0 = "rejected_tier0"
    REJECTED_STAGE_A = "rejected_stage_a"
    REJECTED_STAGE_B = "rejected_stage_b"
    REVISION_REQUESTED = "revision_requested"
    SUBMITTING = "submitting"
    AWAITING_MY_CLICK = "awaiting_my_click"
    SENT = "sent"
    STUCK = "stuck"
    FAILED = "failed"


@dataclass
class JobPosting:
    source: str
    external_id: str
    title: str
    company: str
    description: str
    url: str
    apply_method: ApplyMethod
    posted_date: str | None = None
    apply_email: str | None = None
    discovered_at: str = field(default_factory=utcnow_iso)
    id: int | None = None


@dataclass
class Application:
    job_posting_id: int
    resume_variant: ResumeVariant
    resume_choice_reason: str | None = None
    cover_letter: str | None = None
    status: ApplicationStatus = ApplicationStatus.FOUND
    relevance_score: float | None = None
    gmail_thread_id: str | None = None
    gmail_last_message_id: str | None = None
    revision_count: int = 0  # Stage B (cover-letter) revisions only
    apply_log: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    tier0_score: float | None = None
    tier0_resume_hint: ResumeVariant | None = None
    stage_a_approved_at: str | None = None
    stage_a_revision_count: int = 0
    cover_letter_requirement: CoverLetterRequirement = CoverLetterRequirement.FULL_LETTER
    cover_letter_char_limit: int | None = None
    id: int | None = None


@dataclass
class Tier1TrainingExample:
    """One labeled (job posting -> relevance + resume decision) example for
    fine-tuning the tier-1 local classifier. `source` distinguishes real,
    production-logged examples from offline-generated synthetic ones."""

    job_posting_id: int | None
    title: str
    company: str
    description: str
    relevant: bool
    relevance_score: float
    relevance_reason: str
    resume_variant: ResumeVariant | None = None
    resume_reason: str | None = None
    source: str = "production"  # "production" | "synthetic"
    created_at: str = field(default_factory=utcnow_iso)
    id: int | None = None


@dataclass
class Tier1DecisionLog:
    """One row per tier1-routed posting, for tracking real-world local-model
    vs. Claude agreement over time (see jobfinder.ai.tier1_classifier)."""

    job_posting_id: int | None
    path: str  # "claude" | "claude_fallback" | "tier1" | "tier1_spot_checked"
    relevance_confidence: float | None = None
    resume_confidence: float | None = None
    agreement: bool | None = None
    created_at: str = field(default_factory=utcnow_iso)
    id: int | None = None
