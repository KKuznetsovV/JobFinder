"""Dataclasses for the two core records: JobPosting and Application.

These mirror the DB schema (see schema.sql) but stay decoupled from sqlite3
Row objects so the rest of the codebase can pass plain, typed objects around.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApplyMethod(str, Enum):
    EMAIL = "email"
    COMPANY_SITE = "company_site"
    LINKEDIN_EASY_APPLY = "linkedin_easy_apply"


class ResumeVariant(str, Enum):
    FULLSTACK = "fullstack"
    PROJECT_MANAGER = "project_manager"


class ApplicationStatus(str, Enum):
    FOUND = "found"
    NOTIFIED = "notified"
    APPROVED = "approved"
    REJECTED = "rejected"
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
    revision_count: int = 0
    apply_log: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    id: int | None = None
