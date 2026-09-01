"""Base interface for deterministic ATS (Applicant Tracking System) form
adapters.

Each adapter targets one ATS platform's well-known, publicly-documented
common application-form structure (field names/ids that are stable across
most companies using that platform). Real companies sometimes customize
their forms beyond this, so adapters raise a clear `ATSFormError` when a
required field/button can't be found rather than silently skipping it or
guessing - the local agent catches this and marks the application STUCK for
manual follow-up instead of submitting an incomplete/wrong application.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from jobfinder.db.models import Application, JobPosting


class ATSFormError(RuntimeError):
    """Raised when a required field or control can't be found on the page."""


class ATSAdapter(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def matches(url: str) -> bool:
        """Return True if this adapter handles the given job posting URL."""

    @abstractmethod
    def fill_and_submit(self, page, application: Application, job: JobPosting) -> None:
        """Fill the application form and submit it. Raise ATSFormError (or
        let a Playwright TimeoutError propagate) if a required field/button
        isn't found - never submit a partially-filled form."""


def split_applicant_name() -> tuple[str, str]:
    """Split config.APPLICANT_NAME into (first, last). Last name is
    everything after the first space, joined back together."""
    from jobfinder import config

    parts = config.APPLICANT_NAME.strip().split(" ", 1)
    first = parts[0]
    last = parts[1] if len(parts) > 1 else ""
    return first, last


def resume_file_for(application: Application) -> str:
    from jobfinder.resume.cache import resume_path

    return str(resume_path(application.resume_variant))


def fill_if_present(page, selector: str, value: str) -> bool:
    """Fill `selector` with `value` if it exists on the page and value is
    non-empty. Returns whether it was filled - used for optional fields."""
    if not value:
        return False
    locator = page.locator(selector)
    if locator.count() == 0:
        return False
    locator.first.fill(value)
    return True


def require(page, selector: str, description: str):
    """Return the first matching locator, or raise ATSFormError if not found."""
    locator = page.locator(selector)
    if locator.count() == 0:
        raise ATSFormError(f"Required field not found: {description} ({selector})")
    return locator.first
