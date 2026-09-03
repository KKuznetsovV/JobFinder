"""Classify what cover-letter treatment a job posting's application form
actually needs, before Stage B (cover-letter generation) runs - so postings
that don't want a full letter (or any letter at all) don't waste a
generation + fact-check Claude call on content that gets thrown away or
doesn't fit.

Two sources, cheapest first:
- Recognized ATS templates (Greenhouse/Lever/Workday/Comeet/SmartRecruiters/
  JazzHR): free, static per-template knowledge declared directly on each
  `ATSAdapter` subclass (see `jobfinder.local.ats`).
- Everything else (LinkedIn Easy Apply, or a company_site posting on an
  unrecognized/bespoke career page): a cheap heuristic text scan over the
  posting body for explicit signals. Defaults to FULL_LETTER when no signal
  is found - skipping a wanted letter is worse than generating an unneeded
  one.
"""
from __future__ import annotations

import re

from jobfinder.db.models import Application, ApplyMethod, CoverLetterRequirement, JobPosting, utcnow_iso

_NONE_PATTERNS = [
    re.compile(r"no cover letter (?:is )?(?:needed|required|necessary)", re.IGNORECASE),
    re.compile(r"cover letters? (?:are|is) not (?:needed|required|necessary)", re.IGNORECASE),
    re.compile(r"do(?:n't| not) (?:need|require) a cover letter", re.IGNORECASE),
    re.compile(r"without (?:a )?cover letter", re.IGNORECASE),
    re.compile(r"please do not (?:submit|include|attach) a cover letter", re.IGNORECASE),
]

_SHORT_PATTERNS = [
    re.compile(r"cover letter (?:is )?optional", re.IGNORECASE),
    re.compile(r"brief (?:cover letter|note|statement)", re.IGNORECASE),
    re.compile(r"short (?:cover letter|note)", re.IGNORECASE),
    re.compile(r"include a (?:brief|short) note", re.IGNORECASE),
    re.compile(r"a few sentences", re.IGNORECASE),
]

_CHAR_LIMIT_RE = re.compile(r"(\d+)\s*characters?\b", re.IGNORECASE)

# What the local apply step can report having actually found on the real
# form, mapped back to the classification enum for mismatch comparison.
_FIELD_FOUND_TO_REQUIREMENT = {
    "full_letter_field": CoverLetterRequirement.FULL_LETTER,
    "short_field": CoverLetterRequirement.SHORT_NOTE,
    "no_field": CoverLetterRequirement.NONE,
}


def _heuristic_from_text(text: str) -> tuple[CoverLetterRequirement, int | None]:
    """Cheap text-scan fallback for postings whose actual application form
    isn't known until apply-time. Never guesses NONE/SHORT_NOTE without an
    explicit signal - defaults to FULL_LETTER."""
    if not text:
        return CoverLetterRequirement.FULL_LETTER, None

    for pattern in _NONE_PATTERNS:
        if pattern.search(text):
            return CoverLetterRequirement.NONE, None

    for pattern in _SHORT_PATTERNS:
        if pattern.search(text):
            limit_match = _CHAR_LIMIT_RE.search(text)
            char_limit = int(limit_match.group(1)) if limit_match else None
            return CoverLetterRequirement.SHORT_NOTE, char_limit

    return CoverLetterRequirement.FULL_LETTER, None


def classify_for_posting(job: JobPosting) -> tuple[CoverLetterRequirement, int | None]:
    """Return (requirement, char_limit) for this posting's application."""
    if job.apply_method == ApplyMethod.COMPANY_SITE:
        from jobfinder.local.ats import get_adapter_for_url  # local import: avoid ai -> local import cycle

        adapter = get_adapter_for_url(job.url)
        if adapter is not None:
            return adapter.cover_letter_requirement, adapter.cover_letter_char_limit

    return _heuristic_from_text(job.description or "")


def adapt_cover_letter_locally(
    cover_letter: str | None,
    requirement: CoverLetterRequirement,
    char_limit: int | None = None,
) -> str | None:
    """Adapt an already-approved letter to what the real application form
    turns out to need at apply-time, without calling Claude again - e.g. the
    real LinkedIn Easy Apply form has no message field after all, or only a
    short one. Used by the local apply step (see README's
    "cover-letter requirement detection" section)."""
    if not cover_letter:
        return cover_letter

    if requirement == CoverLetterRequirement.NONE:
        return None

    if requirement == CoverLetterRequirement.SHORT_NOTE:
        sentences = re.split(r"(?<=[.!?])\s+", cover_letter.strip())
        short = " ".join(sentences[:2]).strip()
        if char_limit and len(short) > char_limit:
            short = short[:char_limit].rstrip()
        return short

    return cover_letter


def log_if_classification_mismatched(application: Application, field_found: str | None) -> None:
    """Compare what the local apply step actually found on the real form
    against the upfront classification stored on `application`, and log a
    visible mismatch if they disagree - no Claude re-call to "fix" it, just
    make the miss visible so the heuristic can be improved over time.

    `field_found` is the apply step's own report of what it saw on the page:
    "full_letter_field" | "short_field" | "no_field" (or anything else/None,
    meaning nothing conclusive was reported).
    """
    actual = _FIELD_FOUND_TO_REQUIREMENT.get(field_found or "")
    if actual is None or actual == application.cover_letter_requirement:
        return
    application.apply_log.append(
        {
            "ts": utcnow_iso(),
            "event": "cover_letter_classification_mismatch",
            "detail": f"expected={application.cover_letter_requirement.value} actual={actual.value}",
        }
    )
