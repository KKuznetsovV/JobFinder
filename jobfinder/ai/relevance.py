"""Relevance filtering: score a JobPosting against the candidate's resumes
using Claude, before spending any further effort (resume selection, cover
letter generation) on it.

Uses a forced tool call so the score/reason come back as structured data
rather than needing to parse free-text from the model.
"""
from __future__ import annotations

from jobfinder import config
from jobfinder.ai.client import create_message_with_retry, get_client
from jobfinder.db.models import JobPosting
from jobfinder.resume.cache import get_or_parse_resume

_REPORT_TOOL = {
    "name": "report_relevance",
    "description": "Report a structured relevance score for how well the job posting matches the candidate's resumes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "description": "0.0 (no fit at all) to 1.0 (excellent fit). Use the full range.",
            },
            "reason": {
                "type": "string",
                "description": "One or two sentence justification grounded in the candidate profile and job description.",
            },
        },
        "required": ["score", "reason"],
    },
}


def build_candidate_profile() -> str:
    """Concatenate the raw text of both resume variants into a single profile
    string. Grounded only in what's actually in the resumes - never invents
    skills or experience."""
    from jobfinder.db.models import ResumeVariant

    parts = []
    for variant in (ResumeVariant.FULLSTACK, ResumeVariant.PROJECT_MANAGER):
        try:
            resume = get_or_parse_resume(variant)
        except (FileNotFoundError, ValueError):
            continue
        parts.append(f"--- Resume variant: {variant.value} ---\n{resume['raw_text']}")
    return "\n\n".join(parts)


def score_relevance(
    job: JobPosting, candidate_profile: str | None = None, client=None
) -> tuple[float, str]:
    """Return (score, reason) for how relevant `job` is to the candidate.

    `candidate_profile` defaults to `build_candidate_profile()` if not
    supplied (mainly so tests can pass in a fixed profile string).
    """
    if candidate_profile is None:
        candidate_profile = build_candidate_profile()
    if client is None:
        client = get_client()

    message = create_message_with_retry(
        client,
        model=config.MODEL_ID,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": (
                    "You are a job-relevance classifier for a personal job-search "
                    "assistant. Given a candidate's resumes and a job posting, decide "
                    "how relevant the job is to the candidate, based only on the "
                    "provided resume content. Do not assume skills or experience not "
                    "present in the resumes. Always call report_relevance."
                ),
            }
        ],
        tools=[_REPORT_TOOL],
        tool_choice={"type": "tool", "name": "report_relevance"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Candidate profile:\n{candidate_profile}\n\n"
                    f"Job posting:\nTitle: {job.title}\n"
                    f"Company: {job.company}\n"
                    f"Description: {job.description}"
                ),
            }
        ],
    )

    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "report_relevance":
            data = block.input
            score = float(data["score"])
            score = max(0.0, min(1.0, score))
            return score, str(data["reason"])

    raise ValueError("Claude did not call report_relevance as expected")


def is_relevant(job: JobPosting, candidate_profile: str | None = None, client=None) -> tuple[bool, float, str]:
    """Return (passes_threshold, score, reason)."""
    score, reason = score_relevance(job, candidate_profile=candidate_profile, client=client)
    return score >= config.RELEVANCE_THRESHOLD, score, reason
