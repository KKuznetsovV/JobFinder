"""Choose which resume variant (fullstack vs project_manager) to use for a
given job posting.

Fast path: keyword heuristics on the job title/description. Falls back to a
Claude comparison (grounded in both resumes' actual raw text) only when the
heuristic is ambiguous (no keywords matched, or keywords for both variants
matched).
"""
from __future__ import annotations

from jobfinder import config
from jobfinder.ai.client import create_message_with_retry, get_client
from jobfinder.db.models import JobPosting, ResumeVariant
from jobfinder.resume.cache import get_or_parse_resume

PROJECT_MANAGER_KEYWORDS = [
    "project manager",
    "program manager",
    "product manager",
    "scrum master",
    "delivery manager",
    "pmo",
]

FULLSTACK_KEYWORDS = [
    "full stack",
    "fullstack",
    "full-stack",
    "backend",
    "back-end",
    "frontend",
    "front-end",
    "software engineer",
    "software developer",
    "swe",
]

_SELECT_TOOL = {
    "name": "select_resume",
    "description": "Select which resume variant best fits the job posting.",
    "input_schema": {
        "type": "object",
        "properties": {
            "variant": {
                "type": "string",
                "enum": [ResumeVariant.FULLSTACK.value, ResumeVariant.PROJECT_MANAGER.value],
            },
            "reason": {
                "type": "string",
                "description": "One or two sentence justification grounded in the resumes and job description.",
            },
        },
        "required": ["variant", "reason"],
    },
}


def _keyword_hits(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def heuristic_select(job: JobPosting) -> ResumeVariant | None:
    """Return a ResumeVariant if the keyword heuristic is unambiguous, else
    None (meaning: fall back to the Claude comparison)."""
    text = f"{job.title}\n{job.description}"
    is_pm = _keyword_hits(text, PROJECT_MANAGER_KEYWORDS)
    is_fullstack = _keyword_hits(text, FULLSTACK_KEYWORDS)

    if is_pm and not is_fullstack:
        return ResumeVariant.PROJECT_MANAGER
    if is_fullstack and not is_pm:
        return ResumeVariant.FULLSTACK
    return None


def _claude_select(job: JobPosting, client=None) -> tuple[ResumeVariant, str]:
    if client is None:
        client = get_client()

    fullstack_resume = get_or_parse_resume(ResumeVariant.FULLSTACK)
    pm_resume = get_or_parse_resume(ResumeVariant.PROJECT_MANAGER)

    message = create_message_with_retry(
        client,
        model=config.MODEL_ID,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": (
                    "You help a job applicant pick which of their two resumes better "
                    "fits a job posting. Base your decision only on the content of the "
                    "two resumes provided and the job posting. Always call select_resume."
                ),
            }
        ],
        tools=[_SELECT_TOOL],
        tool_choice={"type": "tool", "name": "select_resume"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"--- Resume: fullstack ---\n{fullstack_resume['raw_text']}\n\n"
                    f"--- Resume: project_manager ---\n{pm_resume['raw_text']}\n\n"
                    f"Job posting:\nTitle: {job.title}\n"
                    f"Company: {job.company}\n"
                    f"Description: {job.description}"
                ),
            }
        ],
    )

    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "select_resume":
            data = block.input
            return ResumeVariant(data["variant"]), str(data["reason"])

    raise ValueError("Claude did not call select_resume as expected")


def select_resume(job: JobPosting, client=None) -> tuple[ResumeVariant, str]:
    """Return (variant, reason). Tries the keyword heuristic first; falls
    back to a Claude comparison of both resumes when ambiguous."""
    variant = heuristic_select(job)
    if variant is not None:
        return variant, f"Keyword heuristic matched '{variant.value}' resume."

    return _claude_select(job, client=client)
