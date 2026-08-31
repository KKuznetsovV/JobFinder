"""Generate a cover letter for a job posting, grounded strictly in the
selected resume's content, then run a fact-check pass to catch anything the
model invented that isn't actually in the resume.

Uses explicit prompt-caching breakpoints on the resume text + style
reference + static instructions (the parts that repeat across many calls),
since those combined comfortably clear Sonnet 5's 1024-token minimum
cacheable prompt size.
"""
from __future__ import annotations

from jobfinder import config
from jobfinder.ai.client import create_message_with_retry, get_client
from jobfinder.db.models import JobPosting, ResumeVariant
from jobfinder.resume.cache import get_or_parse_resume

_FACT_CHECK_TOOL = {
    "name": "report_fact_check",
    "description": (
        "Report whether the cover letter draft is fully grounded in the candidate's "
        "resume, and provide a corrected version if not."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "approved": {
                "type": "boolean",
                "description": "True if the draft contains no claims beyond what's in the resume.",
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific unsupported/invented claims found, if any.",
            },
            "corrected_letter": {
                "type": "string",
                "description": (
                    "The final cover letter text to use - either the original draft "
                    "unchanged (if approved) or a corrected version with invented "
                    "claims removed/fixed."
                ),
            },
        },
        "required": ["approved", "issues", "corrected_letter"],
    },
}


def _load_style_reference() -> str:
    try:
        return config.STYLE_REFERENCE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _generate_draft(job: JobPosting, resume: dict, style_reference: str, client) -> str:
    message = create_message_with_retry(
        client,
        model=config.MODEL_ID,
        max_tokens=config.MAX_TOKENS_DEFAULT,
        system=[
            {
                "type": "text",
                "text": (
                    "You write cover letters for a job applicant. Ground every claim "
                    "strictly in the resume text provided - never invent skills, "
                    "employers, dates, or achievements not present in the resume. "
                    "Write roughly 200-300 words. Match the tone/register of the "
                    "provided style reference where reasonable, but keep it "
                    "professional enough for a cover letter. "
                    f"Sign the letter as '{config.APPLICANT_NAME}'."
                ),
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": (
                    f"Candidate resume (source of truth - do not go beyond this):\n"
                    f"{resume['raw_text']}\n\n"
                    f"Style reference (tone guidance only, not content):\n{style_reference}"
                ),
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Write a cover letter for this job posting:\n"
                    f"Title: {job.title}\n"
                    f"Company: {job.company}\n"
                    f"Description: {job.description}"
                ),
            }
        ],
    )
    return "".join(block.text for block in message.content if getattr(block, "type", None) == "text")


def _fact_check(draft: str, resume: dict, client) -> tuple[str, list[str]]:
    message = create_message_with_retry(
        client,
        model=config.MODEL_ID,
        max_tokens=config.MAX_TOKENS_DEFAULT,
        system=[
            {
                "type": "text",
                "text": (
                    "You fact-check a cover letter draft against the candidate's resume. "
                    "Flag any claim (skill, employer, title, dates, achievement) in the "
                    "draft that is not supported by the resume text. Always call "
                    "report_fact_check."
                ),
            }
        ],
        tools=[_FACT_CHECK_TOOL],
        tool_choice={"type": "tool", "name": "report_fact_check"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Resume:\n{resume['raw_text']}\n\nCover letter draft:\n{draft}"
                ),
            }
        ],
    )

    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "report_fact_check":
            data = block.input
            return str(data["corrected_letter"]), list(data.get("issues", []))

    raise ValueError("Claude did not call report_fact_check as expected")


def generate_cover_letter(
    job: JobPosting, resume_variant: ResumeVariant, client=None
) -> tuple[str, list[str]]:
    """Return (final_cover_letter, fact_check_issues). fact_check_issues is
    empty if the draft was approved as-is."""
    if client is None:
        client = get_client()

    resume = get_or_parse_resume(resume_variant)
    style_reference = _load_style_reference()

    draft = _generate_draft(job, resume, style_reference, client)
    final_letter, issues = _fact_check(draft, resume, client)
    return final_letter, issues
