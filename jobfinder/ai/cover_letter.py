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
from jobfinder.db.models import CoverLetterRequirement, JobPosting, ResumeVariant
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


_FULL_LETTER_MAX_WORDS = 150
_SHORT_NOTE_MAX_WORDS = 70
_LENGTH_OVERAGE_TOLERANCE = 0.15  # flag/regenerate only if overshooting the target by more than this


def _word_count(text: str) -> int:
    return len(text.split())


def _length_target_description(requirement: CoverLetterRequirement, char_limit: int | None) -> str:
    if requirement == CoverLetterRequirement.SHORT_NOTE and char_limit:
        return f"under {char_limit} characters"
    max_words = _SHORT_NOTE_MAX_WORDS if requirement == CoverLetterRequirement.SHORT_NOTE else _FULL_LETTER_MAX_WORDS
    return f"under {max_words} words"


def _exceeds_length_target(
    text: str, requirement: CoverLetterRequirement, char_limit: int | None
) -> bool:
    """True if `text` overshoots its target length (full_letter/short_note word
    target, or short_note's known char_limit if set) by more than
    `_LENGTH_OVERAGE_TOLERANCE`."""
    if requirement == CoverLetterRequirement.SHORT_NOTE and char_limit:
        return len(text) > char_limit * (1 + _LENGTH_OVERAGE_TOLERANCE)
    max_words = _SHORT_NOTE_MAX_WORDS if requirement == CoverLetterRequirement.SHORT_NOTE else _FULL_LETTER_MAX_WORDS
    return _word_count(text) > max_words * (1 + _LENGTH_OVERAGE_TOLERANCE)


def _generate_draft(
    job: JobPosting,
    resume: dict,
    style_reference: str,
    client,
    revision_feedback: str | None = None,
    previous_letter: str | None = None,
    requirement: CoverLetterRequirement = CoverLetterRequirement.FULL_LETTER,
    char_limit: int | None = None,
) -> str:
    if revision_feedback and previous_letter:
        user_content = (
            f"Revise the cover letter draft below per the user's feedback. Keep it "
            f"grounded strictly in the resume - never invent anything new.\n\n"
            f"Previous draft:\n{previous_letter}\n\n"
            f"User feedback: {revision_feedback}\n\n"
            f"Job posting:\nTitle: {job.title}\n"
            f"Company: {job.company}\n"
            f"Description: {job.description}"
        )
    else:
        user_content = (
            f"Write a cover letter for this job posting:\n"
            f"Title: {job.title}\n"
            f"Company: {job.company}\n"
            f"Description: {job.description}"
        )

    if requirement == CoverLetterRequirement.SHORT_NOTE:
        length_instruction = (
            "Write a SHORT note, not a full cover letter - one or two sentences, "
            "roughly 40-70 words. No greeting or sign-off: this goes directly into "
            "a small inline application-form text box, not an email."
        )
        if char_limit:
            length_instruction += f" Stay under {char_limit} characters, no exceptions."
        sign_off_instruction = ""
    else:
        length_instruction = (
            "Write roughly 100-150 words in three short paragraphs: an opening naming "
            "the role and one concrete reason you're a fit, a middle paragraph on one "
            "specific relevant experience from the resume, and a brief closing. Do not "
            "restate the whole resume."
        )
        sign_off_instruction = f" Sign the letter as '{config.APPLICANT_NAME}'."

    targeting_instruction = (
        " Identify 2-3 concrete, specific details from the job posting text below "
        "(e.g. a named technology, responsibility, or requirement it mentions) and "
        "reference at least two of them directly, so the letter clearly responds to "
        "this exact posting rather than reading as generic. Base these details only "
        "on what the posting text actually says - never invent details about the "
        "company or role."
    )

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
                    f"{length_instruction}{targeting_instruction} Match the tone/register "
                    "of the provided style reference where reasonable, but keep it "
                    f"professional enough for a cover letter.{sign_off_instruction}"
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
        messages=[{"role": "user", "content": user_content}],
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
    job: JobPosting,
    resume_variant: ResumeVariant,
    client=None,
    revision_feedback: str | None = None,
    previous_letter: str | None = None,
    requirement: CoverLetterRequirement = CoverLetterRequirement.FULL_LETTER,
    char_limit: int | None = None,
) -> tuple[str, list[str]]:
    """Return (final_cover_letter, fact_check_issues). fact_check_issues is
    empty if the draft was approved as-is.

    Pass `revision_feedback` + `previous_letter` to revise an existing letter
    per the user's feedback instead of writing a fresh draft. Pass
    `requirement`/`char_limit` (see `jobfinder.db.models.CoverLetterRequirement`)
    to target a short inline-form note instead of a full letter.

    If the fact-checked draft overshoots its length target (100-150 words for
    full_letter, 40-70 words or the given char_limit for short_note) by more
    than ~15%, one extra generate+fact-check round is run asking the model to
    shorten it - this never happens for a well-behaved draft.
    """
    if client is None:
        client = get_client()

    resume = get_or_parse_resume(resume_variant)
    style_reference = _load_style_reference()

    draft = _generate_draft(
        job,
        resume,
        style_reference,
        client,
        revision_feedback=revision_feedback,
        previous_letter=previous_letter,
        requirement=requirement,
        char_limit=char_limit,
    )
    final_letter, issues = _fact_check(draft, resume, client)

    if _exceeds_length_target(final_letter, requirement, char_limit):
        shorten_feedback = f"Too long - cut it down to {_length_target_description(requirement, char_limit)} while keeping the same content and grounding."
        draft = _generate_draft(
            job,
            resume,
            style_reference,
            client,
            revision_feedback=shorten_feedback,
            previous_letter=final_letter,
            requirement=requirement,
            char_limit=char_limit,
        )
        final_letter, more_issues = _fact_check(draft, resume, client)
        issues = issues + more_issues

    return final_letter, issues
