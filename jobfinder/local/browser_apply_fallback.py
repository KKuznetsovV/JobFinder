"""Fallback application filler for job postings whose company career page
isn't a recognized ATS platform (`jobfinder.local.ats.get_adapter_for_url`
returned None), or whose recognized adapter raised `ATSFormError` (e.g.
Workday's multi-step wizard).

Per an explicit user preference, this uses CUSTOM accessibility-tree-based
tools (list_elements/click_by_ref/fill_by_ref/upload_by_ref, with
take_screenshot only as a last-resort fallback) driven by Claude tool-use,
rather than Anthropic's built-in computer-use/browser-use toolsets (which are
screenshot+pixel-coordinate driven). This is a deliberate deviation - see the
README's "Assumptions and deviations" section.

Never invents form values: only the applicant's name/email (config), the
resume file path, the cover letter text, and the job posting fields are
provided to the model - it must ground every filled value in those.
"""
from __future__ import annotations

import logging
from typing import Any

import anthropic
from playwright.sync_api import Locator, Page

from jobfinder import config
from jobfinder.ai import cover_letter_requirement
from jobfinder.ai.client import create_message_with_retry, get_client
from jobfinder.db.models import Application, ApplicationStatus, JobPosting
from jobfinder.resume.cache import resume_path

logger = logging.getLogger(__name__)

MAX_STEPS = 40

_INTERACTIVE_SELECTOR = "input, textarea, select, button, a[href], [role='button'], [contenteditable='true']"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_elements",
        "description": "List the interactive elements currently visible on the page, each tagged with a ref id.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "click_by_ref",
        "description": "Click the element with the given ref (from the most recent list_elements call).",
        "input_schema": {
            "type": "object",
            "properties": {"ref": {"type": "string"}},
            "required": ["ref"],
        },
    },
    {
        "name": "fill_by_ref",
        "description": "Fill a text input/textarea with the given ref.",
        "input_schema": {
            "type": "object",
            "properties": {"ref": {"type": "string"}, "value": {"type": "string"}},
            "required": ["ref", "value"],
        },
    },
    {
        "name": "upload_by_ref",
        "description": "Upload a file (e.g. the resume) to a file-input element with the given ref.",
        "input_schema": {
            "type": "object",
            "properties": {"ref": {"type": "string"}, "file_path": {"type": "string"}},
            "required": ["ref", "file_path"],
        },
    },
    {
        "name": "take_screenshot",
        "description": "Take a screenshot as a last-resort fallback when the accessibility tree isn't enough to understand the layout.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "finish",
        "description": "Call when the application has been fully submitted, or if you get stuck and can't proceed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["submitted", "stuck"]},
                "note": {"type": "string"},
                "cover_letter_field_found": {
                    "type": "string",
                    "enum": ["full_letter_field", "short_field", "no_field"],
                    "description": (
                        "What kind of cover-letter field you actually found on this form, "
                        "if any - used to check the upfront classification was right. Omit "
                        "if not applicable/unsure."
                    ),
                },
            },
            "required": ["status", "note"],
        },
    },
]


def snapshot_elements(page: Page) -> tuple[str, dict[str, Locator]]:
    """Return (text_description, ref_map) where ref_map maps ref strings to
    Playwright Locators for the currently visible interactive elements."""
    locator = page.locator(_INTERACTIVE_SELECTOR)
    count = locator.count()
    ref_map: dict[str, Locator] = {}
    lines: list[str] = []
    for i in range(count):
        element = locator.nth(i)
        ref = f"e{i}"
        ref_map[ref] = element
        tag = element.evaluate("el => el.tagName.toLowerCase()")
        role = element.get_attribute("role") or tag
        input_type = element.get_attribute("type") or ""
        name = (
            element.get_attribute("aria-label")
            or element.get_attribute("placeholder")
            or element.inner_text()
            or ""
        ).strip()[:80]
        type_suffix = f" type={input_type}" if input_type else ""
        lines.append(f"[{ref}] <{tag}{type_suffix}> {role}: {name!r}")
    return "\n".join(lines), ref_map


def _build_system_prompt(application: Application, job: JobPosting) -> list[dict[str, Any]]:
    resume_file = str(resume_path(application.resume_variant))
    adapted_cover_letter = cover_letter_requirement.adapt_cover_letter_locally(
        application.cover_letter, application.cover_letter_requirement, application.cover_letter_char_limit
    )
    if adapted_cover_letter:
        cover_letter_instruction = (
            f"Cover letter text to use if the form has a cover-letter field:\n{adapted_cover_letter}"
        )
    else:
        cover_letter_instruction = (
            "No cover letter was drafted for this application (it isn't needed). If the "
            "form unexpectedly has a cover-letter field, leave it blank rather than inventing "
            "text, and report cover_letter_field_found accordingly when you finish."
        )
    return [
        {
            "type": "text",
            "text": (
                "You are filling out a job application form in a real browser on behalf "
                "of a candidate, using only the tools provided. Use list_elements to see "
                "what's on the page, then click_by_ref/fill_by_ref/upload_by_ref to "
                "interact with it. Only use take_screenshot if the accessibility tree "
                "genuinely isn't enough to proceed. Never invent information beyond what's "
                "given below. When the form has been fully submitted, call finish with "
                "status='submitted'. If you get stuck (a required field you can't "
                "determine a value for, a CAPTCHA, a broken page, etc.), call finish with "
                "status='stuck' and explain why in note. Whenever you call finish, also "
                "report cover_letter_field_found based on what you actually saw on this "
                "form.\n\n"
                f"Applicant name: {config.APPLICANT_NAME}\n"
                f"Applicant email: {config.GMAIL_USER_EMAIL}\n"
                f"Applicant phone: {config.APPLICANT_PHONE or 'not provided - call finish with status=stuck if a phone field is required'}\n"
                f"Resume file path (for upload_by_ref): {resume_file}\n"
                f"Job: {job.title} at {job.company}\nJob URL: {job.url}\n\n"
                f"{cover_letter_instruction}"
            ),
        }
    ]


def execute_tool(
    page: Page, ref_map: dict[str, Locator], name: str, tool_input: dict[str, Any]
) -> tuple[str, dict[str, Locator]]:
    """Execute one tool call. Returns (result_text, possibly-updated ref_map)."""
    try:
        if name == "list_elements":
            text, ref_map = snapshot_elements(page)
            return text, ref_map
        if name == "click_by_ref":
            ref_map[tool_input["ref"]].click()
            return "clicked", ref_map
        if name == "fill_by_ref":
            ref_map[tool_input["ref"]].fill(tool_input["value"])
            return "filled", ref_map
        if name == "upload_by_ref":
            ref_map[tool_input["ref"]].set_input_files(tool_input["file_path"])
            return "uploaded", ref_map
        if name == "take_screenshot":
            page.screenshot()
            return "screenshot taken", ref_map
        return f"unknown tool: {name}", ref_map
    except KeyError:
        return f"error: unknown ref '{tool_input.get('ref')}' - call list_elements again", ref_map
    except Exception as exc:  # noqa: BLE001 - report any Playwright error back to the model, don't crash
        return f"error: {exc}", ref_map


def run_apply_loop(
    page: Page,
    application: Application,
    job: JobPosting,
    client: anthropic.Anthropic | None = None,
    max_steps: int = MAX_STEPS,
) -> ApplicationStatus:
    """Drive the accessibility-tree tool loop to fill and submit the
    application. Returns ApplicationStatus.SENT on a successful submission,
    ApplicationStatus.STUCK otherwise (explicit stuck call, tool loop
    exhausted, or the model stopped calling tools without finishing)."""
    if client is None:
        client = get_client()

    system = _build_system_prompt(application, job)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "Begin filling out the job application on the current page. Call list_elements first.",
        }
    ]
    ref_map: dict[str, Locator] = {}

    for _ in range(max_steps):
        message = create_message_with_retry(
            client, model=config.MODEL_ID, max_tokens=2048, system=system, tools=TOOLS, messages=messages
        )
        messages.append({"role": "assistant", "content": message.content})

        tool_results: list[dict[str, Any]] = []
        finished_status = None
        for block in message.content:
            if block.type != "tool_use":
                continue
            if block.name == "finish":
                finished_status = block.input["status"]
                logger.info("browser-apply-fallback finished: %s - %s", finished_status, block.input.get("note"))
                raw_field_found = block.input.get("cover_letter_field_found")
                cover_letter_requirement.log_if_classification_mismatched(
                    application, raw_field_found if isinstance(raw_field_found, str) else None
                )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": "acknowledged"}
                )
                continue
            result_text, ref_map = execute_tool(page, ref_map, block.name, block.input)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})

        if finished_status is not None:
            return ApplicationStatus.SENT if finished_status == "submitted" else ApplicationStatus.STUCK

        if not tool_results:
            logger.warning("browser-apply-fallback: model produced no tool calls, stopping.")
            break

        messages.append({"role": "user", "content": tool_results})

    logger.warning("browser-apply-fallback: exhausted %d steps without finishing.", max_steps)
    return ApplicationStatus.STUCK
