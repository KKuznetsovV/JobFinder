"""LinkedIn Easy Apply filler: fills the entire Easy Apply form but always
stops ONE STEP BEFORE the final submit, leaving the browser open and ready
for the user to review and click "Submit application" themselves.

Reuses the same custom accessibility-tree tools as
`jobfinder.local.browser_apply_fallback` (list_elements/click_by_ref/
fill_by_ref/upload_by_ref/take_screenshot), but with a stricter "finish" tool
that can only report "ready_for_submit" or "stuck" - never "submitted" - and
a code-level safeguard that refuses to click anything that looks like the
final submit button, so a prompt-injection or model mistake can't cause an
actual auto-submission of a LinkedIn application.
"""
from __future__ import annotations

import logging
import re

from jobfinder import config
from jobfinder.ai.client import create_message_with_retry, get_client
from jobfinder.db.models import Application, ApplicationStatus, JobPosting
from jobfinder.local.browser_apply_fallback import _execute_tool, snapshot_elements
from jobfinder.resume.cache import resume_path

logger = logging.getLogger(__name__)

MAX_STEPS = 20

_SUBMIT_TEXT_RE = re.compile(r"submit\s+application", re.IGNORECASE)

TOOLS = [
    {
        "name": "list_elements",
        "description": "List the interactive elements currently visible on the page, each tagged with a ref id.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "click_by_ref",
        "description": "Click the element with the given ref (from the most recent list_elements call). Use this for 'Next'/'Review' buttons, never the final submit button.",
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
        "description": (
            "Call once the form is fully filled and the ONLY remaining action is to click the "
            "final 'Submit application' button - you must NEVER click that button yourself. "
            "Use status='ready_for_submit'. Use status='stuck' if you can't proceed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ready_for_submit", "stuck"]},
                "note": {"type": "string"},
            },
            "required": ["status", "note"],
        },
    },
]


def _build_system_prompt(application: Application, job: JobPosting) -> list[dict]:
    resume_file = str(resume_path(application.resume_variant))
    return [
        {
            "type": "text",
            "text": (
                "You are filling out a LinkedIn Easy Apply form on behalf of a candidate, "
                "using only the tools provided. Use list_elements to see what's on the page, "
                "then click_by_ref/fill_by_ref/upload_by_ref to progress through each step "
                "(contact info, resume, screening questions, etc.). Never invent information "
                "beyond what's given below. "
                "CRITICAL: you must NEVER click the final 'Submit application' button - the "
                "human always does that themselves. Once every field is filled and the only "
                "remaining action would be that final submit click, call finish with "
                "status='ready_for_submit' and stop. If you get stuck, call finish with "
                "status='stuck'.\n\n"
                f"Applicant name: {config.APPLICANT_NAME}\n"
                f"Applicant email: {config.GMAIL_USER_EMAIL}\n"
                f"Applicant phone: {config.APPLICANT_PHONE or 'not provided - call finish with status=stuck if a phone field is required'}\n"
                f"Resume file path (for upload_by_ref): {resume_file}\n"
                f"Job: {job.title} at {job.company}\nJob URL: {job.url}\n\n"
                f"Cover letter text to use if the form has a cover-letter/message field:\n"
                f"{application.cover_letter}"
            ),
        }
    ]


def _guarded_execute_tool(page, ref_map: dict[str, object], name: str, tool_input: dict):
    if name == "click_by_ref":
        element = ref_map.get(tool_input.get("ref"))
        if element is not None:
            try:
                text = (element.inner_text() or "") + " " + (element.get_attribute("aria-label") or "")
            except Exception:  # noqa: BLE001 - if we can't inspect it, fall through to normal execution
                text = ""
            if _SUBMIT_TEXT_RE.search(text):
                return (
                    "refused: this looks like the final submit button - do not click it, "
                    "call finish with status='ready_for_submit' instead",
                    ref_map,
                )
    return _execute_tool(page, ref_map, name, tool_input)


def run_easy_apply_loop(
    page, application: Application, job: JobPosting, client=None, max_steps: int = MAX_STEPS
) -> ApplicationStatus:
    """Fill the LinkedIn Easy Apply form, stopping before the final submit
    click. Returns ApplicationStatus.AWAITING_MY_CLICK when ready for the
    user to submit, ApplicationStatus.STUCK otherwise."""
    if client is None:
        client = get_client()

    system = _build_system_prompt(application, job)
    messages = [
        {
            "role": "user",
            "content": "Begin filling out the LinkedIn Easy Apply form on the current page. Call list_elements first.",
        }
    ]
    ref_map: dict[str, object] = {}

    for _ in range(max_steps):
        message = create_message_with_retry(
            client, model=config.MODEL_ID, max_tokens=2048, system=system, tools=TOOLS, messages=messages
        )
        messages.append({"role": "assistant", "content": message.content})

        tool_results = []
        finished_status = None
        for block in message.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            if block.name == "finish":
                finished_status = block.input["status"]
                logger.info("browser-apply-linkedin finished: %s - %s", finished_status, block.input.get("note"))
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": "acknowledged"}
                )
                continue
            result_text, ref_map = _guarded_execute_tool(page, ref_map, block.name, block.input)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})

        if finished_status is not None:
            return (
                ApplicationStatus.AWAITING_MY_CLICK
                if finished_status == "ready_for_submit"
                else ApplicationStatus.STUCK
            )

        if not tool_results:
            logger.warning("browser-apply-linkedin: model produced no tool calls, stopping.")
            break

        messages.append({"role": "user", "content": tool_results})

    logger.warning("browser-apply-linkedin: exhausted %d steps without finishing.", max_steps)
    return ApplicationStatus.STUCK
