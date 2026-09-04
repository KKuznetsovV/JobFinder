"""Poll a notified Application's Gmail thread for the user's reply, classify
it via Claude into approve/reject/revise/ambiguous, and update the
Application accordingly.

Two-stage approval: `process_stage_a_reply` handles replies to the Stage A
(job+resume match) email - approve triggers cover-letter generation (the
expensive Claude call) via `jobfinder.pipeline.process_stage_a_approval`;
reject/revise never reach that call. `process_reply` (unchanged in
mechanics) now exclusively handles Stage B (cover-letter) replies.

Never submits an application without an unambiguous "approve" tied to a
specific Application's thread (matched by `config.APP_ID_HEADER` when the
notification was sent, and by `gmail_thread_id` thereafter). Ambiguous
replies trigger a clarifying email rather than guessing. Revision requests
are capped at `config.MAX_REVISION_ROUNDS` (tracked separately per stage:
`revision_count` for Stage B, `stage_a_revision_count` for Stage A).
"""
from __future__ import annotations

import base64
from email.message import EmailMessage
from typing import Any

import anthropic
from googleapiclient.discovery import Resource  # type: ignore

from jobfinder import config
from jobfinder.ai.client import create_message_with_retry, get_client
from jobfinder.db import store
from jobfinder.db.models import Application, ApplicationStatus, JobPosting, ResumeVariant, utcnow_iso
from jobfinder.gmail import notify
from jobfinder.gmail.client import get_thread, send_raw_message

_CLASSIFY_TOOL: dict[str, Any] = {
    "name": "classify_reply",
    "description": "Classify the user's reply to an application approval-request email.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": ["approve", "reject", "revise", "ambiguous"],
            },
            "feedback": {
                "type": "string",
                "description": (
                    "For 'revise', the specific feedback to apply. For 'ambiguous', "
                    "a short note on what's unclear. Empty string otherwise."
                ),
            },
        },
        "required": ["classification", "feedback"],
    },
}


def _extract_message_text(payload: dict[str, Any]) -> str:
    """Prefer text/plain; fall back to text/html (best-effort, tags left in)."""

    def _decode(body_data: str) -> str:
        padded = body_data + "=" * (-len(body_data) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")

    def _walk(part: dict[str, Any], wanted_mime: str) -> str | None:
        body_data = part.get("body", {}).get("data")
        if part.get("mimeType") == wanted_mime and body_data:
            return _decode(body_data)
        for sub_part in part.get("parts", []):
            found = _walk(sub_part, wanted_mime)
            if found:
                return found
        return None

    return _walk(payload, "text/plain") or _walk(payload, "text/html") or ""


def _latest_reply_text(service: Resource, application: Application) -> tuple[str | None, str | None]:
    """Return (message_id, text) for the newest message in the thread that
    isn't the notification message we already sent, or (None, None) if there
    is no such reply yet."""
    if application.gmail_thread_id is None:
        return None, None
    thread = get_thread(service, application.gmail_thread_id)
    messages = thread.get("messages", [])
    for message in reversed(messages):
        if message["id"] == application.gmail_last_message_id:
            continue
        return message["id"], _extract_message_text(message.get("payload", {}))
    return None, None


def _classify_reply(
    reply_text: str, client: anthropic.Anthropic | None = None, stage_context: str = ""
) -> tuple[str, str]:
    if client is None:
        client = get_client()

    system_text = (
        "Classify a user's email reply to a job-application approval request "
        "as one of: approve, reject, revise, ambiguous. Only classify as "
        "'approve' if the reply is unambiguous. If unsure, use 'ambiguous'. "
        "Always call classify_reply."
    )
    if stage_context:
        system_text += f" {stage_context}"

    message = create_message_with_retry(
        client,
        model=config.MODEL_ID,
        max_tokens=512,
        system=[{"type": "text", "text": system_text}],
        tools=[_CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "classify_reply"},
        messages=[{"role": "user", "content": reply_text}],
    )

    for block in message.content:
        if block.type == "tool_use" and block.name == "classify_reply":
            data: dict[str, Any] = block.input  # type: ignore[assignment]
            return str(data["classification"]), str(data.get("feedback", ""))

    raise ValueError("Claude did not call classify_reply as expected")


def _send_clarifying_email(
    service: Resource, application: Application, job: JobPosting, note: str, subject: str | None = None
) -> None:
    reply = EmailMessage()
    reply["To"] = config.GMAIL_USER_EMAIL
    reply["From"] = config.GMAIL_USER_EMAIL
    reply["Subject"] = subject or f"Re: Cover letter ready: {job.title} at {job.company}"
    reply[config.APP_ID_HEADER] = str(application.id)
    reply.set_content(
        f"I couldn't tell what you'd like to do with this application. {note}\n\n"
        "Please reply with APPROVE, REJECT, or specific revision feedback."
    )
    send_raw_message(service, reply.as_bytes(), thread_id=application.gmail_thread_id)


def _other_resume_variant(variant: ResumeVariant) -> ResumeVariant:
    return ResumeVariant.PROJECT_MANAGER if variant == ResumeVariant.FULLSTACK else ResumeVariant.FULLSTACK


def _stage_a_subject(job: JobPosting) -> str:
    return f"Re: New match: {job.title} at {job.company} \u2014 resume + job review"


def process_stage_a_reply(
    service: Resource,
    application: Application,
    job: JobPosting,
    db_path: str | None = None,
    client: anthropic.Anthropic | None = None,
) -> Application:
    """Check for a new reply to the Stage A (job + resume match) notification
    and act on it. No-op (returns application unchanged) if there's no new
    reply yet.

    Approve triggers `jobfinder.pipeline.process_stage_a_approval`, which
    generates the cover letter (the expensive Claude call) and sends the
    Stage B email - so that call only ever happens after this approval.
    Reject and "stuck" revise never reach it.
    """
    from jobfinder import pipeline  # local import: pipeline doesn't import this module

    message_id, reply_text = _latest_reply_text(service, application)
    if message_id is None or reply_text is None:
        return application

    classification, feedback = _classify_reply(
        reply_text,
        client=client,
        stage_context=(
            "This is Stage A: the user is only approving the job match and resume choice - "
            "no cover letter has been drafted yet. 'Revise' here means asking for the other "
            "resume variant, not editing a cover letter."
        ),
    )
    application.gmail_last_message_id = message_id

    if classification == "approve":
        application.stage_a_approved_at = utcnow_iso()
        store.append_apply_log(
            application, "stage_a_approved", "User approved job+resume via email reply.", db_path=db_path
        )
        application = pipeline.process_stage_a_approval(
            application, job, service, db_path=db_path, client=client
        )
    elif classification == "reject":
        application.status = ApplicationStatus.REJECTED_STAGE_A
        store.update_application(application, db_path=db_path)
        store.append_apply_log(
            application, "rejected_stage_a", "User rejected via email reply.", db_path=db_path
        )
    elif classification == "revise":
        if application.stage_a_revision_count >= config.MAX_REVISION_ROUNDS:
            store.update_application(application, db_path=db_path)
            _send_clarifying_email(
                service,
                application,
                job,
                f"You've reached the {config.MAX_REVISION_ROUNDS}-revision limit for this application.",
                subject=_stage_a_subject(job),
            )
            store.append_apply_log(application, "stage_a_revision_limit_reached", feedback, db_path=db_path)
        else:
            application.stage_a_revision_count += 1
            application.resume_variant = _other_resume_variant(application.resume_variant)
            application.resume_choice_reason = feedback or "User requested the other resume."
            store.update_application(application, db_path=db_path)
            store.append_apply_log(application, "stage_a_revision_applied", feedback, db_path=db_path)
            application = notify.send_stage_a_revision(service, application, job, db_path=db_path)
    else:  # ambiguous
        store.update_application(application, db_path=db_path)
        _send_clarifying_email(service, application, job, feedback, subject=_stage_a_subject(job))
        store.append_apply_log(application, "stage_a_ambiguous_reply", feedback, db_path=db_path)

    return application


def process_reply(
    service: Resource,
    application: Application,
    job: JobPosting,
    db_path: str | None = None,
    client: anthropic.Anthropic | None = None,
) -> Application:
    """Check for a new reply in the Stage B (cover-letter) application's
    thread and act on it. No-op (returns application unchanged) if there's
    no new reply yet."""
    message_id, reply_text = _latest_reply_text(service, application)
    if message_id is None or reply_text is None:
        return application

    classification, feedback = _classify_reply(
        reply_text,
        client=client,
        stage_context=(
            "This is Stage B: the user already approved the job+resume match and is now "
            "reviewing the drafted cover letter before final submission."
        ),
    )
    application.gmail_last_message_id = message_id

    if classification == "approve":
        application.status = ApplicationStatus.APPROVED
        store.update_application(application, db_path=db_path)
        store.append_apply_log(application, "approved", "User approved via email reply.", db_path=db_path)
    elif classification == "reject":
        application.status = ApplicationStatus.REJECTED_STAGE_B
        store.update_application(application, db_path=db_path)
        store.append_apply_log(
            application, "rejected_stage_b", "User rejected via email reply.", db_path=db_path
        )
    elif classification == "revise":
        if application.revision_count >= config.MAX_REVISION_ROUNDS:
            store.update_application(application, db_path=db_path)
            _send_clarifying_email(
                service,
                application,
                job,
                f"You've reached the {config.MAX_REVISION_ROUNDS}-revision limit for this application.",
            )
            store.append_apply_log(
                application, "revision_limit_reached", feedback, db_path=db_path
            )
        else:
            application.revision_count += 1
            application.status = ApplicationStatus.REVISION_REQUESTED
            store.update_application(application, db_path=db_path)
            store.append_apply_log(application, "revision_requested", feedback, db_path=db_path)
    else:  # ambiguous
        store.update_application(application, db_path=db_path)
        _send_clarifying_email(service, application, job, feedback)
        store.append_apply_log(application, "ambiguous_reply", feedback, db_path=db_path)

    return application
