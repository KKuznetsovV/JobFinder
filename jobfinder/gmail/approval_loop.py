"""Poll a notified Application's Gmail thread for the user's reply, classify
it via Claude into approve/reject/revise/ambiguous, and update the
Application accordingly.

Never submits an application without an unambiguous "approve" tied to a
specific Application's thread (matched by `config.APP_ID_HEADER` when the
notification was sent, and by `gmail_thread_id` thereafter). Ambiguous
replies trigger a clarifying email rather than guessing. Revision requests
are capped at `config.MAX_REVISION_ROUNDS`.
"""
from __future__ import annotations

import base64
from email.message import EmailMessage

from jobfinder import config
from jobfinder.ai.client import create_message_with_retry, get_client
from jobfinder.db import store
from jobfinder.db.models import Application, ApplicationStatus, JobPosting
from jobfinder.gmail.client import get_thread, send_raw_message

_CLASSIFY_TOOL = {
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


def _extract_message_text(payload: dict) -> str:
    """Prefer text/plain; fall back to text/html (best-effort, tags left in)."""

    def _decode(body_data: str) -> str:
        padded = body_data + "=" * (-len(body_data) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")

    def _walk(part: dict, wanted_mime: str) -> str | None:
        body_data = part.get("body", {}).get("data")
        if part.get("mimeType") == wanted_mime and body_data:
            return _decode(body_data)
        for sub_part in part.get("parts", []):
            found = _walk(sub_part, wanted_mime)
            if found:
                return found
        return None

    return _walk(payload, "text/plain") or _walk(payload, "text/html") or ""


def _latest_reply_text(service, application: Application) -> tuple[str | None, str | None]:
    """Return (message_id, text) for the newest message in the thread that
    isn't the notification message we already sent, or (None, None) if there
    is no such reply yet."""
    thread = get_thread(service, application.gmail_thread_id)
    messages = thread.get("messages", [])
    for message in reversed(messages):
        if message["id"] == application.gmail_last_message_id:
            continue
        return message["id"], _extract_message_text(message.get("payload", {}))
    return None, None


def _classify_reply(reply_text: str, client=None) -> tuple[str, str]:
    if client is None:
        client = get_client()

    message = create_message_with_retry(
        client,
        model=config.MODEL_ID,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": (
                    "Classify a user's email reply to a job-application approval request "
                    "as one of: approve, reject, revise, ambiguous. Only classify as "
                    "'approve' if the reply is unambiguous. If unsure, use 'ambiguous'. "
                    "Always call classify_reply."
                ),
            }
        ],
        tools=[_CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "classify_reply"},
        messages=[{"role": "user", "content": reply_text}],
    )

    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "classify_reply":
            data = block.input
            return str(data["classification"]), str(data.get("feedback", ""))

    raise ValueError("Claude did not call classify_reply as expected")


def _send_clarifying_email(service, application: Application, job: JobPosting, note: str) -> None:
    reply = EmailMessage()
    reply["To"] = config.GMAIL_USER_EMAIL
    reply["From"] = config.GMAIL_USER_EMAIL
    reply["Subject"] = f"Re: Approve application: {job.title} at {job.company}"
    reply[config.APP_ID_HEADER] = str(application.id)
    reply.set_content(
        f"I couldn't tell what you'd like to do with this application. {note}\n\n"
        "Please reply with APPROVE, REJECT, or specific revision feedback."
    )
    send_raw_message(service, reply.as_bytes(), thread_id=application.gmail_thread_id)


def process_reply(
    service, application: Application, job: JobPosting, db_path: str | None = None, client=None
) -> Application:
    """Check for a new reply in the application's thread and act on it.
    No-op (returns application unchanged) if there's no new reply yet."""
    message_id, reply_text = _latest_reply_text(service, application)
    if message_id is None:
        return application

    classification, feedback = _classify_reply(reply_text, client=client)
    application.gmail_last_message_id = message_id

    if classification == "approve":
        application.status = ApplicationStatus.APPROVED
        store.update_application(application, db_path=db_path)
        store.append_apply_log(application, "approved", "User approved via email reply.", db_path=db_path)
    elif classification == "reject":
        application.status = ApplicationStatus.REJECTED
        store.update_application(application, db_path=db_path)
        store.append_apply_log(application, "rejected", "User rejected via email reply.", db_path=db_path)
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
