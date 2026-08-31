"""Send an approval-request email for an Application awaiting the user's
sign-off, and mark it as notified.

The email includes the full cover letter, the resume choice + reasoning,
and the intended submission method, and carries `config.APP_ID_HEADER`
(a custom RFC822 header) so replies can be tied back to the exact
Application row regardless of subject-line mangling by mail clients.
"""
from __future__ import annotations

from email.message import EmailMessage

from jobfinder import config
from jobfinder.db import store
from jobfinder.db.models import Application, ApplicationStatus, JobPosting
from jobfinder.gmail.client import send_raw_message

_APPLY_METHOD_LABELS = {
    "email": "Direct email application",
    "company_site": "Automatic submission via the company's application site",
    "linkedin_easy_apply": "LinkedIn Easy Apply (I will fill the form and pause before the final click)",
}


def build_approval_email(application: Application, job: JobPosting) -> EmailMessage:
    if application.id is None:
        raise ValueError("Application must be inserted (have an id) before notifying")

    message = EmailMessage()
    message["To"] = config.GMAIL_USER_EMAIL
    message["From"] = config.GMAIL_USER_EMAIL
    message["Subject"] = f"Approve application: {job.title} at {job.company}"
    message[config.APP_ID_HEADER] = str(application.id)

    submission_label = _APPLY_METHOD_LABELS.get(job.apply_method.value, job.apply_method.value)
    body = (
        f"Job: {job.title} at {job.company}\n"
        f"Link: {job.url}\n"
        f"Submission method: {submission_label}\n"
        f"Relevance score: {application.relevance_score}\n\n"
        f"Resume chosen: {application.resume_variant.value}\n"
        f"Reason: {application.resume_choice_reason}\n\n"
        f"--- Cover letter ---\n{application.cover_letter}\n\n"
        f"Reply APPROVE to send this application, REJECT to skip it, or describe "
        f"what to change to request a revision."
    )
    message.set_content(body)
    return message


def send_approval_request(
    service, application: Application, job: JobPosting, db_path: str | None = None
) -> Application:
    """Send the approval-request email, then update the Application's
    gmail_thread_id/status in the datastore. Returns the updated Application."""
    message = build_approval_email(application, job)
    response = send_raw_message(service, message.as_bytes())

    application.gmail_thread_id = response["threadId"]
    application.gmail_last_message_id = response["id"]
    application.status = ApplicationStatus.NOTIFIED
    store.update_application(application, db_path=db_path)
    store.append_apply_log(application, "notified", "Sent approval-request email.", db_path=db_path)
    return application
