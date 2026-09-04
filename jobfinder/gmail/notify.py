"""Send approval-request emails for the two-stage approval flow, and mark
applications as notified.

Stage A (`send_stage_a_request`): sent right after resume selection, before
any cover letter is generated - job details, chosen resume + one-line
reason, no cover letter. Cheap to produce, meant to let the user reject a
bad match before any cover-letter LLM call is made.

Stage B (`send_approval_request`, unchanged from the original single-stage
flow apart from its subject line): sent once Stage A is approved - includes
the full cover letter and asks for final sign-off before submission.

Both carry `config.APP_ID_HEADER` (a custom RFC822 header) so replies can be
tied back to the exact Application row regardless of subject-line mangling
by mail clients.
"""
from __future__ import annotations

from email.message import EmailMessage

from googleapiclient.discovery import Resource  # type: ignore

from jobfinder import config
from jobfinder.db import store
from jobfinder.db.models import Application, ApplicationStatus, JobPosting
from jobfinder.gmail.client import send_raw_message

_APPLY_METHOD_LABELS = {
    "email": "Direct email application",
    "company_site": "Automatic submission via the company's application site",
    "linkedin_easy_apply": "LinkedIn Easy Apply (I will fill the form and pause before the final click)",
}


def build_stage_a_email(application: Application, job: JobPosting, revised: bool = False) -> EmailMessage:
    """Build the Stage A (job + resume match) approval-request email. No
    cover letter exists yet at this point - that's only generated after
    Stage A is approved, to avoid spending a Claude call on a bad match.
    """
    if application.id is None:
        raise ValueError("Application must be inserted (have an id) before notifying")

    message = EmailMessage()
    message["To"] = config.GMAIL_USER_EMAIL
    message["From"] = config.GMAIL_USER_EMAIL
    subject_prefix = "Re: " if revised else ""
    message["Subject"] = f"{subject_prefix}New match: {job.title} at {job.company} \u2014 resume + job review"
    message[config.APP_ID_HEADER] = str(application.id)

    submission_label = _APPLY_METHOD_LABELS.get(job.apply_method.value, job.apply_method.value)
    summary = (job.description or "")[:500]
    tier0_line = (
        f"Local match score: {application.tier0_score:.2f}\n" if application.tier0_score is not None else ""
    )
    body = (
        f"Job: {job.title} at {job.company}\n"
        f"Link: {job.url}\n"
        f"Submission method: {submission_label}\n"
        f"Relevance score: {application.relevance_score}\n"
        f"{tier0_line}\n"
        f"Resume chosen: {application.resume_variant.value}\n"
        f"Reason: {application.resume_choice_reason}\n\n"
        f"Job summary: {summary}\n\n"
        f"No cover letter has been drafted yet. Reply APPROVE to have me draft one and "
        f"continue, REJECT to skip this one, or tell me if you'd rather use the other resume."
    )
    message.set_content(body)
    return message


def send_stage_a_request(
    service: Resource, application: Application, job: JobPosting, db_path: str | None = None
) -> Application:
    """Send the initial Stage A email and mark the application as pending
    Stage A approval."""
    message = build_stage_a_email(application, job)
    response = send_raw_message(service, message.as_bytes())

    application.gmail_thread_id = response["threadId"]
    application.gmail_last_message_id = response["id"]
    application.status = ApplicationStatus.PENDING_STAGE_A_APPROVAL
    store.update_application(application, db_path=db_path)
    store.append_apply_log(
        application, "stage_a_notified", "Sent stage A (job+resume) approval email.", db_path=db_path
    )
    return application


def send_stage_a_revision(
    service: Resource, application: Application, job: JobPosting, db_path: str | None = None
) -> Application:
    """Re-send the Stage A email in the same thread after the resume choice
    was corrected per the user's feedback."""
    message = build_stage_a_email(application, job, revised=True)
    response = send_raw_message(service, message.as_bytes(), thread_id=application.gmail_thread_id)

    application.gmail_last_message_id = response["id"]
    application.status = ApplicationStatus.PENDING_STAGE_A_APPROVAL
    store.update_application(application, db_path=db_path)
    store.append_apply_log(
        application, "stage_a_notified", "Resent stage A email after resume correction.", db_path=db_path
    )
    return application


def build_approval_email(application: Application, job: JobPosting) -> EmailMessage:
    if application.id is None:
        raise ValueError("Application must be inserted (have an id) before notifying")

    message = EmailMessage()
    message["To"] = config.GMAIL_USER_EMAIL
    message["From"] = config.GMAIL_USER_EMAIL
    message["Subject"] = f"Cover letter ready: {job.title} at {job.company}"
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
    service: Resource, application: Application, job: JobPosting, db_path: str | None = None
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


def build_stage_b_no_letter_email(application: Application, job: JobPosting) -> EmailMessage:
    """Build the lightweight Stage B email for postings whose application
    form has no cover-letter field at all (`cover_letter_requirement=NONE`) -
    same final-sign-off ask as `build_approval_email`, just without a cover
    letter section since none was drafted."""
    if application.id is None:
        raise ValueError("Application must be inserted (have an id) before notifying")

    message = EmailMessage()
    message["To"] = config.GMAIL_USER_EMAIL
    message["From"] = config.GMAIL_USER_EMAIL
    message["Subject"] = f"Ready to apply: {job.title} at {job.company} (no cover letter needed)"
    message[config.APP_ID_HEADER] = str(application.id)

    submission_label = _APPLY_METHOD_LABELS.get(job.apply_method.value, job.apply_method.value)
    body = (
        f"Job: {job.title} at {job.company}\n"
        f"Link: {job.url}\n"
        f"Submission method: {submission_label}\n"
        f"Relevance score: {application.relevance_score}\n\n"
        f"Resume chosen: {application.resume_variant.value}\n"
        f"Reason: {application.resume_choice_reason}\n\n"
        f"This application's form has no cover-letter field, so none was drafted.\n\n"
        f"Reply APPROVE to send this application, or REJECT to skip it."
    )
    message.set_content(body)
    return message


def send_stage_b_no_letter_request(
    service: Resource, application: Application, job: JobPosting, db_path: str | None = None
) -> Application:
    """Send the lightweight no-cover-letter Stage B email, then update the
    Application's gmail_thread_id/status in the datastore, same as
    `send_approval_request`."""
    message = build_stage_b_no_letter_email(application, job)
    response = send_raw_message(service, message.as_bytes())

    application.gmail_thread_id = response["threadId"]
    application.gmail_last_message_id = response["id"]
    application.status = ApplicationStatus.NOTIFIED
    store.update_application(application, db_path=db_path)
    store.append_apply_log(
        application, "notified", "Sent lightweight approval-request email (no cover letter needed).", db_path=db_path
    )
    return application


def build_revision_email(application: Application, job: JobPosting) -> EmailMessage:
    if application.id is None:
        raise ValueError("Application must be inserted (have an id) before notifying")

    message = EmailMessage()
    message["To"] = config.GMAIL_USER_EMAIL
    message["From"] = config.GMAIL_USER_EMAIL
    message["Subject"] = f"Re: Cover letter ready: {job.title} at {job.company}"
    message[config.APP_ID_HEADER] = str(application.id)

    submission_label = _APPLY_METHOD_LABELS.get(job.apply_method.value, job.apply_method.value)
    body = (
        f"Updated per your feedback.\n\n"
        f"Job: {job.title} at {job.company}\n"
        f"Link: {job.url}\n"
        f"Submission method: {submission_label}\n"
        f"Relevance score: {application.relevance_score}\n\n"
        f"Resume chosen: {application.resume_variant.value}\n"
        f"Reason: {application.resume_choice_reason}\n\n"
        f"--- Revised cover letter ---\n{application.cover_letter}\n\n"
        f"Reply APPROVE to send this application, REJECT to skip it, or describe "
        f"what to change to request another revision."
    )
    message.set_content(body)
    return message


def send_revision_request(
    service: Resource, application: Application, job: JobPosting, db_path: str | None = None
) -> Application:
    """Re-send the approval-request email in the same thread after the cover
    letter was revised per the user's feedback. Returns the updated Application."""
    message = build_revision_email(application, job)
    response = send_raw_message(service, message.as_bytes(), thread_id=application.gmail_thread_id)

    application.gmail_last_message_id = response["id"]
    application.status = ApplicationStatus.NOTIFIED
    store.update_application(application, db_path=db_path)
    store.append_apply_log(
        application, "notified", "Sent revised approval-request email.", db_path=db_path
    )
    return application
