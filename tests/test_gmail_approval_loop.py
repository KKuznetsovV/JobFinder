import base64
from types import SimpleNamespace

from jobfinder import config
from jobfinder.db import store
from jobfinder.db.models import (
    Application,
    ApplicationStatus,
    ApplyMethod,
    JobPosting,
    ResumeVariant,
)
from jobfinder.gmail import approval_loop


def _job():
    return JobPosting(
        source="gotfriends",
        external_id="1",
        title="Backend Engineer",
        company="Acme",
        description="Build things.",
        url="https://example.com/jobs/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )


def _application(tmp_path, job, **overrides):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    saved_job = store.insert_job_posting(job, db_path)
    app = Application(
        job_posting_id=saved_job.id,
        resume_variant=ResumeVariant.FULLSTACK,
        cover_letter="Dear hiring team, ...",
        status=ApplicationStatus.NOTIFIED,
        gmail_thread_id="thread-1",
        gmail_last_message_id="msg-notify",
        **overrides,
    )
    saved = store.insert_application(app, db_path)
    return saved, db_path


def _b64(text):
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def _thread_with_reply(reply_text, message_id="msg-reply"):
    return {
        "messages": [
            {"id": "msg-notify", "payload": {"mimeType": "text/plain", "body": {}}},
            {
                "id": message_id,
                "payload": {
                    "mimeType": "text/plain",
                    "body": {"data": _b64(reply_text)},
                },
            },
        ]
    }


def test_no_new_reply_returns_unchanged(tmp_path, mocker):
    job = _job()
    application, db_path = _application(tmp_path, job)
    mocker.patch(
        "jobfinder.gmail.approval_loop.get_thread",
        return_value={"messages": [{"id": "msg-notify", "payload": {}}]},
    )

    result = approval_loop.process_reply(object(), application, job, db_path=db_path)

    assert result.status == ApplicationStatus.NOTIFIED


def test_approve_reply_sets_approved(tmp_path, mocker):
    job = _job()
    application, db_path = _application(tmp_path, job)
    mocker.patch(
        "jobfinder.gmail.approval_loop.get_thread", return_value=_thread_with_reply("APPROVE")
    )
    mocker.patch(
        "jobfinder.gmail.approval_loop._classify_reply", return_value=("approve", "")
    )

    result = approval_loop.process_reply(object(), application, job, db_path=db_path)

    assert result.status == ApplicationStatus.APPROVED
    reloaded = store.get_application(application.id, db_path=db_path)
    assert reloaded.status == ApplicationStatus.APPROVED


def test_reject_reply_sets_rejected(tmp_path, mocker):
    job = _job()
    application, db_path = _application(tmp_path, job)
    mocker.patch(
        "jobfinder.gmail.approval_loop.get_thread", return_value=_thread_with_reply("no thanks")
    )
    mocker.patch(
        "jobfinder.gmail.approval_loop._classify_reply", return_value=("reject", "")
    )

    result = approval_loop.process_reply(object(), application, job, db_path=db_path)

    assert result.status == ApplicationStatus.REJECTED


def test_revise_reply_increments_count_and_sets_status(tmp_path, mocker):
    job = _job()
    application, db_path = _application(tmp_path, job)
    mocker.patch(
        "jobfinder.gmail.approval_loop.get_thread",
        return_value=_thread_with_reply("please mention Python more"),
    )
    mocker.patch(
        "jobfinder.gmail.approval_loop._classify_reply",
        return_value=("revise", "mention Python more"),
    )

    result = approval_loop.process_reply(object(), application, job, db_path=db_path)

    assert result.status == ApplicationStatus.REVISION_REQUESTED
    assert result.revision_count == 1


def test_revise_reply_at_cap_sends_clarifying_email_instead(tmp_path, mocker):
    job = _job()
    application, db_path = _application(
        tmp_path, job, revision_count=config.MAX_REVISION_ROUNDS
    )
    mocker.patch(
        "jobfinder.gmail.approval_loop.get_thread",
        return_value=_thread_with_reply("please revise again"),
    )
    mocker.patch(
        "jobfinder.gmail.approval_loop._classify_reply",
        return_value=("revise", "revise again"),
    )
    send_spy = mocker.patch("jobfinder.gmail.approval_loop.send_raw_message")

    result = approval_loop.process_reply(object(), application, job, db_path=db_path)

    assert result.revision_count == config.MAX_REVISION_ROUNDS
    send_spy.assert_called_once()


def test_ambiguous_reply_sends_clarifying_email_and_keeps_status(tmp_path, mocker):
    job = _job()
    application, db_path = _application(tmp_path, job)
    mocker.patch(
        "jobfinder.gmail.approval_loop.get_thread",
        return_value=_thread_with_reply("hmm not sure"),
    )
    mocker.patch(
        "jobfinder.gmail.approval_loop._classify_reply",
        return_value=("ambiguous", "unclear what you want"),
    )
    send_spy = mocker.patch("jobfinder.gmail.approval_loop.send_raw_message")

    result = approval_loop.process_reply(object(), application, job, db_path=db_path)

    assert result.status == ApplicationStatus.NOTIFIED
    send_spy.assert_called_once()


def test_extract_message_text_prefers_plain():
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": _b64("plain text")}},
            {"mimeType": "text/html", "body": {"data": _b64("<p>html text</p>")}},
        ],
    }
    assert approval_loop._extract_message_text(payload) == "plain text"
