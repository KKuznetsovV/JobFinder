import base64

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
        resume_choice_reason="keyword match",
        status=ApplicationStatus.PENDING_STAGE_A_APPROVAL,
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
                "payload": {"mimeType": "text/plain", "body": {"data": _b64(reply_text)}},
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

    result = approval_loop.process_stage_a_reply(object(), application, job, db_path=db_path)

    assert result.status == ApplicationStatus.PENDING_STAGE_A_APPROVAL


def test_approve_triggers_cover_letter_generation_via_pipeline(tmp_path, mocker):
    job = _job()
    application, db_path = _application(tmp_path, job)
    mocker.patch(
        "jobfinder.gmail.approval_loop.get_thread", return_value=_thread_with_reply("APPROVE")
    )
    mocker.patch("jobfinder.gmail.approval_loop._classify_reply", return_value=("approve", ""))
    stage_a_approval_spy = mocker.patch(
        "jobfinder.pipeline.process_stage_a_approval",
        side_effect=lambda app, job, service, db_path=None, client=None: app,
    )

    result = approval_loop.process_stage_a_reply(object(), application, job, db_path=db_path)

    stage_a_approval_spy.assert_called_once()
    assert result.stage_a_approved_at is not None
    reloaded = store.get_application(application.id, db_path=db_path)
    events = [entry["event"] for entry in reloaded.apply_log]
    assert "stage_a_approved" in events


def test_reject_sets_rejected_stage_a_without_cover_letter_call(tmp_path, mocker):
    job = _job()
    application, db_path = _application(tmp_path, job)
    mocker.patch(
        "jobfinder.gmail.approval_loop.get_thread", return_value=_thread_with_reply("no thanks")
    )
    mocker.patch("jobfinder.gmail.approval_loop._classify_reply", return_value=("reject", ""))
    cover_letter_spy = mocker.patch("jobfinder.ai.cover_letter.generate_cover_letter")

    result = approval_loop.process_stage_a_reply(object(), application, job, db_path=db_path)

    assert result.status == ApplicationStatus.REJECTED_STAGE_A
    cover_letter_spy.assert_not_called()


def test_revise_switches_resume_variant_and_resends_stage_a(tmp_path, mocker):
    job = _job()
    application, db_path = _application(tmp_path, job)
    mocker.patch(
        "jobfinder.gmail.approval_loop.get_thread",
        return_value=_thread_with_reply("use the PM resume instead"),
    )
    mocker.patch(
        "jobfinder.gmail.approval_loop._classify_reply",
        return_value=("revise", "use the PM resume instead"),
    )
    send_spy = mocker.patch(
        "jobfinder.gmail.approval_loop.notify.send_stage_a_revision",
        side_effect=lambda service, app, job, db_path=None: app,
    )

    result = approval_loop.process_stage_a_reply(object(), application, job, db_path=db_path)

    assert result.status == ApplicationStatus.PENDING_STAGE_A_APPROVAL
    assert result.resume_variant == ResumeVariant.PROJECT_MANAGER
    assert result.stage_a_revision_count == 1
    send_spy.assert_called_once()


def test_revise_at_cap_sends_clarifying_email_instead(tmp_path, mocker):
    job = _job()
    application, db_path = _application(
        tmp_path, job, stage_a_revision_count=config.MAX_REVISION_ROUNDS
    )
    mocker.patch(
        "jobfinder.gmail.approval_loop.get_thread",
        return_value=_thread_with_reply("try the other resume again"),
    )
    mocker.patch(
        "jobfinder.gmail.approval_loop._classify_reply",
        return_value=("revise", "try again"),
    )
    send_spy = mocker.patch("jobfinder.gmail.approval_loop.send_raw_message")

    result = approval_loop.process_stage_a_reply(object(), application, job, db_path=db_path)

    assert result.stage_a_revision_count == config.MAX_REVISION_ROUNDS
    assert result.resume_variant == ResumeVariant.FULLSTACK  # unchanged
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
        return_value=("ambiguous", "unclear"),
    )
    send_spy = mocker.patch("jobfinder.gmail.approval_loop.send_raw_message")

    result = approval_loop.process_stage_a_reply(object(), application, job, db_path=db_path)

    assert result.status == ApplicationStatus.PENDING_STAGE_A_APPROVAL
    send_spy.assert_called_once()
