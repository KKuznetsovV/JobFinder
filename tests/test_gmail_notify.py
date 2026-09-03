from jobfinder import config
from jobfinder.db import store
from jobfinder.db.models import (
    Application,
    ApplicationStatus,
    ApplyMethod,
    JobPosting,
    ResumeVariant,
)
from jobfinder.gmail import notify


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


def _application(tmp_path, job):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    saved_job = store.insert_job_posting(job, db_path)
    app = Application(
        job_posting_id=saved_job.id,
        resume_variant=ResumeVariant.FULLSTACK,
        resume_choice_reason="Keyword heuristic matched 'fullstack' resume.",
        cover_letter="Dear hiring team, ...",
        status=ApplicationStatus.FOUND,
        relevance_score=0.8,
    )
    return store.insert_application(app, db_path), db_path


def test_build_approval_email_includes_app_id_header_and_content(tmp_path):
    job = _job()
    application, _ = _application(tmp_path, job)

    message = notify.build_approval_email(application, job)

    assert message[config.APP_ID_HEADER] == str(application.id)
    assert "Cover letter ready" in message["Subject"]
    body = message.get_content()
    assert "Dear hiring team" in body
    assert "fullstack" in body
    assert job.title in body


def test_build_approval_email_requires_persisted_application(tmp_path):
    job = _job()
    app = Application(
        job_posting_id=1,
        resume_variant=ResumeVariant.FULLSTACK,
        cover_letter="x",
    )
    try:
        notify.build_approval_email(app, job)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_send_approval_request_updates_application(tmp_path, mocker):
    job = _job()
    application, db_path = _application(tmp_path, job)
    fake_service = object()
    mocker.patch(
        "jobfinder.gmail.notify.send_raw_message",
        return_value={"threadId": "thread-1", "id": "msg-1"},
    )

    updated = notify.send_approval_request(fake_service, application, job, db_path=db_path)

    assert updated.gmail_thread_id == "thread-1"
    assert updated.gmail_last_message_id == "msg-1"
    assert updated.status == ApplicationStatus.NOTIFIED

    reloaded = store.get_application(application.id, db_path=db_path)
    assert reloaded.status == ApplicationStatus.NOTIFIED
    assert reloaded.gmail_thread_id == "thread-1"


def test_build_stage_b_no_letter_email_omits_cover_letter_section(tmp_path):
    job = _job()
    application, _ = _application(tmp_path, job)
    application.cover_letter = None

    message = notify.build_stage_b_no_letter_email(application, job)

    assert message[config.APP_ID_HEADER] == str(application.id)
    assert "no cover letter needed" in message["Subject"].lower()
    body = message.get_content()
    assert "no cover-letter field" in body.lower()
    assert "APPROVE" in body


def test_send_stage_b_no_letter_request_updates_application(tmp_path, mocker):
    job = _job()
    application, db_path = _application(tmp_path, job)
    application.cover_letter = None
    fake_service = object()
    mocker.patch(
        "jobfinder.gmail.notify.send_raw_message",
        return_value={"threadId": "thread-1", "id": "msg-1"},
    )

    updated = notify.send_stage_b_no_letter_request(fake_service, application, job, db_path=db_path)

    assert updated.gmail_thread_id == "thread-1"
    assert updated.status == ApplicationStatus.NOTIFIED

    reloaded = store.get_application(application.id, db_path=db_path)
    assert reloaded.status == ApplicationStatus.NOTIFIED


def test_build_revision_email_is_a_reply_with_revised_letter(tmp_path):
    job = _job()
    application, _ = _application(tmp_path, job)
    application.cover_letter = "Shorter revised letter."

    message = notify.build_revision_email(application, job)

    assert message["Subject"].startswith("Re: Cover letter ready")
    body = message.get_content()
    assert "Shorter revised letter." in body


def test_send_revision_request_keeps_thread_and_sets_notified(tmp_path, mocker):
    job = _job()
    application, db_path = _application(tmp_path, job)
    application.gmail_thread_id = "thread-1"
    application.status = ApplicationStatus.REVISION_REQUESTED
    store.update_application(application, db_path=db_path)
    send_spy = mocker.patch(
        "jobfinder.gmail.notify.send_raw_message",
        return_value={"threadId": "thread-1", "id": "msg-2"},
    )

    updated = notify.send_revision_request(object(), application, job, db_path=db_path)

    assert updated.status == ApplicationStatus.NOTIFIED
    assert updated.gmail_last_message_id == "msg-2"
    send_spy.assert_called_once()
    assert send_spy.call_args.kwargs["thread_id"] == "thread-1"
