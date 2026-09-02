from jobfinder.ai.tier0_filter import Tier0Result
from jobfinder.db import store
from jobfinder.db.models import ApplicationStatus, ApplyMethod, JobPosting, ResumeVariant
from jobfinder import pipeline, config


def _job(db_path):
    posting = JobPosting(
        source="gotfriends",
        external_id="1",
        title="Backend Engineer",
        company="Acme",
        description="Build things with Python.",
        url="https://example.com/jobs/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )
    return store.insert_job_posting(posting, db_path)


def test_process_new_posting_requires_persisted_job(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = JobPosting(
        source="gotfriends",
        external_id="1",
        title="x",
        company="Acme",
        description="x",
        url="https://example.com/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )
    try:
        pipeline.process_new_posting(job, gmail_service=object(), db_path=db_path)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_process_new_posting_returns_none_when_not_relevant(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    mocker.patch(
        "jobfinder.pipeline.tier0_filter.score_posting",
        return_value=Tier0Result(score=1.0, resume_hint=None),
    )
    mocker.patch(
        "jobfinder.ai.relevance.is_relevant", return_value=(False, 0.1, "not a fit")
    )

    result = pipeline.process_new_posting(job, gmail_service=object(), db_path=db_path)

    assert result is None


def test_process_new_posting_rejects_below_tier0_threshold(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    mocker.patch(
        "jobfinder.pipeline.tier0_filter.score_posting",
        return_value=Tier0Result(score=0.01, resume_hint=ResumeVariant.FULLSTACK),
    )
    relevance_spy = mocker.patch("jobfinder.ai.relevance.is_relevant")

    result = pipeline.process_new_posting(job, gmail_service=object(), db_path=db_path)

    assert result is None
    relevance_spy.assert_not_called()

    applications = store.list_applications_by_status(
        ApplicationStatus.REJECTED_TIER0, db_path=db_path
    )
    assert len(applications) == 1
    assert applications[0].tier0_score == 0.01
    assert applications[0].tier0_resume_hint == ResumeVariant.FULLSTACK
    events = [entry["event"] for entry in applications[0].apply_log]
    assert "rejected_tier0" in events


def test_process_new_posting_creates_application_and_sends_stage_a(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)

    mocker.patch(
        "jobfinder.pipeline.tier0_filter.score_posting",
        return_value=Tier0Result(score=0.9, resume_hint=ResumeVariant.FULLSTACK),
    )
    mocker.patch(
        "jobfinder.ai.relevance.is_relevant", return_value=(True, 0.9, "great fit")
    )
    mocker.patch(
        "jobfinder.ai.resume_selector.select_resume",
        return_value=(ResumeVariant.FULLSTACK, "keyword match"),
    )
    cover_letter_spy = mocker.patch("jobfinder.pipeline.cover_letter_module.generate_cover_letter")
    notify_spy = mocker.patch(
        "jobfinder.pipeline.notify.send_stage_a_request",
        side_effect=lambda service, app, job, db_path=None: app,
    )

    result = pipeline.process_new_posting(job, gmail_service=object(), db_path=db_path)

    assert result is not None
    assert result.cover_letter is None
    assert result.status == ApplicationStatus.FOUND
    assert result.relevance_score == 0.9
    assert result.resume_variant == ResumeVariant.FULLSTACK
    notify_spy.assert_called_once()
    cover_letter_spy.assert_not_called()  # the expensive call must wait for Stage A approval

    reloaded = store.get_application(result.id, db_path=db_path)
    events = [entry["event"] for entry in reloaded.apply_log]
    assert "relevance_passed" in events


def test_process_stage_a_approval_generates_cover_letter_and_notifies(tmp_path, mocker):
    from jobfinder.db.models import Application

    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    application = Application(
        job_posting_id=job.id,
        resume_variant=ResumeVariant.FULLSTACK,
        resume_choice_reason="keyword match",
        status=ApplicationStatus.PENDING_STAGE_A_APPROVAL,
        relevance_score=0.9,
    )
    application = store.insert_application(application, db_path)

    mocker.patch(
        "jobfinder.pipeline.cover_letter_module.generate_cover_letter",
        return_value=("Final letter.", ["invented claim x"]),
    )
    notify_spy = mocker.patch(
        "jobfinder.pipeline.notify.send_approval_request",
        side_effect=lambda service, app, job, db_path=None: app,
    )

    result = pipeline.process_stage_a_approval(application, job, gmail_service=object(), db_path=db_path)

    assert result.cover_letter == "Final letter."
    notify_spy.assert_called_once()

    reloaded = store.get_application(result.id, db_path=db_path)
    events = [entry["event"] for entry in reloaded.apply_log]
    assert "cover_letter_generated" in events
    assert "fact_check_issues" in events



def test_process_revision_request_regenerates_and_resends(tmp_path, mocker):
    from jobfinder.db.models import Application, ApplicationStatus

    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    application = Application(
        job_posting_id=job.id,
        resume_variant=ResumeVariant.FULLSTACK,
        cover_letter="Long previous letter.",
        status=ApplicationStatus.REVISION_REQUESTED,
        gmail_thread_id="thread-1",
        gmail_last_message_id="msg-notify",
    )
    application = store.insert_application(application, db_path)
    store.append_apply_log(application, "revision_requested", "Make it shorter.", db_path=db_path)

    cover_letter_spy = mocker.patch(
        "jobfinder.pipeline.cover_letter_module.generate_cover_letter",
        return_value=("Shorter letter.", []),
    )
    notify_spy = mocker.patch(
        "jobfinder.pipeline.notify.send_revision_request",
        side_effect=lambda service, app, job, db_path=None: app,
    )

    result = pipeline.process_revision_request(application, job, gmail_service=object(), db_path=db_path)

    assert result.cover_letter == "Shorter letter."
    cover_letter_spy.assert_called_once()
    assert cover_letter_spy.call_args.kwargs["revision_feedback"] == "Make it shorter."
    assert cover_letter_spy.call_args.kwargs["previous_letter"] == "Long previous letter."
    notify_spy.assert_called_once()

    reloaded = store.get_application(result.id, db_path=db_path)
    events = [entry["event"] for entry in reloaded.apply_log]
    assert "revision_applied" in events
