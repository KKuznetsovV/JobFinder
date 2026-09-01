from datetime import datetime, timedelta, timezone

from jobfinder import config
from jobfinder.db import store
from jobfinder.db.models import (
    Application,
    ApplicationStatus,
    ApplyMethod,
    JobPosting,
    ResumeVariant,
)
from jobfinder.local.agent import ApplyAgent


def _job(db_path, apply_method=ApplyMethod.COMPANY_SITE, external_id="1"):
    posting = JobPosting(
        source="gotfriends",
        external_id=external_id,
        title="Backend Engineer",
        company="Acme",
        description="Build things.",
        url="https://example.com/jobs/" + external_id,
        apply_method=apply_method,
    )
    return store.insert_job_posting(posting, db_path)


def _approved_application(db_path, job):
    app = Application(
        job_posting_id=job.id,
        resume_variant=ResumeVariant.FULLSTACK,
        cover_letter="Dear hiring team, ...",
        status=ApplicationStatus.APPROVED,
    )
    return store.insert_application(app, db_path)


def test_process_once_marks_stuck_when_no_handler_registered(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    application = _approved_application(db_path, job)

    agent = ApplyAgent(db_path=db_path)
    processed = agent.process_once()

    assert len(processed) == 1
    assert processed[0].status == ApplicationStatus.STUCK
    reloaded = store.get_application(application.id, db_path=db_path)
    assert reloaded.status == ApplicationStatus.STUCK


def test_process_once_uses_registered_handler(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path, apply_method=ApplyMethod.LINKEDIN_EASY_APPLY)
    application = _approved_application(db_path, job)

    agent = ApplyAgent(db_path=db_path)
    agent.register_handler(
        ApplyMethod.LINKEDIN_EASY_APPLY,
        lambda app, job: ApplicationStatus.AWAITING_MY_CLICK,
    )

    processed = agent.process_once()

    assert processed[0].status == ApplicationStatus.AWAITING_MY_CLICK


def test_process_once_respects_daily_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DAILY_APPLICATION_CAP", 1)
    db_path = tmp_path / "test.db"
    store.init_db(db_path)

    job1 = _job(db_path, external_id="1")
    job2 = _job(db_path, external_id="2")
    _approved_application(db_path, job1)
    _approved_application(db_path, job2)

    agent = ApplyAgent(db_path=db_path)
    agent.register_handler(
        ApplyMethod.COMPANY_SITE, lambda app, job: ApplicationStatus.SENT
    )

    processed = agent.process_once()

    assert len(processed) == 1
    assert processed[0].status == ApplicationStatus.SENT
    remaining_approved = store.list_applications_by_status(
        ApplicationStatus.APPROVED, db_path=db_path
    )
    assert len(remaining_approved) == 1


def test_process_once_handler_exception_marks_stuck_with_log(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    application = _approved_application(db_path, job)

    agent = ApplyAgent(db_path=db_path)

    def _boom(app, job):
        raise RuntimeError("browser crashed")

    agent.register_handler(ApplyMethod.COMPANY_SITE, _boom)

    processed = agent.process_once()

    assert processed[0].status == ApplicationStatus.STUCK
    reloaded = store.get_application(application.id, db_path=db_path)
    assert reloaded.apply_log[-1]["detail"] == "browser crashed"


def test_run_forever_respects_max_iterations(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    agent = ApplyAgent(db_path=db_path)
    sleep_spy = mocker.patch("jobfinder.local.agent.time.sleep")

    agent.run_forever(poll_interval_seconds=0.01, max_iterations=3)

    assert sleep_spy.call_count == 2
