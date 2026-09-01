from jobfinder import tracking
from jobfinder.db import store
from jobfinder.db.models import (
    Application,
    ApplicationStatus,
    ApplyMethod,
    JobPosting,
    ResumeVariant,
)


def _job(db_path, external_id="1"):
    posting = JobPosting(
        source="gotfriends",
        external_id=external_id,
        title="Backend Engineer",
        company="Acme",
        description="Build things.",
        url="https://example.com/jobs/" + external_id,
        apply_method=ApplyMethod.COMPANY_SITE,
    )
    return store.insert_job_posting(posting, db_path)


def _application(db_path, job, status):
    app = Application(
        job_posting_id=job.id,
        resume_variant=ResumeVariant.FULLSTACK,
        cover_letter="Dear hiring team, ...",
        status=status,
    )
    return store.insert_application(app, db_path)


def test_status_summary_counts_by_status(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job1 = _job(db_path, "1")
    job2 = _job(db_path, "2")
    _application(db_path, job1, ApplicationStatus.SENT)
    _application(db_path, job2, ApplicationStatus.STUCK)

    summary = tracking.status_summary(db_path=db_path)

    assert summary["sent"] == 1
    assert summary["stuck"] == 1
    assert summary["found"] == 0


def test_application_timeline_returns_apply_log(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    application = _application(db_path, job, ApplicationStatus.FOUND)
    store.append_apply_log(application, "notified", "sent email", db_path=db_path)

    timeline = tracking.application_timeline(application.id, db_path=db_path)

    assert len(timeline) == 1
    assert timeline[0]["event"] == "notified"


def test_application_timeline_returns_empty_for_missing_application(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)

    assert tracking.application_timeline(999, db_path=db_path) == []


def test_list_needs_attention_includes_stuck_and_awaiting_click(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job1 = _job(db_path, "1")
    job2 = _job(db_path, "2")
    job3 = _job(db_path, "3")
    _application(db_path, job1, ApplicationStatus.STUCK)
    _application(db_path, job2, ApplicationStatus.AWAITING_MY_CLICK)
    _application(db_path, job3, ApplicationStatus.SENT)

    needs_attention = tracking.list_needs_attention(db_path=db_path)

    assert len(needs_attention) == 2
    statuses = {app.status for app in needs_attention}
    assert statuses == {ApplicationStatus.STUCK, ApplicationStatus.AWAITING_MY_CLICK}
