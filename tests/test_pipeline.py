from jobfinder.db import store
from jobfinder.db.models import ApplyMethod, JobPosting, ResumeVariant
from jobfinder import pipeline


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
        "jobfinder.pipeline.relevance.is_relevant", return_value=(False, 0.1, "not a fit")
    )

    result = pipeline.process_new_posting(job, gmail_service=object(), db_path=db_path)

    assert result is None


def test_process_new_posting_creates_application_and_notifies(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)

    mocker.patch(
        "jobfinder.pipeline.relevance.is_relevant", return_value=(True, 0.9, "great fit")
    )
    mocker.patch(
        "jobfinder.pipeline.resume_selector.select_resume",
        return_value=(ResumeVariant.FULLSTACK, "keyword match"),
    )
    mocker.patch(
        "jobfinder.pipeline.cover_letter_module.generate_cover_letter",
        return_value=("Final letter.", ["invented claim x"]),
    )
    notify_spy = mocker.patch(
        "jobfinder.pipeline.notify.send_approval_request",
        side_effect=lambda service, app, job, db_path=None: app,
    )

    result = pipeline.process_new_posting(job, gmail_service=object(), db_path=db_path)

    assert result is not None
    assert result.cover_letter == "Final letter."
    assert result.relevance_score == 0.9
    assert result.resume_variant == ResumeVariant.FULLSTACK
    notify_spy.assert_called_once()

    reloaded = store.get_application(result.id, db_path=db_path)
    events = [entry["event"] for entry in reloaded.apply_log]
    assert "relevance_passed" in events
    assert "fact_check_issues" in events
