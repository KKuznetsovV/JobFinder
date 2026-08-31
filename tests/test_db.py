from jobfinder.db import store
from jobfinder.db.models import (
    Application,
    ApplicationStatus,
    ApplyMethod,
    JobPosting,
    ResumeVariant,
)


def test_init_db_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    with store.get_conn(db_path) as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {"job_postings", "applications", "processed_gmail_messages"} <= tables


def test_insert_and_dedup_job_posting(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    posting = JobPosting(
        source="devjobs",
        external_id="abc123",
        title="Backend Engineer",
        company="Acme",
        description="Build things.",
        url="https://example.com/jobs/abc123",
        apply_method=ApplyMethod.EMAIL,
        apply_email="jobs@acme.example",
    )
    saved = store.insert_job_posting(posting, db_path)
    assert saved is not None
    assert saved.id is not None

    duplicate = JobPosting(
        source="devjobs",
        external_id="abc123",
        title="Backend Engineer (dup)",
        company="Acme",
        description="Build things.",
        url="https://example.com/jobs/abc123",
        apply_method=ApplyMethod.EMAIL,
    )
    assert store.insert_job_posting(duplicate, db_path) is None
    assert store.job_posting_exists("devjobs", "abc123", db_path)


def test_application_lifecycle(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    posting = store.insert_job_posting(
        JobPosting(
            source="gotfriends",
            external_id="xyz",
            title="PM",
            company="Beta",
            description="Manage things.",
            url="https://example.com/jobs/xyz",
            apply_method=ApplyMethod.COMPANY_SITE,
        ),
        db_path,
    )
    app = store.insert_application(
        Application(
            job_posting_id=posting.id,
            resume_variant=ResumeVariant.PROJECT_MANAGER,
        ),
        db_path,
    )
    assert app.id is not None
    assert app.status == ApplicationStatus.FOUND

    app.status = ApplicationStatus.NOTIFIED
    app.gmail_thread_id = "thread-1"
    store.update_application(app, db_path)

    fetched = store.get_application(app.id, db_path)
    assert fetched.status == ApplicationStatus.NOTIFIED
    assert fetched.gmail_thread_id == "thread-1"

    by_thread = store.get_application_by_thread("thread-1", db_path)
    assert by_thread.id == app.id

    store.append_apply_log(fetched, "notified_sent", "email dispatched", db_path)
    logged = store.get_application(app.id, db_path)
    assert logged.apply_log[-1]["event"] == "notified_sent"


def test_processed_gmail_messages(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    assert not store.is_gmail_message_processed("msg-1", db_path)
    store.mark_gmail_message_processed("msg-1", "job_alert", db_path)
    assert store.is_gmail_message_processed("msg-1", db_path)
