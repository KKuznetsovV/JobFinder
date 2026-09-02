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


def test_init_db_migrates_pre_tier0_applications_table(tmp_path):
    """A DB created before the tier0_score/tier0_resume_hint columns (and the
    'rejected_tier0' status) existed must be upgraded in place, without losing
    any pre-existing application rows."""
    db_path = tmp_path / "test.db"
    with store._connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE job_postings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL, external_id TEXT NOT NULL, title TEXT NOT NULL,
                company TEXT NOT NULL, description TEXT NOT NULL, url TEXT NOT NULL,
                posted_date TEXT, apply_method TEXT NOT NULL, apply_email TEXT,
                discovered_at TEXT NOT NULL, UNIQUE (source, external_id)
            );
            CREATE TABLE applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_posting_id INTEGER NOT NULL REFERENCES job_postings(id),
                resume_variant TEXT NOT NULL CHECK (resume_variant IN ('fullstack', 'project_manager')),
                resume_choice_reason TEXT, cover_letter TEXT,
                status TEXT NOT NULL DEFAULT 'found' CHECK (status IN (
                    'found', 'notified', 'approved', 'rejected',
                    'revision_requested', 'submitting', 'awaiting_my_click',
                    'sent', 'stuck', 'failed'
                )),
                relevance_score REAL, gmail_thread_id TEXT, gmail_last_message_id TEXT,
                revision_count INTEGER NOT NULL DEFAULT 0,
                apply_log TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO job_postings (source, external_id, title, company, description, url, "
            "apply_method, discovered_at) VALUES ('gotfriends', '1', 'PM', 'Acme', 'x', "
            "'https://example.com/1', 'company_site', '2024-01-01T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO applications (job_posting_id, resume_variant, status, apply_log, "
            "created_at, updated_at) VALUES (1, 'fullstack', 'found', '[]', "
            "'2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00')"
        )
        conn.commit()

    store.init_db(db_path)  # should migrate in place, not wipe existing rows

    preserved = store.get_application(1, db_path)
    assert preserved is not None
    assert preserved.resume_variant == ResumeVariant.FULLSTACK
    assert preserved.tier0_score is None
    assert preserved.tier0_resume_hint is None

    # new columns/status are now usable
    app = store.insert_application(
        Application(
            job_posting_id=1,
            resume_variant=ResumeVariant.FULLSTACK,
            status=ApplicationStatus.REJECTED_TIER0,
            tier0_score=0.1,
            tier0_resume_hint=ResumeVariant.FULLSTACK,
        ),
        db_path,
    )
    reloaded = store.get_application(app.id, db_path)
    assert reloaded.status == ApplicationStatus.REJECTED_TIER0
    assert reloaded.tier0_score == 0.1
    assert reloaded.tier0_resume_hint == ResumeVariant.FULLSTACK


def test_init_db_migrates_post_tier0_pre_stage_a_applications_table(tmp_path):
    """A DB created after the tier0 columns were added but before the Stage A
    columns (stage_a_approved_at/stage_a_revision_count) existed must also be
    upgraded in place, preserving existing rows and their tier0 data."""
    db_path = tmp_path / "test.db"
    with store._connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE job_postings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL, external_id TEXT NOT NULL, title TEXT NOT NULL,
                company TEXT NOT NULL, description TEXT NOT NULL, url TEXT NOT NULL,
                posted_date TEXT, apply_method TEXT NOT NULL, apply_email TEXT,
                discovered_at TEXT NOT NULL, UNIQUE (source, external_id)
            );
            CREATE TABLE applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_posting_id INTEGER NOT NULL REFERENCES job_postings(id),
                resume_variant TEXT NOT NULL CHECK (resume_variant IN ('fullstack', 'project_manager')),
                resume_choice_reason TEXT, cover_letter TEXT,
                status TEXT NOT NULL DEFAULT 'found' CHECK (status IN (
                    'found', 'notified', 'approved', 'rejected', 'rejected_tier0',
                    'revision_requested', 'submitting', 'awaiting_my_click',
                    'sent', 'stuck', 'failed'
                )),
                relevance_score REAL, gmail_thread_id TEXT, gmail_last_message_id TEXT,
                revision_count INTEGER NOT NULL DEFAULT 0,
                apply_log TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                tier0_score REAL, tier0_resume_hint TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO job_postings (source, external_id, title, company, description, url, "
            "apply_method, discovered_at) VALUES ('gotfriends', '1', 'PM', 'Acme', 'x', "
            "'https://example.com/1', 'company_site', '2024-01-01T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO applications (job_posting_id, resume_variant, status, apply_log, "
            "created_at, updated_at, tier0_score, tier0_resume_hint) VALUES (1, 'fullstack', "
            "'found', '[]', '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00', 0.55, 'fullstack')"
        )
        conn.commit()

    store.init_db(db_path)  # should migrate in place, not wipe existing rows

    preserved = store.get_application(1, db_path)
    assert preserved is not None
    assert preserved.tier0_score == 0.55
    assert preserved.tier0_resume_hint == ResumeVariant.FULLSTACK
    assert preserved.stage_a_approved_at is None
    assert preserved.stage_a_revision_count == 0

    app = store.insert_application(
        Application(
            job_posting_id=1,
            resume_variant=ResumeVariant.FULLSTACK,
            status=ApplicationStatus.PENDING_STAGE_A_APPROVAL,
            stage_a_revision_count=2,
        ),
        db_path,
    )
    reloaded = store.get_application(app.id, db_path)
    assert reloaded.status == ApplicationStatus.PENDING_STAGE_A_APPROVAL
    assert reloaded.stage_a_revision_count == 2

