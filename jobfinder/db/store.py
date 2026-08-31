"""Thin SQLite access layer shared by the cloud and local components.

Both processes point `JOBFINDER_DB_PATH` at the same file. SQLite's file
locking is enough for a single-user, low-frequency polling workload -
no message queue needed.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from jobfinder import config
from jobfinder.db.models import (
    Application,
    ApplicationStatus,
    ApplyMethod,
    JobPosting,
    ResumeVariant,
    utcnow_iso,
)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    """Create tables if they don't exist yet. Safe to call on every startup."""
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


@contextmanager
def get_conn(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _row_to_job_posting(row: sqlite3.Row) -> JobPosting:
    return JobPosting(
        id=row["id"],
        source=row["source"],
        external_id=row["external_id"],
        title=row["title"],
        company=row["company"],
        description=row["description"],
        url=row["url"],
        posted_date=row["posted_date"],
        apply_method=ApplyMethod(row["apply_method"]),
        apply_email=row["apply_email"],
        discovered_at=row["discovered_at"],
    )


def _row_to_application(row: sqlite3.Row) -> Application:
    return Application(
        id=row["id"],
        job_posting_id=row["job_posting_id"],
        resume_variant=ResumeVariant(row["resume_variant"]),
        resume_choice_reason=row["resume_choice_reason"],
        cover_letter=row["cover_letter"],
        status=ApplicationStatus(row["status"]),
        relevance_score=row["relevance_score"],
        gmail_thread_id=row["gmail_thread_id"],
        gmail_last_message_id=row["gmail_last_message_id"],
        revision_count=row["revision_count"],
        apply_log=json.loads(row["apply_log"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# --- job_postings ------------------------------------------------------------

def insert_job_posting(posting: JobPosting, db_path: Path | str | None = None) -> JobPosting | None:
    """Insert a posting, returning it with its new id, or None if it's a dedup hit."""
    with get_conn(db_path) as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO job_postings
                    (source, external_id, title, company, description, url,
                     posted_date, apply_method, apply_email, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    posting.source,
                    posting.external_id,
                    posting.title,
                    posting.company,
                    posting.description,
                    posting.url,
                    posting.posted_date,
                    posting.apply_method.value,
                    posting.apply_email,
                    posting.discovered_at,
                ),
            )
        except sqlite3.IntegrityError:
            return None  # already seen this (source, external_id)
        posting.id = cur.lastrowid
        return posting


def job_posting_exists(source: str, external_id: str, db_path: Path | str | None = None) -> bool:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM job_postings WHERE source = ? AND external_id = ?",
            (source, external_id),
        ).fetchone()
        return row is not None


def get_job_posting(posting_id: int, db_path: Path | str | None = None) -> JobPosting | None:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM job_postings WHERE id = ?", (posting_id,)).fetchone()
        return _row_to_job_posting(row) if row else None


# --- applications ------------------------------------------------------------

def insert_application(app: Application, db_path: Path | str | None = None) -> Application:
    with get_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO applications
                (job_posting_id, resume_variant, resume_choice_reason, cover_letter,
                 status, relevance_score, gmail_thread_id, gmail_last_message_id,
                 revision_count, apply_log, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                app.job_posting_id,
                app.resume_variant.value,
                app.resume_choice_reason,
                app.cover_letter,
                app.status.value,
                app.relevance_score,
                app.gmail_thread_id,
                app.gmail_last_message_id,
                app.revision_count,
                json.dumps(app.apply_log),
                app.created_at,
                app.updated_at,
            ),
        )
        app.id = cur.lastrowid
        return app


def get_application(app_id: int, db_path: Path | str | None = None) -> Application | None:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
        return _row_to_application(row) if row else None


def get_application_by_thread(thread_id: str, db_path: Path | str | None = None) -> Application | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM applications WHERE gmail_thread_id = ?", (thread_id,)
        ).fetchone()
        return _row_to_application(row) if row else None


def list_applications_by_status(
    status: ApplicationStatus, db_path: Path | str | None = None
) -> list[Application]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM applications WHERE status = ? ORDER BY id", (status.value,)
        ).fetchall()
        return [_row_to_application(row) for row in rows]


def count_applications_sent_since(since_iso: str, db_path: Path | str | None = None) -> int:
    """Used by the local component to enforce the daily application cap."""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM applications WHERE status = 'sent' AND updated_at >= ?",
            (since_iso,),
        ).fetchone()
        return row["n"]


def update_application(app: Application, db_path: Path | str | None = None) -> None:
    app.updated_at = utcnow_iso()
    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE applications SET
                resume_variant = ?, resume_choice_reason = ?, cover_letter = ?,
                status = ?, relevance_score = ?, gmail_thread_id = ?,
                gmail_last_message_id = ?, revision_count = ?, apply_log = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                app.resume_variant.value,
                app.resume_choice_reason,
                app.cover_letter,
                app.status.value,
                app.relevance_score,
                app.gmail_thread_id,
                app.gmail_last_message_id,
                app.revision_count,
                json.dumps(app.apply_log),
                app.updated_at,
                app.id,
            ),
        )


def append_apply_log(app: Application, event: str, detail: str = "", db_path: Path | str | None = None) -> None:
    app.apply_log.append({"ts": utcnow_iso(), "event": event, "detail": detail})
    update_application(app, db_path)


# --- processed_gmail_messages -------------------------------------------------

def is_gmail_message_processed(message_id: str, db_path: Path | str | None = None) -> bool:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM processed_gmail_messages WHERE message_id = ?", (message_id,)
        ).fetchone()
        return row is not None


def mark_gmail_message_processed(
    message_id: str, purpose: str, db_path: Path | str | None = None
) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO processed_gmail_messages (message_id, purpose, processed_at) "
            "VALUES (?, ?, ?)",
            (message_id, purpose, utcnow_iso()),
        )
