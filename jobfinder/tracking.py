"""Query helpers for observing Application lifecycle state: status counts,
per-application event timelines, and applications needing manual attention.
"""
from __future__ import annotations

from jobfinder.db import store
from jobfinder.db.models import Application, ApplicationStatus


def status_summary(db_path: str | None = None) -> dict[str, int]:
    """Return a count of applications per status."""
    counts = {status.value: 0 for status in ApplicationStatus}
    for status in ApplicationStatus:
        counts[status.value] = len(store.list_applications_by_status(status, db_path=db_path))
    return counts


def application_timeline(app_id: int, db_path: str | None = None) -> list[dict]:
    """Return the ordered apply_log entries for one application, or an empty
    list if the application doesn't exist."""
    application = store.get_application(app_id, db_path=db_path)
    if application is None:
        return []
    return application.apply_log


def list_needs_attention(db_path: str | None = None) -> list[Application]:
    """Applications in states that need manual attention: STUCK (a submit
    attempt failed) or AWAITING_MY_CLICK (LinkedIn Easy Apply filled and
    paused, ready for the user's final click)."""
    stuck = store.list_applications_by_status(ApplicationStatus.STUCK, db_path=db_path)
    awaiting = store.list_applications_by_status(ApplicationStatus.AWAITING_MY_CLICK, db_path=db_path)
    return stuck + awaiting
