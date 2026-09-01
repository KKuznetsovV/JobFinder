"""Local "hands" agent: runs on the user's machine, polls the shared SQLite
datastore for approved applications, and dispatches each one to a submission
handler based on the job posting's apply_method.

This branch only builds the polling/dispatch skeleton and daily-cap
enforcement. The actual browser-automation handlers (ATS templates,
LinkedIn Easy Apply fill-and-pause, generic fallback) are registered here by
later branches - see `jobfinder.local.ats`, `jobfinder.local.apply_linkedin`,
and `jobfinder.local.apply_fallback` once built. A handler with no submission
logic registered yet raises NotImplementedError, which is caught and recorded
as a STUCK application rather than crashing the whole agent.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable

from jobfinder import config
from jobfinder.db import store
from jobfinder.db.models import Application, ApplicationStatus, ApplyMethod, JobPosting

logger = logging.getLogger(__name__)

LOCAL_POLL_INTERVAL_SECONDS = 30

# A handler receives (application, job_posting) and returns the resulting
# ApplicationStatus (typically SENT or AWAITING_MY_CLICK). It may raise on
# failure, which the agent records as STUCK.
ApplyHandler = Callable[[Application, JobPosting], ApplicationStatus]


def _start_of_today_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


class ApplyAgent:
    def __init__(
        self,
        db_path: str | None = None,
        handlers: dict[ApplyMethod, ApplyHandler] | None = None,
    ):
        self.db_path = db_path
        self.handlers: dict[ApplyMethod, ApplyHandler] = handlers or {}

    def register_handler(self, apply_method: ApplyMethod, handler: ApplyHandler) -> None:
        self.handlers[apply_method] = handler

    def _daily_cap_remaining(self) -> int:
        sent_today = store.count_applications_sent_since(
            _start_of_today_iso(), db_path=self.db_path
        )
        return max(0, config.DAILY_APPLICATION_CAP - sent_today)

    def process_once(self) -> list[Application]:
        """Process as many APPROVED applications as the daily cap allows.
        Returns the list of Applications that were processed (attempted)."""
        approved = store.list_applications_by_status(
            ApplicationStatus.APPROVED, db_path=self.db_path
        )
        remaining = self._daily_cap_remaining()
        processed: list[Application] = []

        for application in approved:
            if remaining <= 0:
                logger.info("Daily application cap reached, deferring remaining applications.")
                break

            job = store.get_job_posting(application.job_posting_id, db_path=self.db_path)
            if job is None:
                logger.error("No job posting found for application %s, marking failed.", application.id)
                application.status = ApplicationStatus.FAILED
                store.update_application(application, db_path=self.db_path)
                store.append_apply_log(application, "failed", "Job posting missing.", db_path=self.db_path)
                processed.append(application)
                continue

            application.status = ApplicationStatus.SUBMITTING
            store.update_application(application, db_path=self.db_path)
            store.append_apply_log(
                application, "submitting", f"apply_method={job.apply_method.value}", db_path=self.db_path
            )

            try:
                handler = self.handlers.get(job.apply_method)
                if handler is None:
                    raise NotImplementedError(
                        f"No apply handler registered for apply_method={job.apply_method.value}"
                    )
                application.status = handler(application, job)
                store.append_apply_log(
                    application, "handler_completed", application.status.value, db_path=self.db_path
                )
            except Exception as exc:  # noqa: BLE001 - any handler failure -> STUCK, never crash the agent
                logger.exception("Apply handler failed for application %s", application.id)
                application.status = ApplicationStatus.STUCK
                store.append_apply_log(application, "error", str(exc), db_path=self.db_path)

            store.update_application(application, db_path=self.db_path)
            processed.append(application)
            if application.status == ApplicationStatus.SENT:
                remaining -= 1

        return processed

    def run_forever(self, poll_interval_seconds: float | None = None, max_iterations: int | None = None) -> None:
        interval = poll_interval_seconds if poll_interval_seconds is not None else LOCAL_POLL_INTERVAL_SECONDS
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            self.process_once()
            iterations += 1
            if max_iterations is None or iterations < max_iterations:
                time.sleep(interval)
