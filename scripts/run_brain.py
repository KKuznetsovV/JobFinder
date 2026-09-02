"""Cloud "brain" entrypoint: polls job sources, runs the relevance/resume/
cover-letter pipeline for new postings, sends approval-request emails, and
polls Gmail for the user's approve/reject/revise replies.

Run:
    python scripts/run_brain.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobfinder import config, pipeline
from jobfinder.ai import tier0_filter
from jobfinder.db import store
from jobfinder.db.models import ApplicationStatus
from jobfinder.gmail.approval_loop import process_reply
from jobfinder.gmail.auth import get_gmail_service
from jobfinder.sources.devjobs import DevJobsSource
from jobfinder.sources.gotfriends import GotFriendsSource
from jobfinder.sources.jobinfo import JobInfoSource
from jobfinder.sources.linkedin_email import LinkedInEmailSource
from jobfinder.sources.scheduler import poll_once

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jobfinder.brain")

SOURCES = [GotFriendsSource(), DevJobsSource(), JobInfoSource(), LinkedInEmailSource()]


def main() -> None:
    store.init_db(config.DB_PATH)
    service = get_gmail_service()

    while True:
        new_postings = poll_once(SOURCES, db_path=config.DB_PATH)
        tier0_scores = []
        for job in new_postings:
            tier0_result = tier0_filter.score_posting(job)
            tier0_scores.append(tier0_result.score)
            try:
                pipeline.process_new_posting(
                    job, service, db_path=config.DB_PATH, tier0_result=tier0_result
                )
            except Exception:
                logger.exception("Pipeline failed for job posting %s", job.id)

        if tier0_scores:
            stats = tier0_filter.summarize_scores(tier0_scores)
            logger.info(
                "tier0 batch stats: count=%d passed=%d rejected=%d min=%.3f max=%.3f median=%.3f "
                "(LLM calls avoided=%d)",
                stats["count"],
                stats["passed"],
                stats["rejected"],
                stats["min"],
                stats["max"],
                stats["median"],
                stats["rejected"],
            )

        for application in store.list_applications_by_status(
            ApplicationStatus.NOTIFIED, db_path=config.DB_PATH
        ):
            job = store.get_job_posting(application.job_posting_id, db_path=config.DB_PATH)
            try:
                process_reply(service, application, job, db_path=config.DB_PATH)
            except Exception:
                logger.exception("Approval-reply check failed for application %s", application.id)

        for application in store.list_applications_by_status(
            ApplicationStatus.REVISION_REQUESTED, db_path=config.DB_PATH
        ):
            job = store.get_job_posting(application.job_posting_id, db_path=config.DB_PATH)
            try:
                pipeline.process_revision_request(application, job, service, db_path=config.DB_PATH)
            except Exception:
                logger.exception("Revision handling failed for application %s", application.id)

        time.sleep(config.JOB_POLL_INTERVAL_HOURS * 3600)


if __name__ == "__main__":
    main()
