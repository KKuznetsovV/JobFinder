"""Periodic scheduler that polls every registered JobSource and persists new
postings. Used by the cloud component's main loop (jobfinder/cloud_main.py,
added in a later branch).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from jobfinder import config
from jobfinder.db.models import JobPosting
from jobfinder.sources.base import JobSource, fetch_and_store

logger = logging.getLogger(__name__)


def poll_once(sources: list[JobSource], db_path: Path | str | None = None) -> list[JobPosting]:
    """Run every source exactly once, isolating failures per-source so one
    broken feed can't stop the others from being polled.

    Returns all newly inserted postings across all sources combined.
    """
    all_new: list[JobPosting] = []
    for source in sources:
        try:
            new_postings = fetch_and_store(source, db_path)
            logger.info("source=%s new_postings=%d", source.name, len(new_postings))
            all_new.extend(new_postings)
        except Exception:
            logger.exception("source=%s failed to fetch", source.name)
    return all_new


def run_forever(
    sources: list[JobSource],
    poll_interval_hours: float | None = None,
    db_path: Path | str | None = None,
    max_iterations: int | None = None,
) -> None:
    """Poll all sources every `poll_interval_hours`, forever.

    `max_iterations` lets tests/manual runs stop after a fixed number of
    polls instead of looping indefinitely.
    """
    interval_seconds = (poll_interval_hours or config.JOB_POLL_INTERVAL_HOURS) * 3600
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        poll_once(sources, db_path)
        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            break
        time.sleep(interval_seconds)
