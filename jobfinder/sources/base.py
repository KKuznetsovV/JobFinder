"""Common interface all job sources implement, plus the dedupe-and-store
helper used by the scheduler to persist newly discovered postings.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from jobfinder.db import store
from jobfinder.db.models import JobPosting


class JobSource(ABC):
    """A pluggable source of job postings (RSS feed, email digest, scraper).

    Each source is responsible only for producing JobPosting objects; dedup
    against the shared datastore is handled centrally by `fetch_and_store`,
    so individual sources don't need to know about SQLite at all.
    """

    #: Short, stable identifier stored in JobPosting.source and used as part
    #: of the (source, external_id) dedup key. Must be unique across sources.
    name: str

    @abstractmethod
    def fetch(self) -> list[JobPosting]:
        """Return the current batch of postings visible to this source.

        Implementations should NOT filter out already-seen postings - that's
        handled by `fetch_and_store`'s dedup step - but should still avoid
        pointlessly expensive/duplicate network calls internally.
        """
        raise NotImplementedError


def fetch_and_store(source: JobSource, db_path: Path | str | None = None) -> list[JobPosting]:
    """Fetch postings from one source and persist only the genuinely new ones.

    Returns the newly inserted JobPosting objects (with their ids populated);
    postings already present in the datastore are silently skipped.
    """
    newly_inserted: list[JobPosting] = []
    for posting in source.fetch():
        if posting.source != source.name:
            posting.source = source.name
        saved = store.insert_job_posting(posting, db_path)
        if saved is not None:
            newly_inserted.append(saved)
    return newly_inserted
