from jobfinder.db import store
from jobfinder.db.models import ApplyMethod, JobPosting
from jobfinder.sources import scheduler
from jobfinder.sources.base import JobSource, fetch_and_store


class FakeSource(JobSource):
    name = "fake"

    def __init__(self, postings):
        self._postings = postings

    def fetch(self):
        return self._postings


class BrokenSource(JobSource):
    name = "broken"

    def fetch(self):
        raise RuntimeError("boom")


def _posting(external_id, source="fake"):
    return JobPosting(
        source=source,
        external_id=external_id,
        title="Engineer",
        company="Acme",
        description="desc",
        url=f"https://example.com/{external_id}",
        apply_method=ApplyMethod.COMPANY_SITE,
    )


def test_fetch_and_store_dedupes_across_calls(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    source = FakeSource([_posting("1"), _posting("2")])

    first = fetch_and_store(source, db_path)
    assert len(first) == 2

    # Same source returning identical postings again (e.g. an RSS feed that
    # hasn't rolled over yet) should not produce duplicate inserts.
    second = fetch_and_store(source, db_path)
    assert second == []

    source._postings.append(_posting("3"))
    third = fetch_and_store(source, db_path)
    assert [p.external_id for p in third] == ["3"]


def test_fetch_and_store_forces_source_name(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    source = FakeSource([_posting("1", source="wrong")])
    [saved] = fetch_and_store(source, db_path)
    assert saved.source == "fake"


def test_poll_once_isolates_failures(tmp_path):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    good = FakeSource([_posting("1")])
    broken = BrokenSource()

    new_postings = scheduler.poll_once([broken, good], db_path)

    assert [p.external_id for p in new_postings] == ["1"]


def test_run_forever_respects_max_iterations(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    source = FakeSource([_posting("1")])
    sleep_mock = mocker.patch("jobfinder.sources.scheduler.time.sleep")

    scheduler.run_forever(
        [source], poll_interval_hours=0.001, db_path=db_path, max_iterations=2
    )

    # Sleeps only between iterations, not after the final one.
    assert sleep_mock.call_count == 1
