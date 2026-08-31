from jobfinder.sources.jobinfo import JobInfoSource


def test_jobinfo_source_returns_empty_and_warns(caplog):
    with caplog.at_level("WARNING"):
        postings = JobInfoSource().fetch()

    assert postings == []
    assert any("Cloudflare" in record.message for record in caplog.records)
