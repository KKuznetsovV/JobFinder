from jobfinder.db.models import ApplyMethod
from jobfinder.sources.gotfriends import GotFriendsSource, _parse_category_page

SAMPLE_CATEGORY_PAGE = """
<html><body>
<a href="/jobslobby/ai/">AI overview (category link, no id)</a>
<a href="/jobslobby/ai/llm-engineer/142628/">NLP Data Scientist at a funded startup</a>
<a href="/jobslobby/ai/ai-engineer/118511/?utm=footer">Machine Learning for a medical startup</a>
<a href="/jobslobby/ai/llm-engineer/142628/?utm=footer">NLP Data Scientist at a funded startup</a>
</body></html>
"""


def test_parse_category_page_extracts_only_individual_postings():
    jobs = _parse_category_page(SAMPLE_CATEGORY_PAGE)

    assert len(jobs) == 2  # category overview link and duplicate footer link excluded
    assert jobs[0]["external_id"] == "142628"
    assert jobs[0]["title"] == "NLP Data Scientist at a funded startup"
    assert jobs[0]["url"].startswith("/jobslobby/ai/llm-engineer/142628/")
    assert jobs[1]["external_id"] == "118511"


def test_gotfriends_source_fetch_builds_postings(mocker):
    fake_response = mocker.Mock(text=SAMPLE_CATEGORY_PAGE)
    mocker.patch(
        "jobfinder.sources.gotfriends.scrape_utils.polite_get", return_value=fake_response
    )

    source = GotFriendsSource(category_slugs=["ai"])
    postings = source.fetch()

    assert len(postings) == 2
    assert postings[0].source == "gotfriends"
    assert postings[0].apply_method == ApplyMethod.COMPANY_SITE
    assert postings[0].external_id == "142628"


def test_gotfriends_source_isolates_per_category_failures(mocker):
    def fake_polite_get(url):
        if "software" in url:
            raise RuntimeError("boom")
        return mocker.Mock(text=SAMPLE_CATEGORY_PAGE)

    mocker.patch(
        "jobfinder.sources.gotfriends.scrape_utils.polite_get", side_effect=fake_polite_get
    )

    source = GotFriendsSource(category_slugs=["software", "ai"])
    postings = source.fetch()

    # The broken "software" category is skipped; "ai" still yields postings.
    assert len(postings) == 2
