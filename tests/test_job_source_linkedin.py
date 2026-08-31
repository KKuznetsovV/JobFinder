import base64

from jobfinder.db import store
from jobfinder.db.models import ApplyMethod
from jobfinder.sources.linkedin_email import (
    LinkedInEmailSource,
    _decode_html_body,
    _parse_job_cards,
)

SAMPLE_HTML = """
<html><body>
<table>
  <tr><td>
    <a href="https://www.linkedin.com/jobs/view/1111111111/?trk=abc">Senior Backend Engineer</a>
    <br/>Acme Corp
  </td></tr>
  <tr><td>
    <a href="https://www.linkedin.com/jobs/view/2222222222/?trk=xyz">Product Manager</a>
    <br/>Beta Inc
  </td></tr>
</table>
</body></html>
"""


def _b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def test_parse_job_cards_extracts_title_company_and_clean_url():
    cards = _parse_job_cards(SAMPLE_HTML)

    assert len(cards) == 2
    assert cards[0]["external_id"] == "1111111111"
    assert cards[0]["title"] == "Senior Backend Engineer"
    assert cards[0]["company"] == "Acme Corp"
    assert cards[0]["url"] == "https://www.linkedin.com/jobs/view/1111111111/"

    assert cards[1]["external_id"] == "2222222222"
    assert cards[1]["company"] == "Beta Inc"


def test_parse_job_cards_dedupes_repeated_links_in_same_email():
    html = SAMPLE_HTML + '<a href="https://www.linkedin.com/jobs/view/1111111111/?trk=footer">Senior Backend Engineer</a>'
    cards = _parse_job_cards(html)
    assert len(cards) == 2  # the repeated footer link to job 1111111111 isn't duplicated


def test_decode_html_body_simple_payload():
    payload = {"mimeType": "text/html", "body": {"data": _b64url("<p>hello</p>")}}
    assert _decode_html_body(payload) == "<p>hello</p>"


def test_decode_html_body_nested_multipart_payload():
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": _b64url("plain text")}},
            {"mimeType": "text/html", "body": {"data": _b64url("<p>nested html</p>")}},
        ],
    }
    assert _decode_html_body(payload) == "<p>nested html</p>"


def _fake_message(message_id: str, html: str) -> dict:
    return {
        "id": message_id,
        "payload": {"mimeType": "text/html", "body": {"data": _b64url(html)}},
    }


def test_fetch_returns_postings_and_marks_message_processed(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)

    fake_service = mocker.Mock()
    mocker.patch(
        "jobfinder.sources.linkedin_email.gmail_client.list_message_ids",
        return_value=["msg-1"],
    )
    mocker.patch(
        "jobfinder.sources.linkedin_email.gmail_client.get_message",
        return_value=_fake_message("msg-1", SAMPLE_HTML),
    )

    source = LinkedInEmailSource(gmail_service=fake_service, db_path=db_path)
    postings = source.fetch()

    assert len(postings) == 2
    assert postings[0].source == "linkedin"
    assert postings[0].apply_method == ApplyMethod.LINKEDIN_EASY_APPLY
    assert postings[0].company == "Acme Corp"
    assert store.is_gmail_message_processed("msg-1", db_path)


def test_fetch_skips_already_processed_messages(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    store.mark_gmail_message_processed("msg-1", "job_alert", db_path)

    mocker.patch(
        "jobfinder.sources.linkedin_email.gmail_client.list_message_ids",
        return_value=["msg-1"],
    )
    get_message_mock = mocker.patch(
        "jobfinder.sources.linkedin_email.gmail_client.get_message"
    )

    source = LinkedInEmailSource(gmail_service=mocker.Mock(), db_path=db_path)
    postings = source.fetch()

    assert postings == []
    get_message_mock.assert_not_called()
