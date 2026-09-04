"""Parse LinkedIn job-alert digest emails (received via Gmail) into JobPosting
objects.

Per the project's constraints, this NEVER scrapes linkedin.com directly -
LinkedIn's own emailed job-alert digest (sent to the user's Gmail after they
set up a job alert search on linkedin.com) is the only data source. See the
README's "Setting up a LinkedIn job alert" section for how to configure that
search on LinkedIn's site.

ASSUMPTION (no real sample alert email was available while building this):
LinkedIn's alert email markup changes over time and typically only includes
a short title/company/location per job, not the full job description. This
parser extracts what's available and stores a short summary in `description`
rather than the full posting text; the browser-automation fallback (later
branch) fetches the real page content at apply time regardless. If the
markup drifts from what's assumed here, `_parse_job_cards` is the place to
adjust the extraction logic - it's isolated from the rest of the pipeline.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

from bs4 import BeautifulSoup
from googleapiclient.discovery import Resource  # type: ignore

from jobfinder.db import store
from jobfinder.db.models import ApplyMethod, JobPosting
from jobfinder.gmail import client as gmail_client
from jobfinder.sources.base import JobSource

# The known, stable sender address LinkedIn uses for job-alert digest emails.
GMAIL_QUERY = "from:jobalerts-noreply@linkedin.com"

_JOB_LINK_RE = re.compile(r"/jobs/view/(\d+)")


def _decode_html_body(payload: dict) -> str:
    """Extract and base64url-decode the text/html part of a Gmail message payload."""

    def _walk(part: dict) -> str | None:
        body_data = part.get("body", {}).get("data")
        if part.get("mimeType") == "text/html" and body_data:
            padded = body_data + "=" * (-len(body_data) % 4)
            return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
        for sub_part in part.get("parts", []):
            found = _walk(sub_part)
            if found:
                return found
        return None

    return _walk(payload) or ""


def _guess_company(anchor) -> str:
    """Best-effort company name: the first other bit of text near the job link."""
    container = anchor.find_parent(["td", "div", "table"])
    if not container:
        return "Unknown (see posting)"
    title = anchor.get_text(strip=True)
    for text in container.stripped_strings:
        text = text.strip()
        if text and text != title:
            return text
    return "Unknown (see posting)"


def _parse_job_cards(html: str) -> list[dict]:
    """Best-effort extraction of job cards from a LinkedIn job-alert email."""
    soup = BeautifulSoup(html, "lxml")
    cards: list[dict] = []
    seen_ids: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        match = _JOB_LINK_RE.search(anchor["href"])
        if not match:
            continue
        job_id = match.group(1)
        if job_id in seen_ids:
            continue
        title = anchor.get_text(strip=True)
        if not title:
            continue
        seen_ids.add(job_id)
        cards.append(
            {
                "external_id": job_id,
                "title": title,
                "company": _guess_company(anchor),
                "url": anchor["href"].split("?")[0],
            }
        )
    return cards


class LinkedInEmailSource(JobSource):
    name = "linkedin"

    def __init__(self, gmail_service: Resource | None = None, db_path: Path | str | None = None):
        self._gmail_service = gmail_service
        self._db_path = db_path

    def _service(self) -> Resource:
        if self._gmail_service is None:
            from jobfinder.gmail.auth import get_gmail_service

            self._gmail_service = get_gmail_service()
        return self._gmail_service

    def fetch(self) -> list[JobPosting]:
        service = self._service()
        postings: list[JobPosting] = []
        for message_id in gmail_client.list_message_ids(service, GMAIL_QUERY):
            if store.is_gmail_message_processed(message_id, self._db_path):
                continue
            message = gmail_client.get_message(service, message_id)
            html = _decode_html_body(message.get("payload", {}))
            for card in _parse_job_cards(html):
                postings.append(
                    JobPosting(
                        source=self.name,
                        external_id=card["external_id"],
                        title=card["title"],
                        company=card["company"],
                        description=(
                            f"{card['title']} at {card['company']} - via LinkedIn job "
                            "alert email. Full description not included in the alert; "
                            "the applying agent visits the URL for full details."
                        ),
                        url=card["url"],
                        apply_method=ApplyMethod.LINKEDIN_EASY_APPLY,
                    )
                )
            store.mark_gmail_message_processed(message_id, "job_alert", self._db_path)
        return postings
