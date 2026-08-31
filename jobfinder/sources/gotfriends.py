"""GotFriends (gotfriends.co.il) job board source.

GotFriends is an Israeli tech staffing agency; postings are listed under
their own brand and applying goes through their submission form rather than
the hiring company's own ATS (agency placement model), so these are stored
with ApplyMethod.COMPANY_SITE (a generic "apply via this URL's web form").

No RSS feed is available (confirmed 2026-08-31), so this scrapes the public
category listing pages directly, respecting robots.txt and rate limiting
(see scrape_utils). To limit request volume per poll cycle, only the short
listing-page blurb is stored as `description` - the browser-apply agent
fetches the full posting page at apply time regardless.
"""
from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup

from jobfinder.db.models import ApplyMethod, JobPosting
from jobfinder.sources import scrape_utils
from jobfinder.sources.base import JobSource

logger = logging.getLogger(__name__)

BASE_URL = "https://www.gotfriends.co.il"

# Known top-level job categories from the site's listing hub. Categories are
# added/renamed occasionally - if the site restructures, update this list.
CATEGORY_SLUGS = [
    "software",
    "ai",
    "hardware",
    "datasecurity",
    "algorithm",
    "qa",
    "executive-position",
    "graduates",
    "system",
    "projects",
    "bibig_data",
]

# Individual job postings end in a numeric id, e.g.
# /jobslobby/ai/llm-engineer/142628/ - category/overview links don't.
_JOB_LINK_RE = re.compile(r"/jobslobby/([^/]+)/([^/]+)/(\d+)/?(?:$|\?)")


def _parse_category_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    seen_ids: set[str] = set()
    jobs: list[dict] = []
    for anchor in soup.find_all("a", href=True):
        match = _JOB_LINK_RE.search(anchor["href"])
        if not match:
            continue
        job_id = match.group(3)
        if job_id in seen_ids:
            continue
        title = anchor.get_text(strip=True)
        if not title:
            continue
        seen_ids.add(job_id)
        jobs.append(
            {
                "external_id": job_id,
                "title": title,
                "url": anchor["href"].split("?")[0],
            }
        )
    return jobs


class GotFriendsSource(JobSource):
    name = "gotfriends"

    def __init__(self, category_slugs: list[str] | None = None):
        self.category_slugs = category_slugs or CATEGORY_SLUGS

    def fetch(self) -> list[JobPosting]:
        postings: list[JobPosting] = []
        for slug in self.category_slugs:
            url = f"{BASE_URL}/jobslobby/{slug}/"
            try:
                response = scrape_utils.polite_get(url)
            except Exception:
                logger.exception("gotfriends: failed to fetch category '%s'", slug)
                continue
            for job in _parse_category_page(response.text):
                postings.append(
                    JobPosting(
                        source=self.name,
                        external_id=job["external_id"],
                        title=job["title"],
                        company="See posting (GotFriends is a staffing agency; employer is disclosed on apply)",
                        description=(
                            f"{job['title']} - listed in the '{slug}' category on GotFriends. "
                            "Full job description not fetched during polling; the applying "
                            "agent visits the URL for full details."
                        ),
                        url=job["url"],
                        apply_method=ApplyMethod.COMPANY_SITE,
                    )
                )
        return postings
