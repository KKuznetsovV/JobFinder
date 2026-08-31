"""DevJobs (devjobs.co.il) job board source.

DevJobs' listing page is client-side rendered (confirmed via live inspection
on 2026-08-31 - a plain HTTP GET returns none of the job cards), so this uses
a real-Chrome render (scrape_utils.render_page_html) rather than requests.get.

Each card exposes the *actual* application destination via a `data-url`
attribute on its "Apply now" button: many listings are aggregated from
LinkedIn (data-url points at a linkedin.com/jobs/view/... page) while others
point directly at a company's own site/ATS. apply_method is set accordingly
so the local agent picks the right flow (semi-automatic LinkedIn vs. fully
automatic company-site).
"""
from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup

from jobfinder.db.models import ApplyMethod, JobPosting
from jobfinder.sources import scrape_utils
from jobfinder.sources.base import JobSource

logger = logging.getLogger(__name__)

LISTING_URL = "https://devjobs.co.il/jobs-grid"
_JOB_ID_RE = re.compile(r"/job-details/(\d+)")


def _parse_listing_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    jobs: list[dict] = []
    seen_ids: set[str] = set()

    for card in soup.select("div.card-grid-2"):
        title_anchor = card.select_one("a.name-job")
        company_anchor = card.select_one("a.profession")
        if not title_anchor or not company_anchor:
            continue

        job_id_match = _JOB_ID_RE.search(title_anchor.get("href", ""))
        if not job_id_match:
            continue
        job_id = job_id_match.group(1)
        if job_id in seen_ids:
            continue
        seen_ids.add(job_id)

        apply_anchor = card.select_one("a.btn-apply-now")
        apply_url = (apply_anchor.get("data-url") if apply_anchor else None) or title_anchor["href"]

        location = card.select_one("span.location-small")
        posted = card.select_one("span.card-time")
        skills = [
            li.get_text(strip=True) for li in card.select("ul.courses li") if li.get_text(strip=True)
        ]

        jobs.append(
            {
                "external_id": job_id,
                "title": title_anchor.get_text(strip=True),
                "company": company_anchor.get_text(strip=True),
                "location": location.get_text(strip=True) if location else "",
                "posted_date": posted.get_text(strip=True) if posted else None,
                "skills": skills,
                "apply_url": apply_url,
            }
        )
    return jobs


class DevJobsSource(JobSource):
    name = "devjobs"

    def __init__(self, listing_url: str = LISTING_URL):
        self.listing_url = listing_url

    def fetch(self) -> list[JobPosting]:
        try:
            html = scrape_utils.render_page_html(
                self.listing_url, wait_selector='a[href*="/job-details/"]'
            )
        except Exception:
            logger.exception("devjobs: failed to render listing page")
            return []

        postings: list[JobPosting] = []
        for job in _parse_listing_page(html):
            apply_method = (
                ApplyMethod.LINKEDIN_EASY_APPLY
                if "linkedin.com" in job["apply_url"]
                else ApplyMethod.COMPANY_SITE
            )
            skills_text = f" Skills: {', '.join(job['skills'])}." if job["skills"] else ""
            postings.append(
                JobPosting(
                    source=self.name,
                    external_id=job["external_id"],
                    title=job["title"],
                    company=job["company"],
                    description=(
                        f"{job['title']} at {job['company']}, {job['location']}."
                        f"{skills_text} Listed on DevJobs."
                    ),
                    url=job["apply_url"],
                    posted_date=job["posted_date"],
                    apply_method=apply_method,
                )
            )
        return postings
