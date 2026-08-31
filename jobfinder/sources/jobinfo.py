"""JobInfo (jobinfo.co.il) job source - INTENTIONALLY NOT IMPLEMENTED.

jobinfo.co.il serves a Cloudflare "Just a moment..." JS challenge to
automated requests (confirmed via a real Chrome browser session on
2026-08-31, not just a plain HTTP client - so this isn't a simple
user-agent/header issue). Building anything to solve that challenge
(headless-detection evasion, CAPTCHA/challenge solving, etc.) would mean
deliberately bypassing an active anti-bot security control, which this
project does not do.

Per an explicit decision with the user, this source is a documented no-op:
`fetch()` always returns an empty list and logs a warning once per process,
rather than silently pretending to work. If JobInfo later offers an email
job-alert subscription (like LinkedIn's) or an official API, a real
implementation could be added following the `linkedin_email.py` or
`gotfriends.py` pattern instead.
"""
from __future__ import annotations

import logging

from jobfinder.db.models import JobPosting
from jobfinder.sources.base import JobSource

logger = logging.getLogger(__name__)


class JobInfoSource(JobSource):
    name = "jobinfo"

    def fetch(self) -> list[JobPosting]:
        logger.warning(
            "jobinfo: skipped - jobinfo.co.il is behind Cloudflare bot protection "
            "and this project does not attempt to bypass anti-bot controls. "
            "See jobfinder/sources/jobinfo.py module docstring."
        )
        return []
