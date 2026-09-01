"""Local "hands" agent entrypoint: polls the shared datastore for approved
applications and submits them via a real, logged-in Chrome session.

Run scripts/seed_chrome_profile.py once first to log in, then:
    python scripts/run_local_agent.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobfinder import config
from jobfinder.db import store
from jobfinder.db.models import ApplicationStatus, ApplyMethod
from jobfinder.local.agent import ApplyAgent
from jobfinder.local.ats import get_adapter_for_url
from jobfinder.local.ats.base import ATSFormError
from jobfinder.local.browser_apply_fallback import run_apply_loop as run_fallback_loop
from jobfinder.local.browser_apply_linkedin import run_easy_apply_loop
from jobfinder.local.browser_session import browser_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jobfinder.local_agent")


def _apply_company_site(application, job) -> ApplicationStatus:
    with browser_session(headless=False) as page:
        page.goto(job.url)
        adapter = get_adapter_for_url(job.url)
        if adapter is not None:
            try:
                adapter.fill_and_submit(page, application, job)
                return ApplicationStatus.SENT
            except ATSFormError:
                logger.info("ATS adapter couldn't finish %s, falling back to accessibility-tree loop.", job.url)
        return run_fallback_loop(page, application, job)


def _apply_linkedin_easy_apply(application, job) -> ApplicationStatus:
    with browser_session(headless=False) as page:
        page.goto(job.url)
        return run_easy_apply_loop(page, application, job)


def main() -> None:
    store.init_db(config.DB_PATH)
    agent = ApplyAgent(db_path=config.DB_PATH)
    agent.register_handler(ApplyMethod.COMPANY_SITE, _apply_company_site)
    agent.register_handler(ApplyMethod.LINKEDIN_EASY_APPLY, _apply_linkedin_easy_apply)
    agent.run_forever()


if __name__ == "__main__":
    main()
