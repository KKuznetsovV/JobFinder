"""Real, persistent Chrome browser session management for the local apply
agent.

Uses Playwright's `launch_persistent_context` with `channel="chrome"` (the
user's actual installed Chrome, never plain Chromium) and a persistent
profile directory (`config.CHROME_USER_DATA_DIR`). Once the user logs into
LinkedIn/ATS sites once via `scripts/seed_chrome_profile.py`, later automated
runs reuse that real, cookie-based logged-in session. This project never
stores or types LinkedIn/ATS passwords - the user logs in manually, once.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from jobfinder import config

logger = logging.getLogger(__name__)


def launch_persistent_context(playwright, headless: bool = False):
    config.CHROME_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return playwright.chromium.launch_persistent_context(
        user_data_dir=str(config.CHROME_USER_DATA_DIR),
        channel="chrome",
        headless=headless,
    )


@contextmanager
def browser_session(headless: bool = False) -> Iterator:
    """Yield a Page backed by the persistent, real-Chrome profile. Reuses an
    already-open tab if the persistent context restores one, otherwise opens
    a new page."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        context = launch_persistent_context(playwright, headless=headless)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            yield page
        finally:
            context.close()
