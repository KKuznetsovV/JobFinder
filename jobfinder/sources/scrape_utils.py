"""Shared helpers for HTML-scraping job sources: robots.txt compliance, rate
limiting, and a real-Chrome fallback for JavaScript-rendered or lightly
bot-protected pages.

Used by gotfriends/devjobs/jobinfo - none of the three exposes an RSS feed
(confirmed by direct check while building this), so a polite scraper is the
only option for them, per the project's "prefer RSS/email digest; scrape
only as a respectful fallback" instruction.
"""
from __future__ import annotations

import random
import time
import urllib.robotparser
from urllib.parse import urlparse

import requests

from jobfinder import config

_robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def _robots_parser(url: str) -> urllib.robotparser.RobotFileParser | None:
    # Fetch robots.txt ourselves with our identifying User-Agent instead of
    # urllib.robotparser's default read() (which uses a generic
    # "Python-urllib/x.y" UA): some sites 403 that generic UA on the robots.txt
    # request itself, which stdlib then misreads as "disallow everything" even
    # though the real robots.txt permits fetching.
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in _robots_cache:
        parser = urllib.robotparser.RobotFileParser()
        robots_url = f"{origin}/robots.txt"
        parser.set_url(robots_url)
        try:
            response = requests.get(
                robots_url, headers={"User-Agent": config.USER_AGENT}, timeout=15.0
            )
            if response.status_code in (401, 403):
                parser.disallow_all = True
            elif 400 <= response.status_code < 500:
                parser.allow_all = True
            elif response.status_code >= 500:
                parser = None  # robots.txt server error: fail open, don't fetch
            else:
                parser.parse(response.text.splitlines())
        except Exception:
            parser = None  # robots.txt itself unreachable
        _robots_cache[origin] = parser
    return _robots_cache[origin]


def can_fetch(url: str, user_agent: str | None = None) -> bool:
    parser = _robots_parser(url)
    if parser is None:
        return True
    return parser.can_fetch(user_agent or config.USER_AGENT, url)


def rate_limit_sleep() -> None:
    time.sleep(random.uniform(config.SCRAPE_MIN_DELAY_SECONDS, config.SCRAPE_MAX_DELAY_SECONDS))


def polite_get(url: str, timeout: float = 15.0) -> requests.Response:
    """A robots.txt-respecting, rate-limited GET with a real, identifying user agent."""
    if not can_fetch(url):
        raise PermissionError(f"robots.txt disallows fetching {url}")
    rate_limit_sleep()
    response = requests.get(url, headers={"User-Agent": config.USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response


def render_page_html(url: str, wait_selector: str | None = None, timeout_ms: int = 20000) -> str:
    """Render a JS-heavy or lightly bot-protected page with a real Chrome
    browser and return the fully rendered HTML. Still respects robots.txt
    and rate limiting - this is a heavier fallback, not a bypass technique.
    """
    if not can_fetch(url):
        raise PermissionError(f"robots.txt disallows fetching {url}")
    rate_limit_sleep()
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        try:
            page = browser.new_page(user_agent=config.USER_AGENT)
            page.goto(url, timeout=timeout_ms)
            if wait_selector:
                page.wait_for_selector(wait_selector, timeout=timeout_ms)
            html = page.content()
        finally:
            browser.close()
    return html
