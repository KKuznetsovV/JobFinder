import urllib.robotparser

import pytest
import requests

from jobfinder.sources import scrape_utils


def _robots_disallowing(paths: list[str]) -> urllib.robotparser.RobotFileParser:
    parser = urllib.robotparser.RobotFileParser()
    lines = ["User-agent: *"] + [f"Disallow: {p}" for p in paths]
    parser.parse(lines)
    return parser


def test_can_fetch_respects_robots_disallow(monkeypatch):
    scrape_utils._robots_cache.clear()
    monkeypatch.setitem(
        scrape_utils._robots_cache,
        "https://example.com",
        _robots_disallowing(["/private/"]),
    )
    assert scrape_utils.can_fetch("https://example.com/public/page") is True
    assert scrape_utils.can_fetch("https://example.com/private/page") is False


def test_can_fetch_fails_open_when_robots_unreachable(monkeypatch):
    scrape_utils._robots_cache.clear()
    monkeypatch.setitem(scrape_utils._robots_cache, "https://example.com", None)
    assert scrape_utils.can_fetch("https://example.com/anything") is True


def test_polite_get_raises_when_disallowed(monkeypatch):
    monkeypatch.setattr(scrape_utils, "can_fetch", lambda url, user_agent=None: False)
    with pytest.raises(PermissionError):
        scrape_utils.polite_get("https://example.com/blocked")


def test_polite_get_sleeps_and_calls_requests(monkeypatch, mocker):
    monkeypatch.setattr(scrape_utils, "can_fetch", lambda url, user_agent=None: True)
    sleep_mock = mocker.patch.object(scrape_utils, "rate_limit_sleep")
    fake_response = mocker.Mock()
    fake_response.raise_for_status = mocker.Mock()
    get_mock = mocker.patch.object(requests, "get", return_value=fake_response)

    result = scrape_utils.polite_get("https://example.com/ok")

    sleep_mock.assert_called_once()
    get_mock.assert_called_once()
    fake_response.raise_for_status.assert_called_once()
    assert result is fake_response


def test_render_page_html_raises_when_disallowed(monkeypatch):
    monkeypatch.setattr(scrape_utils, "can_fetch", lambda url, user_agent=None: False)
    with pytest.raises(PermissionError):
        scrape_utils.render_page_html("https://example.com/blocked")


def test_render_page_html_uses_real_chrome_channel(monkeypatch, mocker):
    monkeypatch.setattr(scrape_utils, "can_fetch", lambda url, user_agent=None: True)
    mocker.patch.object(scrape_utils, "rate_limit_sleep")

    fake_page = mocker.Mock()
    fake_page.content.return_value = "<html>rendered</html>"
    fake_browser = mocker.Mock()
    fake_browser.new_page.return_value = fake_page
    fake_chromium = mocker.Mock()
    fake_chromium.launch.return_value = fake_browser
    fake_playwright_ctx = mocker.Mock()
    fake_playwright_ctx.chromium = fake_chromium
    fake_playwright_cm = mocker.MagicMock()
    fake_playwright_cm.__enter__.return_value = fake_playwright_ctx

    mocker.patch("playwright.sync_api.sync_playwright", return_value=fake_playwright_cm)

    html = scrape_utils.render_page_html("https://example.com/ok", wait_selector=".job-card")

    fake_chromium.launch.assert_called_once_with(channel="chrome", headless=True)
    fake_page.wait_for_selector.assert_called_once()
    fake_browser.close.assert_called_once()
    assert html == "<html>rendered</html>"
