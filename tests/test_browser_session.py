from jobfinder import config
from jobfinder.local import browser_session


def test_launch_persistent_context_uses_real_chrome_channel(mocker, tmp_path):
    mocker.patch.object(config, "CHROME_USER_DATA_DIR", tmp_path / "chrome-profile")
    fake_context = mocker.Mock()
    fake_chromium = mocker.Mock()
    fake_chromium.launch_persistent_context.return_value = fake_context
    fake_playwright = mocker.Mock()
    fake_playwright.chromium = fake_chromium

    result = browser_session.launch_persistent_context(fake_playwright, headless=True)

    fake_chromium.launch_persistent_context.assert_called_once_with(
        user_data_dir=str(tmp_path / "chrome-profile"), channel="chrome", headless=True
    )
    assert result is fake_context
    assert (tmp_path / "chrome-profile").exists()


def test_browser_session_reuses_existing_page_and_closes_context(mocker):
    fake_page = mocker.Mock()
    fake_context = mocker.Mock()
    fake_context.pages = [fake_page]
    fake_playwright_ctx = mocker.Mock()
    fake_playwright_cm = mocker.MagicMock()
    fake_playwright_cm.__enter__.return_value = fake_playwright_ctx
    mocker.patch("playwright.sync_api.sync_playwright", return_value=fake_playwright_cm)
    mocker.patch.object(browser_session, "launch_persistent_context", return_value=fake_context)

    with browser_session.browser_session(headless=True) as page:
        assert page is fake_page

    fake_context.close.assert_called_once()


def test_browser_session_opens_new_page_when_no_pages_exist(mocker):
    fake_new_page = mocker.Mock()
    fake_context = mocker.Mock()
    fake_context.pages = []
    fake_context.new_page.return_value = fake_new_page
    fake_playwright_ctx = mocker.Mock()
    fake_playwright_cm = mocker.MagicMock()
    fake_playwright_cm.__enter__.return_value = fake_playwright_ctx
    mocker.patch("playwright.sync_api.sync_playwright", return_value=fake_playwright_cm)
    mocker.patch.object(browser_session, "launch_persistent_context", return_value=fake_context)

    with browser_session.browser_session(headless=True) as page:
        assert page is fake_new_page

    fake_context.new_page.assert_called_once()
