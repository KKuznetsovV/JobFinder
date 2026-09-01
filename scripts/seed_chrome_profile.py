"""One-time manual setup: opens a real, persistent Chrome profile and waits
for the user to log into LinkedIn (and any ATS sites they expect to apply
through) by hand.

Run this once before using the local apply agent:
    python scripts/seed_chrome_profile.py

This project never stores or types your passwords - Playwright's persistent
context writes the real Chrome profile (cookies, local storage, login
session) to `config.CHROME_USER_DATA_DIR` on disk as you browse normally, and
later automated runs reuse that same profile directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobfinder.local.browser_session import browser_session


def main() -> None:
    print("Opening a real Chrome window using the persistent profile...")
    print("Log into LinkedIn and any other sites you'll apply through, then")
    print("press Enter here when you're done.")
    with browser_session(headless=False) as page:
        page.goto("https://www.linkedin.com/login")
        input("Press Enter after logging in to save the session...")
    print("Chrome profile saved. You can now run the local apply agent.")


if __name__ == "__main__":
    main()
