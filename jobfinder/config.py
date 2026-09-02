"""Central configuration: model IDs, paths, thresholds, and env-derived secrets.

Both the cloud component and the local component import this module, so keep it
free of heavy dependencies and side effects beyond loading `.env`.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Model configuration -----------------------------------------------------
# Single place to change the Claude model used for generation, classification,
# resume selection, fact-checking, email-reply classification, and the
# browser-automation reasoning step. Current Sonnet snapshot as of 2026-08-31
# (see https://platform.claude.com/docs/en/about-claude/models/overview).
MODEL_ID = "claude-sonnet-5"
MAX_TOKENS_DEFAULT = 4096

# --- Paths ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("JOBFINDER_DB_PATH", "data/jobfinder.db"))
RESUME_DIR = BASE_DIR / "resumes"
RESUME_CACHE_DIR = BASE_DIR / "data" / "resume_cache"
STYLE_REFERENCE_PATH = BASE_DIR / "config" / "style_reference.txt"
CHROME_USER_DATA_DIR = Path(os.environ.get("CHROME_USER_DATA_DIR", "chrome-profile/"))

# --- Identity --------------------------------------------------------------
APPLICANT_NAME = os.environ.get("APPLICANT_NAME", "Kirill Kuznetsov")
APPLICANT_PHONE = os.environ.get("APPLICANT_PHONE", "")

# --- Anthropic -------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Required only for multi-workspace personal API keys (Console -> Settings ->
# Workspaces); leave unset for a workspace-scoped or single-workspace key.
ANTHROPIC_WORKSPACE_ID = os.environ.get("ANTHROPIC_WORKSPACE_ID", "")

# --- Gmail (least-privilege scopes: read to poll, send to notify/dispatch) --
GMAIL_CLIENT_SECRET_FILE = Path(
    os.environ.get("GMAIL_CLIENT_SECRET_FILE", "credentials/gmail_client_secret.json")
)
GMAIL_TOKEN_FILE = Path(os.environ.get("GMAIL_TOKEN_FILE", "credentials/gmail_token.json"))
GMAIL_USER_EMAIL = os.environ.get("GMAIL_USER_EMAIL", "")
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

# --- Scheduling / thresholds / pacing ---------------------------------------
JOB_POLL_INTERVAL_HOURS = 3
RELEVANCE_THRESHOLD = 0.35  # 0-1 score from the relevance-filter Claude call
MAX_REVISION_ROUNDS = 3
DAILY_APPLICATION_CAP = int(os.environ.get("DAILY_APPLICATION_CAP", "15"))
SCRAPE_MIN_DELAY_SECONDS = 3
SCRAPE_MAX_DELAY_SECONDS = 8
USER_AGENT = "JobFinderBot/0.1 (+personal job-search assistant; single user, low volume)"

# Custom Gmail header used to tie a notification thread to an Application row.
# Never rely on subject text alone: Gmail (and replies) can mangle/rewrite it.
APP_ID_HEADER = "X-App-ID"
