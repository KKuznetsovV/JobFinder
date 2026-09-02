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
RELEVANCE_THRESHOLD = 0.42  # 0-1 score from the relevance-filter Claude call
MAX_REVISION_ROUNDS = 3
DAILY_APPLICATION_CAP = int(os.environ.get("DAILY_APPLICATION_CAP", "25"))

# Local (no-API-call) embedding pre-filter that runs before the Claude relevance
# call. 0-1 cosine similarity against the candidate's resumes; postings below
# this score never reach the relevance/resume-selection/cover-letter LLM calls.
TIER0_MIN_SIMILARITY = float(os.environ.get("TIER0_MIN_SIMILARITY", "0.35"))
TIER0_MODEL_NAME = os.environ.get("TIER0_MODEL_NAME", "all-MiniLM-L6-v2")

# Tier-1 local fine-tuned classifier (LoRA-tuned Qwen2.5-1.5B-Instruct, see
# scripts/tier1/) that can replace the Claude relevance-filter and
# resume-selection calls for postings that already passed tier-0. Kill-switch
# defaults to off (no trained model shipped yet / instant revert if something
# looks wrong): false means every posting goes through the pre-tier1
# all-Claude path, identical to before this feature existed.
TIER1_MODEL_ENABLED = os.environ.get("TIER1_MODEL_ENABLED", "false").lower() == "true"
# 0-1 confidence cutoff, applied to BOTH the relevance and resume-selection
# decisions; below this on either, fall back to the real Claude calls for
# that posting instead of trusting the local model. Start conservative -
# retune once jobfinder.db.store.summarize_tier1_decisions() shows the
# model's real agreement rate with Claude over enough spot-checks.
TIER1_CONFIDENCE_MIN = float(os.environ.get("TIER1_CONFIDENCE_MIN", "0.85"))
# Fraction of confident local-model decisions to double-check against Claude
# anyway, purely to log agreement/disagreement (0.1 = roughly 1 in 10).
TIER1_AGREEMENT_CHECK_RATE = float(os.environ.get("TIER1_AGREEMENT_CHECK_RATE", "0.1"))
TIER1_MODEL_PATH = Path(
    os.environ.get("TIER1_MODEL_PATH", str(BASE_DIR / "models" / "tier1_classifier.q4_k_m.gguf"))
)

SCRAPE_MIN_DELAY_SECONDS = 3
SCRAPE_MAX_DELAY_SECONDS = 8
USER_AGENT = "JobFinderBot/0.1 (+personal job-search assistant; single user, low volume)"

# Custom Gmail header used to tie a notification thread to an Application row.
# Never rely on subject text alone: Gmail (and replies) can mangle/rewrite it.
APP_ID_HEADER = "X-App-ID"
