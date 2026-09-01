"""Shared Anthropic client wrapper with retry handling for transient errors.

Centralizing this avoids every AI-calling module (relevance filter, resume
selector, cover-letter generator, reply classifier, browser-agent reasoning)
reimplementing its own retry loop.
"""
from __future__ import annotations

import logging
import time

import anthropic

from jobfinder import config

logger = logging.getLogger(__name__)

# NOTE: OverloadedError (HTTP 529) is a sibling of InternalServerError, not a
# subclass of it - both extend APIStatusError. Easy to miss since it's less
# commonly documented than 429/500/503, but it's a real, observed-in-production
# transient error and must be retried like the others.
RETRYABLE_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
)

MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 1.0


def get_client() -> anthropic.Anthropic:
    headers = {}
    if config.ANTHROPIC_WORKSPACE_ID:
        headers["anthropic-workspace-id"] = config.ANTHROPIC_WORKSPACE_ID
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, default_headers=headers)


def create_message_with_retry(client: anthropic.Anthropic, **kwargs):
    """Call `client.messages.create(**kwargs)`, retrying on transient errors
    with exponential backoff. Raises the last exception if all retries are
    exhausted."""
    attempt = 0
    while True:
        try:
            return client.messages.create(**kwargs)
        except RETRYABLE_EXCEPTIONS as exc:
            attempt += 1
            if attempt > MAX_RETRIES:
                logger.error("Anthropic call failed after %d retries: %s", MAX_RETRIES, exc)
                raise
            backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Anthropic call raised %s (attempt %d/%d), retrying in %.1fs",
                type(exc).__name__,
                attempt,
                MAX_RETRIES,
                backoff,
            )
            time.sleep(backoff)
