"""Thin, low-level wrappers around the Gmail API resource from auth.get_gmail_service().

Kept deliberately minimal (list/get/send) - the notification-sending logic
(feature/gmail-notify) and reply-polling/classification logic
(feature/gmail-approval-loop) are built on top of these primitives in later
modules, not here.
"""
from __future__ import annotations

import base64
from typing import Any

from googleapiclient.discovery import Resource


def list_message_ids(service: Resource, query: str, user_id: str = "me") -> list[str]:
    """Return all message ids matching a Gmail search query, following pagination."""
    ids: list[str] = []
    page_token = None
    while True:
        response = (
            service.users()
            .messages()
            .list(userId=user_id, q=query, pageToken=page_token)
            .execute()
        )
        ids.extend(m["id"] for m in response.get("messages", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return ids


def get_message(
    service: Resource, message_id: str, user_id: str = "me", format: str = "full"
) -> dict[str, Any]:
    return service.users().messages().get(userId=user_id, id=message_id, format=format).execute()


def get_thread(service: Resource, thread_id: str, user_id: str = "me") -> dict[str, Any]:
    return service.users().threads().get(userId=user_id, id=thread_id, format="full").execute()


def send_raw_message(
    service: Resource,
    raw_rfc822_bytes: bytes,
    user_id: str = "me",
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Send a pre-built RFC822 message.

    Callers build the MIME message (including any custom headers like
    config.APP_ID_HEADER) and pass the raw bytes here; this function only
    handles the base64url encoding and the API call.
    """
    body: dict[str, Any] = {"raw": base64.urlsafe_b64encode(raw_rfc822_bytes).decode("ascii")}
    if thread_id:
        body["threadId"] = thread_id
    return service.users().messages().send(userId=user_id, body=body).execute()
