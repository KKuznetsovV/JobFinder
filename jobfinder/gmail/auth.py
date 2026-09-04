"""Gmail OAuth: interactive first-run login, then silent token refresh.

Uses the least-privilege scopes in config.GMAIL_SCOPES (readonly + send only -
no modify/label scope, since JobFinder never needs to alter the user's inbox
state beyond sending new messages). Credentials are cached in
config.GMAIL_TOKEN_FILE (gitignored) so subsequent runs don't need a browser
popup unless the refresh token itself is revoked.

Pylance strict mode is downgraded to basic for this file only: google-auth /
google-auth-oauthlib ship no py.typed type stubs, so strict-only checks
(reportUnknownMemberType etc.) fire on essentially every call into them with
no actionable fix available.
"""
# pyright: basic
from __future__ import annotations

from pathlib import Path
from typing import cast

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build

from jobfinder import config


def load_credentials(
    client_secret_file: Path | None = None,
    token_file: Path | None = None,
    scopes: list[str] | None = None,
) -> Credentials:
    """Return valid Gmail API credentials.

    Order of preference: a still-valid cached token, then a silent refresh of
    an expired-but-refreshable token, then (only as a last resort) the
    interactive browser-based OAuth flow. The resulting token is always
    written back to `token_file`.
    """
    client_secret_file = Path(client_secret_file or config.GMAIL_CLIENT_SECRET_FILE)
    token_file = Path(token_file or config.GMAIL_TOKEN_FILE)
    scopes = scopes or config.GMAIL_SCOPES

    creds: Credentials | None = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), scopes)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if not client_secret_file.exists():
            raise FileNotFoundError(
                f"Gmail OAuth client secret not found at {client_secret_file}. "
                "Download it from Google Cloud Console (see README 'Authorizing "
                "Gmail') and set GMAIL_CLIENT_SECRET_FILE, then re-run to "
                "complete the interactive login."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), scopes)
        # google-auth-oauthlib's return type also covers external-account creds,
        # which this project's Desktop-app OAuth flow never produces.
        creds = cast(Credentials, flow.run_local_server(port=0))

    assert creds is not None
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(creds.to_json(), encoding="utf-8")
    return creds


def get_gmail_service(
    client_secret_file: Path | None = None,
    token_file: Path | None = None,
    scopes: list[str] | None = None,
) -> Resource:
    """Return an authorized Gmail API client (googleapiclient Resource)."""
    creds = load_credentials(client_secret_file, token_file, scopes)
    return build("gmail", "v1", credentials=creds)
