import base64

from jobfinder.gmail import auth, client


def test_load_credentials_returns_cached_valid_token(tmp_path, mocker):
    token_file = tmp_path / "token.json"
    token_file.write_text("{}", encoding="utf-8")

    fake_creds = mocker.Mock(valid=True)
    mocker.patch.object(auth.Credentials, "from_authorized_user_file", return_value=fake_creds)
    flow_ctor = mocker.patch.object(auth.InstalledAppFlow, "from_client_secrets_file")

    result = auth.load_credentials(
        client_secret_file=tmp_path / "secret.json", token_file=token_file, scopes=["scope"]
    )

    assert result is fake_creds
    flow_ctor.assert_not_called()


def test_load_credentials_refreshes_expired_token(tmp_path, mocker):
    token_file = tmp_path / "token.json"
    token_file.write_text("{}", encoding="utf-8")

    fake_creds = mocker.Mock(valid=False, expired=True, refresh_token="rt")
    fake_creds.to_json.return_value = '{"refreshed": true}'
    mocker.patch.object(auth.Credentials, "from_authorized_user_file", return_value=fake_creds)
    mocker.patch.object(auth, "Request")
    flow_ctor = mocker.patch.object(auth.InstalledAppFlow, "from_client_secrets_file")

    result = auth.load_credentials(
        client_secret_file=tmp_path / "secret.json", token_file=token_file, scopes=["scope"]
    )

    fake_creds.refresh.assert_called_once()
    flow_ctor.assert_not_called()
    assert result is fake_creds
    assert token_file.read_text(encoding="utf-8") == '{"refreshed": true}'


def test_load_credentials_runs_interactive_flow_when_no_token(tmp_path, mocker):
    token_file = tmp_path / "token.json"  # does not exist yet
    secret_file = tmp_path / "secret.json"
    secret_file.write_text("{}", encoding="utf-8")

    fake_creds = mocker.Mock()
    fake_creds.to_json.return_value = '{"new": true}'
    fake_flow = mocker.Mock()
    fake_flow.run_local_server.return_value = fake_creds
    mocker.patch.object(auth.InstalledAppFlow, "from_client_secrets_file", return_value=fake_flow)

    result = auth.load_credentials(
        client_secret_file=secret_file, token_file=token_file, scopes=["scope"]
    )

    fake_flow.run_local_server.assert_called_once_with(port=0)
    assert result is fake_creds
    assert token_file.read_text(encoding="utf-8") == '{"new": true}'


def test_load_credentials_missing_secret_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="Gmail OAuth client secret not found"):
        auth.load_credentials(
            client_secret_file=tmp_path / "missing_secret.json",
            token_file=tmp_path / "missing_token.json",
            scopes=["scope"],
        )


def test_list_message_ids_paginates(mocker):
    service = mocker.Mock()
    execute_mock = service.users.return_value.messages.return_value.list.return_value.execute
    execute_mock.side_effect = [
        {"messages": [{"id": "1"}, {"id": "2"}], "nextPageToken": "p2"},
        {"messages": [{"id": "3"}]},
    ]

    ids = client.list_message_ids(service, "query")

    assert ids == ["1", "2", "3"]
    assert execute_mock.call_count == 2


def test_get_message_and_thread_call_correct_api(mocker):
    service = mocker.Mock()
    client.get_message(service, "msg-1")
    service.users.return_value.messages.return_value.get.assert_called_once_with(
        userId="me", id="msg-1", format="full"
    )

    client.get_thread(service, "thread-1")
    service.users.return_value.threads.return_value.get.assert_called_once_with(
        userId="me", id="thread-1", format="full"
    )


def test_send_raw_message_encodes_and_includes_thread_id(mocker):
    service = mocker.Mock()
    raw = b"From: me\r\nTo: you\r\n\r\nhello"

    client.send_raw_message(service, raw, thread_id="thread-42")

    send_call = service.users.return_value.messages.return_value.send
    _, kwargs = send_call.call_args
    assert kwargs["userId"] == "me"
    assert kwargs["body"]["threadId"] == "thread-42"
    decoded = base64.urlsafe_b64decode(kwargs["body"]["raw"])
    assert decoded == raw
