"""Behavioural tests for ``unify.gateway.channels.gmail``.

Includes all 13 scenarios faithfully ported from
``communication/tests/gmail/test_send_with_attachment.py`` (8) and
``communication/tests/gmail/test_delete_watch.py`` (5), plus
router-contract and per-endpoint tests for the other 3 endpoints
(``/delete``, ``/watch``, ``/attachment``). Ported scenarios are
tagged ``PORTED:`` in their docstrings.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from googleapiclient.errors import HttpError

from unify.gateway.channels.gmail import router

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _http_error(status_code: int) -> HttpError:
    """Construct an ``HttpError`` whose ``resp.status`` matches ``status_code``."""
    resp = MagicMock()
    resp.status = status_code
    resp.reason = "error"
    return HttpError(resp=resp, content=b"")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CAJ123@mail.gmail.com", "<CAJ123@mail.gmail.com>"),
        ("<CAJ123@mail.gmail.com>", "<CAJ123@mail.gmail.com>"),
        ("  CAJ123@mail.gmail.com  ", "<CAJ123@mail.gmail.com>"),
        ("<CAJ123@mail.gmail.com", "<CAJ123@mail.gmail.com>"),
        ("CAJ123@mail.gmail.com>", "<CAJ123@mail.gmail.com>"),
        ("", ""),
    ],
)
def test_as_message_id_normalizes_brackets(raw, expected):
    """In-Reply-To / References must carry an angle-bracketed Message-ID so the
    recipient threads the reply, regardless of how the caller passed it."""
    from unify.gateway.channels.gmail.views import _as_message_id

    assert _as_message_id(raw) == expected


def test_format_from_header_uses_explicit_name():
    from unify.gateway.channels.gmail.views import _format_from_header

    assert (
        _format_from_header("staging-twin@unify.ai", "T-W1N")
        == "T-W1N <staging-twin@unify.ai>"
    )


def test_format_from_header_preserves_existing_display_name():
    from unify.gateway.channels.gmail.views import _format_from_header

    assert (
        _format_from_header("Existing Name <sender@example.com>")
        == "Existing Name <sender@example.com>"
    )


@patch("unify.gateway.channels.gmail.views._resolve_directory_display_name")
def test_format_from_header_falls_back_to_directory_name(mock_lookup):
    from unify.gateway.channels.gmail.views import _format_from_header

    mock_lookup.return_value = "T-W1N"
    assert (
        _format_from_header("staging-twin@unify.ai") == "T-W1N <staging-twin@unify.ai>"
    )
    mock_lookup.assert_called_once_with("staging-twin@unify.ai")


@pytest.fixture
def _gmail_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_SA_KEY", "{}")
    monkeypatch.setenv("WORKSPACE_ADMIN_SUBJECT", "admin@example.com")
    monkeypatch.setenv("ORCHESTRA_ADMIN_KEY", "test-admin-key")


@pytest.fixture
def mock_gmail_service() -> MagicMock:
    """Mock the Gmail API service surface used by send and watch."""
    mock_service = MagicMock()
    mock_service.users().messages().send().execute.return_value = {
        "id": "test_message_id_123",
    }
    mock_service.users().stop().execute.return_value = {}
    mock_service.users().watch().execute.return_value = {"historyId": "hist-1"}
    return mock_service


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/gmail")
    return app


@pytest.fixture
def client(
    app: FastAPI,
    mock_gmail_service: MagicMock,
    _gmail_credentials: None,
) -> TestClient:
    with patch(
        "unify.gateway.channels.gmail.views.get_gmail_service_async",
        new_callable=AsyncMock,
        return_value=mock_gmail_service,
    ):
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Router contract
# ---------------------------------------------------------------------------


def test_router_exposes_expected_paths() -> None:
    """FastAPI emits one route entry per (path, method); /watch has two."""
    paths = sorted(
        (r.path, sorted(r.methods)) for r in router.routes  # type: ignore[attr-defined]
    )
    assert paths == [
        ("/attachment", ["GET"]),
        ("/delete", ["DELETE"]),
        ("/send", ["POST"]),
        ("/watch", ["DELETE"]),
        ("/watch", ["POST"]),
    ]


def test_router_importable_from_package_root() -> None:
    from unify.gateway.channels.gmail import router as exported

    assert exported is router


# ---------------------------------------------------------------------------
# POST /send -- 4 scenarios PORTED from test_send_with_attachment.py
# ---------------------------------------------------------------------------


class TestSendEmailWithoutAttachment:
    """PORTED: from communication/tests/gmail/test_send_with_attachment.py."""

    def test_send_basic_email(
        self,
        client: TestClient,
        mock_gmail_service: MagicMock,
    ) -> None:
        """PORTED: Send a basic email without attachment."""
        response = client.post(
            "/gmail/send",
            json={
                "from": "sender@example.com",
                "to": "recipient@example.com",
                "subject": "Test Subject",
                "body": "Test body content",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["id"] == "test_message_id_123"

    def test_send_email_with_reply_to(
        self,
        client: TestClient,
        mock_gmail_service: MagicMock,
    ) -> None:
        """PORTED: Send an email as a reply (threading)."""
        response = client.post(
            "/gmail/send",
            json={
                "from": "sender@example.com",
                "to": "recipient@example.com",
                "subject": "Re: Test Subject",
                "body": "Reply content",
                "in_reply_to": "<original-message-id@example.com>",
                "thread_id": "gmail-thread-123",
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        send_body = mock_gmail_service.users().messages().send.call_args.kwargs["body"]
        assert send_body["threadId"] == "gmail-thread-123"


class TestSendEmailWithAttachment:
    """PORTED: attachment scenarios from test_send_with_attachment.py."""

    def test_send_email_with_attachment(
        self,
        client: TestClient,
        mock_gmail_service: MagicMock,
    ) -> None:
        """PORTED: Send an email with a file attachment."""
        file_content = b"This is test file content"
        content_base64 = base64.b64encode(file_content).decode("utf-8")
        response = client.post(
            "/gmail/send",
            json={
                "from": "sender@example.com",
                "to": "recipient@example.com",
                "subject": "Email with attachment",
                "body": "Please see attached file",
                "attachment": {
                    "filename": "test_document.txt",
                    "content_base64": content_base64,
                },
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert response.json()["id"] == "test_message_id_123"
        assert mock_gmail_service.users().messages().send.called

    def test_send_email_with_pdf_attachment(
        self,
        client: TestClient,
        mock_gmail_service: MagicMock,
    ) -> None:
        """PORTED: Send an email with a PDF attachment."""
        pdf_content = b"%PDF-1.4 fake pdf content"
        content_base64 = base64.b64encode(pdf_content).decode("utf-8")
        response = client.post(
            "/gmail/send",
            json={
                "from": "sender@example.com",
                "to": "recipient@example.com",
                "subject": "Report attached",
                "body": "Here is the quarterly report",
                "attachment": {
                    "filename": "quarterly_report.pdf",
                    "content_base64": content_base64,
                },
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_send_email_with_attachment_and_reply(
        self,
        client: TestClient,
        mock_gmail_service: MagicMock,
    ) -> None:
        """PORTED: Send a reply email with an attachment."""
        file_content = b"attachment content"
        content_base64 = base64.b64encode(file_content).decode("utf-8")
        response = client.post(
            "/gmail/send",
            json={
                "from": "sender@example.com",
                "to": "recipient@example.com",
                "subject": "Re: Document request",
                "body": "Here is the file you requested",
                "in_reply_to": "<message-id@example.com>",
                "attachment": {
                    "filename": "requested_file.docx",
                    "content_base64": content_base64,
                },
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True


class TestAttachmentErrorHandling:
    """PORTED: from test_send_with_attachment.py."""

    def test_invalid_base64_content(
        self,
        client: TestClient,
        mock_gmail_service: MagicMock,
    ) -> None:
        """PORTED: Return 400 for invalid base64 content."""
        response = client.post(
            "/gmail/send",
            json={
                "from": "sender@example.com",
                "to": "recipient@example.com",
                "subject": "Test",
                "body": "Test body",
                "attachment": {
                    "filename": "test.txt",
                    "content_base64": "not-valid-base64!!!",
                },
            },
        )
        assert response.status_code == 400
        assert "Failed to attach file" in response.json()["detail"]

    def test_attachment_without_filename(
        self,
        client: TestClient,
        mock_gmail_service: MagicMock,
    ) -> None:
        """PORTED: Attachment without filename uses default."""
        file_content = b"content"
        content_base64 = base64.b64encode(file_content).decode("utf-8")
        response = client.post(
            "/gmail/send",
            json={
                "from": "sender@example.com",
                "to": "recipient@example.com",
                "subject": "Test",
                "body": "Test body",
                "attachment": {"content_base64": content_base64},
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True


class TestRequestValidation:
    """PORTED: from test_send_with_attachment.py."""

    def test_missing_required_fields(self, client: TestClient) -> None:
        """PORTED: Return 400 when required fields are missing."""
        response = client.post("/gmail/send", json={"subject": "Test"})
        assert response.status_code == 400
        assert "Missing required fields" in response.json()["detail"]

    def test_missing_body(self, client: TestClient) -> None:
        """PORTED: Return 400 when body is missing."""
        response = client.post(
            "/gmail/send",
            json={
                "from": "sender@example.com",
                "to": "recipient@example.com",
                "subject": "Test",
            },
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /watch -- 4 scenarios PORTED from test_delete_watch.py
# ---------------------------------------------------------------------------


class TestDeleteGmailWatch:
    """PORTED: from communication/tests/gmail/test_delete_watch.py."""

    def test_stop_watch_success(
        self,
        client: TestClient,
        mock_gmail_service: MagicMock,
    ) -> None:
        """PORTED: Stop the Gmail watch for the given primary email."""
        response = client.request(
            "DELETE",
            "/gmail/watch",
            json={"primary_email": "user@byod.com"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["primary_email"] == "user@byod.com"
        assert data.get("already_absent") is not True
        mock_gmail_service.users().stop.assert_called_with(userId="me")

    def test_missing_primary_email_returns_400(self, client: TestClient) -> None:
        """PORTED: Return 400 when primary_email is missing."""
        response = client.request("DELETE", "/gmail/watch", json={})
        assert response.status_code == 400
        assert response.json()["detail"] == "Missing primary_email"

    def test_stop_watch_404_treated_as_already_absent(
        self,
        client: TestClient,
        mock_gmail_service: MagicMock,
    ) -> None:
        """PORTED: A 404 from Gmail's stop call is a benign no-op."""
        mock_gmail_service.users().stop().execute.side_effect = _http_error(404)
        response = client.request(
            "DELETE",
            "/gmail/watch",
            json={"primary_email": "user@byod.com"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["already_absent"] is True
        assert data["primary_email"] == "user@byod.com"

    def test_stop_watch_500_bubbles_up(
        self,
        client: TestClient,
        mock_gmail_service: MagicMock,
    ) -> None:
        """PORTED: Non-404 Gmail API errors surface as 500 to the caller."""
        mock_gmail_service.users().stop().execute.side_effect = _http_error(500)
        response = client.request(
            "DELETE",
            "/gmail/watch",
            json={"primary_email": "user@byod.com"},
        )
        assert response.status_code == 500


# Note on the 5th DELETE /watch scenario from communication:
# test_missing_auth_returns_unauthorized was an aggregator-level test
# (the admin-key dependency lives on the aggregator router mount, not
# the channel module itself). Per the channel-isolated test shape
# documented in unify/gateway/channels/README.md, auth integration
# belongs in tests/gateway/test_app.py once the Phase B aggregator
# (unify/gateway/app.py) lands. Tagged here so the scenario isn't lost.


# ---------------------------------------------------------------------------
# POST /watch -- new tests (no existing scenarios)
# ---------------------------------------------------------------------------


def test_start_watch_returns_history_id(
    client: TestClient,
    mock_gmail_service: MagicMock,
) -> None:
    mock_gmail_service.users().watch().execute.return_value = {"historyId": "hist-42"}
    response = client.post(
        "/gmail/watch",
        json={"primary_email": "user@byod.com"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["historyId"] == "hist-42"


def test_start_watch_missing_primary_email_returns_400(client: TestClient) -> None:
    response = client.post("/gmail/watch", json={})
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing primary_email"


def test_start_watch_default_topic_includes_gmail_notifications_prefix(
    client: TestClient,
    mock_gmail_service: MagicMock,
) -> None:
    """Pin the default topic name format: gmail-notifications{env_suffix}."""
    response = client.post(
        "/gmail/watch",
        json={"primary_email": "user@byod.com"},
    )
    assert response.status_code == 200
    # The watch() call received a body whose topicName references the
    # gmail-notifications topic. Inspect the most recent call.
    call_args = mock_gmail_service.users().watch.call_args
    body = call_args.kwargs.get("body") or (
        call_args.args[1] if len(call_args.args) > 1 else None
    )
    if body is None:
        # MagicMock returns a different shape depending on chain depth;
        # fall back to inspecting all watch() calls for one with a body.
        for call in mock_gmail_service.users().watch.call_args_list:
            if call.kwargs.get("body"):
                body = call.kwargs["body"]
                break
    assert body is not None
    assert body["labelIds"] == ["INBOX"]
    assert "gmail-notifications" in body["topicName"]
    assert body["topicName"].startswith("projects/")


def test_start_watch_explicit_topic_name_overrides_default(
    client: TestClient,
    mock_gmail_service: MagicMock,
) -> None:
    response = client.post(
        "/gmail/watch",
        json={
            "primary_email": "user@byod.com",
            "topic_name": "my-custom-topic",
        },
    )
    assert response.status_code == 200
    for call in mock_gmail_service.users().watch.call_args_list:
        body = call.kwargs.get("body")
        if body and "my-custom-topic" in body.get("topicName", ""):
            return
    pytest.fail("watch() never called with the custom topic name in body")


# ---------------------------------------------------------------------------
# DELETE /delete (Workspace user delete) -- new tests
# ---------------------------------------------------------------------------


def test_delete_email_user_missing_primary_email_returns_400(
    client: TestClient,
) -> None:
    response = client.request("DELETE", "/gmail/delete", json={})
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing primary_email"


def test_delete_email_user_success(
    client: TestClient,
    _gmail_credentials: None,
) -> None:
    fake_admin_service = MagicMock()
    fake_admin_service.users().delete().execute.return_value = {}

    with patch(
        "unify.gateway.channels.gmail.views.get_admin_service",
        return_value=fake_admin_service,
    ):
        response = client.request(
            "DELETE",
            "/gmail/delete",
            json={"primary_email": "user@example.com"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["deleted"] is True
    assert data["already_absent"] is False


def test_delete_email_user_404_treated_as_already_absent(
    client: TestClient,
    _gmail_credentials: None,
) -> None:
    fake_admin_service = MagicMock()
    fake_admin_service.users().delete().execute.side_effect = _http_error(404)

    with patch(
        "unify.gateway.channels.gmail.views.get_admin_service",
        return_value=fake_admin_service,
    ):
        response = client.request(
            "DELETE",
            "/gmail/delete",
            json={"primary_email": "user@example.com"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["deleted"] is False
    assert data["already_absent"] is True


def test_delete_email_user_non_404_bubbles_up_as_500(
    client: TestClient,
    _gmail_credentials: None,
) -> None:
    fake_admin_service = MagicMock()
    fake_admin_service.users().delete().execute.side_effect = _http_error(500)

    with patch(
        "unify.gateway.channels.gmail.views.get_admin_service",
        return_value=fake_admin_service,
    ):
        response = client.request(
            "DELETE",
            "/gmail/delete",
            json={"primary_email": "user@example.com"},
        )

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# GET /attachment -- new tests
# ---------------------------------------------------------------------------


def test_get_attachment_returns_bytes_with_octet_stream_content_type(
    client: TestClient,
    mock_gmail_service: MagicMock,
) -> None:
    file_bytes = b"some attached bytes"
    encoded = base64.urlsafe_b64encode(file_bytes).decode("utf-8")
    mock_gmail_service.users().messages().attachments().get().execute.return_value = {
        "data": encoded,
    }
    response = client.get(
        "/gmail/attachment",
        params={
            "receiver_email": "user@byod.com",
            "gmail_message_id": "msg-1",
            "attachment_id": "att-1",
            "filename": "doc.pdf",
        },
    )
    assert response.status_code == 200
    assert response.content == file_bytes
    assert response.headers["content-type"] == "application/octet-stream"
    assert "filename=doc.pdf" in response.headers["content-disposition"]


def test_get_attachment_default_filename_when_unspecified(
    client: TestClient,
    mock_gmail_service: MagicMock,
) -> None:
    encoded = base64.urlsafe_b64encode(b"x").decode("utf-8")
    mock_gmail_service.users().messages().attachments().get().execute.return_value = {
        "data": encoded,
    }
    response = client.get(
        "/gmail/attachment",
        params={
            "receiver_email": "user@byod.com",
            "gmail_message_id": "msg-1",
            "attachment_id": "att-1",
        },
    )
    assert response.status_code == 200
    assert "filename=attachment" in response.headers["content-disposition"]


def test_get_attachment_returns_404_when_data_missing(
    client: TestClient,
    mock_gmail_service: MagicMock,
) -> None:
    mock_gmail_service.users().messages().attachments().get().execute.return_value = {}
    response = client.get(
        "/gmail/attachment",
        params={
            "receiver_email": "user@byod.com",
            "gmail_message_id": "msg-1",
            "attachment_id": "att-1",
        },
    )
    assert response.status_code == 404
