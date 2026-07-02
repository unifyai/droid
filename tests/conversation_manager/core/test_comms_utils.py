"""
tests/conversation_manager/core/test_comms_utils.py
=============================================================

Unit tests for comms_utils functions, particularly attachment handling.

Marked ``real_comms_functions`` to opt out of the universal comms stub
in the parent conftest.  These tests call the real ``comms_utils``
implementations with mocked dependencies (aiohttp, SESSION_DETAILS, etc.).

Every test that mocks aiohttp asserts the **URL and HTTP method** used for
the request, not just the response.  This catches wrong-service bugs (like
posting to COMMS_URL when the endpoint lives on the adapters service).
"""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unify.conversation_manager.domains import comms_utils

pytestmark = pytest.mark.real_comms_functions

COMMS_URL = "http://comms.test:8080"
ADAPTERS_URL = "http://adapters.test:8081"


def _mock_aiohttp_session(response_json=None, raise_on_status=None, status=200):
    """Build a mock aiohttp session + response pair."""
    import json

    mock_response = MagicMock()
    mock_response.status = status
    if raise_on_status:
        mock_response.raise_for_status = MagicMock(side_effect=raise_on_status)
    else:
        mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(return_value=response_json or {})
    mock_response.text = AsyncMock(
        return_value=json.dumps(response_json or {}),
    )

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
    )
    mock_session.get = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
    )
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    return mock_session


class TestUploadUnifyAttachment:
    """Tests for upload_unify_attachment function."""

    @pytest.mark.asyncio
    async def test_upload_success(self):
        """Successfully uploads file and returns attachment details."""
        mock_session = _mock_aiohttp_session(
            response_json={
                "id": "test-uuid-123",
                "filename": "document.pdf",
                "url": "https://storage.googleapis.com/signed-url",
            },
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.agent_id = 42
            mock_settings.conversation.ADAPTERS_URL = ADAPTERS_URL

            result = await comms_utils.upload_unify_attachment(
                file_content=b"test file content",
                filename="document.pdf",
            )

            assert result["id"] == "test-uuid-123"
            assert result["filename"] == "document.pdf"
            assert "storage.googleapis.com" in result["url"]

            posted_url = mock_session.post.call_args[0][0]
            assert posted_url == f"{ADAPTERS_URL}/unify/attachment"

    @pytest.mark.asyncio
    async def test_upload_int_agent_id_serializes_form_data(self):
        """Integer agent_id must be string-coerced for multipart form serialization.

        aiohttp.FormData cannot serialize int values — calling form_data()
        raises LookupError("Can not serialize value type: <class 'int'>").
        This catches the regression from the str→int agent_id migration.
        """
        mock_session = _mock_aiohttp_session(
            response_json={"id": "uuid", "filename": "f.txt", "url": "https://url"},
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.agent_id = 42
            mock_settings.conversation.ADAPTERS_URL = ADAPTERS_URL

            await comms_utils.upload_unify_attachment(
                file_content=b"test content",
                filename="file.txt",
            )

            form_data = mock_session.post.call_args.kwargs["data"]
            form_data()

    @pytest.mark.asyncio
    async def test_upload_with_custom_assistant_id(self):
        """Uses provided assistant_id instead of session default."""
        mock_session = _mock_aiohttp_session(
            response_json={"id": "uuid", "filename": "file.txt", "url": "https://url"},
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_settings.conversation.ADAPTERS_URL = ADAPTERS_URL

            result = await comms_utils.upload_unify_attachment(
                file_content=b"content",
                filename="file.txt",
                assistant_id="custom-assistant-id",
            )

            assert result["id"] == "uuid"
            mock_session.post.assert_called_once()
            posted_url = mock_session.post.call_args[0][0]
            assert posted_url == f"{ADAPTERS_URL}/unify/attachment"

    @pytest.mark.asyncio
    async def test_upload_failure_returns_error(self):
        """Returns error dict when upload fails."""
        mock_session = _mock_aiohttp_session(
            response_json={"error": "Connection refused"},
            status=503,
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.agent_id = 42
            mock_settings.conversation.ADAPTERS_URL = ADAPTERS_URL

            result = await comms_utils.upload_unify_attachment(
                file_content=b"content",
                filename="file.txt",
            )

            assert result.get("success") is False
            assert "error" in result

            posted_url = mock_session.post.call_args[0][0]
            assert posted_url == f"{ADAPTERS_URL}/unify/attachment"

    @pytest.mark.asyncio
    async def test_upload_posts_to_adapters_not_comms_app(self):
        """The /unify/attachment endpoint is on the adapters service.

        COMMS_URL points to the comms-app (phone, gmail, infra routes).
        The attachment upload endpoint lives on the separate adapters service.
        Using COMMS_URL results in a 404 because the comms-app has no /unify/* routes.
        """
        mock_session = _mock_aiohttp_session(
            response_json={"id": "uuid", "filename": "file.txt", "url": "https://url"},
        )

        comms_app_url = "http://comms.test:8080"
        adapters_url = "http://adapters.test:8081"

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.agent_id = 42
            mock_settings.conversation.COMMS_URL = comms_app_url
            mock_settings.conversation.ADAPTERS_URL = adapters_url

            await comms_utils.upload_unify_attachment(
                file_content=b"content",
                filename="file.txt",
            )

            posted_url = mock_session.post.call_args[0][0]
            assert posted_url.startswith(adapters_url), (
                f"upload_unify_attachment must post to ADAPTERS_URL "
                f"({adapters_url}), not COMMS_URL ({comms_app_url}). "
                f"Got: {posted_url}"
            )
            assert posted_url == f"{adapters_url}/unify/attachment"


class TestSendSms:
    """Tests for send_sms_message_via_number."""

    @pytest.mark.asyncio
    async def test_send_sms_url_and_body(self):
        """Posts to /phone/send-text on COMMS_URL with correct JSON body."""
        mock_session = _mock_aiohttp_session(
            response_json={"success": True, "sid": "SM123"},
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.number = "+15551234567"
            mock_settings.conversation.COMMS_URL = COMMS_URL

            result = await comms_utils.send_sms_message_via_number(
                to_number="+15559876543",
                content="Hello from tests",
            )

            assert result["success"] is True

            posted_url = mock_session.post.call_args[0][0]
            assert posted_url == f"{COMMS_URL}/phone/send-text"

            payload = mock_session.post.call_args.kwargs["json"]
            assert payload["From"] == "+15551234567"
            assert payload["To"] == "+15559876543"
            assert payload["Body"] == "Hello from tests"

    @pytest.mark.asyncio
    async def test_send_sms_collapses_hard_wrapped_body(self):
        """Hard-wrapped SMS bodies are normalized before transport."""
        mock_session = _mock_aiohttp_session(
            response_json={"success": True, "sid": "SM123"},
        )
        wrapped = "Line one of the clue\nLine two of the clue"

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.number = "+15551234567"
            mock_settings.conversation.COMMS_URL = COMMS_URL

            await comms_utils.send_sms_message_via_number(
                to_number="+15559876543",
                content=wrapped,
            )

            payload = mock_session.post.call_args.kwargs["json"]
            assert payload["Body"] == "Line one of the clue Line two of the clue"

    @pytest.mark.asyncio
    async def test_send_sms_no_from_number(self):
        """Returns failure when assistant has no phone number."""
        with patch(
            "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
        ) as mock_session_details:
            mock_session_details.assistant.number = None

            result = await comms_utils.send_sms_message_via_number(
                to_number="+15559876543",
                content="Hello",
            )

            assert result["success"] is False


class TestStartCall:
    """Tests for start_call."""

    @pytest.mark.asyncio
    async def test_start_call_url_and_body(self):
        """Posts to /phone/send-call on COMMS_URL with correct JSON body."""
        mock_session = _mock_aiohttp_session(
            response_json={"success": True, "call_sid": "CA123"},
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.number = "+15551234567"
            mock_settings.conversation.COMMS_URL = COMMS_URL

            result = await comms_utils.start_call(to_number="+15559876543")

            assert result["success"] is True

            posted_url = mock_session.post.call_args[0][0]
            assert posted_url == f"{COMMS_URL}/phone/send-call"

            payload = mock_session.post.call_args.kwargs["json"]
            assert payload["From"] == "+15551234567"
            assert payload["To"] == "+15559876543"

    @pytest.mark.asyncio
    async def test_start_call_no_from_number(self):
        """Returns failure when assistant has no phone number."""
        with patch(
            "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
        ) as mock_session_details:
            mock_session_details.assistant.number = None

            result = await comms_utils.start_call(to_number="+15559876543")

            assert result["success"] is False


class TestSendUnifyMessage:
    """Tests for send_unify_message function with attachments."""

    @pytest.mark.asyncio
    async def test_send_without_attachment(self):
        """Sends message without attachment via Pub/Sub."""
        with (
            patch(
                "unify.conversation_manager.domains.comms_utils._get_publisher",
            ) as mock_get_publisher,
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session,
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_settings.DEPLOY_ENV = "production"
            mock_settings.ENV_SUFFIX = ""
            mock_session.assistant.agent_id = 42

            mock_publisher = MagicMock()
            mock_future = MagicMock()
            mock_future.result.return_value = "message-id-123"
            mock_publisher.publish.return_value = mock_future
            mock_publisher.topic_path.return_value = "projects/test/topics/unity-test"
            mock_get_publisher.return_value = mock_publisher

            result = await comms_utils.send_unify_message(
                content="Hello world",
                contact_id=1,
            )

            assert result["success"] is True
            mock_publisher.publish.assert_called_once()

            call_args = mock_publisher.publish.call_args
            assert call_args[0][0] == "projects/test/topics/unity-test"

    @pytest.mark.asyncio
    async def test_send_with_attachment(self):
        """Sends message with attachment included in Pub/Sub event."""
        with (
            patch(
                "unify.conversation_manager.domains.comms_utils._get_publisher",
            ) as mock_get_publisher,
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session,
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_settings.DEPLOY_ENV = "production"
            mock_settings.ENV_SUFFIX = ""
            mock_session.assistant.agent_id = 42

            mock_publisher = MagicMock()
            mock_future = MagicMock()
            mock_future.result.return_value = "message-id-123"
            mock_publisher.publish.return_value = mock_future
            mock_publisher.topic_path.return_value = "projects/test/topics/unity-test"
            mock_get_publisher.return_value = mock_publisher

            attachment = {
                "id": "att-uuid",
                "filename": "report.pdf",
                "url": "https://storage.googleapis.com/signed-url",
            }

            result = await comms_utils.send_unify_message(
                content="Here's the report",
                contact_id=1,
                attachment=attachment,
            )

            assert result["success"] is True

            call_args = mock_publisher.publish.call_args
            import json

            published_data = json.loads(call_args[0][1].decode("utf-8"))
            assert published_data["event"]["attachments"] == [attachment]


class TestSendEmailViaAddress:
    """Tests for send_email_via_address function with attachments."""

    @pytest.mark.asyncio
    async def test_send_with_attachment(self):
        """Sends email with attachment to /email/send on COMMS_URL."""
        mock_session = _mock_aiohttp_session(
            response_json={"success": True, "id": "email-123"},
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.email = "assistant@test.com"
            mock_settings.conversation.COMMS_URL = COMMS_URL

            attachment = {
                "filename": "report.pdf",
                "content_base64": "UERGIGNvbnRlbnQ=",
            }

            result = await comms_utils.send_email_via_address(
                to=["user@example.com"],
                subject="Report",
                body="Please see attached.",
                attachment=attachment,
            )

            assert result["success"] is True

            call_args = mock_session.post.call_args
            assert call_args[0][0] == f"{COMMS_URL}/email/send"
            assert call_args.kwargs["json"]["attachment"] == attachment

    @pytest.mark.asyncio
    async def test_send_with_cc_and_bcc(self):
        """Sends email with cc and bcc to /email/send on COMMS_URL."""
        mock_session = _mock_aiohttp_session(
            response_json={"success": True, "id": "email-123"},
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.email = "assistant@test.com"
            mock_settings.conversation.COMMS_URL = COMMS_URL

            result = await comms_utils.send_email_via_address(
                to=["user@example.com"],
                cc=["cc1@example.com", "cc2@example.com"],
                bcc=["bcc@example.com"],
                subject="Team Update",
                body="Here's the update.",
                thread_id="gmail-thread-123",
            )

            assert result["success"] is True

            call_args = mock_session.post.call_args
            assert call_args[0][0] == f"{COMMS_URL}/email/send"

            payload = call_args.kwargs["json"]
            assert payload["to"] == ["user@example.com"]
            assert payload["cc"] == ["cc1@example.com", "cc2@example.com"]
            assert payload["bcc"] == ["bcc@example.com"]
            assert payload["thread_id"] == "gmail-thread-123"

    @pytest.mark.asyncio
    async def test_send_uses_provider_agnostic_endpoint(self):
        """All providers route to the provider-agnostic /email/send endpoint."""
        mock_session = _mock_aiohttp_session(
            response_json={"success": True, "id": "email-456"},
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.email = "assistant@outlook.unify.ai"
            mock_settings.conversation.COMMS_URL = COMMS_URL

            result = await comms_utils.send_email_via_address(
                to=["user@example.com"],
                subject="Hello",
                body="Test email.",
            )

            assert result["success"] is True

            posted_url = mock_session.post.call_args[0][0]
            assert posted_url == f"{COMMS_URL}/email/send"

    @pytest.mark.asyncio
    async def test_send_email_collapses_hard_wrapped_body(self):
        """Hard-wrapped email bodies are normalized before transport."""
        mock_session = _mock_aiohttp_session(
            response_json={"success": True, "id": "email-789"},
        )
        wrapped = "First sentence of the quiz.\nSecond sentence here.\n\nClue line."

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.email = "assistant@outlook.unify.ai"
            mock_settings.conversation.COMMS_URL = COMMS_URL

            await comms_utils.send_email_via_address(
                to=["user@example.com"],
                subject="Quiz",
                body=wrapped,
            )

            payload = mock_session.post.call_args.kwargs["json"]
            assert (
                payload["body"]
                == "First sentence of the quiz. Second sentence here.\n\nClue line."
            )


class TestLocalCommsBackends:
    @pytest.mark.asyncio
    async def test_send_sms_uses_local_twilio_backend(self):
        with (
            patch(
                "unify.conversation_manager.domains.comms_utils._use_local_comms",
                return_value=True,
            ),
            patch(
                "unify.conversation_manager.local_providers.twilio.send_sms_message",
                new=AsyncMock(return_value={"success": True, "sid": "SM123"}),
            ) as mock_send_sms,
            patch(
                "unify.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
        ):
            mock_session_details.assistant.number = "+15551234567"

            result = await comms_utils.send_sms_message_via_number(
                to_number="+15559876543",
                content="Hello from tests",
            )

            assert result["success"] is True
            mock_send_sms.assert_awaited_once_with(
                "+15559876543",
                "+15551234567",
                "Hello from tests",
            )

    @pytest.mark.asyncio
    async def test_send_unify_message_mirrors_to_local_outbox_and_pubsub(self):
        """Local-comms mode mirrors the message into the local outbox but
        still publishes to Pub/Sub so the Console SSE bridge can deliver
        it.  Pub/Sub is the single source of truth for the Unify chat
        surface, even when ``LOCAL_COMMS_MODE=local``.
        """
        mock_publisher = MagicMock()
        mock_publisher.topic_path.return_value = "projects/p/topics/unity-1"
        mock_future = MagicMock()
        mock_future.result.return_value = "msg-id-1"
        mock_publisher.publish.return_value = mock_future

        with (
            patch(
                "unify.conversation_manager.domains.comms_utils._use_local_comms",
                return_value=True,
            ),
            patch(
                "unify.conversation_manager.domains.comms_utils._publish_local_outbox_async",
                new=AsyncMock(return_value=True),
            ) as mock_outbox,
            patch(
                "unify.conversation_manager.domains.comms_utils._get_publisher",
                return_value=mock_publisher,
            ),
        ):
            result = await comms_utils.send_unify_message(
                content="Hello world",
                contact_id=7,
            )

            assert result["success"] is True
            mock_outbox.assert_awaited_once()
            outbox_payload = mock_outbox.await_args.args[0]
            assert outbox_payload["thread"] == "unify_message_outbound"
            assert outbox_payload["event"]["content"] == "Hello world"
            assert outbox_payload["event"]["contact_id"] == 7
            mock_publisher.publish.assert_called_once()
            _, publish_kwargs = mock_publisher.publish.call_args
            assert publish_kwargs.get("thread") == "unify_message_outbound"

    @pytest.mark.asyncio
    async def test_send_unify_meet_ring_publishes_incoming_thread(self):
        """The Meet ring publishes a ``unify_meet_incoming`` frame carrying the
        call session id and reason so the Console can show the Answer window."""
        mock_publisher = MagicMock()
        mock_publisher.topic_path.return_value = "projects/p/topics/unity-1"
        mock_future = MagicMock()
        mock_future.result.return_value = "ring-msg-1"
        mock_publisher.publish.return_value = mock_future

        with patch(
            "unify.conversation_manager.domains.comms_utils._get_publisher",
            return_value=mock_publisher,
        ):
            result = await comms_utils.send_unify_meet_ring(
                call_session_id="meet-ring-abc",
                reason="Continuing onboarding on the live call.",
            )

        assert result["success"] is True
        mock_publisher.publish.assert_called_once()
        args, publish_kwargs = mock_publisher.publish.call_args
        assert publish_kwargs.get("thread") == "unify_meet_incoming"
        published = json.loads(args[1].decode("utf-8"))
        assert published["thread"] == "unify_meet_incoming"
        assert published["event"]["call_session_id"] == "meet-ring-abc"
        assert published["event"]["reason"] == "Continuing onboarding on the live call."

    @pytest.mark.asyncio
    async def test_upload_unify_attachment_uses_local_attachment_store(self):
        mock_session = _mock_aiohttp_session(response_json={"success": True})

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils._use_local_comms",
                return_value=True,
            ),
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_settings.conversation.LOCAL_COMMS_PUBLIC_URL = ""
            mock_settings.conversation.LOCAL_COMMS_HOST = "127.0.0.1"
            mock_settings.conversation.LOCAL_COMMS_PORT = 8787

            result = await comms_utils.upload_unify_attachment(
                file_content=b"local file content",
                filename="document.pdf",
            )

            assert result["filename"] == "document.pdf"
            assert result["url"].startswith(
                "http://127.0.0.1:8787/local/comms/attachments/",
            )
            posted_url = mock_session.post.call_args[0][0]
            assert posted_url == "http://127.0.0.1:8787/local/comms/attachments"

    @pytest.mark.asyncio
    async def test_send_email_uses_local_email_backend(self):
        with (
            patch(
                "unify.conversation_manager.domains.comms_utils._use_local_comms",
                return_value=True,
            ),
            patch(
                "unify.conversation_manager.local_providers.email.send_email",
                new=AsyncMock(return_value={"success": True, "id": "email-123"}),
            ) as mock_send_email,
        ):
            result = await comms_utils.send_email_via_address(
                to=["user@example.com"],
                subject="Report",
                body="Please see attached.",
                attachment={
                    "filename": "report.pdf",
                    "content_base64": "UERGIGNvbnRlbnQ=",
                },
            )

            assert result["success"] is True
            mock_send_email.assert_awaited_once()
            kwargs = mock_send_email.await_args.kwargs
            assert kwargs["to"] == ["user@example.com"]
            assert kwargs["subject"] == "Report"

    @pytest.mark.asyncio
    async def test_add_email_attachments_supports_inline_content(self):
        mock_session = _mock_aiohttp_session(response_json={"success": True})
        mock_file_manager = MagicMock()

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.manager_registry.ManagerRegistry.get_file_manager",
                return_value=mock_file_manager,
            ),
        ):
            await comms_utils.add_email_attachments(
                [
                    {
                        "id": "att-1",
                        "filename": "note.txt",
                        "content_base64": base64.b64encode(b"hello").decode("ascii"),
                    },
                ],
                receiver_email="assistant@example.com",
                message_id="",
            )

            mock_session.get.assert_not_called()
            mock_file_manager.save_attachment.assert_called_once_with(
                "att-1",
                "note.txt",
                b"hello",
                auto_ingest=False,
            )


class TestAddEmailAttachmentsRouting:
    """Tests for provider-agnostic routing in add_email_attachments."""

    @pytest.mark.asyncio
    async def test_fetches_via_provider_agnostic_endpoint(self):
        """Attachments are fetched from /email/attachment with unified params."""
        mock_response = MagicMock()
        mock_response.read = AsyncMock(return_value=b"file-bytes")

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
            ),
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_file_manager = MagicMock()

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unify.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
            patch(
                "unify.manager_registry.ManagerRegistry.get_file_manager",
                return_value=mock_file_manager,
            ),
        ):
            mock_settings.conversation.COMMS_URL = COMMS_URL

            await comms_utils.add_email_attachments(
                [{"id": "att-1", "filename": "doc.pdf"}],
                receiver_email="assistant@unify.ai",
                message_id="msg-123",
            )

            get_url = mock_session.get.call_args[0][0]
            assert get_url == f"{COMMS_URL}/email/attachment"

            params = mock_session.get.call_args.kwargs["params"]
            assert params["receiver_email"] == "assistant@unify.ai"
            assert params["message_id"] == "msg-123"
            assert params["attachment_id"] == "att-1"
