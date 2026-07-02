from __future__ import annotations

import pytest

from unify.provider_proxy.classify import (
    KIND_BATCH,
    KIND_FILE_READ,
    KIND_FILE_WRITE,
    KIND_NON_FILE,
    KIND_UNKNOWN,
    classify,
)

# ── Microsoft Graph ──────────────────────────────────────────────────────────


def test_ms_root_children_is_listing_with_parent():
    c = classify("microsoft", "GET", "v1.0/me/drive/root/children", {})
    assert c.kind == KIND_FILE_READ
    assert c.is_listing is True
    assert c.parent is not None and c.parent.item_id == "root"


def test_ms_item_children_uses_explicit_drive_and_item():
    c = classify("microsoft", "GET", "v1.0/drives/D1/items/I1/children", {})
    assert c.kind == KIND_FILE_READ
    assert c.is_listing is True
    assert (c.parent.drive_id, c.parent.item_id) == ("D1", "I1")


def test_ms_get_item_targets_item():
    c = classify("microsoft", "GET", "v1.0/drives/D1/items/I1", {})
    assert c.kind == KIND_FILE_READ
    assert c.is_listing is False
    assert (c.target.drive_id, c.target.item_id) == ("D1", "I1")


def test_ms_content_is_content_read():
    c = classify("microsoft", "GET", "v1.0/drives/D1/items/I1/content", {})
    assert c.kind == KIND_FILE_READ
    assert c.is_content is True
    assert c.target.item_id == "I1"


def test_ms_search_is_listing():
    c = classify("microsoft", "GET", "v1.0/me/drive/root/search(q='x')", {})
    assert c.kind == KIND_FILE_READ
    assert c.is_listing is True
    assert c.is_search is True


@pytest.mark.parametrize("method", ["PATCH", "DELETE"])
def test_ms_item_write_targets_item(method):
    c = classify("microsoft", method, "v1.0/drives/D1/items/I1", {})
    assert c.kind == KIND_FILE_WRITE
    assert (c.target.drive_id, c.target.item_id) == ("D1", "I1")


def test_ms_create_child_is_write_with_parent():
    c = classify("microsoft", "POST", "v1.0/drives/D1/items/I1/children", {})
    assert c.kind == KIND_FILE_WRITE
    assert (c.parent.drive_id, c.parent.item_id) == ("D1", "I1")


def test_ms_batch():
    assert classify("microsoft", "POST", "v1.0/$batch", {}).kind == KIND_BATCH


def test_ms_non_file_passthrough():
    assert classify("microsoft", "GET", "v1.0/me/events", {}).kind == KIND_NON_FILE
    assert classify("microsoft", "GET", "v1.0/me/messages", {}).kind == KIND_NON_FILE


def test_ms_path_get_item_is_read_with_path_target():
    c = classify("microsoft", "GET", "v1.0/me/drive/root:/Finance/report.xlsx", {})
    assert c.kind == KIND_FILE_READ
    assert c.target is not None and c.target.is_path
    assert c.target.path == "Finance/report.xlsx"
    assert c.target.anchor_item_id == "root"


def test_ms_path_children_listing():
    c = classify("microsoft", "GET", "v1.0/me/drive/root:/Finance:/children", {})
    assert c.kind == KIND_FILE_READ
    assert c.is_listing is True
    assert c.parent is not None and c.parent.path == "Finance"


def test_ms_path_content_download():
    c = classify("microsoft", "GET", "v1.0/drives/D1/root:/a/b.txt:/content", {})
    assert c.kind == KIND_FILE_READ
    assert c.is_content is True
    assert (c.target.drive_id, c.target.path) == ("D1", "a/b.txt")


def test_ms_path_content_upload_is_write():
    c = classify("microsoft", "PUT", "v1.0/me/drive/root:/Finance/new.txt:/content", {})
    assert c.kind == KIND_FILE_WRITE
    assert c.target is not None and c.target.path == "Finance/new.txt"


def test_ms_path_upload_session_is_write():
    c = classify(
        "microsoft",
        "POST",
        "v1.0/me/drive/root:/Finance/big.zip:/createUploadSession",
        {},
    )
    assert c.kind == KIND_FILE_WRITE
    assert c.target is not None and c.target.path == "Finance/big.zip"


def test_ms_path_relative_to_item():
    c = classify("microsoft", "GET", "v1.0/drives/D1/items/I1:/sub/file.txt", {})
    assert c.kind == KIND_FILE_READ
    assert c.target.anchor_item_id == "I1"
    assert c.target.path == "sub/file.txt"


def test_ms_unparseable_colon_shape_denied():
    # A colon in a non-drive-base position we cannot parse stays default-deny.
    c = classify("microsoft", "GET", "v1.0/sites/S1/drive/root:/x:", {})
    assert c.kind == KIND_UNKNOWN


def test_ms_list_drives_is_root_listing_unfiltered():
    c = classify("microsoft", "GET", "v1.0/me/drives", {})
    assert c.kind == KIND_FILE_READ
    assert c.root_listing is True
    assert c.is_listing is False


def test_ms_recent_is_per_item_listing():
    c = classify("microsoft", "GET", "v1.0/me/drive/recent", {})
    assert c.kind == KIND_FILE_READ
    assert c.is_listing is True
    assert c.parent is None


# ── Google Drive ─────────────────────────────────────────────────────────────


def test_google_files_list_is_listing():
    c = classify("google", "GET", "drive/v3/files", {})
    assert c.kind == KIND_FILE_READ
    assert c.is_listing is True
    assert c.parent is None


def test_google_get_file_targets_item_with_drive_from_query():
    c = classify("google", "GET", "drive/v3/files/F1", {"driveId": "SD1"})
    assert c.kind == KIND_FILE_READ
    assert (c.target.drive_id, c.target.item_id) == ("SD1", "F1")


def test_google_media_download_is_content():
    c = classify("google", "GET", "drive/v3/files/F1", {"alt": "media"})
    assert c.kind == KIND_FILE_READ
    assert c.is_content is True


def test_google_create_is_write():
    assert classify("google", "POST", "drive/v3/files", {}).kind == KIND_FILE_WRITE


def test_google_update_is_write_targeting_item():
    c = classify("google", "PATCH", "drive/v3/files/F1", {})
    assert c.kind == KIND_FILE_WRITE
    assert c.target.item_id == "F1"


def test_google_non_drive_passthrough():
    c = classify("google", "GET", "calendar/v3/calendars/primary/events", {})
    assert c.kind == KIND_NON_FILE


def test_google_unknown_drive_endpoint_denied():
    c = classify("google", "GET", "drive/v3/changes", {})
    assert c.kind == KIND_UNKNOWN
