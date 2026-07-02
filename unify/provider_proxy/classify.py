"""Classify a proxied Drive/Graph request into a policy decision shape.

The proxy fronts the *full* Microsoft Graph and Google Drive REST surface. Only
file/drive endpoints are subject to the allowlist; everything else (calendar,
mail, contacts, ...) is passed straight through. This module inspects the method
and path and returns a :class:`Classification` describing how the proxy should
enforce the allowlist. Anything that looks like a file/drive endpoint but is not
explicitly recognized is classified ``unknown_file`` and denied by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

KIND_NON_FILE = "non_file"
KIND_FILE_READ = "file_read"
KIND_FILE_WRITE = "file_write"
KIND_UNKNOWN = "unknown_file"
KIND_BATCH = "batch"

_MY_DRIVE = "my-drive"
_MS_DEFAULT_DRIVE = "me"
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


@dataclass
class Locator:
    """A unified handle for an item.

    Either id-addressed (``item_id`` set, ``path is None``) or, for Microsoft
    Graph, path-addressed relative to an anchor (``path`` set,
    ``anchor_item_id`` = ``"root"`` or a concrete item id). Path-addressed
    locators are resolved to concrete ids by the proxy before allow-checks.
    """

    drive_id: str
    item_id: str = ""
    path: str | None = None
    anchor_item_id: str | None = None

    @property
    def is_path(self) -> bool:
        return self.path is not None


@dataclass
class Classification:
    provider: str
    kind: str
    operation: str = ""
    target: Optional[Locator] = None
    parent: Optional[Locator] = None
    is_listing: bool = False
    is_content: bool = False
    is_search: bool = False
    root_listing: bool = False


# ── Microsoft Graph ─────────────────────────────────────────────────────────


def _strip_ms_version(segs: list[str]) -> list[str]:
    if segs and segs[0] in ("v1.0", "beta"):
        return segs[1:]
    return segs


def _ms_split_drive_base(rest: str) -> tuple[str, str] | None:
    """Return ``(drive_id, remainder)`` after the drive base, or None.

    ``remainder`` is the string after ``me/drive`` or ``drives/{id}`` and may
    contain Graph path-addressing colons. Only the personal drive and explicit
    ``drives/{id}`` are supported for path addressing.
    """
    if rest == "me/drive" or rest.startswith("me/drive/"):
        return (_MS_DEFAULT_DRIVE, rest[len("me/drive") :].lstrip("/"))
    if rest.startswith("drives/"):
        remainder = rest[len("drives/") :]
        drive_id, _, after = remainder.partition("/")
        if not drive_id:
            return None
        return (drive_id, after)
    return None


def _classify_ms_path(
    method: str,
    drive_id: str,
    after: str,
    query: dict[str, str],
) -> Classification | None:
    """Classify a Graph path-addressed request (``root:/path:`` forms).

    ``after`` is the remainder after the drive base, e.g. ``root:/A/B:/children``
    or ``items/{id}:/rel/file.txt:/content``. Returns None if the shape is not
    recognized so the caller can default-deny.
    """
    if after.startswith("root:"):
        anchor_id = "root"
        remainder = after[len("root:") :]
    elif after.startswith("items/"):
        body = after[len("items/") :]
        anchor_id, colon, remainder = body.partition(":")
        if not colon or not anchor_id:
            return None
    else:
        return None

    path_part, _, suffix = remainder.partition(":")
    path = path_part.strip("/")
    loc = Locator(drive_id=drive_id, path=path, anchor_item_id=anchor_id)
    write = method in _WRITE_METHODS

    if suffix == "":
        kind = KIND_FILE_WRITE if write else KIND_FILE_READ
        return Classification("microsoft", kind, "path_item", target=loc)
    if suffix == "/children":
        if write:
            return Classification(
                "microsoft",
                KIND_FILE_WRITE,
                "path_children",
                parent=loc,
            )
        return Classification(
            "microsoft",
            KIND_FILE_READ,
            "path_children",
            parent=loc,
            is_listing=True,
        )
    if suffix == "/content":
        kind = KIND_FILE_WRITE if write else KIND_FILE_READ
        return Classification(
            "microsoft",
            kind,
            "path_content",
            target=loc,
            is_content=not write,
        )
    if suffix == "/createUploadSession":
        return Classification("microsoft", KIND_FILE_WRITE, "path_upload", target=loc)
    if suffix == "/copy":
        return Classification("microsoft", KIND_FILE_WRITE, "path_copy", target=loc)
    if suffix.startswith("/search("):
        return Classification(
            "microsoft",
            KIND_FILE_READ,
            "path_search",
            parent=loc,
            is_listing=True,
            is_search=True,
        )
    return None


def _ms_is_file_domain(segs: list[str]) -> bool:
    if segs[:2] == ["me", "drive"]:
        return True
    if segs[:2] == ["me", "drives"]:
        return True
    if segs[:1] == ["drives"]:
        return True
    if segs[:1] == ["shares"]:
        return True
    if segs and segs[0] in ("sites", "groups", "users"):
        return "drive" in segs or "drives" in segs
    return False


def _ms_drive_and_rest(segs: list[str]) -> Optional[tuple[str, list[str]]]:
    """Return ``(drive_id, rest_segs)`` for a supported drive addressing, else None."""
    if segs[:2] == ["me", "drive"]:
        return (_MS_DEFAULT_DRIVE, segs[2:])
    if segs[:1] == ["drives"] and len(segs) >= 2:
        return (segs[1], segs[2:])
    if segs and segs[0] in ("sites", "groups", "users") and "drives" in segs:
        idx = segs.index("drives")
        if len(segs) >= idx + 2:
            return (segs[idx + 1], segs[idx + 2 :])
    return None


def _classify_microsoft(
    method: str,
    segs: list[str],
    query: dict[str, str],
) -> Classification:
    segs = _strip_ms_version(segs)
    if not segs:
        return Classification("microsoft", KIND_NON_FILE, "root")

    if segs == ["$batch"] and method == "POST":
        return Classification("microsoft", KIND_BATCH, "batch")

    if not _ms_is_file_domain(segs):
        return Classification("microsoft", KIND_NON_FILE, "/".join(segs))

    rest_str = "/".join(segs)

    # Graph path addressing (root:/A/B: or items/{id}:/rel:) is resolved to a
    # concrete item by the proxy; anything colon-shaped we cannot parse is
    # default-denied.
    if ":" in rest_str:
        parsed = _ms_split_drive_base(rest_str)
        if parsed is not None:
            drive_id, after = parsed
            path_class = _classify_ms_path(method, drive_id, after, query)
            if path_class is not None:
                return path_class
        return Classification("microsoft", KIND_UNKNOWN, rest_str)

    # me/drives => list of drives (roots); leave unfiltered.
    if segs[:2] == ["me", "drives"]:
        return Classification(
            "microsoft",
            KIND_FILE_READ,
            "list_drives",
            root_listing=True,
        )

    parsed = _ms_drive_and_rest(segs)
    if parsed is None:
        return Classification("microsoft", KIND_UNKNOWN, "/".join(segs))
    drive_id, rest = parsed
    write = method in _WRITE_METHODS

    # Bare drive object (me/drive or drives/{id}) => drive metadata, no item.
    if not rest:
        return Classification("microsoft", KIND_FILE_READ, "get_drive")

    head = rest[0]
    tail = rest[1:]

    if head == "root":
        if tail == ["children"]:
            kind = KIND_FILE_WRITE if write else KIND_FILE_READ
            return Classification(
                "microsoft",
                kind,
                "root_children",
                parent=Locator(drive_id, "root"),
                is_listing=not write,
            )
        if tail and tail[0].startswith("search("):
            return Classification(
                "microsoft",
                KIND_FILE_READ,
                "search",
                parent=Locator(drive_id, "root"),
                is_listing=True,
                is_search=True,
            )
        if not tail:
            return Classification(
                "microsoft",
                KIND_FILE_WRITE if write else KIND_FILE_READ,
                "root_item",
                target=Locator(drive_id, "root"),
            )
        return Classification("microsoft", KIND_UNKNOWN, "/".join(segs))

    if head == "items" and len(rest) >= 2:
        item_id = rest[1]
        sub = rest[2:]
        if sub == ["children"]:
            kind = KIND_FILE_WRITE if write else KIND_FILE_READ
            return Classification(
                "microsoft",
                kind,
                "children",
                parent=Locator(drive_id, item_id),
                is_listing=not write,
            )
        if sub == ["content"]:
            return Classification(
                "microsoft",
                KIND_FILE_WRITE if write else KIND_FILE_READ,
                "content",
                target=Locator(drive_id, item_id),
                is_content=not write,
            )
        if sub and sub[0].startswith("search("):
            return Classification(
                "microsoft",
                KIND_FILE_READ,
                "search",
                parent=Locator(drive_id, item_id),
                is_listing=True,
                is_search=True,
            )
        if sub in (["copy"], ["createUploadSession"]):
            return Classification(
                "microsoft",
                KIND_FILE_WRITE,
                sub[0],
                target=Locator(drive_id, item_id),
            )
        if not sub:
            return Classification(
                "microsoft",
                KIND_FILE_WRITE if write else KIND_FILE_READ,
                "item",
                target=Locator(drive_id, item_id),
            )
        return Classification("microsoft", KIND_UNKNOWN, "/".join(segs))

    # recent / sharedWithMe / delta => cross-folder listings; filter per item.
    if head in ("recent", "sharedWithMe") and not write:
        return Classification(
            "microsoft",
            KIND_FILE_READ,
            head,
            is_listing=True,
        )

    return Classification("microsoft", KIND_UNKNOWN, "/".join(segs))


# ── Google Drive ────────────────────────────────────────────────────────────


def _classify_google(
    method: str,
    segs: list[str],
    query: dict[str, str],
) -> Classification:
    # Non-Drive Google APIs (calendar, people, ...) pass straight through.
    if segs[:2] != ["drive", "v3"]:
        return Classification("google", KIND_NON_FILE, "/".join(segs))

    rest = segs[2:]
    if not rest:
        return Classification("google", KIND_NON_FILE, "drive_root")

    write = method in _WRITE_METHODS
    drive_id = query.get("driveId") or _MY_DRIVE

    if rest == ["drives"]:
        return Classification(
            "google",
            KIND_FILE_READ,
            "list_drives",
            root_listing=True,
        )
    if rest == ["about"]:
        return Classification("google", KIND_NON_FILE, "about")

    if rest == ["files"]:
        if write:
            # Create: parent(s) come from the JSON body; proxy checks them.
            return Classification("google", KIND_FILE_WRITE, "create")
        return Classification("google", KIND_FILE_READ, "list", is_listing=True)

    if rest[:1] == ["files"] and len(rest) >= 2:
        file_id = rest[1]
        sub = rest[2:]
        if sub == ["export"]:
            return Classification(
                "google",
                KIND_FILE_READ,
                "export",
                target=Locator(drive_id, file_id),
                is_content=True,
            )
        if sub == ["copy"]:
            return Classification(
                "google",
                KIND_FILE_WRITE,
                "copy",
                target=Locator(drive_id, file_id),
            )
        if not sub:
            if write:
                return Classification(
                    "google",
                    KIND_FILE_WRITE,
                    "update",
                    target=Locator(drive_id, file_id),
                )
            if (query.get("alt") or "").lower() == "media":
                return Classification(
                    "google",
                    KIND_FILE_READ,
                    "content",
                    target=Locator(drive_id, file_id),
                    is_content=True,
                )
            return Classification(
                "google",
                KIND_FILE_READ,
                "get",
                target=Locator(drive_id, file_id),
            )

    return Classification("google", KIND_UNKNOWN, "/".join(segs))


def classify(
    provider: str,
    method: str,
    rest_path: str,
    query: dict[str, str],
) -> Classification:
    """Classify a proxied request. ``rest_path`` excludes the ``/{provider}`` prefix."""
    segs = [s for s in rest_path.split("/") if s]
    method = method.upper()
    if provider == "microsoft":
        return _classify_microsoft(method, segs, query)
    if provider == "google":
        return _classify_google(method, segs, query)
    return Classification(provider, KIND_UNKNOWN, rest_path)
