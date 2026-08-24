"""Helpers to keep media filenames and paths free of whitespace.

Screen-replay photo_path / photo_path_2 must be viable paths (no spaces,
including unicode whitespace like U+202F from macOS screenshot names) so
they can be stored in JSON and used as S3 key components without quoting.
"""

from __future__ import annotations

import os
import re
import unicodedata


def sanitize_media_filename(name: str) -> str:
    """Return a filename with no whitespace or path separators.

    Spaces, unicode whitespace, and characters that break URLs/S3 keys are
    replaced with underscores. The extension is preserved (lowercased).
    """
    if not name:
        return "media"
    normalized = unicodedata.normalize("NFKC", os.path.basename(name))
    stem, ext = os.path.splitext(normalized)

    def _clean(part: str) -> str:
        chars = []
        for ch in part:
            if ch.isspace() or ch in '/\\:*?"<>|':
                chars.append("_")
            else:
                chars.append(ch)
        return re.sub(r"_+", "_", "".join(chars)).strip("._")

    clean_stem = _clean(stem) or "media"
    clean_ext = _clean(ext.lstrip(".")).lower()
    return f"{clean_stem}.{clean_ext}" if clean_ext else clean_stem


def path_has_whitespace(path: str) -> bool:
    """True if any character in the path is whitespace (unicode included)."""
    return any(ch.isspace() for ch in (path or ""))


def ensure_viable_media_path(path: str) -> str:
    """Rename a file in place if its filename contains whitespace.

    Returns the (possibly new) path. If the source does not exist, returns
    the sanitized destination so the caller can fail on a missing file.
    """
    if not path:
        return path
    directory, filename = os.path.split(path)
    safe_name = sanitize_media_filename(filename)
    dest = os.path.join(directory, safe_name) if directory else safe_name
    if dest == path:
        return path
    if os.path.isfile(path):
        if not os.path.exists(dest):
            os.rename(path, dest)
        return dest
    return dest
