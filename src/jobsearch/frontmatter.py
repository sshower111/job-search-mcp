"""Minimal, lossless frontmatter handling for Obsidian notes.

Parsing uses PyYAML (safe_load). Editing is done line-by-line on the raw text so
that every key we do not touch, comments, key order, quoting style, line endings
and the note body are preserved byte-for-byte.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass

import yaml

_FENCE = "---"
_SAFE_PLAIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.,'()&+/-]*$")


@dataclass
class Note:
    frontmatter: dict
    body: str
    has_frontmatter: bool
    newline: str = "\n"


def _detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _split(text: str) -> tuple[list[str], int] | None:
    """Return (lines, closing_fence_index) when the text starts with a frontmatter block."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n").strip() != _FENCE:
        return None
    for i in range(1, len(lines)):
        if lines[i].rstrip("\r\n").strip() == _FENCE:
            return lines, i
    return None


def _normalise(value):
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


def parse(text: str) -> Note:
    newline = _detect_newline(text)
    split = _split(text)
    if split is None:
        return Note({}, text, False, newline)
    lines, end = split
    raw = "".join(lines[1:end])
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    body = "".join(lines[end + 1 :])
    return Note(_normalise(data), body, True, newline)


def yaml_scalar(value: str | None) -> str:
    """Render a value for a single `key: value` line. Empty -> bare key (Obsidian's empty property)."""
    if value is None or value == "":
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    if _SAFE_PLAIN.match(text) and text.lower() not in {"yes", "no", "true", "false", "null", "on", "off"}:
        return text
    return json.dumps(text, ensure_ascii=False)


def set_fields(text: str, updates: dict[str, str | None]) -> str:
    """Set top-level frontmatter keys in place. Keys not present are appended before the closing fence."""
    split = _split(text)
    if split is None:
        raise ValueError("note has no frontmatter block")
    lines, end = split
    newline = _detect_newline(text)
    remaining = dict(updates)
    out: list[str] = [lines[0]]
    i = 1
    while i < end:
        line = lines[i]
        m = re.match(r"^([A-Za-z0-9_][\w -]*?)\s*:(.*)$", line.rstrip("\r\n"))
        if m and not line.startswith((" ", "\t")) and m.group(1) in remaining:
            key = m.group(1)
            value = yaml_scalar(remaining.pop(key))
            ending = line[len(line.rstrip("\r\n")) :] or newline
            out.append(f"{key}: {value}".rstrip() + ending)
            i += 1
            # Drop continuation lines of the old value (block lists / multi-line scalars).
            while i < end and (lines[i].startswith((" ", "\t", "- ")) or lines[i].strip() == "-"):
                i += 1
            continue
        out.append(line)
        i += 1
    for key, value in remaining.items():
        out.append(f"{key}: {yaml_scalar(value)}".rstrip() + newline)
    out.extend(lines[end:])
    return "".join(out)


def build(fields: dict[str, str | None], body: str, newline: str = "\n") -> str:
    head = [_FENCE] + [f"{k}: {yaml_scalar(v)}".rstrip() for k, v in fields.items()] + [_FENCE]
    return newline.join(head) + newline + body


def append_note_line(text: str, line: str, heading: str = "## Notes") -> str:
    """Append ``line`` at the end of the ``## Notes`` section, creating it at the end if missing."""
    newline = _detect_newline(text)
    lines = text.splitlines(keepends=True)
    start = None
    for idx, raw in enumerate(lines):
        if raw.rstrip("\r\n").strip().lower() == heading.lower():
            start = idx
            break
    if start is None:
        prefix = text if text.endswith(("\n", "\r\n")) or not text else text + newline
        return f"{prefix}{newline}{heading}{newline}{line}{newline}"
    # Section ends at the next heading of the same or higher level.
    level = len(heading) - len(heading.lstrip("#"))
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        m = re.match(r"^(#+)\s", lines[idx])
        if m and len(m.group(1)) <= level:
            end = idx
            break
    insert_at = end
    while insert_at > start + 1 and lines[insert_at - 1].strip() == "":
        insert_at -= 1
    if insert_at > 0 and not lines[insert_at - 1].endswith(("\n", "\r")):
        lines[insert_at - 1] = lines[insert_at - 1] + newline
    lines.insert(insert_at, line + newline)
    return "".join(lines)
