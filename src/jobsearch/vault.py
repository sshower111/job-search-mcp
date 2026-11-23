"""Filesystem access to the Obsidian vault.

Every path is resolved and checked to be inside the vault (and inside the
expected sub-folder) before it is read or written. Writes are atomic (temp file
in the same folder + ``os.replace``) and never delete anything.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import tempfile
from pathlib import Path

from . import frontmatter
from .config import Settings
from .models import (
    PROGRESSION,
    STATUSES,
    TERMINAL,
    TRIAGE_STATUSES,
    Application,
    CreateResult,
    FollowUp,
    Posting,
    PostingSummary,
    SearchHit,
    SeenJobResult,
    UpdateResult,
)
from .safety import detect_injection, one_line

log = logging.getLogger("jobsearch.vault")


class VaultError(Exception):
    """An anticipated, user-facing error (bad path, bad transition, missing file...)."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().casefold()


def seen_key(company: str, role: str) -> tuple[str, str]:
    return _norm(company), _norm(role)


def is_allowed_transition(current: str, new: str) -> bool:
    """Statuses only move forward: To apply -> Applied -> Interview -> Offer.

    Any non-terminal status may move to a terminal one (Rejected / Withdrawn / Skip).
    Terminal statuses are final. Re-setting the same status is a no-op and allowed.
    An empty/unknown current status is treated as "To apply".
    """
    if new not in STATUSES:
        return False
    current = current if current in STATUSES else "To apply"
    if new == current:
        return True
    if current in TERMINAL:
        return False
    if new in TERMINAL:
        return True
    return PROGRESSION.index(new) > PROGRESSION.index(current)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".jobsearch-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return fh.read()


def _safe_filename(text: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", text)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:120] or "Untitled"


_TITLE_PATTERNS = [
    re.compile(r"^(?P<company>.+?) hiring (?P<role>.+?)(?: in (?P<location>.+?))?(?: \| LinkedIn)?$", re.I),
    re.compile(r"^(?P<role>.+?) at (?P<company>.+?)(?: \|.*)?$", re.I),
    re.compile(r"^(?P<role>.+?) - (?P<company>.+?)(?: - (?P<location>.+?))?(?: \| Indeed.*)?$", re.I),
]


def posting_identity(meta: dict, title: str, body: str) -> tuple[str, str, str]:
    """Best-effort (company, role, location) from clipper frontmatter / title / body."""
    company = str(meta.get("company") or "").strip()
    role = str(meta.get("role") or "").strip()
    location = str(meta.get("location") or "").strip()
    if not (company and role):
        for pattern in _TITLE_PATTERNS:
            m = pattern.match(title.strip())
            if m:
                company = company or m.group("company").strip()
                role = role or m.group("role").strip()
                if not location and m.groupdict().get("location"):
                    location = (m.group("location") or "").strip()
                break
    if not company:
        m = re.search(r"^\s*\**company\**\s*:\s*\**\s*(.+)$", body, re.I | re.M)
        company = m.group(1).strip(" *") if m else "Unknown"
    if not location:
        m = re.search(r"^\s*\**location\**\s*:\s*\**\s*(.+)$", body, re.I | re.M)
        location = m.group(1).strip(" *") if m else ""
    return company, role or title or "Unknown role", location


class Vault:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = Path(settings.vault_path).expanduser().resolve()

    # ------------------------------------------------------------ path safety
    def resolve(self, relative: str | os.PathLike[str], within: str | None = None) -> Path:
        """Resolve a vault-relative path, refusing anything that escapes the vault (or ``within``)."""
        rel = str(relative).strip()
        if not rel or "\x00" in rel:
            raise VaultError("Empty or invalid path")
        candidate = Path(rel)
        if candidate.is_absolute() or candidate.drive or rel.startswith(("/", "\\")):
            raise VaultError("Absolute paths are not allowed; use a path relative to the vault")
        base = self.root if within is None else (self.root / within).resolve()
        if not base.is_relative_to(self.root):
            raise VaultError("Configured folder is outside the vault")
        resolved = (base / candidate).resolve()
        if not resolved.is_relative_to(base):
            raise VaultError("Path escapes the allowed folder")
        return resolved

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    @property
    def postings_dir(self) -> Path:
        return self.resolve(self.settings.folders.postings)

    @property
    def applications_dir(self) -> Path:
        return self.resolve(self.settings.folders.applications)

    @property
    def seen_jobs_path(self) -> Path:
        return self.resolve(self.settings.folders.seen_jobs)

    # ------------------------------------------------------------ postings
    def _posting_path(self, file: str) -> Path:
        path = self.resolve(file, within=self.settings.folders.postings)
        if path.suffix.lower() != ".md":
            raise VaultError("Postings must be .md files")
        if not path.is_file():
            raise VaultError(f"Posting not found: {file}")
        return path

    def _summary(self, path: Path, note: frontmatter.Note) -> PostingSummary:
        title = str(note.frontmatter.get("title") or path.stem)
        company, role, location = posting_identity(note.frontmatter, title, note.body)
        link = str(note.frontmatter.get("source") or note.frontmatter.get("url") or
                   note.frontmatter.get("link") or "")
        rel = path.relative_to(self.postings_dir).as_posix()
        return PostingSummary(file=rel, title=title, company=company, role=role, link=link, location=location)

    def list_postings(self) -> list[PostingSummary]:
        folder = self.postings_dir
        if not folder.is_dir():
            return []
        out = []
        for path in sorted(folder.rglob("*.md")):
            if not path.resolve().is_relative_to(folder):
                continue  # symlink pointing outside the folder
            out.append(self._summary(path, frontmatter.parse(read_text(path))))
        return out

    def read_posting(self, file: str) -> Posting:
        path = self._posting_path(file)
        note = frontmatter.parse(read_text(path))
        summary = self._summary(path, note)
        body = note.body
        limit = self.settings.max_posting_chars
        truncated = len(body) > limit
        flags = detect_injection(body + " " + summary.title)
        if flags:
            log.warning("Posting %s matched injection heuristics: %s", summary.file, ",".join(flags))
        return Posting(**summary.model_dump(), body=body[:limit], truncated=truncated, injection_flags=flags)

    def search_postings(self, query: str, limit: int = 10) -> list[SearchHit]:
        q = query.strip()
        if not q:
            raise VaultError("Empty query")
        terms = [t.casefold() for t in q.split()][:10]
        hits: list[SearchHit] = []
        for summary in self.list_postings():
            path = self._posting_path(summary.file)
            text = read_text(path)
            low = text.casefold()
            if all(t in low for t in terms):
                idx = low.find(terms[0])
                snippet = one_line(text[max(0, idx - 80): idx + 160], 240)
                hits.append(SearchHit(file=summary.file, title=summary.title, snippet=snippet))
            if len(hits) >= limit:
                break
        return hits

    # ------------------------------------------------------------ applications
    def _application_path(self, file: str, must_exist: bool = True) -> Path:
        path = self.resolve(file, within=self.settings.folders.applications)
        if path.suffix.lower() != ".md":
            raise VaultError("Application notes must be .md files")
        if must_exist and not path.is_file():
            raise VaultError(f"Application note not found: {file}")
        return path

    def list_applications(self, status: str | None = None, track: str | None = None) -> list[Application]:
        folder = self.applications_dir
        if not folder.is_dir():
            return []
        out = []
        for path in sorted(folder.rglob("*.md")):
            if not path.resolve().is_relative_to(folder):
                continue
            fm = frontmatter.parse(read_text(path)).frontmatter
            fields = {k: ("" if fm.get(k) is None else str(fm.get(k))) for k in Application.model_fields
                      if k != "file"}
            app = Application(file=path.relative_to(folder).as_posix(), **fields)
            if status and _norm(app.status) != _norm(status):
                continue
            if track and _norm(app.track) != _norm(track):
                continue
            out.append(app)
        return out

    def find_application(self, company: str, role: str) -> Application | None:
        key = seen_key(company, role)
        for app in self.list_applications():
            if seen_key(app.company, app.role) == key:
                return app
        return None

    def update_application(
        self,
        file: str,
        status: str | None = None,
        applied: str | None = None,
        follow_up: str | None = None,
        note: str | None = None,
        today: dt.date | None = None,
    ) -> UpdateResult:
        path = self._application_path(file)
        text = read_text(path)
        parsed = frontmatter.parse(text)
        if not parsed.has_frontmatter:
            raise VaultError("Application note has no frontmatter; refusing to edit")
        current = str(parsed.frontmatter.get("status") or "")
        updates: dict[str, str] = {}
        if status is not None:
            if status not in STATUSES:
                raise VaultError(f"Unknown status {status!r}; expected one of {', '.join(STATUSES)}")
            if not is_allowed_transition(current, status):
                raise VaultError(f"Refusing to move status backwards or out of a final state: "
                                 f"{current or '(empty)'} -> {status}")
            if status != current:
                updates["status"] = status
        for key, value in (("applied", applied), ("follow_up", follow_up)):
            if value is not None:
                try:
                    updates[key] = dt.date.fromisoformat(value.strip()).isoformat()
                except ValueError as exc:
                    raise VaultError(f"{key} must be a YYYY-MM-DD date") from exc
        if not updates and not note:
            return UpdateResult(file=file, previous_status=current, changed={}, message="Nothing to change")
        new_text = frontmatter.set_fields(text, updates) if updates else text
        changed = dict(updates)
        if note:
            day = (today or dt.date.today()).isoformat()
            line = f"- {day}: {one_line(note)}"
            new_text = frontmatter.append_note_line(new_text, line)
            changed["note"] = line
        atomic_write(path, new_text)
        log.info("Updated application %s fields=%s", file, ",".join(sorted(changed)))
        return UpdateResult(file=file, previous_status=current, changed=changed, message="Updated")

    def create_application(
        self,
        company: str,
        role: str,
        track: str,
        status: str = "To apply",
        fit: str = "",
        location: str = "",
        link: str = "",
        note: str | None = None,
        today: dt.date | None = None,
    ) -> CreateResult:
        if status not in TRIAGE_STATUSES:
            raise VaultError(f"New application notes may only start as {' or '.join(TRIAGE_STATUSES)}")
        company, role = one_line(company, 120), one_line(role, 160)
        if not company or not role:
            raise VaultError("company and role are required")
        existing = self.find_application(company, role)
        if existing:
            return CreateResult(file=existing.file, created=False, message="Application already exists")
        rel = f"{_safe_filename(company)} - {_safe_filename(role)}.md"
        path = self._application_path(rel, must_exist=False)
        if path.exists():
            return CreateResult(file=rel, created=False, message="A note with this name already exists")
        day = (today or dt.date.today()).isoformat()
        fields = {
            "company": company, "role": role, "track": one_line(track, 40), "status": status,
            "fit": one_line(fit, 40), "applied": "", "follow_up": "", "location": one_line(location, 120),
            "link": one_line(link, 500), "resume": "",
        }
        body = "\n## Notes\n" + f"- {day}: {one_line(note or 'Created by jobagent triage')}\n"
        atomic_write(path, frontmatter.build(fields, body))
        log.info("Created application note %s", rel)
        return CreateResult(file=rel, created=True, message="Created")

    def follow_ups_due(self, on: dt.date) -> list[FollowUp]:
        due = []
        for app in self.list_applications():
            if app.status not in ("Applied", "Interview") or not app.follow_up:
                continue
            try:
                when = dt.date.fromisoformat(app.follow_up[:10])
            except ValueError:
                continue
            if when <= on:
                due.append(FollowUp(file=app.file, company=app.company, role=app.role, status=app.status,
                                    follow_up=when.isoformat(), days_overdue=(on - when).days))
        return sorted(due, key=lambda f: (-f.days_overdue, f.company.casefold()))

    # ------------------------------------------------------------ seen jobs
    def seen_jobs(self) -> list[tuple[str, str, str, str]]:
        path = self.seen_jobs_path
        if not path.is_file():
            return []
        rows = []
        for line in read_text(path).splitlines():
            m = re.match(r"^\s*-\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*$", line)
            if m:
                rows.append((m.group(1), m.group(2), m.group(3), m.group(4)))
        return rows

    def add_seen_job(self, company: str, role: str, link: str = "", today: dt.date | None = None) -> SeenJobResult:
        def clean(value: str, n: int) -> str:
            return one_line(value, n).replace("|", "/")

        company, role, link = clean(company, 120), clean(role, 160), clean(link, 500)
        if not company or not role:
            raise VaultError("company and role are required")
        key = seen_key(company, role)
        for _, c, r, _ in self.seen_jobs():
            if seen_key(c, r) == key:
                return SeenJobResult(added=False, line="", message="Already in Seen Jobs")
        line = f"- {(today or dt.date.today()).isoformat()} | {company} | {role} | {link}".rstrip()
        path = self.seen_jobs_path
        existing = read_text(path) if path.is_file() else ""
        newline = "\r\n" if "\r\n" in existing else "\n"
        if existing and not existing.endswith(("\n", "\r")):
            existing += newline
        atomic_write(path, existing + line + newline)  # append-only: old content is kept verbatim
        return SeenJobResult(added=True, line=line, message="Added")

    # ------------------------------------------------------------ resumes
    def resume_text(self, track: str) -> str:
        rel = self.settings.resumes.get(track)
        if not rel:
            raise VaultError(f"No master resume configured for track {track!r}")
        path = self.resolve(rel)
        if not path.is_file():
            raise VaultError(f"Master resume for {track!r} not found at {rel}")
        return frontmatter.parse(read_text(path)).body
