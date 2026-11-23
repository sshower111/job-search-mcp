"""Shared fixtures: a throwaway copy of the synthetic example vault plus a config file for it."""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import shutil
from pathlib import Path

import pytest
from mcp import Client

from jobsearch.config import Settings, load_settings
from jobsearch.server import build_server

REPO = Path(__file__).resolve().parents[1]
EXAMPLE_VAULT = REPO / "examples" / "vault"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TODAY = dt.date(2026, 1, 10)

CONFIG = """
vault_path = "vault"
[folders]
postings = "Jobs to tailor"
applications = "Job Search/Applications"
summaries = "Job Search/Daily Summaries"
seen_jobs = "Job Search/_system/Seen Jobs.md"
[resumes]
support = "Resumes/Master Resume - Support Engineer.md"
integration = "Resumes/Master Resume - Integration Engineer.md"
ai = "Resumes/Master Resume - AI Engineer.md"
[llm]
enabled = false
"""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("VAULT_PATH", "JOBAGENT_CONFIG", "JOBSEARCH_READ_ONLY", "JOBAGENT_LLM_BASE_URL",
                "JOBAGENT_LLM_MODEL", "JOBAGENT_LLM_ENABLED", "JOBAGENT_LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    shutil.copytree(EXAMPLE_VAULT, tmp_path / "vault")
    (tmp_path / "jobagent.toml").write_text(CONFIG, encoding="utf-8")
    return tmp_path / "vault"


@pytest.fixture
def settings(vault_dir: Path) -> Settings:
    return load_settings(vault_dir.parent / "jobagent.toml")


@pytest.fixture
def add_injection_postings(vault_dir: Path):
    """Copy the prompt-injection fixtures into the vault's postings folder."""
    def _add() -> list[str]:
        names = []
        for src in sorted((FIXTURES / "injection").glob("*.md")):
            shutil.copy(src, vault_dir / "Jobs to tailor" / src.name)
            names.append(src.name)
        return names
    return _add


def snapshot(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def inproc_factory(settings: Settings, chat_factory=None):
    """Client factory for the agent that runs the server in-process (fast tests)."""

    @contextlib.asynccontextmanager
    async def factory(read_only: bool):
        server = build_server(settings.model_copy(update={"read_only": read_only}), chat_factory=chat_factory)
        async with Client(server) as client:
            yield client

    return factory
