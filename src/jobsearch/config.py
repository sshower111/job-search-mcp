"""Settings: loaded from a TOML file, then overridden by environment variables.

Lookup order for the config file:
  1. the path passed explicitly (``--config``)
  2. ``$JOBAGENT_CONFIG``
  3. ``./jobagent.toml`` if it exists

Environment overrides (handy for Claude Desktop / Claude Code registration):
  VAULT_PATH, JOBSEARCH_READ_ONLY, JOBAGENT_LLM_BASE_URL, JOBAGENT_LLM_MODEL,
  JOBAGENT_LLM_ENABLED. The API key is never stored in the config file; the file
  only names the environment variable that holds it (``api_key_env``).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

TRACKS = ("support", "integration", "ai")

DEFAULT_GTA_LOCATIONS = [
    "toronto", "gta", "greater toronto area", "mississauga", "markham", "richmond hill", "vaughan",
    "brampton", "oakville", "scarborough", "north york", "etobicoke", "pickering", "ajax", "whitby",
    "oshawa", "burlington", "milton", "newmarket", "aurora", "thornhill", "concord",
]


class Folders(BaseModel):
    postings: str = "Jobs to tailor"
    applications: str = "Job Search/Applications"
    summaries: str = "Job Search/Daily Summaries"
    seen_jobs: str = "Job Search/_system/Seen Jobs.md"


class MatchingRules(BaseModel):
    max_years: int = 3
    fit_threshold: int = 50
    locations: list[str] = Field(default_factory=lambda: list(DEFAULT_GTA_LOCATIONS))
    allow_remote_canada: bool = True
    skip_french_required: bool = True
    # Role types to skip regardless of skills (the search targets Support/Integration/AI Engineer roles).
    excluded_title_phrases: list[str] = Field(
        default_factory=lambda: ["help desk", "helpdesk", "it support", "desktop support", "service desk"]
    )
    senior_title_words: list[str] = Field(
        default_factory=lambda: [
            "senior", "sr", "lead", "principal", "staff", "manager", "director", "head of",
            "architect", "vp", "vice president",
        ]
    )


class LLMSettings(BaseModel):
    enabled: bool = False
    base_url: str = "http://localhost:11434/v1"  # Ollama default; LM Studio is http://localhost:1234/v1
    model: str = "qwen2.5:7b-instruct"
    api_key_env: str = "JOBAGENT_LLM_API_KEY"
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    max_tool_steps: int = 8

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None


class Settings(BaseModel):
    vault_path: Path = Path(r"C:\Users\Keith\vault")
    folders: Folders = Field(default_factory=Folders)
    resumes: dict[str, str] = Field(
        default_factory=lambda: {
            "support": "Resumes/Master Resume - Support Engineer.md",
            "integration": "Resumes/Master Resume - Integration Engineer.md",
            "ai": "Resumes/Master Resume - AI Engineer.md",
        }
    )
    matching: MatchingRules = Field(default_factory=MatchingRules)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    read_only: bool = False
    max_posting_chars: int = 20_000

    config_file: Path | None = None


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    path: Path | None = None
    if config_path:
        path = Path(config_path)
    elif os.environ.get("JOBAGENT_CONFIG"):
        path = Path(os.environ["JOBAGENT_CONFIG"])
    elif Path("jobagent.toml").exists():
        path = Path("jobagent.toml")

    data: dict = {}
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("rb") as fh:
            data = tomllib.load(fh)

    settings = Settings.model_validate(data)
    if path is not None:
        settings.config_file = path.resolve()
        # A relative vault_path in the config file is relative to the config file's folder.
        if not settings.vault_path.is_absolute() and "vault_path" in data:
            settings.vault_path = (path.resolve().parent / settings.vault_path).resolve()

    env = os.environ
    if env.get("VAULT_PATH"):
        settings.vault_path = Path(env["VAULT_PATH"])
    if env.get("JOBSEARCH_READ_ONLY"):
        settings.read_only = _truthy(env["JOBSEARCH_READ_ONLY"])
    if env.get("JOBAGENT_LLM_BASE_URL"):
        settings.llm.base_url = env["JOBAGENT_LLM_BASE_URL"]
    if env.get("JOBAGENT_LLM_MODEL"):
        settings.llm.model = env["JOBAGENT_LLM_MODEL"]
    if env.get("JOBAGENT_LLM_ENABLED"):
        settings.llm.enabled = _truthy(env["JOBAGENT_LLM_ENABLED"])
    return settings
