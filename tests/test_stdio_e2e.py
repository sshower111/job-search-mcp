"""End-to-end: the agent launches the real server as a subprocess and talks MCP over stdio."""

import asyncio

from jobsearch.agent import TriageAgent
from jobsearch.vault import Vault

from .conftest import TODAY, snapshot


def test_triage_over_stdio_propose_then_apply(settings, vault_dir):
    agent = TriageAgent(settings, chat=None, today=TODAY)  # default factory = stdio subprocess
    before = snapshot(vault_dir)
    proposals = asyncio.run(agent.propose())
    assert snapshot(vault_dir) == before
    assert len(proposals) == 4

    results = asyncio.run(agent.apply(proposals, confirm=lambda p: not p.needs_manual_review))
    assert results and all(r.ok for r in results), results
    apps = {a.company: a.status for a in Vault(settings).list_applications()}
    assert apps["Maplebyte Labs"] == "To apply"
    assert apps["Lumen Quay"] == "Skip"
    assert "Quillstone" not in apps  # flagged for manual review, not written

    # Running again is idempotent: nothing new is created and Seen Jobs is not duplicated.
    seen_before = (vault_dir / "Job Search" / "_system" / "Seen Jobs.md").read_text(encoding="utf-8")
    proposals = asyncio.run(agent.propose())
    asyncio.run(agent.apply(proposals, confirm=lambda p: not p.needs_manual_review))
    assert (vault_dir / "Job Search" / "_system" / "Seen Jobs.md").read_text(encoding="utf-8") == seen_before
