import datetime as dt
import os
import sys

import pytest

from jobsearch import frontmatter
from jobsearch.vault import Vault, VaultError, is_allowed_transition

APP = "Fernhollow Analytics - Support Engineer.md"  # status Applied in the example vault
D = dt.date(2026, 1, 10)


@pytest.fixture
def vault(settings):
    return Vault(settings)


def app_path(vault_dir, name=APP):
    return vault_dir / "Job Search" / "Applications" / name


# ------------------------------------------------------------------ path safety
@pytest.mark.parametrize("bad", [
    "../Resumes/Master Resume - AI Engineer.md",
    "../../outside.md",
    "sub/../../Job Search/Applications/x.md",
    "/etc/passwd",
    "C:\\Windows\\win.ini",
    "\\\\server\\share\\x.md",
    "",
])
def test_read_posting_rejects_traversal(vault, bad):
    with pytest.raises(VaultError):
        vault.read_posting(bad)


def test_update_rejects_traversal(vault):
    with pytest.raises(VaultError):
        vault.update_application("../../Resumes/Master Resume - AI Engineer.md", status="Applied")


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin rights on Windows")
def test_symlink_escape_is_blocked(vault, vault_dir, tmp_path):
    secret = tmp_path / "secret.md"
    secret.write_text("---\ntitle: secret\n---\nsecret", encoding="utf-8")
    os.symlink(secret, vault_dir / "Jobs to tailor" / "link.md")
    with pytest.raises(VaultError):
        vault.read_posting("link.md")
    assert "link.md" not in [p.file for p in vault.list_postings()]


def test_config_folder_outside_vault_is_rejected(settings):
    settings.folders.postings = "../elsewhere"
    with pytest.raises(VaultError):
        Vault(settings).list_postings()


# ------------------------------------------------------------------ status rules
@pytest.mark.parametrize("current,new,ok", [
    ("To apply", "Applied", True), ("Applied", "Interview", True), ("Interview", "Offer", True),
    ("To apply", "Offer", True), ("Applied", "To apply", False), ("Offer", "Interview", False),
    ("Interview", "Rejected", True), ("Applied", "Withdrawn", True), ("To apply", "Skip", True),
    ("Rejected", "Applied", False), ("Skip", "To apply", False), ("Withdrawn", "Rejected", False),
    ("Applied", "Applied", True), ("", "Applied", True), ("Applied", "Hired", False),
])
def test_transitions(current, new, ok):
    assert is_allowed_transition(current, new) is ok


def test_update_moves_forward_and_preserves_everything_else(vault, vault_dir):
    path = app_path(vault_dir)
    before = path.read_text(encoding="utf-8")
    res = vault.update_application(APP, status="Interview", follow_up="2026-01-20", note="Recruiter call\nbooked",
                                   today=D)
    after = path.read_text(encoding="utf-8")
    assert res.previous_status == "Applied"
    assert res.changed["status"] == "Interview"
    fm_before, fm_after = frontmatter.parse(before).frontmatter, frontmatter.parse(after).frontmatter
    assert fm_after["status"] == "Interview" and fm_after["follow_up"] == "2026-01-20"
    for key in set(fm_before) - {"status", "follow_up"}:
        assert fm_after[key] == fm_before[key]
    assert frontmatter.parse(before).body in frontmatter.parse(after).body
    assert "- 2026-01-10: Recruiter call booked" in after  # newline collapsed: no line injection


def test_update_refuses_backwards(vault, vault_dir):
    path = app_path(vault_dir)
    before = path.read_bytes()
    with pytest.raises(VaultError, match="backwards"):
        vault.update_application(APP, status="To apply")
    assert path.read_bytes() == before


def test_update_rejects_unknown_status_and_bad_dates(vault):
    with pytest.raises(VaultError):
        vault.update_application(APP, status="Hired")
    with pytest.raises(VaultError):
        vault.update_application(APP, applied="next tuesday")


def test_update_refuses_note_without_frontmatter(vault, vault_dir):
    app_path(vault_dir, "Plain.md").write_text("# no properties\n", encoding="utf-8")
    with pytest.raises(VaultError, match="frontmatter"):
        vault.update_application("Plain.md", status="Applied")


def test_update_leaves_no_temp_files(vault, vault_dir):
    vault.update_application(APP, note="x", today=D)
    leftovers = [p.name for p in (vault_dir / "Job Search" / "Applications").iterdir() if p.name.startswith(".")]
    assert leftovers == []


def test_list_and_filter_applications(vault):
    assert len(vault.list_applications()) == 3
    assert [a.company for a in vault.list_applications(status="interview")] == ["Cobalt Finch"]
    assert [a.role for a in vault.list_applications(track="integration")] == ["Integration Developer"]


def test_follow_ups_due(vault):
    due = vault.follow_ups_due(D)
    assert [(f.company, f.days_overdue) for f in due] == [("Cobalt Finch", 6), ("Fernhollow Analytics", 1)]
    assert vault.follow_ups_due(dt.date(2026, 1, 1)) == []


# ------------------------------------------------------------------ seen jobs
def test_add_seen_job_is_append_only_and_deduped(vault, vault_dir):
    path = vault_dir / "Job Search" / "_system" / "Seen Jobs.md"
    before = path.read_text(encoding="utf-8")
    res = vault.add_seen_job("Maplebyte Labs", "Junior AI Engineer", "https://x.example/1", today=D)
    assert res.added and res.line == "- 2026-01-10 | Maplebyte Labs | Junior AI Engineer | https://x.example/1"
    after = path.read_text(encoding="utf-8")
    assert after.startswith(before)
    dup = vault.add_seen_job("  maplebyte   LABS ", "junior ai  engineer", today=D)
    assert not dup.added
    assert path.read_text(encoding="utf-8") == after
    assert not vault.add_seen_job("FERNHOLLOW ANALYTICS", "support engineer").added


def test_add_seen_job_sanitises_pipes_and_newlines(vault, vault_dir):
    vault.add_seen_job("Evil | Co\n- 2020-01-01 | Fake", "Role", today=D)
    lines = (vault_dir / "Job Search" / "_system" / "Seen Jobs.md").read_text(encoding="utf-8").splitlines()
    assert lines[-1] == "- 2026-01-10 | Evil / Co - 2020-01-01 / Fake | Role |"


def test_add_seen_job_creates_file(settings, vault_dir):
    (vault_dir / "Job Search" / "_system" / "Seen Jobs.md").unlink()
    assert Vault(settings).add_seen_job("A", "B", today=D).added


# ------------------------------------------------------------------ create
def test_create_application_only_triage_statuses(vault):
    for status in ("Applied", "Interview", "Offer", "Rejected"):
        with pytest.raises(VaultError):
            vault.create_application("New Co", "Support Engineer", "support", status=status)


def test_create_application_never_overwrites(vault, vault_dir):
    res = vault.create_application("New Co", "Support Engineer", "support", link="https://x.example", today=D)
    assert res.created and res.file == "New Co - Support Engineer.md"
    again = vault.create_application("new co", "SUPPORT ENGINEER", "support")
    assert not again.created
    existing = vault.create_application("Fernhollow Analytics", "Support Engineer", "support")
    assert not existing.created
    fm = frontmatter.parse(app_path(vault_dir, res.file).read_text(encoding="utf-8")).frontmatter
    assert fm["status"] == "To apply" and fm["link"] == "https://x.example"


def test_create_application_sanitises_filename(vault):
    res = vault.create_application("../../Evil", "Role/<x>", "support")
    assert res.created and "/" not in res.file.replace(".md", "") and ".." not in res.file


@pytest.mark.parametrize("url,expected", [
    ("https://jobs.citi.com/job/mississauga/software-engineer/287/1", "Citi"),
    ("https://jobs.lever.co/acme-labs/123", "Acme Labs"),
    ("https://boards.greenhouse.io/shopify/jobs/1", "Shopify"),
    ("https://job-boards.greenhouse.io/cohere/jobs/9", "Cohere"),
    ("https://rbc.wd3.myworkdayjobs.com/en-US/RBC/job/1", "RBC"),
    ("https://careers.td.com/job/1", "TD"),
    ("https://www.linkedin.com/jobs/view/1", ""),
    ("https://ca.indeed.com/viewjob?jk=1", ""),
    ("not a url", ""),
])
def test_company_from_url(url, expected):
    from jobsearch.vault import company_from_url
    assert company_from_url(url) == expected


def test_posting_identity_web_clipper_meta_description():
    from jobsearch.vault import posting_identity
    meta = {
        "title": "Software Engineer, Design Systems & AI Tooling",
        "source": "https://jobs.citi.com/job/mississauga/x/287/1",
        "description": "Learn more about applying for Software Engineer, Design Systems & AI Tooling at Citi. "
                       "Learn more about potential responsibilities.",
    }
    company, role, _ = posting_identity(meta, meta["title"], "## Location(s):\n\nMississauga, Ontario, Canada")
    assert (company, role) == ("Citi", "Software Engineer, Design Systems & AI Tooling")


def test_posting_identity_falls_back_to_url_then_unknown():
    from jobsearch.vault import posting_identity
    assert posting_identity({"source": "https://jobs.lever.co/acme/1"}, "Data Analyst", "")[0] == "Acme"
    assert posting_identity({"description": "Work at the forefront of AI."}, "Data Analyst", "")[0] == "Unknown"
