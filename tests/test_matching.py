import pytest

from jobsearch.config import MatchingRules
from jobsearch.matching import (
    french_required,
    gap_from_match,
    is_senior_title,
    location_check,
    score_posting,
    split_sections,
    years_required,
)
from jobsearch.skills import extract_skills

RULES = MatchingRules()
RESUME = "Python, SQL, REST APIs, JSON, Docker, Git, LLMs, RAG, troubleshooting, ServiceNow"


@pytest.mark.parametrize("text,expected", [
    ("0-2 years of experience with Python", 0),
    ("3-5 years of experience in integrations", 3),
    ("Minimum of 5 years of professional experience", 5),
    ("7+ years of experience building APIs", 7),
    ("2+ years’ experience in support", 2),
    ("Our company was founded 20 years ago", None),
    ("1-3 years experience; 6+ years of hands-on Java experience", 6),
])
def test_years_required(text, expected):
    assert years_required(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Bilingual (English and French) required", True),
    ("Fluent in French and English (required)", True),
    ("French is mandatory for this role", True),
    ("French is an asset", False),
    ("Bilingual French is a plus", False),
    ("We have offices in France", False),
])
def test_french(text, expected):
    assert french_required(text) is expected


@pytest.mark.parametrize("loc,body,status", [
    ("Toronto, ON (Hybrid)", "", "ok"),
    ("Richmond Hill, ON", "", "ok"),
    ("Remote, Canada", "", "ok"),
    ("Remote - US only", "", "rejected"),
    ("Vancouver, BC (on-site)", "", "rejected"),
    ("Remote", "Work from anywhere", "unknown"),
    ("", "Our office is in Markham.", "ok"),
    ("", "Nothing about where.", "unknown"),
])
def test_location(loc, body, status):
    assert location_check(loc, body, RULES)[0] == status


@pytest.mark.parametrize("title,hit", [
    ("Senior Support Engineer", "senior"), ("Sr. Integration Developer", "sr"), ("Team Lead, AI", "lead"),
    ("Junior AI Engineer", None), ("Support Engineer II", None), ("Solutions Architect", "architect"),
])
def test_seniority(title, hit):
    assert is_senior_title(title, RULES) == hit


def test_sections_split_required_and_preferred():
    body = "## Requirements\n- Python\n- SQL\n\n## Nice to have\n- Docker\n\n## Benefits\n- Kubernetes gym\n"
    s = split_sections(body)
    assert extract_skills(s.required) == ["Python", "SQL"]
    assert extract_skills(s.preferred) == ["Docker"]
    assert "Kubernetes" not in s.required + s.preferred + s.other


def test_skill_aliases_do_not_overmatch():
    assert extract_skills("Support agents answer tickets") == []
    assert "C#/.NET" in extract_skills("Experience with C# and .NET")
    assert "Go" not in extract_skills("You will go above and beyond")


def _score(body, title="Junior Support Engineer", location="Toronto, ON"):
    return score_posting(posting_file="x.md", title=title, role=title, location=location, body=body,
                         resume_text=RESUME, track="support", rules=RULES)


def test_good_fit_scores_high():
    r = _score("## Requirements\n- 0-2 years of experience\n- Python, SQL, REST APIs\n## Nice to have\n- Docker")
    assert r.fit == "good" and r.score >= 90 and not r.hard_fail


def test_hard_filters_make_fit_bad_even_with_full_skill_match():
    body = "## Requirements\n- Python, SQL"
    assert _score(body, title="Senior Support Engineer").fit == "bad"
    assert _score(body + "\n- 5+ years of experience").fit == "bad"
    assert _score(body + "\n- French required").fit == "bad"
    assert _score(body, location="Calgary, AB").fit == "bad"
    assert _score(body, title="IT Support Specialist").fit == "bad"


def test_no_recognised_skills_is_not_a_good_fit():
    r = _score("## Requirements\n- 0-2 years of experience in embedded C++ and RTOS")
    assert r.fit == "bad"


def test_gap_analysis_structure():
    r = _score("## Requirements\n- Python, Kubernetes\n## Nice to have\n- Terraform\n- 6+ years of experience")
    gap = gap_from_match(r)
    assert gap.missing_required == ["Kubernetes"]
    assert gap.missing_preferred == ["Terraform"]
    assert {s.skill for s in gap.suggestions} == {"Kubernetes", "Terraform"}
    assert all(s.kind in ("learn", "build") and s.suggestion for s in gap.suggestions)
