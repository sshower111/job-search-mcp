"""Deterministic baseline matcher.

Score = skill coverage against the master resume for a track
(80% required skills, 20% preferred skills) minus small penalties.
Hard filters (any one => fit "bad"), mirroring the job-search rules:
  * senior / lead / manager-type titles
  * more than ``max_years`` years of experience required
  * French required (French as an "asset" is fine)
  * location outside the target area (Toronto/GTA by default) and not remote in Canada
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import MatchingRules
from .models import GapAnalysis, MatchResult, Suggestion
from .skills import extract_skills, suggestion_for

_PREFERRED_HEAD = re.compile(r"nice|prefer|bonus|asset|plus|good to have|desirable|extra credit", re.I)
_REQUIRED_HEAD = re.compile(
    r"require|qualif|must|what you.?ll need|what you bring|who you are|you have|you bring|skills|"
    r"experience|about you|what we.?re looking for|looking for", re.I)
_OTHER_HEAD = re.compile(r"about (us|the company)|benefits|perks|why join|what we offer|compensation|"
                         r"equal opportunity|our values", re.I)
_PREFERRED_LINE = re.compile(r"\b(is an asset|an asset|nice to have|preferred|is a plus|a plus|bonus)\b", re.I)


@dataclass
class Sections:
    required: str = ""
    preferred: str = ""
    other: str = ""
    has_required_heading: bool = False
    lines: list[str] = field(default_factory=list)


def _heading_text(line: str) -> str | None:
    s = line.strip()
    m = re.match(r"^#{1,6}\s+(.*)$", s)
    if m:
        return m.group(1).strip()
    m = re.match(r"^\*\*(.+?)\*\*:?$", s)
    if m:
        return m.group(1).strip()
    if s.endswith(":") and 2 < len(s) < 60 and not s.startswith(("-", "*")):
        return s[:-1]
    return None


def split_sections(body: str) -> Sections:
    out = Sections()
    current = "other"
    req, pref, other = [], [], []
    for line in body.splitlines():
        head = _heading_text(line)
        if head is not None:
            if _OTHER_HEAD.search(head):
                current = "skip"
            elif _PREFERRED_HEAD.search(head):
                current = "preferred"
            elif _REQUIRED_HEAD.search(head):
                current = "required"
                out.has_required_heading = True
            else:
                current = "other"
            continue
        if current == "skip":
            continue
        if _PREFERRED_LINE.search(line):
            pref.append(line)
        elif current == "required":
            req.append(line)
        elif current == "preferred":
            pref.append(line)
        else:
            other.append(line)
    out.required, out.preferred, out.other = "\n".join(req), "\n".join(pref), "\n".join(other)
    return out


# ---------------------------------------------------------------- hard filters
_YEARS = re.compile(
    r"(?:(?:minimum|min\.?|at least)\s+(?:of\s+)?)?(\d{1,2})\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*(\d{1,2})\s*)?"
    r"(?:\+\s*)?years?(?:'|’)?\s*(?:of\s+)?(?:[\w/+#.,-]+\s+){0,6}?(?:experience|exp\b)",
    re.I,
)


def years_required(text: str) -> int | None:
    """Strictest (largest) lower bound among stated experience requirements; None if none stated."""
    lows = [int(m.group(1)) for m in _YEARS.finditer(text)]
    lows = [n for n in lows if n <= 25]
    return max(lows) if lows else None


def is_senior_title(title: str, rules: MatchingRules) -> str | None:
    t = title.casefold()
    for word in rules.senior_title_words:
        if re.search(rf"(?<![\w]){re.escape(word.casefold())}(?![\w])", t):
            return word
    return None


_FRENCH_REQUIRED = re.compile(
    r"(fluen\w*|proficien\w*|bilingual\w*|must (?:be able to )?(?:speak|read|write)|required|mandatory|"
    r"essential)[^.\n]{0,60}\bfrench\b|\bfrench\b[^.\n]{0,60}(required|mandatory|essential|is a must|"
    r"must-have)|bilingu\w+\s*\(?(?:english\s*(?:and|/|&)\s*french|french\s*(?:and|/|&)\s*english)",
    re.I,
)
_FRENCH_ASSET = re.compile(r"french[^.\n]{0,40}(asset|plus|nice to have|preferred|bonus)", re.I)


def french_required(text: str) -> bool:
    for m in _FRENCH_REQUIRED.finditer(text):
        window = text[max(0, m.start() - 10): m.end() + 50]
        if not _FRENCH_ASSET.search(window):
            return True
    return False


_REMOTE = re.compile(r"\bremote\b|work from home|\bwfh\b|anywhere in canada", re.I)
_CANADA = re.compile(r"\bcanada\b|\bcanadian\b|\bontario\b|(?-i:,\s*ON\b)", re.I)
_US_ONLY = re.compile(
    r"(us|u\.s\.|united states)[- ]only|must (?:reside|live|be located) in the (?:us|u\.s\.|united states)|"
    r"remote\s*[-–(]\s*(?:us|usa|united states)\)?", re.I)


def location_check(location_field: str, body: str, rules: MatchingRules) -> tuple[str, str]:
    """Return (status, reason) where status is ok / rejected / unknown."""
    loc = location_field or ""
    text = f"{loc}\n{body}"
    gta = [p for p in rules.locations if re.search(rf"(?<![\w]){re.escape(p)}(?![\w])", loc or body, re.I)]
    remote = bool(_REMOTE.search(loc or text))
    if _US_ONLY.search(text):
        return "rejected", "Remote role restricted to the United States"
    if loc:
        if gta:
            return "ok", f"Location in target area ({gta[0].title()})"
        if remote and rules.allow_remote_canada and (_CANADA.search(text)):
            return "ok", "Remote role open in Canada"
        if remote:
            return "unknown", "Remote, but country eligibility is not stated"
        return "rejected", f"Location outside target area ({loc})"
    if gta:
        return "ok", f"Location in target area ({gta[0].title()})"
    if remote and _CANADA.search(text):
        return "ok", "Remote role open in Canada"
    return "unknown", "Location not stated"


# ---------------------------------------------------------------- scoring
def score_posting(
    *,
    posting_file: str,
    title: str,
    role: str,
    location: str,
    body: str,
    resume_text: str,
    track: str,
    rules: MatchingRules,
) -> MatchResult:
    sections = split_sections(body)
    required_text = sections.required if sections.has_required_heading else sections.required + "\n" + \
        sections.other
    req = extract_skills(required_text)
    pref = [s for s in extract_skills(sections.preferred) if s not in req]
    have = set(extract_skills(resume_text))

    matched_req = [s for s in req if s in have]
    matched_pref = [s for s in pref if s in have]
    missing_req = [s for s in req if s not in have]
    missing_pref = [s for s in pref if s not in have]

    # No recognised required skills means no evidence of overlap, so score it low rather than neutral.
    req_cov = len(matched_req) / len(req) if req else (0.25 if pref else 0.0)
    pref_cov = len(matched_pref) / len(pref) if pref else 0.5
    score = 100 * (0.8 * req_cov + 0.2 * pref_cov)

    reasons: list[str] = []
    hard_fail = False

    senior = is_senior_title(role or title, rules)
    if senior:
        hard_fail = True
        reasons.append(f"Seniority: title contains '{senior}'")

    excluded = next((w for w in rules.excluded_title_phrases
                     if re.search(rf"(?<![\w]){re.escape(w.casefold())}(?![\w])", (role or title).casefold())), None)
    if excluded:
        hard_fail = True
        reasons.append(f"Role type excluded by rules: title contains '{excluded}'")

    full_text = f"{title}\n{body}"
    years = years_required(f"{title}\n{sections.required}\n{sections.other}")
    if years is not None and years > rules.max_years:
        hard_fail = True
        reasons.append(f"Requires {years}+ years of experience (target is 0-{rules.max_years})")
    elif years is not None and years == rules.max_years:
        score -= 5
        reasons.append(f"Asks for {years} years, the top of the target range")

    if rules.skip_french_required and french_required(full_text):
        hard_fail = True
        reasons.append("French language required")

    loc_status, loc_reason = location_check(location, body, rules)
    if loc_status == "rejected":
        hard_fail = True
    elif loc_status == "unknown":
        score -= 5
    reasons.append(loc_reason)

    reasons.append(f"Required skills matched {len(matched_req)}/{len(req)}; preferred {len(matched_pref)}/"
                   f"{len(pref)}")
    score_int = max(0, min(100, round(score)))
    fit = "good" if (not hard_fail and score_int >= rules.fit_threshold) else "bad"
    if not hard_fail and fit == "bad":
        reasons.append(f"Score {score_int} below threshold {rules.fit_threshold}")
    return MatchResult(
        posting=posting_file, track=track, score=score_int, fit=fit, hard_fail=hard_fail, reasons=reasons,
        matched_required=matched_req, matched_preferred=matched_pref, missing_required=missing_req,
        missing_preferred=missing_pref, years_required=years, location_status=loc_status,
    )


_TRACK_HINTS = {
    "ai": re.compile(r"\b(ai|ml|machine learning|llms?|genai|generative|deep learning|nlp|data scien\w*|rag|"
                     r"mcp|model context protocol|agents?|agentic|prompt\w*)\b", re.I),
    "integration": re.compile(r"\b(integrat\w*|apis?|middleware|etl|ipaas|mulesoft|boomi|webhooks?|"
                              r"implementation|solutions? engineer)\b", re.I),
    "support": re.compile(r"\b(support|customer|troubleshoot\w*|tickets?|incidents?|escalations?|"
                          r"service desk|help ?desk)\b", re.I),
}


def track_affinity(title: str, body: str, track: str) -> int:
    """How strongly a posting's wording points at a track. Title words count three times as much as body words."""
    pattern = _TRACK_HINTS.get(track)
    if pattern is None:
        return 0
    return 3 * len(pattern.findall(title or "")) + min(len(pattern.findall(body or "")), 30)


def pick_best(results: list[MatchResult], title: str, body: str) -> MatchResult:
    """Highest score wins; on a tie, the track the posting's wording points at.

    Master resumes share most core skills, so a tie across all tracks is common. Without a tie-break the
    first track in TRACKS (support) would always win.
    """
    return max(results, key=lambda r: (r.score, track_affinity(title, body, r.track)))


def gap_from_match(match: MatchResult) -> GapAnalysis:
    suggestions = []
    for skill in match.missing_required + match.missing_preferred:
        kind, text = suggestion_for(skill)
        suggestions.append(Suggestion(skill=skill, kind=kind, suggestion=text))
    blockers = [r for r in match.reasons if match.hard_fail and (
        r.startswith(("Seniority", "Requires", "French", "Location outside", "Remote role restricted",
                       "Role type excluded")))]
    return GapAnalysis(
        posting=match.posting, track=match.track,
        matched=match.matched_required + match.matched_preferred,
        missing_required=match.missing_required, missing_preferred=match.missing_preferred,
        blockers=blockers, suggestions=suggestions,
    )
