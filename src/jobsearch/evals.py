"""Offline evaluation of the fit classifier: baseline vs LLM judge vs hybrid.

labels.csv columns: file, track, expected_fit (good/bad), reasons
``file`` is relative to the postings directory (default: ``<labels dir>/postings``).
``track`` may be support, integration, ai or auto.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import frontmatter
from .config import TRACKS, Settings
from .judge import hybrid, judge_fit
from .llm import ChatModel, LLMError
from .matching import pick_best, score_posting
from .models import MatchResult
from .vault import Vault, posting_identity, read_text


@dataclass
class LabeledPosting:
    file: str
    track: str
    expected_fit: str
    reasons: str


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    errors: int = 0
    rows: list[dict] = field(default_factory=list)

    def add(self, expected: str, predicted: str | None) -> None:
        if predicted is None:
            self.errors += 1
            return
        if expected == "good":
            self.tp += predicted == "good"
            self.fn += predicted != "good"
        else:
            self.fp += predicted == "good"
            self.tn += predicted != "good"

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def summary(self) -> dict:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        accuracy = (self.tp + self.tn) / self.n if self.n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"n": self.n, "errors": self.errors, "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
                "precision": round(precision, 3), "recall": round(recall, 3), "accuracy": round(accuracy, 3),
                "f1": round(f1, 3)}


def load_labels(path: Path) -> list[LabeledPosting]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for i, row in enumerate(rows, start=2):
        fit = (row.get("expected_fit") or "").strip().lower()
        track = (row.get("track") or "auto").strip().lower()
        if fit not in ("good", "bad"):
            raise ValueError(f"{path}:{i}: expected_fit must be good or bad")
        if track not in (*TRACKS, "auto"):
            raise ValueError(f"{path}:{i}: track must be one of {TRACKS} or auto")
        out.append(LabeledPosting(row["file"].strip(), track, fit, (row.get("reasons") or "").strip()))
    return out


def _resolve_inside(base: Path, rel: str) -> Path:
    path = (base / rel).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"Label file path escapes the postings directory: {rel}")
    return path


def run_eval(settings: Settings, labels_path: Path, postings_dir: Path | None = None,
             chat: ChatModel | None = None) -> dict:
    labels = load_labels(labels_path)
    postings_dir = postings_dir or labels_path.parent / "postings"
    vault = Vault(settings)
    resumes = {t: vault.resume_text(t) for t in TRACKS}
    methods = {"baseline": Metrics()}
    if chat is not None:
        methods["llm"], methods["hybrid"] = Metrics(), Metrics()

    for item in labels:
        note = frontmatter.parse(read_text(_resolve_inside(postings_dir, item.file)))
        title = str(note.frontmatter.get("title") or Path(item.file).stem)
        company, role, location = posting_identity(note.frontmatter, title, note.body)
        tracks = TRACKS if item.track == "auto" else (item.track,)
        base: MatchResult = pick_best(
            [score_posting(posting_file=item.file, title=title, role=role, location=location, body=note.body,
                           resume_text=resumes[t], track=t, rules=settings.matching) for t in tracks],
            title, note.body)
        row = {"file": item.file, "expected": item.expected_fit, "baseline": base.fit,
               "baseline_score": base.score, "baseline_reasons": base.reasons}
        methods["baseline"].add(item.expected_fit, base.fit)
        if chat is not None:
            try:
                judged = judge_fit(chat, posting_file=item.file, title=title, body=note.body,
                                   resume_text=resumes[base.track], track=base.track, rules=settings.matching)
                row["llm"], row["llm_score"] = judged.fit, judged.score
                row["hybrid"] = hybrid(base, judged).fit
            except LLMError as exc:
                row["llm"] = row["hybrid"] = None
                row["llm_error"] = str(exc)[:200]
            methods["llm"].add(item.expected_fit, row["llm"])
            methods["hybrid"].add(item.expected_fit, row["hybrid"])
        methods["baseline"].rows.append(row)

    return {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "labels": labels_path.name,
        "n_labeled": len(labels),
        "llm_model": settings.llm.model if chat is not None else None,
        "metrics": {name: m.summary() for name, m in methods.items()},
        "rows": methods["baseline"].rows,
    }


def render_markdown(result: dict) -> str:
    lines = [
        f"# Fit classifier eval - {result['generated']}",
        "",
        f"Labels: `{result['labels']}` ({result['n_labeled']} postings). Positive class = expected_fit 'good'.",
        f"LLM judge: {result['llm_model'] or 'not run (no LLM configured; use --llm)'}",
        "",
        "| Method | N | Precision | Recall | Accuracy | F1 | TP | FP | TN | FN | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in result["metrics"].items():
        lines.append(f"| {name} | {m['n']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['accuracy']:.3f} | "
                     f"{m['f1']:.3f} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | {m['errors']} |")
    lines += ["", "## Per posting", "", "| File | Expected | Baseline (score) | LLM | Hybrid |",
              "|---|---|---|---|---|"]
    for r in result["rows"]:
        mark = "" if r["baseline"] == r["expected"] else " (miss)"
        lines.append(f"| {r['file']} | {r['expected']} | {r['baseline']} ({r['baseline_score']}){mark} | "
                     f"{r.get('llm', '-') or '-'} | {r.get('hybrid', '-') or '-'} |")
    misses = [r for r in result["rows"] if r["baseline"] != r["expected"]]
    if misses:
        lines += ["", "## Baseline misses", ""]
        for r in misses:
            lines.append(f"- `{r['file']}` expected {r['expected']}, got {r['baseline']}: "
                         + "; ".join(r["baseline_reasons"]))
    return "\n".join(lines) + "\n"


def write_report(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    md = out_dir / f"eval-{stamp}.md"
    js = out_dir / f"eval-{stamp}.json"
    md.write_text(render_markdown(result), encoding="utf-8")
    js.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return md, js
