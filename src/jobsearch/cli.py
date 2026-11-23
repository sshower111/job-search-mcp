"""``jobagent`` command line: triage, eval, serve, check."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import sys
from pathlib import Path

from .agent import Proposal, TriageAgent, default_report_path, render_report
from .config import TRACKS, Settings, load_settings
from .evals import render_markdown, run_eval, write_report
from .llm import OpenAICompatibleChat
from .safety import configure_logging
from .vault import Vault, VaultError


def _chat_for(settings: Settings, force: bool | None):
    use = settings.llm.enabled if force is None else force
    return OpenAICompatibleChat(settings.llm) if use else None


def _interactive_confirm(p: Proposal) -> bool:
    print(f"\n{p.company} - {p.role}: {p.fit} ({p.score}), track {p.track}")
    for a in p.actions:
        print(f"   will call {a.tool} {a.arguments.get('status', '')}".rstrip())
    if p.needs_manual_review:
        print("   WARNING: flagged for manual review: " + ", ".join(p.injection_flags + p.security_events))
    answer = input("Apply these changes? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def cmd_triage(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    chat = _chat_for(settings, args.llm)
    today = dt.date.today()
    agent = TriageAgent(settings, chat=chat, today=today)
    proposals = asyncio.run(agent.propose(limit=args.limit))
    method = "LLM tool calling + rules" if chat else "deterministic baseline"

    applied = None
    if args.apply:
        skipped = [p for p in proposals if p.needs_manual_review]
        if skipped:
            print(f"--apply: skipping {len(skipped)} posting(s) flagged for manual review.", file=sys.stderr)
        applied = asyncio.run(agent.apply(proposals, confirm=lambda p: not p.needs_manual_review))
    elif sys.stdin.isatty() and not args.dry_run:
        print(render_report(proposals, today, method))
        applied = asyncio.run(agent.apply(proposals, confirm=_interactive_confirm))
    report = render_report(proposals, today, method, applied)
    path = Path(args.report) if args.report else default_report_path(today)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    if applied is None:
        print(report)
        print("Dry run: no changes written. Re-run with --apply, or interactively, to write proposals.")
    print(f"Report saved to {path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    chat = _chat_for(settings, args.llm)
    labels = Path(args.labels)
    result = run_eval(settings, labels, Path(args.postings_dir) if args.postings_dir else None, chat=chat)
    md, js = write_report(result, Path(args.out))
    print(render_markdown(result))
    print(f"Report written to {md} and {js}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    vault = Vault(settings)
    print(f"Config file : {settings.config_file or '(none, using defaults/env)'}")
    print(f"Vault       : {'found' if vault.root.is_dir() else 'NOT FOUND'}")
    ok = vault.root.is_dir()
    for label, folder in (("Postings", settings.folders.postings), ("Applications", settings.folders.applications)):
        exists = (vault.root / folder).is_dir()
        ok &= exists
        print(f"{label:<12}: {folder} -> {'ok' if exists else 'missing'}")
    for track in TRACKS:
        try:
            vault.resume_text(track)
            print(f"Resume {track:<5}: ok")
        except VaultError as exc:
            ok = False
            print(f"Resume {track:<5}: {exc}")
    print(f"LLM         : {'enabled' if settings.llm.enabled else 'disabled'} ({settings.llm.base_url}, "
          f"{settings.llm.model})")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jobagent", description="Job-search triage agent and evals")
    parser.add_argument("--config", help="Path to jobagent.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("triage", help="Review postings in 'Jobs to tailor' and propose tracker changes")
    t.add_argument("--apply", action="store_true", help="Write proposals without prompting "
                   "(postings flagged for manual review are skipped)")
    t.add_argument("--dry-run", action="store_true", help="Never prompt or write")
    g = t.add_mutually_exclusive_group()
    g.add_argument("--llm", dest="llm", action="store_true", default=None, help="Use the configured LLM")
    g.add_argument("--no-llm", dest="llm", action="store_false", help="Deterministic baseline only")
    t.add_argument("--limit", type=int, help="Only triage the first N postings")
    t.add_argument("--report", help="Where to save the markdown report (default reports/triage-DATE.md)")
    t.set_defaults(func=cmd_triage)

    e = sub.add_parser("eval", help="Evaluate fit classification against labels.csv")
    e.add_argument("--labels", default="evals/labels.csv")
    e.add_argument("--postings-dir", help="Folder the label 'file' paths are relative to")
    e.add_argument("--out", default="evals/reports")
    g = e.add_mutually_exclusive_group()
    g.add_argument("--llm", dest="llm", action="store_true", default=None)
    g.add_argument("--no-llm", dest="llm", action="store_false")
    e.set_defaults(func=cmd_eval)

    c = sub.add_parser("check", help="Validate config, vault folders and resumes")
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("serve", help="Run the MCP server on stdio")
    s.add_argument("--read-only", action="store_true")
    s.set_defaults(func=None)

    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles may not be UTF-8
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    configure_logging()
    if args.command == "serve":
        from .server import main as serve_main

        serve_main((["--config", args.config] if args.config else []) + (["--read-only"] if args.read_only else []))
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
