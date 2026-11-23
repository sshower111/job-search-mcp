import json
import logging
from pathlib import Path

import pytest

from jobsearch.cli import main as cli_main
from jobsearch.config import LLMSettings, load_settings
from jobsearch.evals import Metrics, load_labels, run_eval, write_report
from jobsearch.llm import LLMError, OpenAICompatibleChat, extract_json
from jobsearch.safety import RedactingFilter, redact

REPO = Path(__file__).resolve().parents[1]
LABELS = REPO / "evals" / "labels.csv"


# ------------------------------------------------------------------ LLM client
def test_openai_compatible_request_shape(monkeypatch):
    monkeypatch.setenv("JOBAGENT_LLM_API_KEY", "sk-test-123456789")
    captured = {}

    def transport(url, headers, body, timeout):
        captured.update(url=url, headers=headers, payload=json.loads(body))
        return {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_posting", "arguments": "{}"}}]}}]}

    chat = OpenAICompatibleChat(LLMSettings(base_url="http://localhost:1234/v1/", model="m"), transport)
    reply = chat.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "x"}}])
    assert captured["url"] == "http://localhost:1234/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test-123456789"
    assert captured["payload"]["model"] == "m" and captured["payload"]["tools"]
    assert reply["tool_calls"][0]["function"]["name"] == "read_posting"


def test_llm_bad_response_and_scheme():
    chat = OpenAICompatibleChat(LLMSettings(), lambda *a: {"oops": 1})
    with pytest.raises(LLMError):
        chat.chat([{"role": "user", "content": "x"}])
    with pytest.raises(LLMError):
        OpenAICompatibleChat(LLMSettings(base_url="file:///etc")).chat([{"role": "user", "content": "x"}])


@pytest.mark.parametrize("text,expected", [
    ('{"fit": "good"}', {"fit": "good"}),
    ('Sure!\n```json\n{"fit": "bad", "reasons": ["x"]}\n```', {"fit": "bad", "reasons": ["x"]}),
    ('Here: {"a": {"b": 1}} trailing', {"a": {"b": 1}}),
    ("no json here", None),
    (None, None),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


# ------------------------------------------------------------------ config
def test_env_overrides(monkeypatch, tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('vault_path = "v"\n[llm]\nenabled = false\n', encoding="utf-8")
    assert load_settings(cfg).vault_path == (tmp_path / "v").resolve()
    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "other"))
    monkeypatch.setenv("JOBSEARCH_READ_ONLY", "1")
    monkeypatch.setenv("JOBAGENT_LLM_ENABLED", "true")
    s = load_settings(cfg)
    assert s.vault_path == tmp_path / "other" and s.read_only and s.llm.enabled


# ------------------------------------------------------------------ evals
def test_metrics_math():
    m = Metrics()
    for exp, pred in [("good", "good"), ("good", "bad"), ("bad", "good"), ("bad", "bad"), ("bad", None)]:
        m.add(exp, pred)
    s = m.summary()
    assert (s["tp"], s["fn"], s["fp"], s["tn"], s["errors"]) == (1, 1, 1, 1, 1)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["accuracy"] == 0.5


def test_labels_file_is_valid():
    labels = load_labels(LABELS)
    assert len(labels) == 15
    assert all((LABELS.parent / "postings" / item.file).is_file() for item in labels)


def test_eval_runs_baseline_and_fake_llm(settings, tmp_path):
    result = run_eval(settings, LABELS)
    assert set(result["metrics"]) == {"baseline"}
    assert result["metrics"]["baseline"]["n"] == 15

    class AlwaysGood:
        def chat(self, messages, tools=None):
            return {"content": '{"fit": "good", "score": 80, "reasons": ["ok"]}'}

    result = run_eval(settings, LABELS, chat=AlwaysGood())
    assert result["metrics"]["llm"]["recall"] == 1.0
    # Hybrid applies the hard rules, so it can only be at least as precise as "always good".
    assert result["metrics"]["hybrid"]["precision"] >= result["metrics"]["llm"]["precision"]
    md, js = write_report(result, tmp_path)
    assert md.read_text(encoding="utf-8").startswith("# Fit classifier eval")
    assert json.loads(js.read_text(encoding="utf-8"))["n_labeled"] == 15


def test_eval_label_path_traversal(settings, tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("file,track,expected_fit,reasons\n../../secret.md,auto,good,x\n", encoding="utf-8")
    (tmp_path / "postings").mkdir()
    with pytest.raises(ValueError):
        run_eval(settings, labels)


# ------------------------------------------------------------------ logging / PII
def test_redaction():
    text = ("Mail jane.doe@example.com or call (416) 555-0199 / +1 647-555-0123; "
            r"vault C:\Users\Jane\vault and /home/jane/vault key sk-abcdefghijklmnop")
    out = redact(text)
    for secret in ("jane.doe@example.com", "555-0199", "647-555-0123", "Jane", "jane", "sk-abcdefghijklmnop"):
        assert secret not in out


def test_redacting_filter_on_records():
    record = logging.LogRecord("jobsearch", logging.WARNING, __file__, 1, "contact %s", ("a@b.co",), None)
    RedactingFilter().filter(record)
    assert record.getMessage() == "contact [email]"


def test_logs_do_not_contain_posting_or_resume_text(settings, caplog, add_injection_postings):
    from jobsearch.vault import Vault

    add_injection_postings()
    logger = logging.getLogger("jobsearch")
    logger.propagate = True
    with caplog.at_level(logging.DEBUG, logger="jobsearch"):
        vault = Vault(settings)
        for p in vault.list_postings():
            vault.read_posting(p.file)
        vault.update_application("Brightkettle - Integration Developer.md", note="Called alex.sample@example.com")
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "example.com" not in joined and "Called" not in joined
    assert "talent@" not in joined and "Requirements" not in joined


# ------------------------------------------------------------------ CLI
def test_cli_check_and_dry_run(settings, tmp_path, capsys, monkeypatch):
    cfg = str(settings.config_file)
    assert cli_main(["--config", cfg, "check"]) == 0
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    report = tmp_path / "r.md"
    assert cli_main(["--config", cfg, "triage", "--no-llm", "--dry-run", "--report", str(report)]) == 0
    out = capsys.readouterr().out
    assert "Dry run: no changes written" in out
    assert "Maplebyte Labs" in report.read_text(encoding="utf-8")
