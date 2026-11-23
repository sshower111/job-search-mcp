from jobsearch import frontmatter

NOTE = """---
company: Acme Widgets
role: Support Engineer
# a comment Obsidian users sometimes leave
tags:
  - job
  - support
status: To apply
link: "https://example.com/a?b=c"
---
# Body

Some text.
"""


def test_parse_reads_fields_and_body():
    note = frontmatter.parse(NOTE)
    assert note.has_frontmatter
    assert note.frontmatter["company"] == "Acme Widgets"
    assert note.frontmatter["tags"] == ["job", "support"]
    assert note.body.startswith("# Body")


def test_parse_without_frontmatter():
    note = frontmatter.parse("# Just a note\n")
    assert not note.has_frontmatter
    assert note.frontmatter == {}


def test_set_fields_only_touches_requested_keys():
    out = frontmatter.set_fields(NOTE, {"status": "Applied", "applied": "2026-01-05"})
    before, after = NOTE.splitlines(), out.splitlines()
    assert "status: Applied" in after
    assert "applied: 2026-01-05" in after
    # Every other line is preserved verbatim and in order.
    untouched = [line for line in before if not line.startswith("status:")]
    assert [line for line in after if not line.startswith(("status:", "applied:"))] == untouched


def test_set_fields_preserves_crlf():
    crlf = NOTE.replace("\n", "\r\n")
    out = frontmatter.set_fields(crlf, {"status": "Applied"})
    assert "status: Applied\r\n" in out
    assert "\n" not in out.replace("\r\n", "")


def test_set_fields_replaces_block_list_value():
    out = frontmatter.set_fields(NOTE, {"tags": "job"})
    parsed = frontmatter.parse(out).frontmatter
    assert parsed["tags"] == "job"
    assert parsed["status"] == "To apply"


def test_yaml_scalar_quotes_risky_values():
    assert frontmatter.yaml_scalar("https://x.y/z") == '"https://x.y/z"'
    assert frontmatter.yaml_scalar("yes") == '"yes"'
    assert frontmatter.yaml_scalar("To apply") == "To apply"
    assert frontmatter.yaml_scalar("a\nb: c") == '"a b: c"'
    assert frontmatter.yaml_scalar("") == ""


def test_append_note_creates_section():
    out = frontmatter.append_note_line(NOTE, "- 2026-01-05: hello")
    assert out.endswith("## Notes\n- 2026-01-05: hello\n")
    assert out.startswith(NOTE)


def test_append_note_into_existing_section_before_next_heading():
    text = NOTE + "\n## Notes\n- 2026-01-01: first\n\n## Interview prep\nstuff\n"
    out = frontmatter.append_note_line(text, "- 2026-01-02: second")
    assert "- 2026-01-01: first\n- 2026-01-02: second\n\n## Interview prep" in out
