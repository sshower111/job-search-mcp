import json

import pytest
from mcp import Client

from jobsearch.server import READ_TOOLS, WRITE_TOOLS, build_server

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def data(result):
    assert not result.is_error, result.content
    d = result.structured_content
    return d["result"] if isinstance(d, dict) and set(d) == {"result"} else d


async def test_tool_list_and_read_only_mode(settings):
    async with Client(build_server(settings)) as c:
        names = {t.name for t in (await c.list_tools()).tools}
    assert names == set(READ_TOOLS) | set(WRITE_TOOLS)
    async with Client(build_server(settings.model_copy(update={"read_only": True}))) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert names == set(READ_TOOLS)
        result = await c.call_tool("update_application", {"file": "x.md", "status": "Offer"})
        assert result.is_error


async def test_read_tools(settings):
    async with Client(build_server(settings)) as c:
        postings = data(await c.call_tool("list_postings", {}))
        assert len(postings) == 4
        hit = data(await c.call_tool("search_postings", {"query": "bilingual french"}))
        assert [h["file"] for h in hit] == ["Technical Support Engineer - Lumen Quay.md"]
        apps = data(await c.call_tool("list_applications", {"status": "Applied"}))
        assert [a["company"] for a in apps] == ["Fernhollow Analytics"]
        due = data(await c.call_tool("follow_ups_due", {"date": "2026-01-10"}))
        assert len(due) == 2
        m = data(await c.call_tool("match_score", {"posting": "Junior AI Engineer - Maplebyte Labs.md"}))
        assert m["track"] == "ai" and m["fit"] == "good"
        g = data(await c.call_tool("gap_analysis", {"posting": "Senior Integration Engineer - Northwind Pipes.md",
                                                     "track": "integration"}))
        assert g["blockers"] and "Java" in g["missing_required"]


async def test_read_posting_flags_injection(settings):
    async with Client(build_server(settings)) as c:
        p = data(await c.call_tool("read_posting", {"file": "Support Engineer - Quillstone (injection test).md"}))
    assert p["untrusted"] is True
    assert "ignore-instructions" in p["injection_flags"]


async def test_path_traversal_through_tools(settings):
    async with Client(build_server(settings)) as c:
        for tool, args in [
            ("read_posting", {"file": "../Resumes/Master Resume - AI Engineer.md"}),
            ("match_score", {"posting": "../../jobagent.toml"}),
            ("update_application", {"file": "../../Resumes/Master Resume - AI Engineer.md", "note": "x"}),
        ]:
            result = await c.call_tool(tool, args)
            assert result.is_error
            assert "escapes" in result.content[0].text or "not allowed" in result.content[0].text


async def test_write_tools(settings, vault_dir):
    async with Client(build_server(settings)) as c:
        r = data(await c.call_tool("update_application", {
            "file": "Brightkettle - Integration Developer.md", "status": "Applied", "applied": "2026-01-06"}))
        assert r["changed"]["status"] == "Applied"
        back = await c.call_tool("update_application", {
            "file": "Brightkettle - Integration Developer.md", "status": "To apply"})
        assert back.is_error and "backwards" in back.content[0].text
        bad_create = await c.call_tool("create_application", {
            "company": "X", "role": "Y", "track": "ai", "status": "Offer"})
        assert bad_create.is_error
        seen = data(await c.call_tool("add_seen_job", {"company": "brightkettle", "role": "INTEGRATION developer"}))
        assert seen["added"] is False


async def test_resources(settings):
    async with Client(build_server(settings)) as c:
        templates = (await c.list_resource_templates()).resource_templates
        assert any(t.uri_template == "resume://{track}" for t in templates)
        text = (await c.read_resource("resume://integration")).contents[0].text
        assert "Integration" in text or "Sync Bridge" in text
        index = (await c.read_resource("resume://index")).contents[0].text
        assert "resume://ai" in index


async def test_llm_method_disabled_and_with_fake_chat(settings):
    async with Client(build_server(settings)) as c:
        r = await c.call_tool("match_score", {"posting": "Junior AI Engineer - Maplebyte Labs.md", "method": "llm"})
        assert r.is_error and "disabled" in r.content[0].text

    class Judge:
        def chat(self, messages, tools=None):
            return {"content": json.dumps({"fit": "good", "score": 99, "reasons": ["looks great"]})}

    async with Client(build_server(settings, chat_factory=Judge)) as c:
        r = data(await c.call_tool("match_score", {
            "posting": "Senior Integration Engineer - Northwind Pipes.md", "method": "hybrid"}))
        assert r["fit"] == "bad" and r["method"] == "hybrid"  # rules veto the LLM
        r = data(await c.call_tool("match_score", {
            "posting": "Junior AI Engineer - Maplebyte Labs.md", "method": "llm"}))
        assert r["fit"] == "good" and r["score"] == 99
