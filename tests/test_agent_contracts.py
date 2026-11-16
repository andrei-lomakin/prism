from prism.agents.contracts import parse_agent_reply


def test_parse_plain_text():
    r = parse_agent_reply("просто текст")
    assert r.action == "done"
    assert "просто" in r.reply


def test_parse_fenced_json():
    raw = (
        '```json\n{"reply":"ok","action":"tool","tool":"search_kb","tool_args":{"query":"x"}}\n```'
    )
    r = parse_agent_reply(raw)
    assert r.action == "tool"
    assert r.tool == "search_kb"
    assert r.tool_args["query"] == "x"


def test_parse_alias_message():
    r = parse_agent_reply('{"message":"hi","action":"done"}')
    assert r.reply == "hi"
