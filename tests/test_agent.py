"""
Testar agentloopen utan riktig LLM eller embeddingmodell.
Kör: python -m pytest tests -q
"""
import json
import sys
import types

# Stubba src.rag så testet inte laddar LaBSE eller anropar API:t
fake_rag = types.ModuleType("src.rag")
fake_rag.CHAT_URL, fake_rag.MODEL_NAME, fake_rag.LLM_API_KEY = "x", "x", None


def _search(query, docs, top_k=4):
    return [
        {"doc_name": d["name"], "page_no": 1, "text": f"{d['name']}: Soliditet {d['sol']} %", "score": 0.9}
        for d in docs
    ][:top_k]


fake_rag.search = _search
fake_rag.summarize = lambda docs, **k: ("sammanfattning", [])
fake_rag.compare = lambda docs, **k: ("jämförelse", [])
sys.modules["src.rag"] = fake_rag

from src.agent import run_agent  # noqa: E402

DOCS = {
    "alfa.md": {"name": "alfa.md", "pages": 1, "sol": 38,
                "chunks": [{"page_no": 1, "text": "VD har ordet om året."},
                           {"page_no": 6, "text": "Femårsöversikt " + "x " * 300 + "Soliditet, % 38,0 36,1"}]},
    "beta.md": {"name": "beta.md", "pages": 1, "sol": 22,
                "chunks": [{"page_no": 2, "text": "Hyresintäkter 2 112 2 021"}]},
}


def call(i, name, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class ScriptedLLM:
    """Spelar upp förbestämda svar och sparar vad den fick se."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, messages, tools, final=False):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


def test_agent_uses_tools_then_answers():
    llm = ScriptedLLM([
        {"tool_calls": [call(1, "list_documents")]},
        {"tool_calls": [call(2, "search_documents", query="soliditet", document="alfa.md"),
                        call(3, "search_documents", query="soliditet", document="beta.md")]},
        {"content": "Alfa är stabilast (alfa.md, s. 1)."},
    ])
    r = run_agent("Vem är stabilast?", DOCS, chat=llm)

    assert r["stopped"] == "answered"
    assert r["steps"] == 3
    assert [t["tool"] for t in r["trace"]] == ["list_documents", "search_documents", "search_documents"]
    assert r["sources"] == [{"doc": "alfa.md", "page": 1}]  # bara citerade källor
    assert r["searched_pages"] == 2
    # Verktygsresultatet ska ha nått LLM:en
    last_input = llm.seen[-1]
    assert any(m["role"] == "tool" and "38" in m["content"] for m in last_input)


def test_unknown_document_is_reported_back_not_crash():
    llm = ScriptedLLM([
        {"tool_calls": [call(1, "search_documents", query="x", document="finns_inte.pdf")]},
        {"content": "Dokumentet finns inte."},
    ])
    r = run_agent("?", DOCS, chat=llm)
    assert r["trace"][0]["result"].startswith("FEL: Okänt dokument")
    assert r["stopped"] == "answered"


def test_unknown_tool_and_bad_json():
    bad = {"id": "c9", "type": "function", "function": {"name": "search_documents", "arguments": "{trasig"}}
    llm = ScriptedLLM([
        {"tool_calls": [call(1, "hacka_systemet"), bad]},
        {"content": "ok"},
    ])
    r = run_agent("?", DOCS, chat=llm)
    assert "finns inte" in r["trace"][0]["result"]
    assert "JSON" in r["trace"][1]["result"]


def test_calculate_tool():
    llm = ScriptedLLM([
        {"tool_calls": [call(1, "calculate", expression="(184-162)/162*100"),
                        call(2, "calculate", expression="7,7 - 1,7")]},
        {"content": "ok"},
        {"content": "ok"},  # skickas tillbaka en gång: calculate läser inget innehåll
    ])
    r = run_agent("?", DOCS, chat=llm)
    assert r["trace"][0]["result"].endswith("= 13.5802")
    assert r["trace"][1]["result"].endswith("= 6.0")


def test_calculate_blocks_code():
    from src.agent import ToolError, safe_eval
    import pytest

    for bad in ["__import__('os').system('dir')", "open('x')", "2**10000", "1/0", "abc"]:
        with pytest.raises(ToolError):
            safe_eval(bad)


def test_repeated_search_is_blocked():
    llm = ScriptedLLM([
        {"tool_calls": [call(1, "search_documents", query="Soliditet", document="alfa.md")]},
        {"tool_calls": [call(2, "search_documents", query="soliditet ", document="alfa.md")]},
        {"content": "ok"},
    ])
    r = run_agent("?", DOCS, chat=llm)
    assert "redan gjort exakt denna sökning" in r["trace"][1]["result"]


def test_keyword_search_finds_table_row_deep_in_chunk():
    llm = ScriptedLLM([
        {"tool_calls": [call(1, "keyword_search", term="soliditet"),
                        call(2, "keyword_search", term="Nettoomsättning", document="beta.md")]},
        {"content": "ok"},
    ])
    r = run_agent("?", DOCS, chat=llm)
    assert "38,0" in r["trace"][0]["result"] and '"page": 6' in r["trace"][0]["result"]
    assert "förekommer inte" in r["trace"][1]["result"]


def test_warnings_catch_false_claims():
    from src.agent import check_answer
    w = check_answer("Tillväxt -2,3 % (beräknat med calculate). Soliditet saknas i dokumentet.", [])
    assert len(w) == 3
    assert check_answer("Soliditet 41,3 % (fil.pdf, s. 35).", [{"tool": "keyword_search"}]) == []


def test_max_steps_forces_final_answer():
    loop = [{"tool_calls": [call(i, "list_documents")]} for i in range(3)]
    llm = ScriptedLLM(loop + [{"content": "Slut på steg."}])
    r = run_agent("?", DOCS, max_steps=3, chat=llm)
    assert r["stopped"] == "max_steps"
    assert r["answer"] == "Slut på steg."
    assert len(r["trace"]) == 3


def test_table_row_pairs_values_with_years():
    from src.agent import table_row
    text = "TIO ÅR Belopp i mkr 2024 2023 2022 Nettoomsättning 7 281 7 453 9 657 Soliditet, % 41,3 44,4 45,4 Skuld"
    pos = text.find("Soliditet")
    assert table_row(text, pos, len("Soliditet")) == "2024: 41,3 | 2023: 44,4 | 2022: 45,4"
    pos = text.find("Nettoomsättning")
    assert table_row(text, pos, len("Nettoomsättning")) == "2024: 7281 | 2023: 7453 | 2022: 9657"


def test_table_row_refuses_to_guess():
    from src.agent import table_row
    text = "2024 2023 2022 Soliditet, % 41,3 44,4 Skuld"  # 2 värden, 3 år
    assert table_row(text, text.find("Soliditet"), 9) is None
    assert table_row("Soliditet 41,3", 0, 9) is None  # ingen årtalsrubrik


def test_cited_accepts_page_without_filename():
    from src.agent import _cited
    sources = [{"doc_name": "ge.pdf", "page_no": 35}, {"doc_name": "bb.pdf", "page_no": 6}]
    assert _cited("Soliditet 41,3 % (s. 35)", sources) == [{"doc": "ge.pdf", "page": 35}]
    assert _cited("Omsättning (s. 3, 6)", sources) == [{"doc": "bb.pdf", "page": 6}]


def test_prompt_is_not_tied_to_annual_reports():
    from src.agent import SYSTEM_PROMPT, TOOLS
    text = (SYSTEM_PROMPT + json.dumps(TOOLS, ensure_ascii=False)).lower()
    for word in ["soliditet", "hyresintäkter", "nettoomsättning", "koncern"]:
        assert word not in text


def test_general_knowledge_only_when_allowed():
    llm = ScriptedLLM([{"content": "ok"}, {"content": "ok"}, {"content": "ok"}, {"content": "ok"}])
    run_agent("?", DOCS, chat=llm)
    run_agent("?", DOCS, chat=llm, allow_general=True)
    assert "Allmän kunskap" not in llm.seen[0][0]["content"]
    assert "Allmän kunskap" in llm.seen[2][0]["content"]


def test_answer_without_reading_is_sent_back_once():
    llm = ScriptedLLM([
        {"tool_calls": [call(1, "list_documents")]},
        {"content": "Dokumentet verkar vara ett CV."},          # gissning från filnamn
        {"tool_calls": [call(2, "summarize_document", document="alfa.md")]},
        {"content": "Sammanfattning (alfa.md, s. 1)."},
    ])
    r = run_agent("Vad handlar dokumenten om?", DOCS, chat=llm)
    tools = [t["tool"] for t in r["trace"]]
    assert tools == ["list_documents", "guardrail", "summarize_document"]
    assert r["answer"] == "Sammanfattning (alfa.md, s. 1)."


def test_guardrail_only_once():
    llm = ScriptedLLM([{"content": "gissning"}, {"content": "gissning igen"}])
    r = run_agent("?", DOCS, chat=llm)
    assert r["answer"] == "gissning igen" and r["stopped"] == "answered"
