"""
agent.py – Dokumentkollens agent.

Byggd utan ramverk: function calling + en while-loop.
LLM:en får ett mål, väljer själv verktyg, läser resultatet och
bestämmer nästa steg – tills den kan svara eller når max antal steg.

Flöde per steg:
    1. Skicka konversationen + verktygslistan till LLM:en
    2. Vill den anropa verktyg? -> kör dem, lägg resultatet i konversationen, loopa
    3. Svarar den med text? -> klart
"""

import ast
import json
import operator
import os
import re
from typing import Any, Callable, Dict, List, Optional

import requests

import sys

# Streamlit-appen importerar "rag", terminalen/testerna "src.rag". Återanvänd den som redan
# är laddad – annars laddas LaBSE två gånger (dubbelt minne, kan krascha Streamlit Cloud).
if "rag" in sys.modules:
    from rag import CHAT_URL, LLM_API_KEY, MODEL_NAME, compare, search, summarize
else:
    from src.rag import CHAT_URL, LLM_API_KEY, MODEL_NAME, compare, search, summarize

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))
RESULT_MAX_CHARS = 4000  # håller kontexten liten

SYSTEM_PROMPT = """Du är Dokumentkollens analysagent. Du besvarar frågor om användarens
inlästa dokument – vilken typ som helst: avtal, rapporter, manualer, protokoll, CV, policyer,
artiklar – genom att använda verktygen.

Arbetssätt:
- Börja med list_documents om du inte vet vilka dokument som finns.
- Välj sökverktyg efter vad du letar efter:
  * keyword_search: exakta saker – namn, siffror, belopp, datum, paragrafnummer,
    produktkoder, termer och tabellrader.
  * search_documents: begrepp och innehåll uttryckt med andra ord – syfte, risker,
    villkor, åsikter, "vad säger dokumentet om …".
  Är du osäker: prova båda.
- Sök med ord som troligen står i dokumentet, inte bara med frågans ord.
  Tänk på synonymer och fackord (t.ex. "uppsägning" / "avtalets upphörande").
- Sök per dokument när du jämför dokument.
- Om en sökning inte ger svaret: byt ord helt eller byt verktyg. Upprepa aldrig samma sökning.
- Om en träff har fältet "table": använd det, det parar värden med rätt årtal.
  Ta aldrig en siffra ur en tabell om du inte säkert ser vilken rad och kolumn den hör till.
- Räkna aldrig i huvudet, inte ens enkla skillnader. Använd calculate för varje beräkning.
- Skillnad mellan två procentsatser anges i procentenheter.
- Säg aldrig att du använt ett verktyg som du inte anropat.

Svaret:
- Påståenden från dokumenten anges med källa: (exakt filnamn, s. X).
- Egna slutsatser markeras "Bedömning:" och får inte påstå något som dokumentet inte säger.
- Hittar du inte något: skriv "hittades inte i sökningarna", aldrig "saknas i dokumentet".
- Svara på samma språk som frågan, kort och strukturerat. Anpassa formen efter frågan:
  en enkel fråga får ett kort svar, inte rubriker."""

GENERAL_KNOWLEDGE_RULE = """
Allmän kunskap är tillåten när dokumenten inte räcker eller frågan inte gäller dem.
Märk då det tydligt med "Allmän kunskap (ej från dokumenten):" och ange aldrig sidkälla för det.
Blanda aldrig ihop det med vad dokumenten säger."""


# -------------------------------------------------
# Verktyg (schema som LLM:en ser)
# -------------------------------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "Lista inlästa dokument (filnamn och antal sidor).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Semantisk sökning. Returnerar textutdrag med filnamn, sida och relevans.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Vad du letar efter, med egna ord."},
                    "document": {"type": "string", "description": "Valfritt: sök bara i detta filnamn."},
                    "top_k": {"type": "integer", "description": "Antal utdrag (1-10). Standard 4."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": "Sammanfatta ett dokument. Dyrt – använd bara när en översikt behövs.",
            "parameters": {
                "type": "object",
                "properties": {"document": {"type": "string", "description": "Filnamn."}},
                "required": ["document"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_documents",
            "description": "Färdig jämförelserapport mellan dokument. Dyrt – föredra riktade sökningar för enskilda siffror.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {"type": "string", "description": "Tema, t.ex. 'risker'."},
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Valfritt: filnamn att jämföra. Standard: alla.",
                    },
                },
                "required": ["focus"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": (
                "Exakt ordsökning (som Ctrl+F). Bäst för namn, siffror, datum, paragrafer, "
                "termer och tabellrader. Ger fältet 'table' när en rad hör till en årtalstabell."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "description": "Ord eller fras som står ordagrant i texten."},
                    "document": {"type": "string", "description": "Valfritt: bara detta filnamn."},
                    "max_hits": {"type": "integer", "description": "Antal träffar (1-8). Standard 4."},
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Räknar ut ett matematiskt uttryck exakt. Tillåtet: tal, + - * / ** % och parenteser. "
                "Använd decimalpunkt. Exempel: '(184-162)/162*100' för tillväxt i procent."
            ),
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Uttrycket, t.ex. '38-22'."}},
                "required": ["expression"],
            },
        },
    },
]


# -------------------------------------------------
# Verktygsimplementationer
# -------------------------------------------------
class ToolError(Exception):
    """Fel som skickas tillbaka till LLM:en så den kan rätta sig."""


def _pick(docs: Dict[str, Dict[str, Any]], names: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not names:
        return list(docs.values())
    missing = [n for n in names if n not in docs]
    if missing:
        raise ToolError(f"Okänt dokument: {', '.join(missing)}. Finns: {', '.join(docs)}")
    return [docs[n] for n in names]


def _hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for h in hits:
        row = {"doc": h["doc_name"], "page": h["page_no"], "score": round(h["score"], 3)}
        if h.get("table"):
            row["table"] = h["table"]  # tolkad tabellrad först – det är den LLM:en ska använda
        row["text"] = h["text"][:700]
        out.append(row)
    return out


_YEAR_RUN = re.compile(r"(?:(?:19|20)\d{2}\s+){2,}(?:19|20)\d{2}")
_VALUE = re.compile(r"-?\d+(?:,\d+)?")
_WORD = re.compile(r"[A-Za-zÅÄÖåäöÉé]{2,}")


def table_row(text: str, pos: int, term_len: int) -> Optional[str]:
    """Parar ihop en tabellrad med närmaste årtalsrubrik ovanför.

    PDF-text tappar kolumnerna: 'Belopp i mkr 2024 2023 … 2015 … Soliditet, % 41,3 44,4 … 41,2'.
    Utan årtal tar LLM:en lätt fel kolumn. Returnerar t.ex. '2024: 41,3 | 2023: 44,4 | …'
    eller None om antalet värden inte matchar antalet årtal (då gissar vi inte).
    """
    headers = [m for m in _YEAR_RUN.finditer(text, 0, pos)]
    if not headers:
        return None
    years = headers[-1].group().split()

    rest = text[pos + term_len:]
    stop = _WORD.search(rest)
    values = _VALUE.findall(rest[: stop.start()] if stop else rest)
    if len(values) != len(years):
        values = _merge_thousands(values)  # '7 281 7 453' -> ['7281', '7453']
    if len(values) != len(years):
        return None  # osäkert – hellre ingen tolkning än fel år
    return " | ".join(f"{y}: {v}" for y, v in zip(years, values))


def _merge_thousands(tokens: List[str]) -> List[str]:
    """Slår ihop svenska tusentalsgrupper: ['2', '112', '-1', '139'] -> ['2112', '-1139']."""
    out: List[str] = []
    open_group = False  # får senaste värdet en tusentalsgrupp till?
    for t in tokens:
        if open_group and len(t) == 3 and t.isdigit():
            out[-1] += t
            open_group = False  # max en grupp: värden upp till 999 999
        else:
            out.append(t)
            open_group = bool(re.fullmatch(r"-?\d{1,3}", t))
    return out


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Räknar ut ett uttryck utan eval(): bara tal och räknesätt, ingen kod kan köras."""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("För stor exponent.")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ToolError("Otillåtet uttryck. Använd bara tal och + - * / ** % ( ).")

    try:
        tree = ast.parse(expression.replace(",", ".").replace("−", "-"), mode="eval")
    except SyntaxError:
        raise ToolError(f"Kunde inte tolka uttrycket: {expression}")
    try:
        return walk(tree.body)
    except ZeroDivisionError:
        raise ToolError("Division med noll.")


def build_tools(docs: Dict[str, Dict[str, Any]], sources: List[Dict[str, Any]]) -> Dict[str, Callable[..., Any]]:
    """Kopplar verktygsnamn till funktioner. `sources` samlar alla träffar för källistan."""

    def list_documents() -> Any:
        return [{"name": d["name"], "pages": d["pages"]} for d in docs.values()]

    tried: set = set()

    def search_documents(query: str, document: Optional[str] = None, top_k: int = 4) -> Any:
        selected = _pick(docs, [document] if document else None)
        key = (" ".join(query.lower().split()), document)
        if key in tried:
            raise ToolError(
                "Du har redan gjort exakt denna sökning. Byt till synonymer eller fackord, "
                "eller prova keyword_search med ett exakt ord som borde stå i texten."
            )
        tried.add(key)
        # Minst 4 träffar: med top_k=1 räcker det att fel chunk vinner för att missa svaret
        hits = search(query, selected, top_k=max(4, min(int(top_k), 10)))
        sources.extend(hits)
        return _hits(hits) or "Inga träffar."

    def summarize_document(document: str) -> Any:
        text, hits = summarize(_pick(docs, [document]))
        sources.extend(hits)
        return text

    def compare_documents(focus: str, documents: Optional[List[str]] = None) -> Any:
        selected = _pick(docs, documents)
        if len(selected) < 2:
            raise ToolError("Minst två dokument krävs för jämförelse.")
        text, hits = compare(selected, focus=focus)
        sources.extend(hits)
        return text

    def keyword_search(term: str, document: Optional[str] = None, max_hits: int = 4) -> Any:
        """Exakt ordsökning. Hittar tabeller och nyckeltal som semantisk sökning missar."""
        needle = term.lower().strip()
        if len(needle) < 3:
            raise ToolError("Söktermen måste ha minst 3 tecken.")
        found = []
        for doc in _pick(docs, [document] if document else None):
            for chunk in doc["chunks"]:
                low = chunk["text"].lower()
                count = low.count(needle)
                if count:
                    pos, table = low.find(needle), None
                    for m in re.finditer(re.escape(needle), low):  # första förekomsten som är en tabellrad
                        table = table_row(chunk["text"], m.start(), len(needle))
                        if table:
                            pos = m.start()
                            break
                    start = max(0, pos - 150)  # utdrag centrerat kring träffen
                    hit = {"doc_name": doc["name"], "page_no": chunk["page_no"],
                           "text": chunk["text"][start:start + 700], "score": float(count) + (5 if table else 0)}
                    if table:
                        hit["table"] = table
                    found.append(hit)
        found.sort(key=lambda h: h["score"], reverse=True)
        found = found[: max(1, min(int(max_hits), 8))]
        sources.extend(found)
        return _hits(found) or f"Ordet '{term}' förekommer inte i texten."

    def calculate(expression: str) -> Any:
        result = safe_eval(expression)
        return f"{expression} = {round(result, 4)}"

    return {
        "list_documents": list_documents,
        "search_documents": search_documents,
        "summarize_document": summarize_document,
        "compare_documents": compare_documents,
        "keyword_search": keyword_search,
        "calculate": calculate,
    }


# -------------------------------------------------
# LLM-anrop med verktyg
# -------------------------------------------------
def _chat(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], final: bool = False) -> Dict[str, Any]:
    """Ett anrop till chat-API:t (OpenAI-format, fungerar mot Mistral). Returnerar assistant-meddelandet."""
    if not CHAT_URL or not MODEL_NAME:
        raise RuntimeError("CHAT_URL eller LLM_MODEL saknas i .env")

    payload: Dict[str, Any] = {"model": MODEL_NAME, "messages": messages, "temperature": 0.1}
    if not final:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    r = requests.post(CHAT_URL, json=payload, headers=headers, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:500]}")
    return r.json()["choices"][0]["message"]


def _run_tool(registry: Dict[str, Callable[..., Any]], name: str, raw_args: Any) -> str:
    """Kör ett verktyg. Alla fel blir text som LLM:en kan läsa och agera på."""
    if name not in registry:
        return f"FEL: verktyget '{name}' finns inte."
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except json.JSONDecodeError:
        return "FEL: argumenten var inte giltig JSON."
    try:
        result = registry[name](**args)
    except ToolError as e:
        return f"FEL: {e}"
    except TypeError as e:
        return f"FEL: fel argument: {e}"
    except Exception as e:  # verktyget kraschade – låt agenten veta, krascha inte loopen
        return f"FEL: {type(e).__name__}: {e}"
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return text[:RESULT_MAX_CHARS]


# -------------------------------------------------
# Agentloopen
# -------------------------------------------------
def run_agent(
    question: str,
    docs: Dict[str, Dict[str, Any]],
    max_steps: int = MAX_STEPS,
    on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    chat: Callable[..., Dict[str, Any]] = _chat,
    allow_general: bool = False,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Kör agenten på en fråga.

    Returnerar {"answer", "trace", "sources", "steps", "stopped"}.
    `on_step` anropas efter varje verktygsanrop (för loggning/UI).
    `chat` kan bytas ut i tester.
    `allow_general`: tillåt allmän kunskap utöver dokumenten (tydligt märkt).
    `language`: "sv" eller "en" tvingar svarsspråk (annars frågans språk).
    """
    sources: List[Dict[str, Any]] = []
    registry = build_tools(docs, sources)
    trace: List[Dict[str, Any]] = []
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT
            + (GENERAL_KNOWLEDGE_RULE if allow_general else "")
            + {"sv": "\nSvara på svenska.", "en": "\nAnswer in English."}.get(language or "", "")},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        msg = chat(messages, TOOLS)
        calls = msg.get("tool_calls") or []

        if not calls:  # LLM:en svarar -> klart
            return _result(msg.get("content") or "", trace, sources, step, "answered")

        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments", "{}")
            output = _run_tool(registry, name, args)
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": output})
            event = {
                "step": step, "tool": name, "args": args, "result": output[:300],
                # Strukturerat för UI:t (räknat på hela resultatet, inte det avkortade)
                "ok": not output.startswith("FEL"),
                "pages": sorted({int(p) for p in re.findall(r'"page": (\d+)', output)}),
                "table": '"table"' in output,
            }
            trace.append(event)
            if on_step:
                on_step(event)

    # Maxsteg nått: tvinga fram ett svar utan fler verktyg
    messages.append({
        "role": "user",
        "content": "Du har nått max antal steg. Svara nu med det du har hittat och säg vad som saknas.",
    })
    msg = chat(messages, TOOLS, final=True)
    return _result(msg.get("content") or "", trace, sources, max_steps, "max_steps")


def _cited(answer: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Källor som svaret faktiskt hänvisar till: (filnamn, s. X) som också fanns i en sökträff."""
    seen, out = set(), []
    for h in sources:
        key = (h["doc_name"], h["page_no"])
        if key in seen:
            continue
        seen.add(key)
        page = rf"s\.\s*(?:\d+\s*,\s*)*{h['page_no']}(?!\d)"  # klarar även 's. 3, 46'
        with_name = rf"{re.escape(h['doc_name'])}[^)\n]*?{page}"
        # Bara '(s. 35)': godkänn om sidan bara förekommer i ett dokument bland träffarna
        unique = sum(1 for d, p in {(x["doc_name"], x["page_no"]) for x in sources} if p == h["page_no"]) == 1
        if re.search(with_name, answer) or (unique and re.search(page, answer)):
            out.append({"doc": h["doc_name"], "page": h["page_no"]})
    return out


def check_answer(answer: str, trace: List[Dict[str, Any]]) -> List[str]:
    """Enkla kontroller av svaret mot vad agenten faktiskt gjorde (guardrails)."""
    used = {t["tool"] for t in trace}
    low = answer.lower()
    warnings = []
    if "calculate" in low and "calculate" not in used:
        warnings.append("Svaret påstår att calculate användes, men verktyget anropades aldrig.")
    if "saknas i dokumentet" in low:
        warnings.append("Svaret säger 'saknas i dokumentet' – agenten kan bara veta att den inte hittade det.")
    if re.search(r"\d+[,.]?\d*\s*%", answer) and "calculate" not in used and re.search(r"tillväxt|ökning|minskning|skillnad", low):
        warnings.append("Svaret innehåller en beräknad förändring utan att calculate användes.")
    return warnings


def _result(answer: str, trace, sources, steps: int, stopped: str) -> Dict[str, Any]:
    answer = answer.strip()
    return {
        "answer": answer,
        "trace": trace,
        "warnings": check_answer(answer, trace),
        "sources": _cited(answer, sources),  # bara det som citeras
        "searched_pages": len({(h["doc_name"], h["page_no"]) for h in sources}),
        "steps": steps,
        "stopped": stopped,
    }
