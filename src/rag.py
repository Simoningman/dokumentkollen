"""
rag.py — Dokumentkollen session engine
Allt lever i minnet under sessionen. Inget sparas till disk.

Flöde: build_doc() skapar ett dokumentobjekt (chunks + embeddings) från
extraherade sidor → search() hittar relevanta utdrag → analysfunktionerna
bygger prompt och anropar LLM:en.
"""

from dotenv import load_dotenv

load_dotenv()

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# Konfiguration
# -------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/LaBSE")
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_DEFAULT = int(os.getenv("TOP_K", "8"))
OUTPUT_LANGUAGE = os.getenv("OUTPUT_LANGUAGE", "sv")  # "sv" eller "en"

CHAT_URL = os.getenv("CHAT_URL")
MODEL_NAME = os.getenv("LLM_MODEL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

SNIPPET_MAX_LEN = 900

# Blandade sv/en-söktermer — LaBSE är flerspråkig och hittar
# relevant innehåll på båda språken.
SUMMARY_SEARCH_QUERY = (
    "sammanfattning syfte mål bakgrund beslut slutsats resultat huvudpunkter "
    "summary purpose goal background decision conclusion results key points"
)


# -------------------------------------------------
# Embeddings & chunkning
# -------------------------------------------------
def _get_embedder() -> SentenceTransformer:
    if not hasattr(_get_embedder, "_model"):
        _get_embedder._model = SentenceTransformer(EMBEDDING_MODEL)
    return _get_embedder._model


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    model = _get_embedder()
    embeddings = model.encode(
        texts, show_progress_bar=False, normalize_embeddings=True
    )
    return np.asarray(embeddings, dtype=np.float32)


def chunk_text(
    text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    if not text or not text.strip():
        return []
    words = text.split()
    chunks: List[str] = []
    step = max(1, max_tokens - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_tokens]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks


# -------------------------------------------------
# Sessionsdokument (i minnet, aldrig på disk)
# -------------------------------------------------
def build_doc(name: str, pages: List[Tuple[int, str]]) -> Dict[str, Any]:
    """Skapar ett sessionsdokument: chunks + embeddings i minnet.

    pages: lista av (sidnummer, text).
    """
    chunks: List[Dict[str, Any]] = []
    for page_no, text in pages:
        for chunk in chunk_text(text or ""):
            chunks.append({"page_no": page_no, "text": chunk})

    embeddings = embed_texts([c["text"] for c in chunks])
    full_text = "\n".join(text for _, text in pages if text)

    return {
        "name": name,
        "pages": len(pages),
        "chunks": chunks,
        "embeddings": embeddings,
        "full_text": full_text,
    }


def search(
    query: str,
    docs: List[Dict[str, Any]],
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:
    """Semantisk sökning över sessionsdokumenten. Returnerar bästa träffarna."""
    if not docs:
        return []

    query_vec = embed_texts([query])[0]
    hits: List[Dict[str, Any]] = []

    for doc in docs:
        if doc["embeddings"].shape[0] == 0:
            continue
        scores = doc["embeddings"] @ query_vec
        for chunk, score in zip(doc["chunks"], scores):
            hits.append(
                {
                    "doc_name": doc["name"],
                    "page_no": chunk["page_no"],
                    "text": chunk["text"],
                    "score": float(score),
                }
            )

    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


def _trim_snippet(text: str, max_len: int = SNIPPET_MAX_LEN) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _build_context(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""
    parts = []
    for hit in hits:
        parts.append(
            f"[Dokument: {hit['doc_name']} | sida {hit['page_no']} | "
            f"relevans {hit['score']:.3f}]\n{_trim_snippet(hit['text'])}"
        )
    return "\n\n".join(parts)


# -------------------------------------------------
# LLM
# -------------------------------------------------
def _clean_llm_output(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"^```(markdown)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)

    lines = text.splitlines()
    while lines:
        last = lines[-1].strip()
        incomplete = (
            last.endswith((",", ":", ";", "-", "–", "|", "och", "and", "eller", "or"))
            or (last.startswith(("-", "*", "|")) and len(last) <= 2)
        )
        if incomplete:
            lines.pop()
        else:
            break
    return "\n".join(lines).strip()


def call_llm(prompt: str) -> str:
    if not CHAT_URL:
        return "[Fel: CHAT_URL saknas i .env]"
    if not MODEL_NAME:
        return "[Fel: LLM_MODEL saknas i .env]"

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    try:
        response = requests.post(CHAT_URL, json=payload, headers=headers, timeout=90)
        if response.status_code != 200:
            return f"[LLM HTTP {response.status_code}] {response.text[:800]}"
        data = response.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )
        return _clean_llm_output(content)
    except Exception as exc:
        return f"[Fel vid API-anrop: {type(exc).__name__}: {exc}]"


# -------------------------------------------------
# Promptregler
# -------------------------------------------------
def _lang(language: Optional[str]) -> str:
    return language if language in ("sv", "en") else OUTPUT_LANGUAGE


def _rules(language: str) -> str:
    if language == "en":
        return """
- Use only information from the context.
- Be concrete, clear and factual. Do not guess.
- Write in English.
- Include figures, amounts, dates and percentages when present.
- If something is missing from the material, state it clearly.
- Write compactly, avoid repetition, prioritize the most important points.
- Adapt structure, headings and tone to the document type (e.g. contract, report, study, minutes, manual, CV).
- Keep each section short: max 2 short paragraphs or 3-5 bullets.
- When referring to a document, use its file name.
- End with complete sentences. Never write an incomplete bullet or row.
""".strip()
    return """
- Använd endast information från kontexten.
- Var konkret, tydlig och saklig. Gissa inte.
- Skriv på svenska.
- Ta med siffror, belopp, datum och procentsatser när de finns.
- Om något saknas i underlaget, skriv det tydligt.
- Skriv kompakt, undvik upprepningar, prioritera det viktigaste.
- Anpassa struktur, rubriker och ton efter dokumenttypen (t.ex. avtal, rapport, studie, protokoll, manual, CV).
- Håll varje sektion kort: max 2 korta stycken eller 3–5 punkter.
- När du hänvisar till ett dokument, använd dokumentets filnamn.
- Avsluta med kompletta meningar. Skriv aldrig en ofullständig punkt eller rad.
""".strip()


# -------------------------------------------------
# Analysfunktioner
# -------------------------------------------------
def summarize(
    docs: List[Dict[str, Any]],
    top_k: int = TOP_K_DEFAULT,
    language: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    language = _lang(language)
    hits = search(SUMMARY_SEARCH_QUERY, docs, top_k=top_k)
    context = _build_context(hits)

    if language == "en":
        prompt = f"""
You are a clear analyst who writes informative, factual and well-structured summaries.

TASK:
Summarize the material below. Highlight its purpose, main content, key conclusions
and the most significant points. If several documents are included, treat them as
one combined body of material and describe the overall picture.

IMPORTANT INSTRUCTIONS:
{_rules(language)}

CONTEXT (document excerpts):
{context}

OUTPUT FORMAT:
## Summary
1-2 short paragraphs capturing what the document is and says.

## Key points
3-6 bullets with the most essential information.

## Conclusion
1-2 sentences.

Only include sections and points that add value. Mention gaps in the material
only if they are significant, and then as one bullet under Key points.

ANSWER:
""".strip()
    else:
        prompt = f"""
Du är en tydlig analytiker som skriver informativa, sakliga och välstrukturerade sammanfattningar.

UPPGIFT:
Sammanfatta underlaget nedan. Lyft fram syfte, huvudinnehåll, viktiga slutsatser
och det mest betydelsefulla. Om flera dokument ingår, behandla dem som ett
gemensamt underlag och beskriv helhetsbilden.

VIKTIGA INSTRUKTIONER:
{_rules(language)}

KONTEXT (utdrag från dokument):
{context}

LEVERANSFORMAT:
## Sammanfattning
1–2 korta stycken som fångar vad dokumentet är och säger.

## Viktigaste punkterna
3–6 punkter med det mest väsentliga.

## Slutsats
1–2 meningar.

Ta bara med sektioner och punkter som tillför värde. Nämn brister i underlaget
endast om de är väsentliga, i så fall som en punkt under Viktigaste punkterna.

SVAR:
""".strip()

    return call_llm(prompt), hits


def analyze_custom(
    focus: str,
    docs: List[Dict[str, Any]],
    top_k: int = TOP_K_DEFAULT,
    language: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Fri analys: användarens eget fokus styr både sökning och prompt."""
    language = _lang(language)
    hits = search(focus, docs, top_k=top_k)
    context = _build_context(hits)

    if language == "en":
        prompt = f"""
You are a skilled analyst who writes clear, factual and well-structured analyses.

TASK:
Analyze the material below with the following focus, defined by the user:
"{focus}"

IMPORTANT INSTRUCTIONS:
{_rules(language)}

CONTEXT (document excerpts):
{context}

OUTPUT FORMAT:
## Quick overview
- Most important finding related to the focus
- What stands out most
- What should be followed up

## Analysis
Structure into suitable sections based on the focus.

## Conclusions
## Missing/unclear

ANSWER:
""".strip()
    else:
        prompt = f"""
Du är en skicklig analytiker som skriver tydliga, sakliga och välstrukturerade analyser.

UPPGIFT:
Analysera underlaget nedan med följande fokus, definierat av användaren:
"{focus}"

VIKTIGA INSTRUKTIONER:
{_rules(language)}

KONTEXT (utdrag från dokument):
{context}

LEVERANSFORMAT:
## Snabböversikt
- Viktigaste iakttagelsen kopplad till fokuset
- Det som sticker ut mest
- Vad som bör följas upp

## Analys
Strukturera i lämpliga sektioner utifrån fokuset.

## Slutsatser
## Saknas/oklart

SVAR:
""".strip()

    return call_llm(prompt), hits


def answer_query(
    query: str,
    docs: List[Dict[str, Any]],
    top_k: int = TOP_K_DEFAULT,
    language: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    language = _lang(language)
    hits = search(query, docs, top_k=top_k)
    context = _build_context(hits)

    if language == "en":
        prompt = f"""
You are a helpful assistant that answers strictly from the context.

INSTRUCTIONS:
{_rules(language)}
- Answer briefly and usefully: normally at most 6-10 short bullets or 3-6 short paragraphs.

CONTEXT (document excerpts):
{context}

QUESTION:
{query}

ANSWER:
""".strip()
    else:
        prompt = f"""
Du är en hjälpsam assistent som svarar endast utifrån kontexten.

INSTRUKTIONER:
{_rules(language)}
- Svara kort och användbart: normalt högst 6–10 korta punkter eller 3–6 korta stycken.

KONTEXT (utdrag från dokument):
{context}

FRÅGA:
{query}

SVAR:
""".strip()

    return call_llm(prompt), hits


def compare(
    docs: List[Dict[str, Any]],
    top_k: int = TOP_K_DEFAULT,
    language: Optional[str] = None,
    focus: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Jämför två eller flera dokument. Valfritt fokus styr jämförelsen."""
    language = _lang(language)
    query = focus.strip() if focus and focus.strip() else SUMMARY_SEARCH_QUERY

    per_doc_k = max(3, top_k // len(docs)) if docs else top_k
    all_hits: List[Dict[str, Any]] = []
    blocks: List[str] = []
    for index, doc in enumerate(docs, start=1):
        doc_hits = search(query, [doc], top_k=per_doc_k)
        all_hits.extend(doc_hits)
        blocks.append(
            f"DOKUMENT {index} – {doc['name']}:\n{_build_context(doc_hits)}"
        )
    documents_text = "\n\n".join(blocks)

    focus_line_sv = f'\nJämförelsen ska fokusera på: "{focus.strip()}".' if focus and focus.strip() else ""
    focus_line_en = f'\nThe comparison should focus on: "{focus.strip()}".' if focus and focus.strip() else ""

    if language == "en":
        prompt = f"""
You are a skilled analyst. Write a clear, structured and compact comparison
report between the documents below.{focus_line_en}

IMPORTANT INSTRUCTIONS:
{_rules(language)}
- Highlight differences and similarities concretely.
- If information is missing in one document, state it clearly.

DOCUMENT MATERIAL:
{documents_text}

OUTPUT FORMAT:
## Quick overview
- Main difference
- Most important similarity
- Area to follow up

## Overall assessment
## Key similarities
## Key differences
## What stands out most
## Conclusions
## Missing/unclear

ANSWER:
""".strip()
    else:
        prompt = f"""
Du är en skicklig analytiker. Skriv en tydlig, strukturerad och kompakt
jämförelserapport mellan dokumenten nedan.{focus_line_sv}

VIKTIGA INSTRUKTIONER:
{_rules(language)}
- Lyft fram skillnader och likheter konkret.
- Om information saknas i något dokument, skriv det tydligt.

DOKUMENTUNDERLAG:
{documents_text}

LEVERANSFORMAT:
## Snabböversikt
- Huvudskillnad
- Viktigaste likhet
- Område att följa upp

## Samlad bedömning
## Viktiga likheter
## Viktiga skillnader
## Det som sticker ut mest
## Slutsatser
## Saknas/oklart

SVAR:
""".strip()

    return call_llm(prompt), all_hits

def ask(
    text: str,
    docs: List[Dict[str, Any]],
    top_k: int = TOP_K_DEFAULT,
    language: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Fri inmatning: besvarar direkta frågor kort, ger strukturerad
    analys när inmatningen är ett tema/fokus."""
    language = _lang(language)
    hits = search(text, docs, top_k=top_k)
    context = _build_context(hits)

    if language == "en":
        prompt = f"""
You are a skilled analyst who answers strictly from the context.

THE USER WROTE:
"{text}"

TASK:
- If the input is a direct question: answer it clearly and concisely
  (at most 6-10 short bullets or 3-6 short paragraphs).
- If the input is a topic or focus area: write a short structured analysis
  with a few clear sections and a brief conclusion.
- Only include what adds value. Do not pad with truisms or empty sections.

IMPORTANT INSTRUCTIONS:
{_rules(language)}

CONTEXT (document excerpts):
{context}

ANSWER:
""".strip()
    else:
        prompt = f"""
Du är en skicklig analytiker som svarar endast utifrån kontexten.

ANVÄNDAREN SKREV:
"{text}"

UPPGIFT:
- Om inmatningen är en direkt fråga: besvara den tydligt och kort
  (högst 6–10 korta punkter eller 3–6 korta stycken).
- Om inmatningen är ett tema eller fokusområde: skriv en kort strukturerad
  analys med några tydliga sektioner och en kort slutsats.
- Ta bara med det som tillför värde. Fyll inte ut med självklarheter
  eller sektioner utan innehåll.

VIKTIGA INSTRUKTIONER:
{_rules(language)}

KONTEXT (utdrag från dokument):
{context}

SVAR:
""".strip()

    return call_llm(prompt), hits