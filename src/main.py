"""
main.py — Dokumentkollen
Ladda upp dokument, analysera, ladda ner som PDF. Inget sparas.

Kör med: streamlit run src/main.py
"""

import os
from datetime import datetime

import streamlit as st

# Stäng av TensorFlow-stöd i Transformers
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import rag
from utils import extract_pages, create_pdf_bytes

# -------------------------------------------------
# Grundinställningar
# -------------------------------------------------
TOP_K = 8  # antal textutdrag som skickas till LLM:en

ACCENT = "#2dd4bf"
BG = "#0e1420"
CARD = "#18202e"
TEXT = "#e6edf3"
MUTED = "#8a94a3"

# Flaggbilder (Twemoji, MIT/CC-BY-licensierade)
FLAG_SE = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f1f8-1f1ea.svg"
FLAG_GB = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f1ec-1f1e7.svg"

# All UI-text i båda språken
TEXTS = {
    "sv": {
        "app_name": "Dokumentkollen",
        "tagline": "FÖRSTÅ DINA DOKUMENT PÅ SEKUNDER",
        "empty": "Dra in ett eller flera dokument för att börja.",
        "pages": "sidor",
        "page": "sida",
        "reading": "Läser in",
        "read_error": "Kunde inte läsa",
        "placeholder": "Vad vill du veta om dokumenten?",
        "summarize": "✨ Sammanfatta",
        "compare": "⚖️ Jämför",
        "summary_title": "Sammanfattning",
        "compare_title": "Jämförelse",
        "analyzing": "Analyserar...",
        "sources": "Källor",
        "excerpts": "utdrag",
        "relevance": "relevans",
        "download": "⬇️ Ladda ner som PDF",
    },
    "en": {
        "app_name": "Document Check",
        "tagline": "UNDERSTAND YOUR DOCUMENTS IN SECONDS",
        "empty": "Drop one or more documents to get started.",
        "pages": "pages",
        "page": "page",
        "reading": "Reading",
        "read_error": "Could not read",
        "placeholder": "What do you want to know about the documents?",
        "summarize": "✨ Summarize",
        "compare": "⚖️ Compare",
        "summary_title": "Summary",
        "compare_title": "Comparison",
        "analyzing": "Analyzing...",
        "sources": "Sources",
        "excerpts": "excerpts",
        "relevance": "relevance",
        "download": "⬇️ Download as PDF",
    },
}

st.set_page_config(
    page_title="Dokumentkollen",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------
# Tema
# -------------------------------------------------
st.markdown(
    f"""
<style>
    #MainMenu, footer, header {{ visibility: hidden; }}
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none; }}

    .stApp {{ background: {BG}; }}
    .block-container {{ padding-top: 1.6rem; max-width: 780px; }}

    p, span, div, label, li {{ color: {TEXT} !important; }}
    hr {{ border-color: #232e40 !important; }}

    /* ── Språkflaggor uppe till höger ── */
    .st-key-langbar [data-testid="stButtonGroup"] {{
        justify-content: flex-end;
    }}
    .st-key-langbar [data-testid="stButtonGroup"] button {{
        background: {CARD} !important;
        border: 1px solid #2a3547 !important;
        border-radius: 100px !important;
        min-height: 0 !important;
        padding: 5px 12px 5px 32px !important;
        font-size: 0.78rem !important;
        color: {MUTED} !important;
        background-repeat: no-repeat !important;
        background-position: 10px center !important;
        background-size: 17px auto !important;
    }}
    .st-key-langbar [data-testid="stButtonGroup"] button:nth-of-type(1) {{
        background-image: url('{FLAG_SE}') !important;
    }}
    .st-key-langbar [data-testid="stButtonGroup"] button:nth-of-type(2) {{
        background-image: url('{FLAG_GB}') !important;
    }}
    .st-key-langbar button[data-testid="stBaseButton-pillsActive"],
    .st-key-langbar [data-testid="stButtonGroup"] button[aria-checked="true"] {{
        border-color: {ACCENT} !important;
    }}
    .st-key-langbar [data-testid="stButtonGroup"] button p {{
        color: {TEXT} !important;
        font-size: 0.78rem !important;
    }}

    /* Knappar */
    .stButton > button, .stDownloadButton > button,
    .stFormSubmitButton > button {{
        background: {CARD} !important;
        border: 1px solid #2a3547 !important;
        color: {TEXT} !important;
        border-radius: 10px !important;
        transition: border-color 0.2s;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: {ACCENT} !important;
        color: {ACCENT} !important;
    }}
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {{
        background: {ACCENT} !important;
        color: {BG} !important;
        border: none !important;
        font-weight: 700 !important;
    }}

    /* Inputs */
    .stTextInput input {{
        background: {CARD} !important;
        border: 1px solid #2a3547 !important;
        border-radius: 10px !important;
        color: {TEXT} !important;
        padding: 10px 14px;
    }}
    .stTextInput input:focus {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 1px {ACCENT}44 !important;
    }}
    .stTextInput input::placeholder {{ color: {MUTED} !important; }}

    /* Expanders */
    .stExpander {{
        background: {CARD} !important;
        border: 1px solid #2a3547 !important;
        border-radius: 12px !important;
    }}
    .stExpander summary {{
        color: {MUTED} !important;
        font-size: 0.85rem !important;
    }}
    .stExpander summary:hover {{ color: {ACCENT} !important; }}
    .stExpander summary p {{ color: inherit !important; }}

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {{
        background: {CARD} !important;
        border: 1px dashed #34455e !important;
        border-radius: 14px !important;
    }}
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small {{ color: {MUTED} !important; }}

    /* ── Kommandorad — fält och knappar i exakt samma höjd (52px) ── */
    .st-key-askbar [data-baseweb="input"],
    .st-key-askbar [data-baseweb="base-input"] {{
        height: 52px !important;
        border-radius: 14px !important;
    }}
    .st-key-askbar .stTextInput input {{
        height: 100% !important;
        font-size: 1rem !important;
        padding: 0 18px !important;
        border-radius: 14px !important;
        border: 1px solid #34455e !important;
    }}
    .st-key-askbar .stButton > button,
    .st-key-askbar [data-testid="stBaseButton-secondary"],
    .st-key-askbar [data-testid="stBaseButton-primary"] {{
        height: 52px !important;
        min-height: 52px !important;
        max-height: 52px !important;
        padding: 0 10px !important;
        border-radius: 14px !important;
        font-size: 0.92rem !important;
        white-space: nowrap !important;
    }}
    .st-key-askbar [data-testid="stTextInputRootElement"] {{
        border: none !important;
    }}

    /* ── Resultatvy — tidningslik typografi ── */
    .st-key-result {{
        background: {CARD};
        border: 1px solid #2a3547;
        border-radius: 14px;
        padding: 26px 30px 20px;
        margin-top: 8px;
    }}
    .st-key-result h1, .st-key-result h2 {{
        color: {ACCENT} !important;
        font-size: 1.08rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border-bottom: 1px solid #2a3547;
        padding-bottom: 6px !important;
        margin: 22px 0 10px !important;
    }}
    .st-key-result h1:first-child, .st-key-result h2:first-child {{
        margin-top: 0 !important;
    }}
    .st-key-result h3 {{
        color: {TEXT} !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin: 16px 0 6px !important;
    }}
    .st-key-result p, .st-key-result li {{
        font-size: 0.9rem !important;
        line-height: 1.7 !important;
        color: #c9d4e0 !important;
    }}
    .st-key-result li {{ margin-bottom: 4px !important; }}
    .st-key-result ul, .st-key-result ol {{ margin: 4px 0 12px !important; }}
    .st-key-result strong {{ color: {TEXT} !important; }}
    .st-key-result table {{ font-size: 0.85rem; border-collapse: collapse; }}
    .st-key-result th, .st-key-result td {{
        border: 1px solid #2a3547 !important;
        padding: 6px 10px !important;
    }}
    .st-key-result th {{ color: {ACCENT} !important; }}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Språkval — flaggor uppe till höger
# -------------------------------------------------
with st.container(key="langbar"):
    _, flag_col = st.columns([3, 1.2])
    with flag_col:
        flag = st.pills(
            "Språk", ["Svenska", "English"], selection_mode="single",
            default="Svenska", label_visibility="collapsed", key="lang_flag",
        )
lang = "en" if flag == "English" else "sv"
t = TEXTS[lang]

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    f"<div style='text-align:center'>"
    f"<div style='font-size:2.2rem'>🔎</div>"
    f"<div style='color:{ACCENT};font-size:1.7rem;font-weight:700'>{t['app_name']}</div>"
    f"<div style='color:{MUTED};font-size:0.62rem;letter-spacing:3px'>{t['tagline']}</div>"
    f"</div><br>",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "docs" not in st.session_state:
    st.session_state.docs = {}  # namn -> dokumentobjekt (chunks + embeddings i minnet)
if "result" not in st.session_state:
    st.session_state.result = None

# -------------------------------------------------
# 1. Ladda upp
# -------------------------------------------------
uploads = st.file_uploader(
    "Ladda upp dokument",
    type=["pdf", "docx", "txt", "md", "csv", "html"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

# Bearbeta nya filer, släpp borttagna
current_names = {up.name for up in (uploads or [])}
for name in list(st.session_state.docs.keys()):
    if name not in current_names:
        del st.session_state.docs[name]
        st.session_state.result = None

for up in uploads or []:
    if up.name not in st.session_state.docs:
        with st.spinner(f"{t['reading']} {up.name}..."):
            try:
                pages = extract_pages(up)
                st.session_state.docs[up.name] = rag.build_doc(up.name, pages)
            except Exception as exc:
                st.error(f"{t['read_error']} {up.name}: {exc}")

docs = list(st.session_state.docs.values())

if not docs:
    st.markdown(
        f"<div style='color:{MUTED};text-align:center;padding:28px 0;font-size:0.9rem'>"
        f"{t['empty']}</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# Kompakt dokumentrad
doc_line = "  ·  ".join(f"📄 {d['name']} ({d['pages']} {t['pages']})" for d in docs)
st.markdown(
    f"<div style='background:{CARD};border:1px solid #2a3547;border-radius:12px;"
    f"padding:10px 16px;margin:4px 0 18px;color:{MUTED};font-size:0.8rem;"
    f"text-align:center'>{doc_line}</div>",
    unsafe_allow_html=True,
)


# -------------------------------------------------
# 2. Fråga eller sammanfatta
# -------------------------------------------------
def _run(kind: str, title: str, fn, *args, **kwargs) -> None:
    with st.spinner(t["analyzing"]):
        answer, hits = fn(*args, **kwargs)
    st.session_state.result = {
        "type": kind, "title": title, "answer": answer, "hits": hits,
    }
    st.rerun()


# Kommandorad — åtgärder till vänster, frågefält, skicka. Enter skickar.
with st.container(key="askbar"):
    if len(docs) >= 2:
        col_sum, col_cmp, col_in, col_send = st.columns(
            [1.7, 1.4, 4.4, 0.7], vertical_alignment="center")
    else:
        col_sum, col_in, col_send = st.columns(
            [1.8, 5, 0.7], vertical_alignment="center")
        col_cmp = None

    with col_sum:
        do_summary = st.button(t["summarize"], use_container_width=True)
    do_compare = False
    if col_cmp is not None:
        with col_cmp:
            do_compare = st.button(t["compare"], use_container_width=True)
    with col_in:
        query = st.text_input(
            "Fråga", placeholder=t["placeholder"],
            label_visibility="collapsed", key="query_input",
        )
    with col_send:
        send = st.button("➤", type="primary", use_container_width=True)

# Enter i fältet ger en rerun med nytt värde — kör då automatiskt
new_query = query.strip() and query.strip() != st.session_state.get("done_query")

if do_summary:
    _run("summary", t["summary_title"],
         rag.summarize, docs, top_k=TOP_K, language=lang)
elif do_compare:
    _run("compare", t["compare_title"],
         rag.compare, docs, top_k=TOP_K, language=lang)
elif (send and query.strip()) or new_query:
    st.session_state.done_query = query.strip()
    _run("ask", query.strip(),
         rag.ask, query.strip(), docs, top_k=TOP_K, language=lang)

# -------------------------------------------------
# 3. Resultat
# -------------------------------------------------
res = st.session_state.result
if res:
    st.markdown("<div style='margin-top:26px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{MUTED};font-size:0.7rem;letter-spacing:2px;"
        f"text-transform:uppercase;margin-bottom:8px'>📋 {res['title']}</div>",
        unsafe_allow_html=True,
    )
    with st.container(key="result"):
        st.markdown(res["answer"])

    if res["hits"]:
        with st.expander(f"📚 {t['sources']} ({len(res['hits'])} {t['excerpts']})"):
            for hit in res["hits"]:
                st.markdown(
                    f"<div style='background:{BG};border:1px solid #2a3547;"
                    f"border-radius:10px;padding:10px 14px;margin-bottom:8px'>"
                    f"<div style='color:{ACCENT};font-size:0.75rem;margin-bottom:4px'>"
                    f"{hit['doc_name']} · {t['page']} {hit['page_no']} · "
                    f"{t['relevance']} {hit['score']:.0%}</div>"
                    f"<div style='color:{MUTED};font-size:0.8rem;line-height:1.5'>"
                    f"{hit['text'][:280]}…</div></div>",
                    unsafe_allow_html=True,
                )

    st.download_button(
        t["download"],
        data=create_pdf_bytes(res["title"], res["answer"]),
        file_name=f"dokumentkollen_{datetime.now():%Y%m%d_%H%M}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )