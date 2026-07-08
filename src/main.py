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
        "max_docs": "Max 2 dokument — de extra filerna hoppades över.",
        "drop_title": "Dra in dina dokument här",
        "drop_hint": "Max 2 dokument · PDF, DOCX, TXT, MD, CSV, HTML",
        "drop_title_full": "Max antal dokument uppnått",
        "drop_hint_full": "Ta bort ett dokument för att kunna ladda upp ett nytt",
        "browse": "Bläddra",
        "summarize": "✨ Sammanfatta",
        "compare": "⚖️ Jämför",
        "summary_title": "Sammanfattning",
        "compare_title": "Jämförelse",
        "analyzing": "Analyserar...",
        "based_on": "Baserat på",
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
        "max_docs": "Max 2 documents — the extra files were skipped.",
        "drop_title": "Drop your documents here",
        "drop_hint": "Max 2 documents · PDF, DOCX, TXT, MD, CSV, HTML",
        "drop_title_full": "Maximum number of documents reached",
        "drop_hint_full": "Remove a document to upload a new one",
        "browse": "Browse",
        "summarize": "✨ Summarize",
        "compare": "⚖️ Compare",
        "summary_title": "Summary",
        "compare_title": "Comparison",
        "analyzing": "Analyzing...",
        "based_on": "Based on",
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
    /* Göm uppladdarens egen fillista — vi visar egna kort istället */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderPagination"],
    [data-testid="stFileUploaderDeleteBtn"] {{ display: none !important; }}

    /* Ersätt dropzonens engelska texter med egna (sätts per språk nedan) */
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {{
        display: none !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] > div::before {{
        content: var(--drop-title);
        color: {MUTED};
        font-size: 0.95rem;
        display: block;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] > div::after {{
        content: var(--drop-hint);
        color: #5c6570;
        font-size: 0.75rem;
        display: block;
        margin-top: 2px;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        font-size: 0 !important;
        padding: 8px 16px !important;
    }}
    [data-testid="stFileUploaderDropzone"] button::after {{
        content: var(--browse-label);
        font-size: 0.85rem;
        color: {TEXT};
    }}

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
    .st-key-send_btn button {{
        color: {ACCENT} !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }}
    .st-key-send_btn button:hover {{
        border-color: {ACCENT} !important;
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

# Dropzonens texter styrs via CSS-variabler så de följer språkvalet
st.markdown(
    f"<style>:root {{ --drop-title: '{t['drop_title']}'; "
    f"--drop-hint: '{t['drop_hint']}'; "
    f"--browse-label: '{t['browse']}'; }}</style>",
    unsafe_allow_html=True,
)

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
MAX_DOCS = 2
if "removed_files" not in st.session_state:
    st.session_state.removed_files = set()

is_full = len(st.session_state.docs) >= MAX_DOCS

# Vid fullt: lås uppladdaren (oklickbar, nedtonad) och byt dess texter
if is_full:
    st.markdown(
        f"""<style>
        [data-testid="stFileUploaderDropzone"] {{
            pointer-events: none !important;
            opacity: 0.45 !important;
        }}
        :root {{
            --drop-title: '{t["drop_title_full"]}';
            --drop-hint: '{t["drop_hint_full"]}';
        }}
        </style>""",
        unsafe_allow_html=True,
    )

uploads = st.file_uploader(
    "Ladda upp dokument",
    type=["pdf", "docx", "txt", "md", "csv", "html"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
current_names = {up.name for up in (uploads or [])}
st.session_state.removed_files &= current_names

overflow = False
for up in uploads or []:
    if up.name in st.session_state.removed_files:
        continue
    if up.name in st.session_state.docs:
        continue
    if len(st.session_state.docs) >= MAX_DOCS:
        st.session_state.removed_files.add(up.name)
        overflow = True
        continue
    with st.spinner(f"{t['reading']} {up.name}..."):
        try:
            pages = extract_pages(up)
            st.session_state.docs[up.name] = rag.build_doc(up.name, pages)
        except Exception as exc:
            st.error(f"{t['read_error']} {up.name}: {exc}")
if overflow:
    st.warning(t["max_docs"])
if not is_full and len(st.session_state.docs) >= MAX_DOCS:
    st.rerun()  # lås uppladdaren direkt när gränsen nås

docs = list(st.session_state.docs.values())

# Dokumentkort — staplade under varandra, med borttagningsknapp
for d in docs:
    card_col, x_col = st.columns([8, 0.7], vertical_alignment="center")
    with card_col:
        st.markdown(
            f"<div style='background:{CARD};border:1px solid #2a3547;"
            f"border-radius:12px;padding:10px 16px;display:flex;"
            f"align-items:baseline;gap:10px'>"
            f"<span style='color:{TEXT};font-size:0.88rem;font-weight:500;"
            f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>"
            f"📄 {d['name']}</span>"
            f"<span style='color:{MUTED};font-size:0.72rem;flex-shrink:0'>"
            f"{d['pages']} {t['pages']}</span></div>",
            unsafe_allow_html=True,
        )
    with x_col:
        if st.button("✕", key=f"rm_{d['name']}", use_container_width=True):
            st.session_state.removed_files.add(d["name"])
            del st.session_state.docs[d["name"]]
            st.session_state.result = None
            st.rerun()
if docs:
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

if not docs:
    st.markdown(
        f"<div style='color:{MUTED};text-align:center;padding:28px 0;font-size:0.9rem'>"
        f"{t['empty']}</div>",
        unsafe_allow_html=True,
    )
    st.stop()



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
        send = st.button("➤", key="send_btn", use_container_width=True)

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
        # Gruppera per dokument: namn -> sorterade unika sidor
        by_doc: dict = {}
        for hit in res["hits"]:
            by_doc.setdefault(hit["doc_name"], set()).add(hit["page_no"])
        parts = []
        for name, page_set in by_doc.items():
            pages_sorted = sorted(page_set)
            label = t["page"] if len(pages_sorted) == 1 else t["pages"]
            page_str = ", ".join(str(p) for p in pages_sorted)
            parts.append(f"📄 {name} ({label} {page_str})")
        st.markdown(
            f"<div style='color:{MUTED};font-size:0.78rem;margin:10px 2px 14px'>"
            f"{t['based_on']}: {'  ·  '.join(parts)}</div>",
            unsafe_allow_html=True,
        )

    st.download_button(
        t["download"],
        data=create_pdf_bytes(res["title"], res["answer"]),
        file_name=f"dokumentkollen_{datetime.now():%Y%m%d_%H%M}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )