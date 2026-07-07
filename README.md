# 🔎 Dokumentkollen

Understand your documents in seconds. Upload one or more documents, ask questions in plain language, get structured answers with source citations — then download the result as a PDF. Nothing is ever stored: everything lives in memory for your session only.

**Fully bilingual** — switch between Swedish and English with one click, and both the interface and the AI's answers follow.

## Features

- **📄 Multi-format support** — PDF (via Docling with PyMuPDF fallback), Word, text, Markdown, CSV, HTML
- **❓ Ask anything** — free-text questions or analysis focus ("what are the risks?", "summarize the costs")
- **✨ One-click summary** — structured summary with key points and conclusion
- **⚖️ Document comparison** — upload 2+ documents and compare them
- **📚 Source citations** — every answer shows which document and page it came from, with relevance scores
- **⬇️ PDF export** — download any analysis as a formatted PDF report
- **🔒 Session-only** — documents are processed entirely in memory; close the tab and everything is gone

## How it works (RAG architecture)

1. **Extract** — text is pulled from uploaded files page by page (Docling for PDFs)
2. **Chunk & embed** — text is split into overlapping chunks and embedded with a multilingual sentence-transformer (LaBSE), enabling cross-language search
3. **Retrieve** — your question is embedded and matched against the chunks via cosine similarity
4. **Generate** — the top excerpts are sent to an LLM with a language-aware prompt, and the answer cites its sources

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/Simoningman/dokumentkollen.git
cd dokumentkollen
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`) pointing at any OpenAI-compatible chat endpoint — a local Ollama server, OpenAI, or another provider:

```env
CHAT_URL=http://localhost:11434/v1/chat/completions
LLM_MODEL=llama3.1
```

Then run:

```bash
streamlit run src/main.py
```

## Background

Dokumentkollen is a generalized, rebuilt version of a RAG-based document analysis tool I developed during my internship (LIA) at Göteborgsregionen — redesigned from an internal tool for public-sector documents into a session-based product for any document type, with a bilingual interface and no data persistence.

## Author

**Simon Ingman** — Python/AI developer, Göteborg
[GitHub](https://github.com/Simoningman) · Also on GitHub: [Cocktail House](https://github.com/Simoningman/cocktail-house-flutter) (Flutter app on Google Play)