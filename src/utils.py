"""
utils.py — DocLens
Textextraktion (PDF via Docling/PyMuPDF, övriga format) och PDF-export.
Ingen databas — allt hanteras i minnet under sessionen.
"""

import io
import os
import re
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd
from docling.document_extractor import DocumentExtractor

# -------------------------------------------------
# PDF-export (ReportLab) – Platypus
# -------------------------------------------------
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth



def basic_text_stats(text: str) -> dict:
    text = text or ""
    return {
        "tecken": len(text),
        "ord": len(text.split()),
        "rader": len([ln for ln in text.splitlines() if ln.strip()]),
    }


def page_count(pdf_bytes: bytes) -> int:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count


def is_pdf_bytes(data: bytes) -> bool:
    """Kolla om byte-strömmen ser ut som en riktig PDF (magic header)."""
    if not isinstance(data, (bytes, bytearray)):
        return False
    return data.startswith(b"%PDF-")



# -------------------------------------------------
# Text-extraktion
# -------------------------------------------------
def iter_pdf_pages(pdf_bytes: bytes) -> Iterable[Tuple[int, str]]:
    """Fallback: extrahera text per sida med PyMuPDF."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            yield i, page.get_text()


def extract_pdf_with_docling(pdf_bytes: bytes):
    """Extrahera text per sida med Docling (bättre än PyMuPDF)."""
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes)
    pages = []
    for page in result.document.pages:
        pages.append((page.number, page.text))
    return pages


def pdf_page_png_highlight(
    pdf_bytes: bytes, page_no: int, query: str = "", scale: float = 2.0
) -> bytes:
    """Rendera en PDF-sida som PNG (vi skippar highlight för nu)."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_no - 1]
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")


def read_any(upload) -> str:
    """
    Enkel läsare för icke-PDF-format.
    - TXT/MD → text
    - DOCX (eller ZIP-liknande) → python-docx
    - CSV → texttabell
    - HTML → rå HTML
    - Övrigt → bästa möjliga dekodning
    """
    name = upload.name.lower()
    data = upload.getvalue()

    if name.endswith(".txt") or name.endswith(".md"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".docx") or data.startswith(b"PK"):
        try:
            import docx  # type: ignore
            from io import BytesIO

            doc = docx.Document(BytesIO(data))
            parts = []
            for para in doc.paragraphs:
                t = para.text.strip()
                if t:
                    parts.append(t)
            return "\n".join(parts)
        except Exception:
            return data.decode("utf-8", errors="ignore")

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(data))
            return df.to_csv(index=False)
        except Exception:
            return data.decode("utf-8", errors="ignore")

    if name.endswith(".html") or name.endswith(".htm"):
        return data.decode("utf-8", errors="ignore")

    return data.decode("utf-8", errors="ignore")



# -------------------------------------------------
# Sidextraktion (ingen lagring)
# -------------------------------------------------
def extract_pages(upload) -> List[Tuple[int, str]]:
    """Extraherar text per sida från en uppladdad fil, helt i minnet.

    - Riktiga PDF:er → Docling (fallback: PyMuPDF)
    - Övriga format (docx, txt, md, csv, html) → read_any() som en sida
    """
    raw = upload.getvalue()

    if is_pdf_bytes(raw) and upload.name.lower().endswith(".pdf"):
        try:
            pages = extract_pdf_with_docling(raw)
            if pages and any(text.strip() for _, text in pages):
                return pages
        except Exception:
            pass
        return list(iter_pdf_pages(raw))

    text = read_any(upload)
    return [(1, text)]


# -------------------------------------------------
# PDF-export – säker rendering av Markdown-ish text
# -------------------------------------------------
def _escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _replace_code_spans(text: str) -> str:
    """
    Ersätter `kod` med ReportLab-vänlig Courier-text.
    Texten förutsätts redan vara HTML-escapad.
    """
    parts: List[str] = []
    in_code = False
    buf = ""

    for ch in text:
        if ch == "`":
            if in_code:
                parts.append(f"<font face='Courier'>{buf}</font>")
                buf = ""
                in_code = False
            else:
                if buf:
                    parts.append(buf)
                buf = ""
                in_code = True
        else:
            buf += ch

    if buf:
        if in_code:
            parts.append(f"<font face='Courier'>{buf}</font>")
        else:
            parts.append(buf)

    return "".join(parts)


def _safe_bold_italic(text: str) -> str:
    """
    Säker markdown-liknande inline-formattering för ReportLab.
    Hanterar:
    - **fet**
    - *kursiv*
    Undviker aggressiv regex som kan skapa trasiga taggar.
    """
    result: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        if text[i:i + 2] == "**":
            end = text.find("**", i + 2)
            if end != -1:
                inner = text[i + 2:end]
                if inner.strip():
                    result.append(f"<b>{inner}</b>")
                    i = end + 2
                    continue

        if text[i] == "*":
            if i + 1 < n and text[i + 1] == "*":
                result.append(text[i])
                i += 1
                continue

            end = text.find("*", i + 1)
            if end != -1:
                inner = text[i + 1:end]
                if inner.strip():
                    result.append(f"<i>{inner}</i>")
                    i = end + 1
                    continue

        result.append(text[i])
        i += 1

    return "".join(result)


def _md_inline_to_rl(s: str) -> str:
    """
    Säker inline-konvertering till ReportLab Paragraph markup.
    Stöd:
    - **bold**
    - *italic*
    - `code`
    """
    s = _escape_html(s)
    s = _replace_code_spans(s)
    s = _safe_bold_italic(s)
    return s


def _safe_paragraph(text: str, style):
    """
    Försöker skapa Paragraph med markdown-liknande formattering.
    Om ReportLab ändå inte klarar markupen, fallback till ren escaped text.
    """
    try:
        return Paragraph(_md_inline_to_rl(text), style)
    except Exception:
        return Paragraph(_escape_html(text or ""), style)


def _is_table_line(line: str) -> bool:
    line = (line or "").strip()
    return line.startswith("|") and line.endswith("|") and "|" in line[1:-1]


def _is_table_sep(line: str) -> bool:
    line = (line or "").strip()
    if not _is_table_line(line):
        return False
    inner = line.strip("|").strip()
    return all(ch in "-: " or ch == "|" for ch in inner)


def _parse_table(block_lines: List[str]) -> Optional[List[List[str]]]:
    """
    Tar en markdown-tabellblock (inkl header + separator + rader)
    och returnerar matris av celler.
    """
    if len(block_lines) < 2:
        return None

    header = block_lines[0].strip().strip("|")
    sep = block_lines[1]
    if not _is_table_sep(sep):
        return None

    headers = [c.strip() for c in header.split("|")]
    rows: List[List[str]] = [headers]

    for ln in block_lines[2:]:
        ln = ln.strip()
        if not _is_table_line(ln):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        while len(cells) < len(headers):
            cells.append("")
        rows.append(cells[: len(headers)])

    return rows


def _footer(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 9)

    left = 18 * mm
    right = A4[0] - 18 * mm
    y = 12 * mm

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    canvas_obj.drawString(left, y, f"Skapad: {stamp}")

    page_txt = f"Sida {doc.page}"
    w = stringWidth(page_txt, "Helvetica", 9)
    canvas_obj.drawString(right - w, y, page_txt)

    canvas_obj.restoreState()


def create_pdf_bytes(title: str, text: str) -> bytes:
    """
    Skapar en snyggare PDF (A4) från analys-text (markdown-ish) och returnerar bytes.
    - Renderar rubriker, listor, stycken och markdown-tabeller.
    - Lägger sidnummer + timestamp i footer.
    - Har fallback om inline-formattering blir ogiltig för ReportLab.
    """
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="H1",
            parent=styles["Heading1"],
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2",
            parent=styles["Heading2"],
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            leading=14,
            spaceAfter=6,
        )
    )

    story: List = []
    story.append(_safe_paragraph(title, styles["H1"]))
    story.append(Spacer(1, 8))

    raw_lines = (text or "").splitlines()
    i = 0

    while i < len(raw_lines):
        line = raw_lines[i].rstrip("\n")
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("### "):
            story.append(_safe_paragraph(stripped[4:], styles["H2"]))
            story.append(Spacer(1, 4))
            i += 1
            continue

        if stripped.startswith("## "):
            story.append(_safe_paragraph(stripped[3:], styles["H2"]))
            story.append(Spacer(1, 4))
            i += 1
            continue

        if stripped.startswith("# "):
            story.append(_safe_paragraph(stripped[2:], styles["H2"]))
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Tabell-block
        if _is_table_line(stripped):
            block = [stripped]
            j = i + 1
            while j < len(raw_lines) and raw_lines[j].strip():
                ln = raw_lines[j].strip()
                if _is_table_line(ln) or _is_table_sep(ln):
                    block.append(ln)
                    j += 1
                else:
                    break

            table_data = _parse_table(block)
            if table_data:
                cell_data = []
                for row in table_data:
                    cell_row = []
                    for cell in row:
                        cell_row.append(_safe_paragraph(cell, styles["Body"]))
                    cell_data.append(cell_row)

                tbl = Table(cell_data, hAlign="LEFT")
                tbl_style = TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                        ("TOPPADDING", (0, 0), (-1, 0), 6),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
                tbl.setStyle(tbl_style)
                story.append(tbl)
                story.append(Spacer(1, 10))
                i = j
                continue

        # Punktlista
        if stripped.startswith("- ") or stripped.startswith("* "):
            items = []
            j = i
            while j < len(raw_lines):
                s = raw_lines[j].strip()
                if s.startswith("- ") or s.startswith("* "):
                    items.append(s[2:].strip())
                    j += 1
                elif not s:
                    j += 1
                    break
                else:
                    break

            lf = ListFlowable(
                [
                    ListItem(_safe_paragraph(it, styles["Body"]), leftIndent=12)
                    for it in items
                ],
                bulletType="bullet",
                leftIndent=18,
            )
            story.append(lf)
            story.append(Spacer(1, 8))
            i = j
            continue

        # Numrerad lista
        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            items = []
            j = i
            while j < len(raw_lines):
                s = raw_lines[j].strip()
                match_num = re.match(r"^(\d+)\.\s+(.*)$", s)
                if match_num:
                    items.append(match_num.group(2).strip())
                    j += 1
                elif not s:
                    j += 1
                    break
                else:
                    break

            lf = ListFlowable(
                [
                    ListItem(_safe_paragraph(it, styles["Body"]), leftIndent=12)
                    for it in items
                ],
                bulletType="1",
                leftIndent=18,
            )
            story.append(lf)
            story.append(Spacer(1, 8))
            i = j
            continue

        # Vanligt stycke
        para_lines = [stripped]
        j = i + 1
        while j < len(raw_lines) and raw_lines[j].strip():
            peek = raw_lines[j].strip()
            if (
                peek.startswith("# ")
                or peek.startswith("## ")
                or peek.startswith("### ")
                or _is_table_line(peek)
                or peek.startswith("- ")
                or peek.startswith("* ")
                or re.match(r"^\d+\.\s+", peek)
            ):
                break
            para_lines.append(peek)
            j += 1

        paragraph = " ".join(para_lines).strip()
        story.append(_safe_paragraph(paragraph, styles["Body"]))
        i = j

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes