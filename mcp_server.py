"""
mcp_server.py – Dokumentkollen som MCP-server.
Återanvänder samma kärna som Streamlit-appen (src/rag.py, src/utils.py).
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from src.rag import build_doc, compare, search, summarize
from src.utils import extract_pages

load_dotenv()

mcp = MCPServer(
    "dokumentkollen",
    instructions="Läs in dokument och sök, sammanfatta eller jämför dem. "
    "Svaren har källhänvisning till dokument och sida.",
)

DOCS: Dict[str, Dict[str, Any]] = {}


class LocalFile:
    """Efterliknar Streamlits UploadedFile så att extract_pages() kan återanvändas."""

    def __init__(self, path: Path):
        self.name = path.name
        self._data = path.read_bytes()

    def getvalue(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data


def _clean(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Behåller bara JSON-vänliga fält (tar bort t.ex. embeddings)."""
    out = []
    for h in hits:
        row = {}
        for k, v in h.items():
            if hasattr(v, "item") and getattr(v, "size", 1) == 1:
                v = v.item()
            if v is None or isinstance(v, (str, int, float, bool)):
                row[k] = v
        out.append(row)
    return out


@mcp.tool()
def load_document(path: str) -> str:
    """Läs in ett dokument (PDF, DOCX, TXT, MD, CSV, HTML) från en filsökväg och indexera det."""
    p = Path(path).expanduser()
    if not p.is_file():
        return f"Hittar ingen fil: {p}"
    pages = extract_pages(LocalFile(p))
    DOCS[p.name] = build_doc(p.name, pages)
    return f"Läste in {p.name} ({len(pages)} sidor). Inlästa: {', '.join(DOCS)}"


@mcp.tool()
def list_documents() -> List[str]:
    """Lista dokument som är inlästa just nu."""
    return list(DOCS)


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Semantisk sökning i inlästa dokument. Returnerar textutdrag med dokument och sida."""
    return _clean(search(query, list(DOCS.values()), top_k=top_k))


@mcp.tool()
def summarize_documents() -> Dict[str, Any]:
    """Sammanfatta alla inlästa dokument med källhänvisningar."""
    text, hits = summarize(list(DOCS.values()))
    return {"summary": text, "sources": _clean(hits)}


@mcp.tool()
def compare_documents(focus: Optional[str] = None) -> Dict[str, Any]:
    """Jämför inlästa dokument (minst två). focus: valfritt tema, t.ex. 'soliditet'."""
    if len(DOCS) < 2:
        return {"error": "Läs in minst två dokument först."}
    text, hits = compare(list(DOCS.values()), focus=focus)
    return {"comparison": text, "sources": _clean(hits)}


if __name__ == "__main__":
    mcp.run()