"""Testar chunkfiltret i den riktiga rag.py, utan att ladda LaBSE."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("real_rag", Path(__file__).parents[1] / "src" / "rag.py")
rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag)
rag.embed_texts = lambda texts: np.zeros((len(texts), 3), dtype=np.float32)


def test_short_title_page_is_dropped():
    pages = [(1, "ÅRS- OCH HÅLLBARHETSREDOVISNING 2024"), (35, "Soliditet, % 41,3 44,4 " * 10)]
    doc = rag.build_doc("ge.pdf", pages)
    assert [c["page_no"] for c in doc["chunks"]] == [35]


def test_tiny_document_is_kept():
    doc = rag.build_doc("kort.md", [(1, "Bara några få ord här")])
    assert len(doc["chunks"]) == 1


def test_hybrid_search_ranks_exact_term_first():
    # Med lika embeddings avgör ordträffarna: chunken med frågans ord vinner
    doc = rag.build_doc("avtal.pdf", [
        (1, "Allmänna villkor för leverans och betalning gäller för alla beställningar " * 3),
        (7, "Uppsägning av avtalet ska ske skriftligen med tre månaders uppsägningstid " * 3),
    ])
    hits = rag.search("Hur sker uppsägning av avtalet?", [doc], top_k=2)
    assert hits[0]["page_no"] == 7


def test_query_terms_skip_stopwords():
    assert rag.query_terms("Vad säger dokumentet om uppsägning och moms?") == ["säger", "uppsägning", "moms"]
