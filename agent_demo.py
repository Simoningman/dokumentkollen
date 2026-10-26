"""
agent_demo.py – kör Dokumentkollens agent från terminalen.

    python agent_demo.py "Vilket bolag är mest finansiellt stabilt?" fil1.pdf fil2.pdf
    python agent_demo.py --fri "Fråga" fil.pdf    (tillåt allmän kunskap, tydligt märkt)

Utan argument körs testfrågan mot de två testrapporterna i Downloads.
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from mcp_server import LocalFile  # noqa: E402
from src.agent import run_agent  # noqa: E402
from src.rag import build_doc  # noqa: E402
from src.utils import extract_pages  # noqa: E402

DOWNLOADS = Path.home() / "Downloads"
DEFAULT_QUESTION = "Vilket bolag är mest finansiellt stabilt, och varför?"
DEFAULT_FILES = [DOWNLOADS / "arsrapport_alfa_2025.md", DOWNLOADS / "arsrapport_beta_2025.md"]


def main() -> None:
    args = sys.argv[1:]
    allow_general = "--fri" in args
    args = [a for a in args if a != "--fri"]
    question = args[0] if args else DEFAULT_QUESTION
    files = [Path(p) for p in args[1:]] or DEFAULT_FILES

    docs = {}
    for f in files:
        docs[f.name] = build_doc(f.name, extract_pages(LocalFile(f)))
        print(f"Inläst: {f.name}")

    print(f"\nFråga: {question}\n")

    def log(e):
        print(f"  [steg {e['step']}] {e['tool']}({e['args']})")
        print(f"           -> {e['result'][:120]}...")

    result = run_agent(question, docs, on_step=log, allow_general=allow_general)

    print(f"\n--- Svar ({result['steps']} steg, stopp: {result['stopped']}) ---\n")
    print(result["answer"])
    for w in result["warnings"]:
        print(f"\n⚠ VARNING: {w}")
    cited =", ".join(f"{s['doc']} s.{s['page']}" for s in result["sources"]) or "inga"
    print(f"\nCiterade källor: {cited}  (sökte i {result['searched_pages']} sidor)")


if __name__ == "__main__":
    main()
