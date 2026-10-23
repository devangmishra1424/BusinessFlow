"""Prints the lender's documents (policy HTML under data/policies/source/, and the demo borrowers' loan documents under
data/demo/source/) to real, text-layer PDFs with a headless Chromium browser (Edge or Chrome).

Why a browser and not a PDF library: none is installed, the browser is already on every developer machine, and it
lays out tables and Devanagari correctly. The PDFs are committed, so the server never needs a browser: they reach the
knowledge base the way any lender's document does, through the ops upload (POST /policy-documents, or a borrower's
POST /accounts/{id}/documents).

Run (from the repo root): python -m scripts.build_policy_pdfs [--src DIR --out DIR]
Set CHROMIUM_PATH if the browser is not in a usual place.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/microsoft-edge",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
_PRINT_TIMEOUT_SECONDS = 90


def _browser() -> str:
    configured = os.environ.get("CHROMIUM_PATH")
    if configured:
        return configured
    for candidate in _CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("no Chromium browser found -- set CHROMIUM_PATH to Edge or Chrome")


def print_pdf(browser: str, html: Path, pdf: Path) -> None:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.unlink(missing_ok=True)
    subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-first-run",
         f"--print-to-pdf={pdf}", html.resolve().as_uri()],
        check=True, capture_output=True, timeout=_PRINT_TIMEOUT_SECONDS,
    )
    if not pdf.exists() or pdf.stat().st_size == 0:
        raise RuntimeError(f"the browser wrote no PDF for {html.name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=_REPO / "data" / "policies" / "source")
    parser.add_argument("--out", type=Path, default=_REPO / "data" / "policies")
    args = parser.parse_args(argv)
    browser = _browser()
    pages = sorted(args.src.glob("*.html"))
    if not pages:
        raise SystemExit(f"no .html files in {args.src}")
    for html in pages:
        pdf = args.out / f"{html.stem}.pdf"
        print_pdf(browser, html, pdf)
        print(f"{pdf.relative_to(_REPO) if pdf.is_relative_to(_REPO) else pdf}: {pdf.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main(sys.argv[1:])
