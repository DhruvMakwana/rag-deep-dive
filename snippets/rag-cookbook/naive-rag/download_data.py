"""
Fetches a small sample PDF to run the naive RAG pipeline against.

Default: "Attention Is All You Need" (Vaswani et al., 2017) from arXiv —
a short (~15 page), well-structured technical PDF with headings, an
abstract, tables, and references. It's freely downloadable from arXiv
and is the same sample document used across most public RAG tutorials,
so it's a safe, reproducible default with no licensing ambiguity for
an educational demo like this one.

Swap in your own PDF any time — see the README ("Using your own PDF").
"""

import os
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PDF_URL = "https://arxiv.org/pdf/1706.03762"
DEFAULT_PDF_PATH = DATA_DIR / "attention_is_all_you_need.pdf"


def download_sample_pdf(url: str = DEFAULT_PDF_URL, dest: Path = DEFAULT_PDF_PATH) -> Path:
    """Download the sample PDF if it isn't already cached locally."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached PDF: {dest}")
        return dest

    print(f"Downloading sample PDF from {url} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"Saved to {dest} ({dest.stat().st_size / 1024:.0f} KB)")
    return dest


if __name__ == "__main__":
    pdf_url = os.environ.get("SAMPLE_PDF_URL", DEFAULT_PDF_URL)
    download_sample_pdf(pdf_url)
