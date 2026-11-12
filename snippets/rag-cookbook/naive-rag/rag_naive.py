"""
Naive RAG, end to end, from scratch — no LangChain/LlamaIndex orchestration,
so every step is visible: load a PDF -> chunk -> embed -> index -> retrieve
-> generate.

Run:
    python rag_naive.py "What is multi-head attention?"

See the accompanying blog page for a section-by-section walkthrough of
this file, and README.md for setup.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import llm
from download_data import DEFAULT_PDF_PATH, download_sample_pdf

DATA_DIR = Path(__file__).parent / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# --8<-- [start:load_env]
def load_environment() -> None:
    """Load .env from this folder, or (falling back) from a parent directory —
    e.g. if you keep one shared .env above several project folders."""
    local_env = Path(__file__).parent / ".env"
    if local_env.exists():
        load_dotenv(local_env)
    else:
        load_dotenv()  # searches upward from the current working directory
# --8<-- [end:load_env]


# --8<-- [start:load_pdf]
def load_pdf_text(pdf_path: Path) -> str:
    """Extract raw text from every page of a PDF and join it into one string."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
# --8<-- [end:load_pdf]


# --8<-- [start:chunk_text]
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Fixed-size character chunking with overlap — the simplest possible
    strategy, deliberately naive. See the Chunking Strategies page for
    smarter options (recursive, semantic, small-to-big, ...)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]
# --8<-- [end:chunk_text]


# --8<-- [start:embed]
def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Embed a list of strings and L2-normalize each vector, so a plain
    inner-product search (used below) is equivalent to cosine similarity."""
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    return embeddings
# --8<-- [end:embed]


# --8<-- [start:build_index]
def build_index(embeddings: np.ndarray) -> faiss.Index:
    """A flat (brute-force) FAISS index — exact search, fine at this scale
    (a few hundred chunks). See Vector Databases for ANN indexes at scale."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index
# --8<-- [end:build_index]


def build_or_load_index(pdf_path: Path, embed_model: SentenceTransformer, rebuild: bool = False):
    if not rebuild and INDEX_PATH.exists() and CHUNKS_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        chunks = json.loads(CHUNKS_PATH.read_text())
        return index, chunks

    print("Building index (first run, or --rebuild was passed)...")
    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks, embed_model)
    index = build_index(embeddings)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks))
    print(f"Indexed {len(chunks)} chunks.")
    return index, chunks


# --8<-- [start:retrieve]
def retrieve(query: str, index: faiss.Index, chunks: list[str], embed_model: SentenceTransformer, k: int = 4) -> list[str]:
    """Embed the query the same way the chunks were embedded, then pull
    back the k nearest chunks by cosine similarity."""
    query_vec = embed_texts([query], embed_model)
    _scores, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]
# --8<-- [end:retrieve]


# --8<-- [start:prompt]
PROMPT_TEMPLATE = """Answer the question using ONLY the context below. \
If the context doesn't contain the answer, say you don't know — do not guess.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(retrieved_chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)
# --8<-- [end:prompt]


def answer_question(question: str, provider: str | None = None, k: int = 4, rebuild: bool = False) -> str:
    load_environment()

    pdf_path = Path(os.environ.get("PDF_PATH", DEFAULT_PDF_PATH))
    if not pdf_path.exists():
        pdf_path = download_sample_pdf()

    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index, chunks = build_or_load_index(pdf_path, embed_model, rebuild=rebuild)

    retrieved = retrieve(question, index, chunks, embed_model, k=k)
    prompt = build_prompt(question, retrieved)
    return llm.generate(prompt, provider=provider)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive RAG over a sample PDF.")
    parser.add_argument("question", help="The question to ask the document.")
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default=None)
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument("--rebuild", action="store_true", help="Force re-indexing the PDF.")
    args = parser.parse_args()

    print(answer_question(args.question, provider=args.provider, k=args.k, rebuild=args.rebuild))
