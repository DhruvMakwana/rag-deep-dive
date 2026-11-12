"""
Documentation-only companion to vision_rag.py.

Every function below is self-contained (own imports, no cross-function
dependencies) so each one can be extracted and pasted standalone into the
blog. Not run as a script, not kept in sync with vision_rag.py automatically.
"""


# --8<-- [start:render_pages]
def render_pdf_pages(pdf_path: str, dpi: int = 150) -> list:
    """Render every PDF page to a PIL image. This is the entire
    "extraction" step Vision RAG needs — no text, no OCR."""
    import fitz  # pymupdf
    from PIL import Image

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pixmap = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        pages.append(img)
    return pages
# --8<-- [end:render_pages]


# --8<-- [start:vision_model]
def load_vision_model(repo_id: str = "vidore/ColSmolVLM-256M-base"):
    """Load a ColPali-family page-image retriever.

    This checkpoint's `adapter_config.json` ships with
    `base_model_name_or_path` set to a local training path
    (`./models/ColSmolVLM-256M-Base`) that doesn't exist outside the
    original training run. `transformers` auto-detects the adapter config
    and tries to resolve that path as a Hugging Face repo id, which fails.
    The full merged weights are already present in `model.safetensors`, so
    the fix is to patch a local copy of `adapter_config.json` to point at
    the real base model (`HuggingFaceTB/SmolVLM-256M-Base`, from the model
    card) and load from that local copy."""
    import json
    import shutil
    from pathlib import Path

    import torch
    from huggingface_hub import snapshot_download
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor

    snapshot_dir = snapshot_download(repo_id)
    patched_dir = Path("/tmp/colsmolvlm_patched")
    if patched_dir.exists():
        shutil.rmtree(patched_dir)
    shutil.copytree(snapshot_dir, patched_dir)

    for fname in ["adapter_config.json", "adapter2_config.json"]:
        fpath = patched_dir / fname
        if fpath.exists():
            config = json.loads(fpath.read_text())
            config["base_model_name_or_path"] = "HuggingFaceTB/SmolVLM-256M-Base"
            fpath.write_text(json.dumps(config))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ColIdefics3.from_pretrained(str(patched_dir), torch_dtype=torch.float32).to(device).eval()
    processor = ColIdefics3Processor.from_pretrained(str(patched_dir))
    return model, processor
# --8<-- [end:vision_model]


# --8<-- [start:vision_index_search]
def build_page_image_index(pages: list, model, processor) -> list:
    """Embed every page as a multi-vector — one embedding per image patch,
    ColBERT-style — not a single pooled vector. Late interaction scoring
    needs the full patch grid."""
    import torch

    device = next(model.parameters()).device
    embeddings = []
    with torch.no_grad():
        for page in pages:
            batch = processor.process_images([page]).to(device)
            page_embedding = model(**batch)
            embeddings.append(page_embedding[0].cpu())
    return embeddings


def vision_search(query: str, page_embeddings: list, model, processor, k: int = 3) -> list:
    """Score every page with MaxSim (late interaction): for each query
    token embedding, take the best-matching patch embedding on the page,
    then sum those per-token maxima. A page can win because ONE region of
    it — one table cell, one chart legend line — matches strongly, even if
    the rest of the page doesn't."""
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        query_batch = processor.process_queries([query]).to(device)
        query_embedding = model(**query_batch)[0].cpu()

    scores = [
        processor.score_multi_vector(query_embedding.unsqueeze(0), page_embedding.unsqueeze(0)).item()
        for page_embedding in page_embeddings
    ]
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]
# --8<-- [end:vision_index_search]


# --8<-- [start:naive_text_rag]
def naive_text_search(pdf_path: str, query: str, k: int = 3) -> list:
    """The baseline this page compares against: extract text, chunk it,
    embed chunks with a dedicated text embedder. Same pattern used
    throughout this repo."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader
    from sentence_transformers import SentenceTransformer, util

    reader = PdfReader(pdf_path)
    full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    query_embedding = model.encode(query, convert_to_numpy=True)
    sims = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_k = sims.topk(min(k, len(chunks)))
    return [(chunks[i], sims[i].item()) for i in top_k.indices.tolist()]
# --8<-- [end:naive_text_rag]
