"""
Documentation-only companion to multimodal_rag.py.

Not run top-to-bottom, not imported by anything. Every section below is
self-contained — its own imports, its own inlined helpers — so
copy-pasting any single section into a fresh .py works on its own. See
multimodal_rag.py for the actual runnable CLI.

If you change something in multimodal_rag.py, mirror it here too —
these are NOT kept in sync automatically.
"""


# --8<-- [start:extraction]
from pathlib import Path

from PIL import Image


def extract_images_from_pdf(pdf_path: str) -> dict[str, Image.Image]:
    """Pulls the paper's own real embedded figures out of the PDF —
    not stock images, the actual diagrams the authors put in the
    document."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    images = {}
    for page_idx, page in enumerate(reader.pages):
        for img in page.images:
            key = f"page{page_idx}_{Path(img.name).stem}"
            images[key] = img.image
    return images
# --8<-- [end:extraction]


# --8<-- [start:multimodal_encoder]
import numpy as np
from PIL import Image

# jina-clip-v2, not base CLIP — measured directly against base CLIP
# (clip-ViT-B-32) and SigLIP (google/siglip-base-patch16-224) before
# picking this one. Base CLIP scored 0.375 Recall@3 on this recipe's
# 8-question text eval set; SigLIP scored worse still, 0.25.
# jina-clip-v2 scored 0.75 — a real, substantial improvement, with the
# same image-matching accuracy as base CLIP (2/4 on this recipe's test
# queries) and much better-separated confidence scores for irrelevant
# queries. Needs `transformers==4.46.3` pinned — a newer transformers
# breaks its custom modeling code (`clip_loss` was removed from
# `transformers.models.clip.modeling_clip`, which this model's remote
# code still imports) — and `trust_remote_code=True`, since it ships
# custom model code from Jina AI's HuggingFace repo.
CLIP_MODEL_NAME = "jinaai/jina-clip-v2"


class MultimodalEncoder:
    """Wraps jina-clip-v2's AutoModel behind the same simple
    `.encode(...)` interface SentenceTransformer models use elsewhere
    in this repo. jina-clip-v2's own sentence-transformers integration
    didn't route PIL images correctly in testing (raised "Modality
    'image' is not supported"); its documented `AutoModel.encode_text`
    / `encode_image` methods are used directly here instead."""

    def __init__(self, model_name: str = CLIP_MODEL_NAME) -> None:
        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def encode(self, inputs, convert_to_numpy: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        single = isinstance(inputs, (str, Image.Image))
        items = [inputs] if single else list(inputs)
        embeddings = self._model.encode_image(items) if items and isinstance(items[0], Image.Image) else self._model.encode_text(items)
        return embeddings[0] if single else embeddings
# --8<-- [end:multimodal_encoder]


# --8<-- [start:unified]
import numpy as np


def build_unified_index(chunks: list[str], images: dict, clip_model) -> dict:
    """Everything — text chunks AND images — embedded into ONE shared
    CLIP-family space. Enables genuine cross-modal search (a text
    query can directly retrieve an image), at a real cost: even
    jina-clip-v2 (see multimodal_encoder above), the strongest of 3
    text+image models tested, is still a weaker text retriever than a
    dedicated text embedding model (measured: 1.00 -> 0.75 Recall@3 on
    the same 8 factual questions, same corpus — see multimodal_rag.py's
    --compare-text-quality)."""
    text_embs = clip_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    image_names = list(images.keys())
    image_embs = clip_model.encode(list(images.values()), convert_to_numpy=True, show_progress_bar=False)
    return {"chunks": chunks, "chunk_embs": text_embs, "image_names": image_names, "image_embs": image_embs}


def unified_search(query: str, index: dict, clip_model, k: int = 3) -> list[tuple[str, str]]:
    query_vec = clip_model.encode(query, convert_to_numpy=True)
    candidates = [("text", c) for c in index["chunks"]] + [("image", n) for n in index["image_names"]]
    all_embs = np.vstack([index["chunk_embs"], index["image_embs"]])
    sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in all_embs]
    top_indices = np.argsort(sims)[::-1][:k]
    return [candidates[i] for i in top_indices]
# --8<-- [end:unified]


# --8<-- [start:separate_fusion]
import numpy as np
from sentence_transformers import SentenceTransformer

# RRF has no concept of absolute relevance — it only knows RANK within
# a list. With a small image pool, whichever image scores even
# marginally highest still becomes "rank 1" and earns a real fusion
# boost, even when every image is genuinely irrelevant to the query
# (verified directly: all 3 images scored well below this threshold —
# noise, not signal — for a plain-text question, and without a floor
# the highest of the three would still get fused ahead of a relevant
# text chunk). A minimum similarity floor fixes it: don't let a
# modality contribute to fusion at all if its best candidate doesn't
# clear a real relevance bar.
#
# A second, deeper property this surfaced: even with the floor in
# place, RRF still produces an exact tie at rank 0 between modalities
# on every query (1/(k+0+1) is identical for a text chunk or an image
# in that slot) — a structural fact about rank-based fusion, not
# something a tiebreaker can fix, since the top item in any list
# normalizes to the same value by construction. RRF is a consensus
# mechanism, not a magnitude comparator; see unified_search elsewhere
# in this file for how to get a genuine image-only result instead.
CLIP_RELEVANCE_THRESHOLD = 0.28


def build_separate_indexes(chunks: list[str], images: dict, text_model: SentenceTransformer, clip_model) -> dict:
    """Text goes through a DEDICATED text embedder (full retrieval
    quality preserved); images go through the CLIP-family model (the
    only place it's actually needed). Two separate, uncalibrated
    embedding spaces — which is exactly why fusion below uses RRF
    (rank-based) rather than comparing raw similarity scores
    directly."""
    return {
        "chunks": chunks,
        "chunk_embs": text_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False),
        "image_names": list(images.keys()),
        "image_embs": clip_model.encode(list(images.values()), convert_to_numpy=True, show_progress_bar=False),
    }


def reciprocal_rank_fusion(ranked_lists: list[list], k: int = 60) -> list:
    """Same RRF used for multi-query/RAG-Fusion elsewhere in this
    repo — rank-based, not raw-score-based, which matters here since
    the CLIP-family model's image-similarity scores and the text
    embedder's similarity scores come from two different,
    uncalibrated spaces and can't be compared directly.

    A real, non-obvious consequence: EVERY query produces an exact
    tie at rank 0 between modalities — `1/(k+0+1)` is identical
    whether a text chunk or an image sits in that slot, regardless of
    how relevant either actually is. This isn't a bug to patch with a
    tiebreaker (a first attempt at one — normalizing each candidate's
    raw similarity within its own list — didn't work: the top item in
    ANY list normalizes to exactly 1.0 by construction, so the tie
    persists no matter what). It's a real, structural property of
    rank-based fusion: RRF is a CONSENSUS mechanism, not a magnitude
    comparator, and it has no principled way to let one modality's
    confident top pick outright beat another modality's merely-present
    top pick. See `unified_search` above for how to get a genuine
    image-only result instead — a single flat ranking has no second
    list to tie against."""
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: -scores[item])


def fused_search(query: str, index: dict, text_model: SentenceTransformer, clip_model, k: int = 3) -> list[tuple[str, str]]:
    text_query_vec = text_model.encode(query, convert_to_numpy=True)
    text_sims = [float(np.dot(text_query_vec, v) / (np.linalg.norm(text_query_vec) * np.linalg.norm(v) + 1e-8)) for v in index["chunk_embs"]]
    text_ranked = [("text", index["chunks"][i]) for i in np.argsort(text_sims)[::-1]]

    clip_query_vec = clip_model.encode(query, convert_to_numpy=True)
    image_sims = [float(np.dot(clip_query_vec, v) / (np.linalg.norm(clip_query_vec) * np.linalg.norm(v) + 1e-8)) for v in index["image_embs"]]
    ranked_lists = [text_ranked]
    if max(image_sims, default=0.0) >= CLIP_RELEVANCE_THRESHOLD:
        image_order = np.argsort(image_sims)[::-1]
        ranked_lists.append([("image", index["image_names"][i]) for i in image_order if image_sims[i] >= CLIP_RELEVANCE_THRESHOLD])

    fused = reciprocal_rank_fusion(ranked_lists)
    return fused[:k]
# --8<-- [end:separate_fusion]
