"""
Documentation-only companion to multi_doc_rag.py.

Every function below is self-contained (own imports, own constants, no
cross-function dependencies) so each one can be extracted and pasted
standalone into the blog. Not run as a script, not kept in sync with
multi_doc_rag.py automatically.
"""


# --8<-- [start:chunkers]
def chunk_fixed_size(pages: list, size: int = 400, overlap: int = 50) -> list:
    """`pages` is a list of (doc_name, page_number, page_text) tuples --
    chunking per page (rather than guessing offsets in a merged string)
    makes page attribution exact by construction."""
    chunks = []
    chunk_id = 0
    for doc_name, page_num, page_text in pages:
        for start in range(0, len(page_text), size - overlap):
            piece = page_text[start:start + size].strip()
            if len(piece) < 40:
                continue
            chunks.append({"id": chunk_id, "text": piece, "doc": doc_name, "page": page_num})
            chunk_id += 1
    return chunks


def chunk_recursive(pages: list, size: int = 400, overlap: int = 50) -> list:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = []
    chunk_id = 0
    for doc_name, page_num, page_text in pages:
        for piece in splitter.split_text(page_text):
            if len(piece.strip()) < 40:
                continue
            chunks.append({"id": chunk_id, "text": piece.strip(), "doc": doc_name, "page": page_num})
            chunk_id += 1
    return chunks


def chunk_sentence_window(pages: list, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> list:
    """Groups N sentences per chunk with a 1-sentence overlap, instead
    of splitting mid-sentence at a fixed character count -- a cheap
    proxy for "semantic" chunking with no embedding-based splitter."""
    import re

    chunks = []
    chunk_id = 0
    step = max(1, sentences_per_chunk - overlap_sentences)
    for doc_name, page_num, page_text in pages:
        sentences = re.split(r"(?<=[.!?])\s+", page_text)
        for start in range(0, len(sentences), step):
            piece = " ".join(sentences[start:start + sentences_per_chunk]).strip()
            if len(piece) < 40:
                continue
            chunks.append({"id": chunk_id, "text": piece, "doc": doc_name, "page": page_num})
            chunk_id += 1
    return chunks
# --8<-- [end:chunkers]


# --8<-- [start:chunking_eval]
def evaluate_chunking_strategy(chunks: list, eval_set: list, model, k: int = 3) -> dict:
    """`eval_set` items need question/keyword/doc/reference keys.
    Ground truth for a question is every chunk from the RIGHT document
    that contains the keyword -- catching wrong-document attribution as
    a real failure mode, not just wrong-chunk-within-the-right-doc."""
    import numpy as np

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    recalls, hits = [], []
    for item in eval_set:
        relevant = {c["id"] for c in chunks if item["keyword"].lower() in c["text"].lower() and c["doc"] == item["doc"]}
        query_vec = model.encode(item["question"], convert_to_numpy=True, normalize_embeddings=True)
        sims = embeddings @ query_vec
        retrieved = [chunks[i]["id"] for i in np.argsort(-sims)[:k]]

        hits_in_k = sum(1 for cid in retrieved if cid in relevant)
        recalls.append(hits_in_k / len(relevant) if relevant else 0.0)
        hits.append(1.0 if hits_in_k > 0 else 0.0)

    return {"num_chunks": len(chunks), "recall_at_k": float(np.mean(recalls)), "hit_rate_at_k": float(np.mean(hits))}
# --8<-- [end:chunking_eval]


# --8<-- [start:hybrid_rrf]
def hybrid_rrf_fusion(bm25_ranks: list, dense_ranks: list, k_const: int = 60) -> list:
    """Reciprocal Rank Fusion: each ranker contributes 1/(k+rank) per
    item; scores from both rankers are summed. No mechanism exists to
    detect a ranker that's contributing pure noise for a given query --
    RRF trusts both rankers equally by construction."""
    scores = {}
    for rank, cid in enumerate(bm25_ranks, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_const + rank)
    for rank, cid in enumerate(dense_ranks, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_const + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]
# --8<-- [end:hybrid_rrf]


# --8<-- [start:qdrant_multi_doc]
def index_and_query_qdrant(chunks: list, embeddings, query_vec: list, filter_doc: str = None, limit: int = 3):
    """Indexes a multi-document corpus into Qdrant with each chunk's
    source document as metadata, then queries with an optional
    document filter -- useful when a query is genuinely ambiguous
    across documents in the corpus (e.g. "how many layers does the
    model have" could be about any of several papers)."""
    from qdrant_client import QdrantClient, models

    client = QdrantClient(":memory:")
    dim = embeddings.shape[1]
    client.create_collection(
        collection_name="corpus",
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE, hnsw_config=models.HnswConfigDiff(m=32, ef_construct=200)),
    )
    client.upsert(
        collection_name="corpus",
        points=[
            models.PointStruct(id=c["id"], vector=embeddings[i].tolist(), payload={"doc": c["doc"], "page": c["page"]})
            for i, c in enumerate(chunks)
        ],
    )

    query_filter = None
    if filter_doc:
        query_filter = models.Filter(must=[models.FieldCondition(key="doc", match=models.MatchValue(value=filter_doc))])

    result = client.query_points(collection_name="corpus", query=query_vec, limit=limit, query_filter=query_filter)
    return [(p.payload["doc"], p.payload["page"], p.score) for p in result.points]
# --8<-- [end:qdrant_multi_doc]
