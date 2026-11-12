"""
Documentation-only companion to evaluation_metrics.py.

Every function below is self-contained (own imports, own constants, no
cross-function dependencies) so each one can be extracted and pasted
standalone into the blog. Not run as a script, not kept in sync with
evaluation_metrics.py automatically.
"""


# --8<-- [start:retrieval_metrics]
def retrieval_metrics(retrieved: list, relevant: set, k: int) -> dict:
    """Precision@k, Recall@k, Reciprocal Rank, NDCG@k (binary relevance),
    and Hit Rate@k for one query. `retrieved` is a ranked list of chunk
    ids; `relevant` is the real ground-truth relevant set for the query."""
    import math

    top_k = retrieved[:k]
    hits_in_top_k = sum(1 for cid in top_k if cid in relevant)

    precision = hits_in_top_k / k
    recall = hits_in_top_k / len(relevant) if relevant else 0.0

    reciprocal_rank = 0.0
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(1.0 / math.log2(rank + 1) for rank, cid in enumerate(top_k, start=1) if cid in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    hit_rate = 1.0 if hits_in_top_k > 0 else 0.0

    return {"precision": precision, "recall": recall, "reciprocal_rank": reciprocal_rank, "ndcg": ndcg, "hit_rate": hit_rate}
# --8<-- [end:retrieval_metrics]


# --8<-- [start:build_relevance_from_keyword]
def build_relevance_set(chunks: list, keyword: str) -> set:
    """Ground truth built from a real substring match, not hand-picked
    page numbers: the relevant set for a question is every chunk whose
    text actually contains the keyword that answers it."""
    return {c["id"] for c in chunks if keyword.lower() in c["text"].lower()}
# --8<-- [end:build_relevance_from_keyword]


# --8<-- [start:generation_metrics]
def generation_metrics(reference: str, answer: str, embed_fn) -> dict:
    """Classic lexical-overlap metrics (BLEU, ROUGE-L, METEOR) plus
    embedding-based semantic similarity, for one generated answer
    against its reference. `embed_fn(text)` returns a normalized
    embedding vector for cosine similarity via dot product."""
    import sacrebleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    bleu = sacrebleu.sentence_bleu(answer, [reference]).score

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = scorer.score(reference, answer)["rougeL"].fmeasure

    meteor = meteor_score([reference.split()], answer.split())

    ref_vec = embed_fn(reference)
    ans_vec = embed_fn(answer)
    semantic_similarity = float(ref_vec @ ans_vec)

    return {"bleu": bleu, "rouge_l": rouge_l, "meteor": meteor, "semantic_similarity": semantic_similarity}
# --8<-- [end:generation_metrics]
