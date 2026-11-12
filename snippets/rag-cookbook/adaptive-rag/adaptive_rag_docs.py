"""
Documentation-only companion to adaptive_rag.py.

Not run top-to-bottom, not imported by anything. Every function below is
self-contained — its own imports, its own inlined helpers — so
copy-pasting any single section into a fresh .py works on its own. See
adaptive_rag.py for the actual runnable CLI.

If you change something in adaptive_rag.py, mirror it here too — these
are NOT kept in sync automatically.
"""


# --8<-- [start:router]
import re

import llm

ROUTER_PROMPT = """Classify the query below into EXACTLY ONE category, \
based on what problem (if any) it poses for retrieval against a \
technical document. Categories, each with an example:

CLEAR - a single specific fact that is a NAMED thing (an algorithm \
name, a hardware type, a dataset name) — not a numeric parameter value \
— already phrased using the kind of technical vocabulary the document \
itself would use.
  Example: "What optimizer did they use for training?"

VAGUE - a single specific fact whose complete answer is ONE short \
thing, but the question is phrased colloquially, avoiding the \
technical vocabulary the document actually uses.
  Example: "why dont they just use the old-school approach everyone \
used before"

COMPOUND - explicitly asks for two or more distinct, independently \
answerable facts, usually joined by "and".
  Example: "What optimizer did they use, and how many layers does it have?"

BROAD - one open-ended question whose COMPLETE answer requires citing \
several distinct reasons or mechanisms (not one short fact) — decide \
based on whether a full answer needs multiple points, regardless of \
whether the wording sounds casual or technical.
  Example: "What lets this model handle relationships between things \
that are positioned far apart, without losing track over long spans?"

NEEDS_CONTEXT - asks for one narrow NUMERIC or configuration parameter \
value (a rate, a dimension, a count) where knowing the broader \
principle/section it belongs to would help interpret or locate the \
answer.
  Example: "What value did they use for the regularization strength?"

Query: {query}

Respond with ONLY the category name, wrapped in <category></category> tags."""


def route_query(question: str, provider: str | None = None) -> str:
    """One LLM call classifies the query so the pipeline can apply only
    the pre-retrieval fix that category actually needs, instead of
    running every technique on every query."""
    response = llm.generate(ROUTER_PROMPT.format(query=question), provider=provider)
    match = re.search(r"<category>(.*?)</category>", response, re.S)
    category = match.group(1).strip() if match else "CLEAR"
    return category if category in {"CLEAR", "VAGUE", "COMPOUND", "BROAD", "NEEDS_CONTEXT"} else "CLEAR"
# --8<-- [end:router]


# --8<-- [start:pipeline]
import re

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

import llm

ROUTER_PROMPT = """Classify the query below into EXACTLY ONE category, \
based on what problem (if any) it poses for retrieval against a \
technical document. Categories, each with an example:

CLEAR - a single specific fact that is a NAMED thing (an algorithm \
name, a hardware type, a dataset name) — not a numeric parameter value \
— already phrased using the kind of technical vocabulary the document \
itself would use.
  Example: "What optimizer did they use for training?"

VAGUE - a single specific fact whose complete answer is ONE short \
thing, but the question is phrased colloquially, avoiding the \
technical vocabulary the document actually uses.
  Example: "why dont they just use the old-school approach everyone \
used before"

COMPOUND - explicitly asks for two or more distinct, independently \
answerable facts, usually joined by "and".
  Example: "What optimizer did they use, and how many layers does it have?"

BROAD - one open-ended question whose COMPLETE answer requires citing \
several distinct reasons or mechanisms (not one short fact) — decide \
based on whether a full answer needs multiple points, regardless of \
whether the wording sounds casual or technical.
  Example: "What lets this model handle relationships between things \
that are positioned far apart, without losing track over long spans?"

NEEDS_CONTEXT - asks for one narrow NUMERIC or configuration parameter \
value (a rate, a dimension, a count) where knowing the broader \
principle/section it belongs to would help interpret or locate the \
answer.
  Example: "What value did they use for the regularization strength?"

Query: {query}

Respond with ONLY the category name, wrapped in <category></category> tags."""


def route_query(question: str, provider: str | None = None) -> str:
    response = llm.generate(ROUTER_PROMPT.format(query=question), provider=provider)
    match = re.search(r"<category>(.*?)</category>", response, re.S)
    category = match.group(1).strip() if match else "CLEAR"
    return category if category in {"CLEAR", "VAGUE", "COMPOUND", "BROAD", "NEEDS_CONTEXT"} else "CLEAR"


REWRITE_PROMPT = """Rewrite this query to be clearer and more specific \
for searching a technical document. Return ONLY the rewritten query, \
nothing else.

Query: {query}"""

MULTI_QUERY_PROMPT = """Generate 3 different phrasings of the question \
below, each capturing a different angle on the same underlying \
information need. Return one per line, nothing else.

Question: {query}"""

STEP_BACK_PROMPT = """What is a more general question that would help \
answer this specific question? Return ONLY the general question.

Specific question: {query}"""

DECOMPOSE_PROMPT = """If the question below is asking multiple distinct \
things, break it into separate sub-questions, one per line. If it's \
already a single question, return it unchanged on one line.

Question: {query}"""

CANDIDATE_POOL = 15
FINAL_K = 3
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _bi_encoder_retrieve(question, chunks, embed_model, k):
    query_vec = embed_model.encode(question, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in chunk_vecs]
    top_indices = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_indices]


def _reciprocal_rank_fusion(ranked_lists, k=60):
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda c: -scores[c])


def _cross_encoder_rerank(reranker, question, candidates, top_n):
    scores = reranker.predict([(question, c) for c in candidates])
    ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: -x[0])]
    return ranked[:top_n]


def adaptive_retrieve(
    question: str,
    chunks: list[str],
    embed_model: SentenceTransformer,
    provider: str | None = None,
) -> dict:
    """Router -> matching pre-retrieval technique (or none) -> wide
    candidate pool -> cross-encoder rerank.

    Two design decisions that only surfaced from testing the FULL
    pipeline end to end, not each technique in isolation:

    1. A cross-encoder suffers the SAME vocabulary-mismatch problem a
       bi-encoder does — for VAGUE queries, reranking must be scored
       against the REWRITTEN query, not the original colloquial one, or
       it undoes the fix rewriting just made.
    2. Reranking a decomposed COMPOUND question's merged candidate pool
       against one single query collapses it back to one topic —
       undoing decomposition's whole guarantee of independent
       per-sub-question representation. Each sub-question needs its
       own rerank pass and its own guaranteed slots."""
    reranker = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    category = route_query(question, provider=provider)

    if category == "VAGUE":
        rewritten = llm.generate(REWRITE_PROMPT.format(query=question), provider=provider).strip()
        candidates = _bi_encoder_retrieve(rewritten, chunks, embed_model, CANDIDATE_POOL)
        final_chunks = _cross_encoder_rerank(reranker, rewritten, candidates, FINAL_K)

    elif category == "COMPOUND":
        response = llm.generate(DECOMPOSE_PROMPT.format(query=question), provider=provider)
        sub_questions = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
        per_question_pool = max(3, CANDIDATE_POOL // len(sub_questions))
        per_question_final = max(2, FINAL_K // len(sub_questions))
        final_chunks, seen = [], set()
        for sub_q in sub_questions:
            pool = _bi_encoder_retrieve(sub_q, chunks, embed_model, per_question_pool)
            for chunk in _cross_encoder_rerank(reranker, sub_q, pool, per_question_final):
                if chunk not in seen:
                    final_chunks.append(chunk)
                    seen.add(chunk)

    elif category == "BROAD":
        response = llm.generate(MULTI_QUERY_PROMPT.format(query=question), provider=provider)
        variations = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
        variations.append(question)
        ranked_lists = [_bi_encoder_retrieve(v, chunks, embed_model, CANDIDATE_POOL) for v in variations]
        candidates = _reciprocal_rank_fusion(ranked_lists)[:CANDIDATE_POOL]
        final_chunks = _cross_encoder_rerank(reranker, question, candidates, FINAL_K)

    elif category == "NEEDS_CONTEXT":
        general_question = llm.generate(STEP_BACK_PROMPT.format(query=question), provider=provider).strip()
        specific_hits = _bi_encoder_retrieve(question, chunks, embed_model, CANDIDATE_POOL)
        general_hits = _bi_encoder_retrieve(general_question, chunks, embed_model, CANDIDATE_POOL)
        combined, seen = [], set()
        for chunk in specific_hits + general_hits:
            if chunk not in seen:
                combined.append(chunk)
                seen.add(chunk)
        final_chunks = _cross_encoder_rerank(reranker, question, combined[:CANDIDATE_POOL], FINAL_K)

    else:  # CLEAR
        candidates = _bi_encoder_retrieve(question, chunks, embed_model, CANDIDATE_POOL)
        final_chunks = _cross_encoder_rerank(reranker, question, candidates, FINAL_K)

    return {"category": category, "chunks": final_chunks}
# --8<-- [end:pipeline]
