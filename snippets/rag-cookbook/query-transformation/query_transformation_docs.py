"""
Documentation-only companion to query_transformation.py.

Not run top-to-bottom, not imported by anything. Every function below is
self-contained — its own imports, its own inlined helpers — so
copy-pasting any single function into a fresh .py works on its own. See
query_transformation.py for the actual runnable CLI, and any other
recipe folder for why this two-file split exists.

Every function below calls `llm.generate(prompt, provider=provider)` —
llm.py is the same pluggable Anthropic/OpenAI/Ollama wrapper used across
this whole repo (see llm.py in this folder). `provider=None` uses
whatever LLM_PROVIDER is set to in your .env.

If you change something in query_transformation.py, mirror it here too
— these are NOT kept in sync automatically.
"""


# --8<-- [start:rewrite]
from sentence_transformers import SentenceTransformer

import llm

REWRITE_PROMPT = """Rewrite this query to be clearer and more specific \
for searching a technical document. Return ONLY the rewritten query, \
nothing else.

Query: {query}"""


def rewrite_query(query: str, provider: str | None = None) -> str:
    return llm.generate(REWRITE_PROMPT.format(query=query), provider=provider).strip()


def rewrite_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3, provider: str | None = None) -> list[str]:
    import numpy as np

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    rewritten = rewrite_query(question, provider=provider)
    query_vec = embed_model.encode(rewritten, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    sims = [cos(query_vec, v) for v in chunk_vecs]
    top_indices = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_indices]
# --8<-- [end:rewrite]


# --8<-- [start:multi_query]
import numpy as np
from sentence_transformers import SentenceTransformer

import llm

MULTI_QUERY_PROMPT = """Generate 3 different phrasings of the question \
below, each capturing a different angle on the same underlying \
information need. Return one per line, nothing else.

Question: {query}"""


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Each chunk's fused score = sum of 1/(k + rank) across every list
    it appears in. A chunk ranked highly by multiple query variations
    outscores one that only one variation liked."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda c: -scores[c])


def multi_query_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3, provider: str | None = None) -> list[str]:
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    response = llm.generate(MULTI_QUERY_PROMPT.format(query=question), provider=provider)
    variations = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
    variations.append(question)

    ranked_lists = []
    for variant in variations:
        query_vec = embed_model.encode(variant, convert_to_numpy=True)
        chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        sims = [cos(query_vec, v) for v in chunk_vecs]
        top_indices = np.argsort(sims)[::-1][:10]
        ranked_lists.append([chunks[i] for i in top_indices])

    return reciprocal_rank_fusion(ranked_lists)[:k]
# --8<-- [end:multi_query]


# --8<-- [start:hyde]
import numpy as np
from sentence_transformers import SentenceTransformer

import llm

HYDE_PROMPT = "Write a short passage that answers this question: {query}"


def hyde_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3, provider: str | None = None) -> list[str]:
    hypothetical_answer = llm.generate(HYDE_PROMPT.format(query=question), provider=provider)
    query_vec = embed_model.encode(hypothetical_answer, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in chunk_vecs]
    top_indices = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_indices]
# --8<-- [end:hyde]


# --8<-- [start:step_back]
import numpy as np
from sentence_transformers import SentenceTransformer

import llm

STEP_BACK_PROMPT = """What is a more general question that would help \
answer this specific question? Return ONLY the general question.

Specific question: {query}"""


def _naive_retrieve(question, chunks, embed_model, k):
    query_vec = embed_model.encode(question, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in chunk_vecs]
    top_indices = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_indices]


def step_back_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3, provider: str | None = None) -> list[str]:
    general_question = llm.generate(STEP_BACK_PROMPT.format(query=question), provider=provider).strip()

    specific_hits = _naive_retrieve(question, chunks, embed_model, k)
    general_hits = _naive_retrieve(general_question, chunks, embed_model, k)

    combined, seen = [], set()
    for chunk in specific_hits + general_hits:
        if chunk not in seen:
            combined.append(chunk)
            seen.add(chunk)
    return combined[:k]
# --8<-- [end:step_back]


# --8<-- [start:decompose]
import numpy as np
from sentence_transformers import SentenceTransformer

import llm

DECOMPOSE_PROMPT = """If the question below is asking multiple distinct \
things, break it into separate sub-questions, one per line. If it's \
already a single question, return it unchanged on one line.

Question: {query}"""


def decompose_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 6, provider: str | None = None) -> list[str]:
    """Larger k than a single-fact lookup needs — a compound question
    genuinely needs multiple slots, one set per sub-question, not one
    shared top-k the way a single-fact query would use."""

    def naive(q, kk):
        query_vec = embed_model.encode(q, convert_to_numpy=True)
        chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in chunk_vecs]
        top_indices = np.argsort(sims)[::-1][:kk]
        return [chunks[i] for i in top_indices]

    response = llm.generate(DECOMPOSE_PROMPT.format(query=question), provider=provider)
    sub_questions = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]

    per_question_k = max(1, k // len(sub_questions))
    combined, seen = [], set()
    for sub_q in sub_questions:
        for chunk in naive(sub_q, per_question_k):
            if chunk not in seen:
                combined.append(chunk)
                seen.add(chunk)
    return combined[:k]
# --8<-- [end:decompose]


# --8<-- [start:ircot]
import numpy as np
from sentence_transformers import SentenceTransformer

import llm

IRCOT_REASON_PROMPT = """You are answering a question using a document \
retrieval system — you do NOT have the document memorized, but each \
reasoning step you write will be used as a search query against it.

Question: {question}

Retrieved so far:
{context}

Write ONE short, specific, searchable sentence describing the next \
fact you need to look up to make progress (this sentence becomes the \
next search query, so keep it concrete and on-topic — do not ask what \
document this is, assume it's a technical ML paper). If the retrieved \
information above is already enough to answer, start your sentence \
with "ANSWER:" instead."""

FINAL_SYNTHESIS_PROMPT = """Question: {question}

Retrieved passages across {n_hops} rounds of search:
{context}

Using ONLY the passages above, give the final answer in 1-2 sentences. \
If they genuinely don't contain enough to answer, say so plainly."""


def ircot_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3, provider: str | None = None, max_hops: int = 3) -> dict:
    """Interleave reasoning and retrieval: generate one reasoning step,
    use THAT step as the next retrieval query, and repeat — each hop's
    retrieval is informed by the REAL retrieved text seen so far (not
    just the model's own prior reasoning sentences — that distinction
    matters: without real evidence to look at, the model has no way to
    judge whether it actually knows enough yet)."""

    def naive(q, kk):
        query_vec = embed_model.encode(q, convert_to_numpy=True)
        chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in chunk_vecs]
        top_indices = np.argsort(sims)[::-1][:kk]
        return [chunks[i] for i in top_indices]

    context_chunks: list[str] = []
    reasoning_trace: list[str] = []

    for _ in range(max_hops):
        context_text = "\n---\n".join(context_chunks) if context_chunks else "(nothing retrieved yet)"
        step = llm.generate(IRCOT_REASON_PROMPT.format(question=question, context=context_text), provider=provider).strip()
        reasoning_trace.append(step)

        if step.upper().startswith("ANSWER:"):
            break

        for chunk in naive(step, kk=2):
            if chunk not in context_chunks:
                context_chunks.append(chunk)

    final_context = "\n---\n".join(context_chunks) if context_chunks else "(nothing retrieved)"
    final_answer = llm.generate(
        FINAL_SYNTHESIS_PROMPT.format(question=question, n_hops=len(reasoning_trace), context=final_context),
        provider=provider,
    )

    return {"reasoning_trace": reasoning_trace, "retrieved_chunks": context_chunks[:k], "final_answer": final_answer}
# --8<-- [end:ircot]
