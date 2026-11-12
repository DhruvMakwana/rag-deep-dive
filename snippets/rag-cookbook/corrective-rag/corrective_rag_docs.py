"""
Documentation-only companion to corrective_rag.py.

Not run top-to-bottom, not imported by anything. Every function below is
self-contained — its own imports, its own inlined helpers — so
copy-pasting any single function into a fresh .py works on its own. See
corrective_rag.py for the actual runnable CLI.

If you change something in corrective_rag.py, mirror it here too —
these are NOT kept in sync automatically.
"""


# --8<-- [start:evaluator]
import re

import llm

EVALUATOR_PROMPT = """Score how well each passage below helps answer the \
question, on a scale of 1 (irrelevant) to 5 (directly and fully answers \
it). Score EVERY passage, even if most seem irrelevant.

Question: {question}

Passages:
{numbered_passages}

Respond with ONLY the scores, one per passage in order, comma-separated, \
wrapped in <scores></scores> tags. Example for 3 passages: <scores>2,5,1</scores>"""

UPPER_THRESHOLD = 4  # max score >= this -> Correct
LOWER_THRESHOLD = 2  # max score <= this -> Incorrect


def evaluate_retrieval(question: str, chunks: list[str], provider: str | None = None) -> list[int]:
    """One LLM call scores the WHOLE candidate set at once — same
    "batch it, don't loop it" principle as listwise reranking. Falls
    back to a score of 1 for any chunk the model doesn't return a
    parseable score for, so a malformed response degrades toward
    "treat as irrelevant" rather than crashing or silently misaligning
    scores to the wrong chunks."""
    numbered = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    response = llm.generate(EVALUATOR_PROMPT.format(question=question, numbered_passages=numbered), provider=provider)
    match = re.search(r"<scores>(.*?)</scores>", response, re.S)

    scores = [1] * len(chunks)
    if match:
        parts = match.group(1).replace(" ", "").split(",")
        for i, part in enumerate(parts[: len(chunks)]):
            if part.isdigit():
                scores[i] = max(1, min(5, int(part)))
    return scores


def classify(scores: list[int]) -> str:
    top = max(scores) if scores else 1
    if top >= UPPER_THRESHOLD:
        return "Correct"
    if top <= LOWER_THRESHOLD:
        return "Incorrect"
    return "Ambiguous"
# --8<-- [end:evaluator]


# --8<-- [start:refine]
import re

import llm

STRIP_RELEVANCE_PROMPT = """Score how well each short passage below \
helps answer the question, on a scale of 1 (irrelevant) to 5 (directly \
answers it).

Question: {question}

Passages:
{numbered_strips}

Respond with ONLY the scores, comma-separated, wrapped in <scores></scores> \
tags. Example for 3 passages: <scores>2,5,1</scores>"""


def refine_knowledge(question: str, chunks: list[str], provider: str | None = None, keep_threshold: int = 3) -> list[str]:
    """Even a chunk that scored well overall can have sentences that
    don't actually help — split into strips (sentences), re-score each
    one at this finer grain, and keep only the strips that individually
    clear the bar. Same idea as contextual compression, but gated
    specifically on the evaluator's "Correct" verdict rather than
    applied unconditionally to every retrieval."""
    strips = [s.strip() for c in chunks for s in re.split(r"(?<=[.!?])\s+", c) if len(s.strip()) > 20]
    if not strips:
        return chunks

    numbered = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(strips))
    response = llm.generate(STRIP_RELEVANCE_PROMPT.format(question=question, numbered_strips=numbered), provider=provider)
    match = re.search(r"<scores>(.*?)</scores>", response, re.S)

    if not match:
        return chunks

    parts = match.group(1).replace(" ", "").split(",")
    kept = [strips[i] for i, part in enumerate(parts[: len(strips)]) if part.isdigit() and int(part) >= keep_threshold]
    return kept if kept else chunks
# --8<-- [end:refine]


# --8<-- [start:websearch]
import llm

SEARCH_REWRITE_PROMPT = """Rewrite the question below into a short, \
keyword-style web search query — the way you'd type it into a search \
engine, not a full sentence. Return ONLY the search query.

Question: {question}"""


def rewrite_for_search(question: str, provider: str | None = None) -> str:
    return llm.generate(SEARCH_REWRITE_PROMPT.format(question=question), provider=provider).strip()


def web_search(query: str, max_results: int = 3) -> list[str]:
    import os

    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set. Get a free key at https://tavily.com and add it to .env.")

    client = TavilyClient(api_key=api_key)
    response = client.search(query, max_results=max_results)
    return [f"{r['title']}: {r['content']}" for r in response.get("results", [])]
# --8<-- [end:websearch]


# --8<-- [start:pipeline]
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

import llm

RETRIEVE_K = 5


def _naive_retrieve(question, chunks, embed_model, k):
    query_vec = embed_model.encode(question, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in chunk_vecs]
    top_indices = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_indices]


def _score_passages(prompt_prefix, question, passages, provider):
    numbered = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    response = llm.generate(
        f"{prompt_prefix}\n\nQuestion: {question}\n\nPassages:\n{numbered}\n\n"
        f"Respond with ONLY the scores, comma-separated, wrapped in <scores></scores> tags.",
        provider=provider,
    )
    match = re.search(r"<scores>(.*?)</scores>", response, re.S)
    scores = [1] * len(passages)
    if match:
        parts = match.group(1).replace(" ", "").split(",")
        for i, part in enumerate(parts[: len(passages)]):
            if part.isdigit():
                scores[i] = max(1, min(5, int(part)))
    return scores


def _refine_knowledge(question, chunks, provider):
    """The real strip-level refinement (see refine_knowledge above for
    the standalone version with full docs) — split into sentences,
    re-score each one, keep only strips that individually clear the bar."""
    strips = [s.strip() for c in chunks for s in re.split(r"(?<=[.!?])\s+", c) if len(s.strip()) > 20]
    if not strips:
        return chunks
    scores = _score_passages(
        "Score how well each short passage below helps answer the question, "
        "on a scale of 1 (irrelevant) to 5 (directly answers it).",
        question, strips, provider,
    )
    kept = [s for s, sc in zip(strips, scores) if sc >= 3]
    return kept if kept else chunks


def _web_search(question, provider):
    """The real web-search fallback (see rewrite_for_search and
    web_search above for the standalone versions) — rewrite to
    keywords, then a real Tavily API call."""
    search_query = llm.generate(
        f"Rewrite the question below into a short, keyword-style web search query "
        f"— the way you'd type it into a search engine, not a full sentence. "
        f"Return ONLY the search query.\n\nQuestion: {question}",
        provider=provider,
    ).strip()
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(search_query, max_results=3)
    return search_query, [f"{r['title']}: {r['content']}" for r in results.get("results", [])]


def corrective_rag(
    question: str,
    chunks: list[str],
    embed_model: SentenceTransformer,
    provider: str | None = None,
    k: int = RETRIEVE_K,
) -> dict:
    """Retrieve -> evaluate the whole candidate set in one call ->
    branch on the verdict -> generate. Each branch is a genuinely
    different corrective ACTION, not just a different confidence
    label."""
    retrieved = _naive_retrieve(question, chunks, embed_model, k)
    scores = _score_passages(
        "Score how well each passage below helps answer the question, on a scale "
        "of 1 (irrelevant) to 5 (directly and fully answers it). Score EVERY passage, "
        "even if most seem irrelevant.",
        question, retrieved, provider,
    )
    top = max(scores) if scores else 1
    verdict = "Correct" if top >= 4 else "Incorrect" if top <= 2 else "Ambiguous"

    trace = {"retrieved": len(retrieved), "scores": scores, "verdict": verdict}

    if verdict == "Correct":
        context = _refine_knowledge(question, retrieved, provider)
        trace["source"] = "internal (refined)"

    elif verdict == "Incorrect":
        search_query, web_results = _web_search(question, provider)
        context = web_results
        trace["source"] = "web"
        trace["search_query"] = search_query

    else:  # Ambiguous
        refined = _refine_knowledge(question, retrieved, provider)
        search_query, web_results = _web_search(question, provider)
        context = refined + web_results
        trace["source"] = "internal (refined) + web"
        trace["search_query"] = search_query

    trace["context"] = context
    context_text = "\n---\n".join(context) if context else "(no context available)"
    trace["answer"] = llm.generate(
        f"Answer the question using the context below. If context from the web is "
        f"included, you may cite it as such.\n\nContext:\n{context_text}\n\nQuestion: {question}",
        provider=provider,
    )
    return trace
# --8<-- [end:pipeline]
