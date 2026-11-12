"""
Documentation-only companion to retrieval_methods.py.

Not run top-to-bottom, not imported by anything. Every function below is
self-contained — its own imports, its own inlined constants — so
copy-pasting any single function into a fresh .py works on its own. See
retrieval_methods.py for the actual runnable CLI, and chunking-strategies/
or embedding-model-selection/ for why this two-file split exists.

If you change an algorithm in retrieval_methods.py, mirror it here too —
these are NOT kept in sync automatically.
"""


# --8<-- [start:mmr]
import numpy as np
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from sentence_transformers import SentenceTransformer


def mmr_retrieve(
    query: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 5, lambda_mult: float = 0.5
) -> list[str]:
    """Diversify top-k results: greedily pick the chunk that's most
    relevant to the query MINUS most similar to chunks already picked.
    `lambda_mult=1.0` is pure relevance (identical to plain top-k);
    `lambda_mult=0.0` is pure diversity, ignoring the query entirely."""
    query_vec = embed_model.encode(query, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    selected_indices = maximal_marginal_relevance(query_vec, list(chunk_vecs), lambda_mult=lambda_mult, k=k)
    return [chunks[i] for i in selected_indices]
# --8<-- [end:mmr]


# --8<-- [start:self_query]
import json

from sentence_transformers import SentenceTransformer


def _cosine_sim(a, b):
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


SELF_QUERY_PROMPT = """Given the natural-language query below, extract:
1. "semantic_query": the part that should be matched by meaning
2. "filters": any metadata constraints mentioned, as a JSON object with
   only the keys below — omit any key not clearly implied by the query.

Available metadata fields (this is the schema — without it, the model
has no way to know these are valid categorical values, not just query
content; this is exactly what a real self-query retriever's field-info
schema provides):
- "section": one of "remote_work", "expenses", "benefits"
- "year": an integer, e.g. 2023 or 2024

Return ONLY a JSON object like {{"semantic_query": "...", "filters": {{...}}}}.

Query: {query}"""


def self_query_retrieve(query: str, corpus: list[dict], generate_fn, k: int = 3) -> list[dict]:
    """Ask an LLM to split a natural-language query into a semantic part
    and structured metadata filters, then apply the filters BEFORE
    ranking by semantic similarity. `generate_fn` is any `str -> str`
    function that calls an LLM (e.g. an Anthropic/OpenAI client call) —
    swap in whichever provider you use."""
    response = generate_fn(SELF_QUERY_PROMPT.format(query=query))
    parsed = json.loads(response.strip().strip("`").removeprefix("json").strip())
    semantic_query = parsed["semantic_query"]
    filters = parsed.get("filters", {})

    candidates = [doc for doc in corpus if all(doc.get(key) == value for key, value in filters.items())]
    if not candidates:
        return []

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = embed_model.encode(semantic_query, convert_to_numpy=True)
    doc_vecs = embed_model.encode([doc["text"] for doc in candidates], convert_to_numpy=True, show_progress_bar=False)
    scored = sorted(zip(candidates, doc_vecs), key=lambda pair: -_cosine_sim(query_vec, pair[1]))
    return [doc for doc, _ in scored[:k]]
# --8<-- [end:self_query]


# --8<-- [start:hypothetical_questions]
from sentence_transformers import SentenceTransformer


def _cos(a, b):
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


HYPOTHETICAL_QUESTIONS_PROMPT = """Generate 3 short questions that the \
passage below could answer. Return one question per line, nothing else.

Passage: {passage}"""


def build_hypothetical_questions_index(chunks: list[str], generate_fn) -> list[dict]:
    """At INDEX time (not query time), generate plausible questions each
    chunk could answer, and embed those questions instead of the chunk's
    own declarative text — question-to-question similarity at query time
    is often stronger than question-to-answer similarity."""
    index = []
    for chunk in chunks:
        response = generate_fn(HYPOTHETICAL_QUESTIONS_PROMPT.format(passage=chunk))
        questions = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
        for question in questions:
            index.append({"question": question, "chunk": chunk})
    return index


def retrieve_via_hypothetical_questions(
    query: str, question_index: list[dict], embed_model: SentenceTransformer, k: int = 3
) -> list[str]:
    query_vec = embed_model.encode(query, convert_to_numpy=True)
    question_vecs = embed_model.encode(
        [item["question"] for item in question_index], convert_to_numpy=True, show_progress_bar=False
    )
    scored = sorted(zip(question_index, question_vecs), key=lambda pair: -_cos(query_vec, pair[1]))
    seen_chunks, results = set(), []
    for item, _ in scored:
        if item["chunk"] not in seen_chunks:
            results.append(item["chunk"])
            seen_chunks.add(item["chunk"])
        if len(results) >= k:
            break
    return results
# --8<-- [end:hypothetical_questions]


# --8<-- [start:hierarchical]
import numpy as np
from sentence_transformers import SentenceTransformer


def hierarchical_retrieve(
    query: str, chunks: list[str], embed_model: SentenceTransformer, section_size: int = 5, k: int = 3
) -> list[str]:
    """Narrow the search space in two stages instead of searching every
    chunk flat: group chunks into sections, embed each section (here, by
    embedding its concatenated text — production systems often use an
    LLM summary instead for a sharper section-level representation),
    find the best-matching section first, then rank only its chunks."""

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    sections = [chunks[i : i + section_size] for i in range(0, len(chunks), section_size)]
    section_texts = [" ".join(section) for section in sections]

    query_vec = embed_model.encode(query, convert_to_numpy=True)
    section_vecs = embed_model.encode(section_texts, convert_to_numpy=True, show_progress_bar=False)
    best_section_idx = int(np.argmax([cos(query_vec, vec) for vec in section_vecs]))

    candidate_chunks = sections[best_section_idx]
    chunk_vecs = embed_model.encode(candidate_chunks, convert_to_numpy=True, show_progress_bar=False)
    scored = sorted(zip(candidate_chunks, chunk_vecs), key=lambda pair: -cos(query_vec, pair[1]))
    return [chunk for chunk, _ in scored[:k]]
# --8<-- [end:hierarchical]


# --8<-- [start:raptor]
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


RAPTOR_SUMMARY_PROMPT = """Summarize the following related passages in \
2-3 sentences, capturing the common theme:

{passages}"""


def build_raptor_tree(
    chunks: list[str], embed_model: SentenceTransformer, generate_fn, n_clusters: int = 3
) -> dict:
    """Recursively cluster chunks by embedding similarity and summarize
    each cluster with an LLM, building one extra tree layer. (The full
    RAPTOR paper recurses this multiple times up to a single root
    summary; one layer is enough to demonstrate the mechanism without
    the full offline cost.)"""
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    n_clusters = min(n_clusters, len(chunks))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(chunk_vecs)

    summary_nodes = []
    for cluster_id in range(n_clusters):
        cluster_chunks = [c for c, label in zip(chunks, labels) if label == cluster_id]
        if not cluster_chunks:
            continue
        summary = generate_fn(RAPTOR_SUMMARY_PROMPT.format(passages="\n\n".join(cluster_chunks)))
        summary_nodes.append({"summary": summary, "cluster_chunks": cluster_chunks})

    return {"leaves": chunks, "summaries": summary_nodes}


def raptor_retrieve(query: str, tree: dict, embed_model: SentenceTransformer, k: int = 2) -> list[str]:
    """Search leaf chunks AND summary nodes together — a narrow factual
    question tends to match best at leaf level; a broad synthesis
    question tends to match best at a summary node."""
    query_vec = embed_model.encode(query, convert_to_numpy=True)

    candidates = [(leaf, "leaf") for leaf in tree["leaves"]]
    candidates += [(node["summary"], "summary") for node in tree["summaries"]]

    texts = [c[0] for c in candidates]
    vecs = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    scored = sorted(zip(candidates, vecs), key=lambda pair: -_cosine(query_vec, pair[1]))
    return [f"[{kind}] {text}" for (text, kind), _ in scored[:k]]
# --8<-- [end:raptor]


# --8<-- [start:adaptive]
import re

import numpy as np
from sentence_transformers import SentenceTransformer

DRAFT_PROMPT = "Answer this question in one sentence, from general knowledge: {question}"
CONFIDENCE_PROMPT = """Sentence: "{sentence}"

On a scale of 1 (pure guess) to 5 (certain, well-known fact), how \
confident are you this sentence is factually correct? Reply with ONLY \
the number."""


def adaptive_retrieve(
    question: str,
    chunks: list[str],
    embed_model: SentenceTransformer,
    generate_fn,
    confidence_threshold: int = 4,
) -> dict:
    """A simplified stand-in for FLARE: instead of monitoring token-level
    log-probabilities during generation (which needs logprob access most
    hosted chat APIs don't expose the same way), ask the model to
    self-report confidence in its own draft answer, and only retrieve
    (and regenerate) if that self-reported confidence is low. Cruder
    than real FLARE, but the same "retrieve only when needed" shape."""
    draft = generate_fn(DRAFT_PROMPT.format(question=question))
    confidence_response = generate_fn(CONFIDENCE_PROMPT.format(sentence=draft))
    confidence = int(re.search(r"\d", confidence_response).group())

    if confidence >= confidence_threshold:
        return {"answer": draft, "confidence": confidence, "retrieved": False}

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    query_vec = embed_model.encode(question, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    best_idx = int(np.argmax([cos(query_vec, vec) for vec in chunk_vecs]))
    context = chunks[best_idx]

    grounded_prompt = f"Using ONLY this context, answer in one sentence: {context}\n\nQuestion: {question}"
    grounded_answer = generate_fn(grounded_prompt)
    return {"answer": grounded_answer, "confidence": confidence, "retrieved": True, "context_used": context}
# --8<-- [end:adaptive]
