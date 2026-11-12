"""
Documentation-only companion to modular_rag.py.

Not run top-to-bottom, not imported by anything. Every section below is
self-contained — its own imports, its own inlined helpers — so
copy-pasting any single section into a fresh .py works on its own. See
modular_rag.py for the actual runnable CLI.

If you change something in modular_rag.py, mirror it here too — these
are NOT kept in sync automatically.
"""


# --8<-- [start:heuristic]
COMPOUND_MARKERS = [" and ", " as well as "]
VAGUE_MARKERS = ["dont", "didnt", "just ", "how come", "everyone", "like everyone"]
BROAD_MARKERS = ["why is", "why did", "what lets", "what makes", "how does"]
NEEDS_CONTEXT_MARKERS = ["value", "rate did", "parameter"]


def heuristic_router(query: str) -> str:
    q = query.lower()

    if any(marker in q for marker in COMPOUND_MARKERS) and q.count("?") <= 1 and "," in query:
        return "COMPOUND"
    if any(marker in q for marker in VAGUE_MARKERS):
        return "VAGUE"
    if any(marker in q for marker in NEEDS_CONTEXT_MARKERS):
        return "NEEDS_CONTEXT"
    if any(q.startswith(marker) or marker in q for marker in BROAD_MARKERS):
        return "BROAD"
    return "CLEAR"
# --8<-- [end:heuristic]


# --8<-- [start:embedding]
import numpy as np
from sentence_transformers import SentenceTransformer

EXAMPLE_UTTERANCES = {
    "CLEAR": [
        "What dataset was used for evaluation?",
        "What programming language was used?",
        "Which company published this paper?",
    ],
    "VAGUE": [
        "why didnt they just do it the old way",
        "how come nobody thought of this before",
        "whats the deal with skipping that step",
    ],
    "COMPOUND": [
        "What is the accuracy, and what is the training time?",
        "How many layers are there, and what activation function is used?",
        "What is the cost, and what is the latency?",
    ],
    "BROAD": [
        "Why does this approach outperform previous methods?",
        "What makes this design more efficient overall?",
        "How does this system achieve better generalization?",
    ],
    "NEEDS_CONTEXT": [
        "What value was used for the learning rate?",
        "What rate did they set for weight decay?",
        "What parameter controls the batch size?",
    ],
}


def build_router_index(embed_model: SentenceTransformer) -> dict[str, np.ndarray]:
    """One-time setup: embed each category's example utterances and
    average them into a single centroid vector per category."""
    centroids = {}
    for category, examples in EXAMPLE_UTTERANCES.items():
        embeddings = embed_model.encode(examples, convert_to_numpy=True, show_progress_bar=False)
        centroids[category] = embeddings.mean(axis=0)
    return centroids


def embedding_router(query: str, centroids: dict[str, np.ndarray], embed_model: SentenceTransformer) -> str:
    query_vec = embed_model.encode(query, convert_to_numpy=True)
    sims = {
        category: float(np.dot(query_vec, centroid) / (np.linalg.norm(query_vec) * np.linalg.norm(centroid) + 1e-8))
        for category, centroid in centroids.items()
    }
    return max(sims, key=sims.get)
# --8<-- [end:embedding]


# --8<-- [start:llm_router]
import re

import llm

CATEGORIES = ["CLEAR", "VAGUE", "COMPOUND", "BROAD", "NEEDS_CONTEXT"]

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


def llm_router(question: str, provider: str | None = None) -> str:
    response = llm.generate(ROUTER_PROMPT.format(query=question), provider=provider)
    match = re.search(r"<category>(.*?)</category>", response, re.S)
    category = match.group(1).strip() if match else "CLEAR"
    return category if category in CATEGORIES else "CLEAR"
# --8<-- [end:llm_router]
