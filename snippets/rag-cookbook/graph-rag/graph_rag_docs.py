"""
Documentation-only companion to graph_rag.py.

Not run top-to-bottom, not imported by anything. Every function below is
self-contained — its own imports, its own inlined helpers — so
copy-pasting any single function into a fresh .py works on its own. See
graph_rag.py for the actual runnable CLI.

If you change something in graph_rag.py, mirror it here too — these
are NOT kept in sync automatically.
"""


# --8<-- [start:extraction]
import re

import llm

EXTRACTION_PROMPT = """Extract entities and relationships from the \
passage below, for building a knowledge graph. Use short, canonical \
entity names (e.g. "Multi-Head Attention", not "the multi-head \
attention mechanism described here").

Passage:
{chunk}

For each relationship found, output one line as:
EntityA | relationship | EntityB

If no clear entity relationships exist in this passage, output NONE.
Wrap your output in <triples></triples> tags."""


def extract_triples(chunk: str, provider: str | None = None) -> list[tuple[str, str, str]]:
    response = llm.generate(EXTRACTION_PROMPT.format(chunk=chunk), provider=provider)
    match = re.search(r"<triples>(.*?)</triples>", response, re.S)
    if not match or match.group(1).strip().upper() == "NONE":
        return []

    triples = []
    for line in match.group(1).strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append((parts[0], parts[1], parts[2]))
    return triples
# --8<-- [end:extraction]


# --8<-- [start:dedup]
import numpy as np
from sentence_transformers import SentenceTransformer

DEDUP_SIMILARITY_THRESHOLD = 0.88


def deduplicate_entities(entity_names: list[str], embed_model: SentenceTransformer) -> dict[str, str]:
    """Different chunks refer to the same real entity with different
    surface forms ("Encoder" vs "the Encoder stack" vs "encoder").
    Exact-string matching alone misses these — this embeds every
    distinct name and greedily merges anything similar enough to an
    already-seen canonical name, keeping the first-seen form as
    canonical. Returns {raw_name: canonical_name}."""
    unique_names = sorted(set(entity_names))
    if not unique_names:
        return {}

    embeddings = embed_model.encode(unique_names, convert_to_numpy=True, show_progress_bar=False)
    canonical_names: list[str] = []
    canonical_embeddings: list[np.ndarray] = []
    mapping: dict[str, str] = {}

    for name, emb in zip(unique_names, embeddings):
        best_sim, best_idx = -1.0, -1
        for i, cand_emb in enumerate(canonical_embeddings):
            sim = float(np.dot(emb, cand_emb) / (np.linalg.norm(emb) * np.linalg.norm(cand_emb) + 1e-8))
            if sim > best_sim:
                best_sim, best_idx = sim, i

        if best_sim >= DEDUP_SIMILARITY_THRESHOLD:
            mapping[name] = canonical_names[best_idx]
        else:
            canonical_names.append(name)
            canonical_embeddings.append(emb)
            mapping[name] = name

    return mapping
# --8<-- [end:dedup]


# --8<-- [start:communities]
import networkx as nx


def detect_communities(graph: nx.MultiDiGraph) -> list[set[str]]:
    """Louvain community detection — a real, established graph
    clustering algorithm (networkx's built-in implementation), run
    purely on graph structure. No LLM call here; the LLM only
    summarizes the communities this algorithm finds."""
    undirected = nx.Graph(graph)  # collapses parallel/directed edges into a weighted simple graph
    if undirected.number_of_edges() == 0:
        return [{n} for n in undirected.nodes()]
    return list(nx.algorithms.community.louvain_communities(undirected, seed=42))
# --8<-- [end:communities]


# --8<-- [start:local_search]
import numpy as np
from sentence_transformers import SentenceTransformer

import llm


def local_search(
    question: str,
    graph,
    chunks: list[str],
    embed_model: SentenceTransformer,
    provider: str | None = None,
    top_entities: int = 3,
    hops: int = 1,
) -> dict:
    """Entity-link the query to graph nodes by embedding similarity
    (not exact string match — the query rarely uses the graph's exact
    canonical entity names), walk `hops` steps out from the matched
    entities, and gather every triple touched along the way, plus the
    original source chunks those triples came from."""
    nodes = list(graph.nodes())
    if not nodes:
        return {"matched_entities": [], "triples": [], "source_chunks": [], "answer": "(empty graph)"}

    node_embeddings = embed_model.encode(nodes, convert_to_numpy=True, show_progress_bar=False)
    query_vec = embed_model.encode(question, convert_to_numpy=True)
    sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in node_embeddings]
    matched = [nodes[i] for i in np.argsort(sims)[::-1][:top_entities]]

    frontier = set(matched)
    visited_edges = set()
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            for u, v, d in list(graph.out_edges(node, data=True)) + list(graph.in_edges(node, data=True)):
                visited_edges.add((u, v, d["relationship"], d["source_chunk"]))
                next_frontier.update([u, v])
        frontier |= next_frontier

    triples = [(u, rel, v) for (u, v, rel, _) in visited_edges]
    source_chunk_ids = sorted({sc for (_, _, _, sc) in visited_edges})
    source_chunks = [chunks[i] for i in source_chunk_ids]

    facts_text = "\n".join(f"{u} -> {rel} -> {v}" for u, rel, v in triples)
    context = f"Graph facts:\n{facts_text}\n\nSource passages:\n" + "\n---\n".join(source_chunks)
    answer = llm.generate(f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {question}", provider=provider)

    return {"matched_entities": matched, "triples": triples, "source_chunks": source_chunks, "answer": answer}
# --8<-- [end:local_search]


# --8<-- [start:global_search]
import llm

MAP_PROMPT = """Using ONLY the community summary below, write a brief \
partial answer to the question — if this summary isn't relevant to \
the question at all, respond with exactly: NOT RELEVANT

Community summary: {summary}

Question: {question}"""

REDUCE_PROMPT = """Combine the partial answers below into one coherent \
final answer to the question. Ignore any partial answer that said \
"NOT RELEVANT".

Question: {question}

Partial answers:
{partial_answers}"""


def global_search(question: str, community_summaries: list[str], provider: str | None = None) -> dict:
    """Map: ask each community summary independently whether it has a
    partial answer (parallelizable, independent calls — irrelevant
    communities self-report and get filtered out). Reduce: combine
    whatever partial answers came back into one final answer. This is
    what lets a broad synthesis question get answered from a
    whole-corpus view instead of whatever a single top-k happened to
    retrieve — at the real cost of one call per community, every query."""
    partial_answers = []
    for summary in community_summaries:
        resp = llm.generate(MAP_PROMPT.format(summary=summary, question=question), provider=provider).strip()
        if "NOT RELEVANT" not in resp.upper():
            partial_answers.append(resp)

    if not partial_answers:
        return {"partial_answers": [], "answer": "No community summary was relevant to this question."}

    combined = "\n---\n".join(partial_answers)
    answer = llm.generate(REDUCE_PROMPT.format(question=question, partial_answers=combined), provider=provider)
    return {"partial_answers": partial_answers, "answer": answer}
# --8<-- [end:global_search]
