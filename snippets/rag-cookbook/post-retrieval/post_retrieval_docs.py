"""
Documentation-only companion to post_retrieval.py.

Not run top-to-bottom, not imported by anything. Every function below is
self-contained — its own imports, its own inlined helpers — so
copy-pasting any single function into a fresh .py works on its own. See
post_retrieval.py for the actual runnable CLI, and any other recipe
folder for why this two-file split exists.

Functions that call an LLM use `llm.generate(prompt, provider=provider)`
— llm.py is the same pluggable Anthropic/OpenAI/Ollama wrapper used
across this whole repo (see llm.py in this folder). `provider=None`
uses whatever LLM_PROVIDER is set to in your .env. cross_encoder_rerank
needs no LLM key at all — it's fully local.

If you change something in post_retrieval.py, mirror it here too —
these are NOT kept in sync automatically.
"""


# --8<-- [start:cross_encoder]
from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def cross_encoder_rerank(question: str, candidates: list[str], top_n: int = 3) -> list[str]:
    """Bi-encoder retrieval casts a wide net (candidates); the
    cross-encoder reads query+chunk TOGETHER in one forward pass per
    pair (bidirectional attention, no pre-computation possible) and
    outputs a real relevance score per candidate — then keeps only the
    top_n. Too expensive to run over a whole corpus, fine over an
    already-narrowed candidate set."""
    reranker = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    pairs = [(question, c) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: -x[0])]
    return ranked[:top_n]
# --8<-- [end:cross_encoder]


# --8<-- [start:compression]
import llm

COMPRESS_PROMPT = """From the passage below, extract ONLY the sentences \
that are relevant to answering the question. Return them verbatim, \
nothing added. If nothing in the passage is relevant, return "NONE".

Question: {query}

Passage:
{chunk}"""


def contextual_compress(question: str, chunks: list[str], provider: str | None = None) -> list[str]:
    """Runs one compression call per retrieved chunk (independent of
    each other — parallelizable in a real deployment) using a cheap
    LLM call, trimming each chunk down to just the sentences relevant
    to the question. Chunks that compress to "NONE" are dropped."""
    compressed = []
    for chunk in chunks:
        result = llm.generate(COMPRESS_PROMPT.format(query=question, chunk=chunk), provider=provider).strip()
        if result and result.upper() != "NONE":
            compressed.append(result)
    return compressed
# --8<-- [end:compression]


# --8<-- [start:listwise]
import re

import llm

LISTWISE_PROMPT = """You are ranking passages by relevance to a question.

Question: {query}

Passages:
{numbered_passages}

Think through which passages are most relevant if you like. When you \
are done, output your final ranking as passage numbers from MOST to \
LEAST relevant, wrapped in <ranking></ranking> tags, with numbers \
separated by commas and every number included exactly once.

Example: <ranking>3,1,4,2</ranking>"""


def llm_listwise_rerank(question: str, candidates: list[str], top_n: int = 3, provider: str | None = None) -> list[str]:
    """RankGPT-style listwise reranking: instead of scoring each
    candidate independently (pointwise, like a cross-encoder), the LLM
    sees the WHOLE candidate list at once and outputs a full relevance
    ordering in one call — it can reason about candidates relative to
    each other, not just each one in isolation. Costs one LLM call for
    the whole list instead of one lightweight forward pass per
    candidate, and typically has much higher latency than a
    cross-encoder for a comparable-size candidate set.

    An earlier version of this prompt instructed the model to "return
    ONLY the list, nothing else" — the model routinely ignored that and
    explained its reasoning first, and a parser trusting the whole
    response to be just numbers picked up stray digits from that prose
    ("Table 1", "[4]") and silently corrupted the ranking. Fighting a
    model's tendency to explain itself is a losing battle; instead this
    prompt lets it reason freely and asks it to wrap ONLY the final
    answer in an unambiguous <ranking> tag — trivial to extract
    correctly regardless of how much reasoning precedes it."""
    numbered = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(candidates))
    response = llm.generate(
        LISTWISE_PROMPT.format(query=question, numbered_passages=numbered), provider=provider
    ).strip()
    match = re.search(r"<ranking>(.*?)</ranking>", response, re.S)

    order = []
    if match:
        for tok in match.group(1).replace(" ", "").split(","):
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(candidates) and idx not in order:
                    order.append(idx)
    for i in range(len(candidates)):
        if i not in order:
            order.append(i)

    return [candidates[i] for i in order[:top_n]]
# --8<-- [end:listwise]
