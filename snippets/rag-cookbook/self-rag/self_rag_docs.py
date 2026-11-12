"""
Documentation-only companion to self_rag.py.

Not run top-to-bottom, not imported by anything. Every function below is
self-contained — its own imports, its own inlined helpers — so
copy-pasting any single function into a fresh .py works on its own. See
self_rag.py for the actual runnable CLI.

If you change something in self_rag.py, mirror it here too — these are
NOT kept in sync automatically.
"""


# --8<-- [start:relevance]
import re

import llm

RELEVANCE_PROMPT = """Is the passage below relevant to answering the \
question? A passage is relevant if it contains information that would \
help answer the question, even partially.

Question: {question}

Passage:
{chunk}

Respond with ONLY yes or no, wrapped in <relevant></relevant> tags."""


def grade_relevance(question: str, chunk: str, provider: str | None = None) -> bool:
    response = llm.generate(RELEVANCE_PROMPT.format(question=question, chunk=chunk), provider=provider)
    match = re.search(r"<relevant>(.*?)</relevant>", response, re.S)
    return bool(match) and match.group(1).strip().lower() == "yes"
# --8<-- [end:relevance]


# --8<-- [start:generate]
import llm

GENERATE_PROMPT = """Answer the question using ONLY the context below. \
If the context doesn't contain enough to answer, say so explicitly \
rather than guessing.

Context:
{context}

Question: {question}"""


def generate_answer(question: str, chunks: list[str], provider: str | None = None) -> str:
    context = "\n---\n".join(chunks) if chunks else "(no relevant context found)"
    return llm.generate(GENERATE_PROMPT.format(context=context, question=question), provider=provider)
# --8<-- [end:generate]


# --8<-- [start:groundedness]
import numpy as np
from sentence_transformers import CrossEncoder

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


def grade_groundedness(answer: str, chunks: list[str]) -> dict:
    """premise = the relevant retrieved context (concatenated),
    hypothesis = the generated answer. "entailment" means the context
    actually supports the answer; "contradiction" or "neutral" both
    count as NOT grounded — either the context disagrees with the
    answer, or it simply doesn't say enough to support it.

    This uses a local NLI (Natural Language Inference) cross-encoder
    instead of an LLM-judge call for this specific check — locally,
    in ~200-500ms, vs. a full LLM round-trip — and is comparably
    accurate for this kind of QA-over-retrieved-context faithfulness
    check specifically (per recent benchmarking; it's a worse fit for
    open-ended generation with no clear source document).

    Known simplification: this checks the answer as ONE hypothesis
    against the whole context as one premise, rather than decomposing
    the answer into individual claims and checking each separately —
    enough to demonstrate the mechanism, but can miss a
    partially-ungrounded multi-claim answer where most of it is
    supported and only one clause isn't."""
    reranker = CrossEncoder(NLI_MODEL_NAME)

    if not chunks:
        return {"grounded": False, "label": "no_context"}

    context = " ".join(chunks)
    scores = reranker.predict([(context, answer)])[0]
    label = reranker.config.id2label[int(np.argmax(scores))]
    return {"grounded": label == "entailment", "label": label}
# --8<-- [end:groundedness]


# --8<-- [start:usefulness]
import re

import llm

USEFULNESS_PROMPT = """Does the answer below actually address the \
question — regardless of whether it's factually correct? An answer \
that's grounded but off-topic, or that just says "I don't know" when \
a real answer was possible, is NOT useful.

Question: {question}

Answer: {answer}

Respond with ONLY yes or no, wrapped in <useful></useful> tags."""


def grade_usefulness(question: str, answer: str, provider: str | None = None) -> bool:
    response = llm.generate(USEFULNESS_PROMPT.format(question=question, answer=answer), provider=provider)
    match = re.search(r"<useful>(.*?)</useful>", response, re.S)
    return bool(match) and match.group(1).strip().lower() == "yes"
# --8<-- [end:usefulness]


# --8<-- [start:pipeline]
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

import llm

MAX_REGENERATIONS = 1


def _naive_retrieve(question, chunks, embed_model, k):
    query_vec = embed_model.encode(question, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in chunk_vecs]
    top_indices = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_indices]


def self_rag(
    question: str,
    chunks: list[str],
    embed_model: SentenceTransformer,
    provider: str | None = None,
    k: int = 5,
) -> dict:
    """Retrieve -> grade relevance -> generate -> grade groundedness ->
    grade usefulness -> return, regenerate once, or fall back to an
    honest "insufficient support" response instead of a second silent
    failure. See grade_relevance, generate_answer, grade_groundedness,
    and grade_usefulness above for each stage's own docs."""
    import re as _re

    reranker = CrossEncoder("cross-encoder/nli-deberta-v3-base")

    retrieved = _naive_retrieve(question, chunks, embed_model, k)

    def _relevant(q, c):
        resp = llm.generate(
            f"Is the passage below relevant to answering the question? A passage is relevant "
            f"if it contains information that would help answer the question, even partially.\n\n"
            f"Question: {q}\n\nPassage:\n{c}\n\n"
            f"Respond with ONLY yes or no, wrapped in <relevant></relevant> tags.",
            provider=provider,
        )
        m = _re.search(r"<relevant>(.*?)</relevant>", resp, _re.S)
        return bool(m) and m.group(1).strip().lower() == "yes"

    relevant = [c for c in retrieved if _relevant(question, c)]
    trace = {"retrieved": len(retrieved), "relevant": len(relevant), "attempts": []}

    for _ in range(MAX_REGENERATIONS + 1):
        context = "\n---\n".join(relevant) if relevant else "(no relevant context found)"
        answer = llm.generate(
            f"Answer the question using ONLY the context below. If the context doesn't "
            f"contain enough to answer, say so explicitly rather than guessing.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}",
            provider=provider,
        )

        if not relevant:
            groundedness = {"grounded": False, "label": "no_context"}
        else:
            scores = reranker.predict([(" ".join(relevant), answer)])[0]
            label = reranker.config.id2label[int(np.argmax(scores))]
            groundedness = {"grounded": label == "entailment", "label": label}

        useful_resp = llm.generate(
            f'Does the answer below actually address the question — regardless of whether '
            f'it\'s factually correct? An answer that\'s grounded but off-topic, or that just '
            f'says "I don\'t know" when a real answer was possible, is NOT useful.\n\n'
            f"Question: {question}\n\nAnswer: {answer}\n\n"
            f"Respond with ONLY yes or no, wrapped in <useful></useful> tags.",
            provider=provider,
        )
        m = _re.search(r"<useful>(.*?)</useful>", useful_resp, _re.S)
        useful = bool(m) and m.group(1).strip().lower() == "yes"

        trace["attempts"].append({"answer": answer, "groundedness": groundedness, "useful": useful})

        if groundedness["grounded"] and useful:
            trace["final_answer"] = answer
            trace["status"] = "ok"
            return trace

    trace["final_answer"] = (
        "I don't have enough reliably-supported information in the retrieved context to answer this confidently."
    )
    trace["status"] = "fell_back"
    return trace
# --8<-- [end:pipeline]
