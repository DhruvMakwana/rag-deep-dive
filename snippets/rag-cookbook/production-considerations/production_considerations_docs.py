"""
Documentation-only companion to production_considerations.py.

Every function below is self-contained (own imports, own constants, no
cross-function dependencies) so each one can be extracted and pasted
standalone into the blog. Not run as a script, not kept in sync with
production_considerations.py automatically.
"""


# --8<-- [start:retry_backoff]
def demo_retry_backoff() -> str:
    """Retries a failing call with exponential backoff via `tenacity`.
    `retry_if_exception_type` only retries the specific exception a
    transient failure would raise; `stop_after_attempt` bounds the
    worst-case cost of a call that never recovers."""
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    attempts = {"count": 0}

    def flaky_call() -> str:
        attempts["count"] += 1
        if attempts["count"] <= 2:  # fails the first two times, then succeeds
            raise ConnectionError(f"reranker service unavailable (attempt {attempts['count']})")
        return f"reranked results (succeeded on attempt {attempts['count']})"

    @retry(
        retry=retry_if_exception_type(ConnectionError),
        wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def call_with_retry() -> str:
        return flaky_call()

    return call_with_retry()
# --8<-- [end:retry_backoff]


# --8<-- [start:circuit_breaker]
def demo_circuit_breaker(num_calls: int = 6) -> list[str]:
    """Wraps a persistently failing call in a `pybreaker` circuit
    breaker. After `fail_max` consecutive failures the breaker opens
    and short-circuits further calls immediately -- no request is sent
    at all -- until `reset_timeout` elapses."""
    import time

    import pybreaker

    breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=2)

    @breaker
    def call_down_service() -> None:
        time.sleep(0.05)  # simulates real request latency before the failure is known
        raise ConnectionError("reranker service is down")

    log = []
    for i in range(1, num_calls + 1):
        start = time.monotonic()
        try:
            call_down_service()
        except pybreaker.CircuitBreakerError:
            log.append(f"call {i}: short-circuited immediately ({time.monotonic() - start:.4f}s, no request sent)")
        except ConnectionError:
            log.append(f"call {i}: request sent and failed ({time.monotonic() - start:.4f}s)")
    return log
# --8<-- [end:circuit_breaker]


# --8<-- [start:latency_breakdown]
def latency_breakdown(question: str, chunks: list, embed_fn, generate_fn) -> dict:
    """Real per-stage timing across a naive RAG pipeline, not just an
    end-to-end number -- `embed_fn(text)` returns a normalized
    embedding, `generate_fn(prompt)` returns a generated string."""
    import time

    import numpy as np

    timings = {}

    start = time.monotonic()
    query_vec = embed_fn(question)
    timings["embed_query"] = time.monotonic() - start

    start = time.monotonic()
    chunk_embeddings = np.array([embed_fn(c) for c in chunks])
    timings["embed_corpus"] = time.monotonic() - start

    start = time.monotonic()
    sims = chunk_embeddings @ query_vec
    top_k = np.argsort(-sims)[:3]
    retrieved = [chunks[i] for i in top_k]
    timings["retrieval_search"] = time.monotonic() - start

    start = time.monotonic()
    context_block = "\n---\n".join(retrieved)
    generate_fn(f"Answer the question using the context below.\n\nContext:\n{context_block}\n\nQuestion: {question}")
    timings["generation"] = time.monotonic() - start

    return timings
# --8<-- [end:latency_breakdown]


# --8<-- [start:cost_tracking]
def track_generation_usage(prompt: str) -> dict:
    """Extracts real per-call token usage from a local Ollama response
    -- `prompt_eval_count` and `eval_count` are the same shape of data
    a hosted provider exposes via `response.usage`, so this aggregation
    pattern is provider-agnostic."""
    import requests

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen3:4b", "prompt": prompt, "stream": False},
        timeout=180,
    ).json()
    return {"prompt_tokens": response.get("prompt_eval_count", 0), "completion_tokens": response.get("eval_count", 0)}
# --8<-- [end:cost_tracking]


# --8<-- [start:semantic_cache]
def semantic_cache_lookup(query_vec, cache: list, threshold: float = 0.92):
    """Checks a query embedding against a cache of past query
    embeddings; returns the cached answer above `threshold` cosine
    similarity, or None on a miss. `cache` is a list of
    {"query_vec", "question", "answer"} dicts, mutated in place by the
    caller on a miss."""
    best_sim, best_entry = 0.0, None
    for entry in cache:
        sim = float(query_vec @ entry["query_vec"])
        if sim > best_sim:
            best_sim, best_entry = sim, entry

    if best_entry is not None and best_sim >= threshold:
        return best_entry["answer"], best_sim
    return None, best_sim
# --8<-- [end:semantic_cache]
