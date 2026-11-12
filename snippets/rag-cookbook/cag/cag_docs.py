"""
Documentation-only companion to cag.py.

Every function below is self-contained (own imports, no cross-function
dependencies) so each one can be extracted and pasted standalone into the
blog. Not run as a script, not kept in sync with cag.py automatically.
"""


# --8<-- [start:generate_fresh]
def generate_fresh(document_text: str, question: str, model: str) -> tuple:
    """Plain long-context stuffing: the entire document is sent as the
    system prompt on every call, with no cache_control at all. This is
    the baseline CAG's caching layer is compared against -- not RAG, but
    the "just put everything in context, every time" alternative."""
    import time

    import anthropic

    client = anthropic.Anthropic()
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=document_text,
        messages=[{"role": "user", "content": question}],
    )
    elapsed = time.perf_counter() - start

    answer = next(block.text for block in response.content if block.type == "text")
    usage = {
        "input_tokens": response.usage.input_tokens,
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "output_tokens": response.usage.output_tokens,
    }
    return answer, usage, elapsed
# --8<-- [end:generate_fresh]


# --8<-- [start:generate_cached]
def generate_cached(document_text: str, question: str, model: str, ttl: str = "5m") -> tuple:
    """Same document, same question shape, but the system prompt carries a
    `cache_control` breakpoint. The FIRST call against a given document
    writes the cache (billed at a real premium: 1.25x base input price for
    a 5-minute TTL). Every subsequent call within the TTL, against the same
    exact prefix, reads from the cache instead of reprocessing it (billed
    at 0.1x base input price)."""
    import time

    import anthropic

    client = anthropic.Anthropic()
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[{"type": "text", "text": document_text, "cache_control": {"type": "ephemeral", "ttl": ttl}}],
        messages=[{"role": "user", "content": question}],
    )
    elapsed = time.perf_counter() - start

    answer = next(block.text for block in response.content if block.type == "text")
    usage = {
        "input_tokens": response.usage.input_tokens,
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "output_tokens": response.usage.output_tokens,
    }
    return answer, usage, elapsed
# --8<-- [end:generate_cached]


# --8<-- [start:cost_from_usage]
def cost_from_usage(usage: dict) -> float:
    """Real dollar cost for one call, computed from its actual usage
    fields and Sonnet 5's published per-token rates -- not an estimate.
    Cache write and cache read are MULTIPLIERS on the base input price,
    not separate flat rates."""
    INPUT_PRICE_PER_MTOK = 2.00
    OUTPUT_PRICE_PER_MTOK = 10.00
    CACHE_WRITE_5M_MULTIPLIER = 1.25
    CACHE_READ_MULTIPLIER = 0.1

    input_cost = usage["input_tokens"] / 1_000_000 * INPUT_PRICE_PER_MTOK
    cache_write_cost = usage["cache_creation_input_tokens"] / 1_000_000 * INPUT_PRICE_PER_MTOK * CACHE_WRITE_5M_MULTIPLIER
    cache_read_cost = usage["cache_read_input_tokens"] / 1_000_000 * INPUT_PRICE_PER_MTOK * CACHE_READ_MULTIPLIER
    output_cost = usage["output_tokens"] / 1_000_000 * OUTPUT_PRICE_PER_MTOK
    return input_cost + cache_write_cost + cache_read_cost + output_cost
# --8<-- [end:cost_from_usage]


# --8<-- [start:load_document]
def load_document_text(pdf_path: str) -> str:
    """The entire document, not chunks of it -- CAG's whole premise is
    that the corpus is small enough to load in full."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)
# --8<-- [end:load_document]
