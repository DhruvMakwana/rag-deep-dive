# Production Considerations

A RAG pipeline that works in a notebook and one that survives production traffic differ in specific, learnable ways: what happens when a dependency starts failing, what actually costs money at scale, what a generic uptime dashboard can't see, and where throughput actually breaks. This page covers all four with real, tested code for the reliability and cost/monitoring mechanics, and current, sourced numbers for the parts that are inherently about infrastructure scale rather than something to demo on a laptop.

!!! example "Hands-on"
    The reliability, latency, cost, and caching demos below are runnable: [**Production Considerations →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/production-considerations) in the code repo. No paid API needed — generation runs through a local Ollama model.

??? abstract "TL;DR — quick revision"
    - **Retries, fallbacks, and circuit breakers are three different tools, not one** — retries handle transient failures, fallbacks route to a secondary provider, circuit breakers proactively stop sending requests to something that's already failing; this page's own circuit-breaker demo shows the real latency saved by not retrying a doomed call
    - **Semantic caching and prompt caching solve different problems** — semantic caching skips the whole pipeline for a near-duplicate query (this page measured a real 33% hit rate, with a genuine miss right at the similarity threshold boundary); prompt caching (Anthropic's, mechanically confirmed) caches a shared prefix within calls that still each generate fresh output
    - **Standard uptime monitoring cannot see RAG's actual failure mode** — a system serving wrong answers stays fast and returns 200 OK on every request, so retrieval-quality drift needs its own monitoring, not a byproduct of latency/error-rate dashboards
    - **Scalability at real production numbers is a GPU-throughput and region-placement problem for embedding, and a right-sized-model problem for every auxiliary LLM call** — with real published numbers for both, not estimates

## Reliability: retries, fallbacks, and circuit breakers are different tools

It's tempting to treat these as one "handle failures" bucket, but they solve different problems and combine into a layered defense, not a single fallback path.

**Retries** exist for transient failures — a brief rate limit, a cold start, a dropped connection — where trying again with backoff is likely to succeed. `tenacity` is the standard current Python library for this:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/production-considerations/production_considerations_docs.py:retry_backoff"
```

Measured against a call that fails twice, then succeeds:

```text
attempt 1: failed: reranker service unavailable (attempt 1)
attempt 2: failed: reranker service unavailable (attempt 2)
attempt 3: success
```

Retrying blindly has a real cost: too many retries on a call that's failing for a *non*-transient reason just adds latency and can trigger a "retry storm" that makes an already-struggling dependency worse.

**Circuit breakers** exist for the opposite case — a dependency that's failing *persistently*, where retrying is pure waste. Once a failure threshold is crossed, the breaker opens and short-circuits further calls immediately, with no request sent at all, until a cooldown period passes and it allows a single probe request through to check recovery:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/production-considerations/production_considerations_docs.py:circuit_breaker"
```

Measured against a persistently down service, 6 calls:

```text
call 1: request sent and failed (0.0538s)
call 2: request sent and failed (0.0509s)
call 3: short-circuited immediately (0.0557s, no request sent)
call 4: short-circuited immediately (0.0001s, no request sent)
call 5: short-circuited immediately (0.0000s, no request sent)
call 6: short-circuited immediately (0.0000s, no request sent)
```

Only 3 of the 6 calls actually reached the service — the breaker opened after the third failure and every subsequent call returned instantly instead of waiting on a doomed request. That's the real, measured value: not just fewer errors, but genuinely avoided latency.

**Fallbacks** are a third, separate tool — routing to a secondary provider or model when the primary fails, rather than retrying the same one or giving up. Worth knowing that a fallback isn't automatically independent risk: a primary and secondary that share the same upstream infrastructure can fail together, so a fallback is only as good as its actual failure-domain independence from the primary.

LLM-specific reasons this space needs more care than generic microservice circuit-breaker theory: LLM providers have rate limits that can trigger mid-conversation without warning, latency variance that can span 800ms to 15+ seconds on the *same* endpoint, and a failure mode generic breakers don't model at all — a 200 OK response that's silently lower-quality under peak load, which no HTTP-status-based breaker would ever trip on.

!!! note "Circuit-breaker state needs to be shared across workers"
    `pybreaker`'s in-memory state is thread-safe but per-process — running multiple worker processes (the normal case in production) means each one tracks failures independently unless the breaker's state is backed by something shared (Redis is the common choice). A breaker that only sees a fraction of total traffic per process opens far later than intended.

### Graceful degradation when retrieval itself fails

Retry and circuit-breaker logic assumes the *call* can fail. A RAG-specific question is what happens when retrieval succeeds mechanically but returns nothing useful, or the vector DB is unreachable entirely. [Corrective RAG (CRAG)](corrective-rag.md) is the closest thing to a real, research-grade answer to this: it explicitly scores retrieved documents for relevance *after* retrieval, and triggers a live web search as the correction step when they're judged irrelevant — a genuine fallback ladder, not just a retry of the same failing retriever.

The anti-pattern worth naming explicitly: caching one generic "sorry, something went wrong" response as the fallback for every degraded query. Every degraded request then returns identical, unhelpful text regardless of what was actually asked — users notice this immediately, and it's strictly worse than a narrower, query-aware degradation path (a secondary keyword-based retriever, a curated FAQ fallback, or an honest "I don't have enough information" that at least engages with the actual question).

## Cost management: two different caching levers

**Prompt caching** (Anthropic's implementation, confirmed directly from current docs) caches a shared *prefix* of a request — the parts of a prompt (a system prompt, static instructions, a stable block of retrieved context) that stay identical across calls — so only the actually-varying suffix needs fresh processing. Current mechanics: a cache write costs 1.25× the base input-token price for a 5-minute TTL, or 2× for a 1-hour TTL; a cache read costs roughly 0.1× the base input price (a 90% discount) on most current models. The cache clock resets on every read, so a steady stream of requests keeps an entry alive indefinitely without repaying the write cost. There's a minimum cacheable prefix length (1,024 tokens on Sonnet-class models, varying by model tier) — below it, a cache directive is silently ignored, no error. This maps directly onto RAG's repeated-system-prompt-plus-retrieved-context pattern: cache everything up to the final user turn, and only that turn needs fresh processing on every call.

**Semantic caching** solves a different problem: skipping the *entire* pipeline — embedding, retrieval, and generation — for a query that's a near-duplicate of one already answered, not just reusing a prefix within a call that still generates fresh output. Real current tools (GPTCache, Redis's semantic caching) work the same way mechanically: embed the incoming query, check cosine similarity against previously-cached query embeddings, and serve the cached response directly on a hit above some similarity threshold.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/production-considerations/production_considerations_docs.py:semantic_cache"
```

Measured against six real questions, three of them genuine paraphrases of earlier ones, at a 0.85 similarity threshold:

```text
CACHE MISS (best_sim=0.000)  'How many attention heads did they use?'
CACHE MISS (best_sim=0.126)  "What is the model's embedding dimension?"
CACHE HIT  (sim=0.857)  'How many attention heads does the model have?'  ->  matched 'How many attention heads did they use?'
CACHE MISS (best_sim=0.286)  'What optimizer was used for training?'
CACHE HIT  (sim=0.864)  "What's the embedding size of the model?"  ->  matched "What is the model's embedding dimension?"
CACHE MISS (best_sim=0.826)  'What optimizer did they use to train the model?'

2 hits, 4 misses out of 6 queries (33% hit rate)
```

!!! note "The threshold is a real precision/recall trade-off, not a formality"
    At a stricter 0.92 threshold, all three genuine paraphrases missed — even "What's the embedding size of the model?" against "What is the model's embedding dimension?" only reached 0.864 cosine similarity with `all-MiniLM-L6-v2` embeddings, comfortably below 0.92. Lowering to 0.85 caught two of the three real paraphrases, but the third — "What optimizer did they use to train the model?" against "What optimizer was used for training?" — still missed at 0.826, right at the boundary. This is the real trade-off semantic caching guides consistently point at: too high a threshold and genuinely repeated questions still cost a full pipeline run; too low and a cache hit can serve a wrong answer to a question that only *looks* similar. Current guidance from Redis's own semantic-caching documentation frames a false-positive-hit rate above roughly 3–5% as a signal to fix the underlying architecture, not just tighten the threshold further.

Cost drivers worth naming directly, since they set the priority order for where to actually spend optimization effort: LLM generation is consistently the largest ongoing cost line in a production RAG system, vector DB hosting is usually the smallest, and embedding-generation cost is a function of tokens processed × GPU cost per hour ÷ throughput — meaning it scales with corpus size and update frequency, not query volume.

## Monitoring: what standard uptime dashboards cannot see

A RAG system that hallucinates, retrieves the wrong chunks, or silently degrades under load still returns a fast, 200 OK response on every request. Latency, error rate, and throughput — the metrics a generic API monitoring setup already tracks — capture none of the failure modes that actually matter for answer quality. This is the single most important framing for this section: retrieval-quality monitoring has to be built deliberately, not inherited for free from infrastructure monitoring.

**Current tools built specifically for this** (not generic APM): [Arize Phoenix](https://arize.com/blog/llm-tracing-and-observability-with-arize-phoenix/) traces a RAG pipeline as a span tree with dedicated span types for embedding, retrieval (including the actual retrieved chunks and their similarity scores, not just a "retrieval happened" log line), and generation — and runs LLM-assisted evals at both the per-chunk level (a precision@k-style relevance judgment) and the full-answer level. Langfuse and LangSmith offer similar full trace trees with token usage, latency, and cost per span, plus the ability to build eval datasets directly from real production traffic rather than only a static eval set. Helicone takes a lighter-weight, proxy-based approach — no code/SDK change beyond swapping a base URL — trading pipeline-level diagnostic depth for much lower integration effort; it's a better fit for fast cost/usage tracking than for diagnosing exactly where a RAG pipeline went wrong.

**Detecting drift without labeling every live query.** The practical answer isn't waiting for enough labeled production data to retrain a judge — it's running a small, fixed evaluation set (the same kind of curated question/reference set already used throughout this site, via [RAGAS](ragas.md) or the metrics on the [Evaluation Metrics](evaluation-metrics.md) page) against the live system on a recurring schedule, not just at deploy time, and alerting when the score on that fixed set drops. The alert fires because measured quality dropped, not because anything threw an exception — which is exactly the class of failure a normal error-rate dashboard is structurally blind to. A cheaper complementary practice: a periodic manual spot-check of a random sample of real production answers against source documents, to catch failure patterns the fixed eval set doesn't happen to cover.

## Scalability: real numbers, not estimates

**Embedding throughput is a GPU-allocation and region-placement problem at real scale**, not a code-optimization one. A real, published example: generating embeddings for ~30 million product reviews with a 7B-parameter embedding model on a single-region setup needed 47 L4 GPUs and took 20 hours at $710; spreading the same job across 12 regions (mixing spot and on-demand instances to find spare capacity in less-contended regions) cut that to 2.3 hours at $277 — a 9× speedup and 61% cost reduction from placement alone, no model or code change. Batch-size and precision choices matter too: length-based batch sorting to minimize padding waste is commonly cited as a 20–40% compute saving, and mixed-precision (FP16) inference cuts memory with minimal quality loss. On the buy-vs-build question specifically, a real cost-crossover exists: API-based embedding pricing (roughly $0.02–0.06 per million tokens from major providers) beats self-hosting a dedicated GPU below roughly 50–100 million tokens per month of embedding volume — self-hosting only wins once throughput is high enough to keep that GPU consistently busy, not from the first token processed.

**Right-sizing every auxiliary LLM call is a scalability lever, not just a cost one.** Every pipeline-stage LLM call beyond the main generation — routing, relevance grading (CRAG's post-retrieval evaluator), reflection/critique (Self-RAG's `[IsRel]`/`[IsSup]` tokens), reranking, compression — is structurally a narrow classification or scoring task, not open-ended generation, and doesn't need a frontier-model-sized call to do it well. Recent research makes this argument directly for RAG specifically: using full-scale LLMs for binary routing/grading decisions in a high-throughput pipeline introduces real, avoidable computational redundancy, and a small, purpose-tuned model can match a much larger general model's routing accuracy at roughly an order of magnitude lower latency. This is why small/cheap models for grading and routing shows up repeatedly across [Self-RAG](self-rag.md), [CRAG](corrective-rag.md), and [Agentic RAG](agentic-rag.md) — it's the same right-sizing principle applied at every stage that isn't the final answer generation itself, and it compounds: a system with three auxiliary LLM calls per query pays that "narrow task on a frontier model" tax three times over if none of them are right-sized.

## This page's own pipeline, measured

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/production-considerations/production_considerations_docs.py:latency_breakdown"
```

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/production-considerations/production_considerations_docs.py:cost_tracking"
```

Real per-stage latency for a single query against this page's naive RAG pipeline:

```text
embed_query        0.561s  (  5.4%)
embed_corpus       0.457s  (  4.4%)
retrieval_search   0.026s  (  0.2%)
generation         9.343s  ( 90.0%)
total              10.387s
```

Generation dominates end-to-end latency by a wide margin — the same pattern that makes the small-model-for-auxiliary-calls argument above concrete: shaving milliseconds off retrieval matters far less here than making sure every non-essential LLM call in a pipeline isn't accidentally running on the expensive model.

Real per-call token usage, aggregated across 4 questions against a local Ollama model:

```text
generation: 4 calls, 1333 prompt tokens, 2608 completion tokens
avg prompt tokens/call: 333
avg completion tokens/call: 652
```

Completion tokens nearly double prompt tokens here — a real, measured consequence of using a reasoning-capable local model that generates visible thinking tokens before its final answer, worth knowing since it's a genuine, provider-specific cost driver that a naive "prompt tokens are usually the bulk of the cost" assumption would miss.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: Production Considerations">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team's RAG system depends on an external reranking service that has started failing 100% of the time due to an outage on the provider's side. The team's current code retries every failed call up to 5 times with exponential backoff before giving up.",
      "question": "Based on this page, what's the problem with relying on retry logic alone in this specific scenario, and what should be added?",
      "options": [
        "Nothing is wrong -- retries with backoff already handle this case correctly, since exponential backoff naturally spaces out requests enough to ride out a provider-side outage without needing a separate mechanism.",
        "Retries are designed for transient failures where trying again is likely to succeed; a persistently failing dependency should instead trip a circuit breaker, which stops sending requests entirely for a cooldown period rather than repeatedly waiting on a doomed call.",
        "The team should remove retry logic and instead route every reranker call through a fallback to a secondary reranking provider, since fallbacks are the correct tool whenever a dependency is failing.",
        "The fix is to increase the retry count and lengthen the backoff delay between attempts, since a longer backoff window gives a struggling dependency more time to recover before the next attempt."
      ],
      "correct": 1,
      "explanations": [
        "This page states the opposite: retries are designed for transient failures where trying again is likely to succeed -- against a dependency failing 100% of the time, backoff spacing doesn't fix the underlying problem, it just delays wasted latency and risks compounding into a retry storm; a circuit breaker, not more spaced-out retries, is the tool for a persistent failure.",
        "Correct. This page's own measured circuit-breaker demo shows exactly this: after a failure threshold is crossed, further calls are short-circuited instantly (0.0000s, no request sent) instead of each one waiting out a real request that's certain to fail -- a real, measured latency saving on top of not hammering an already-struggling dependency.",
        "This page distinguishes these as different tools for different roles: a fallback routes to a secondary provider (useful, but a separate concern from a failing primary that's still being called), while a circuit breaker is specifically what stops sending wasted requests to a persistently failing dependency -- the scenario's core problem is retries not backing off from a call, not the absence of a secondary provider.",
        "This page states more retries and longer backoff against a non-transient, 100%-failing dependency still just adds latency without addressing the cause -- what actually saves measured latency here, per this page's own circuit-breaker demo, is short-circuiting calls entirely once a failure threshold is crossed, not tuning retry parameters further."
      ]
    },
    {
      "scenario": "A team implements a semantic cache for their RAG chatbot with a very high similarity threshold (0.97) to avoid ever serving a wrong cached answer, and later observes that almost no real user queries ever hit the cache, even when users are clearly asking the same question in slightly different words.",
      "question": "Based on this page's own measured example, what's the most likely explanation?",
      "options": [
        "Semantic caching works better with a stricter, not looser, similarity threshold, since a 0.97 threshold should already be conservative enough to catch nearly all genuine paraphrases while avoiding false hits.",
        "A very high similarity threshold can be stricter than the actual embedding similarity of genuine paraphrases, causing real repeated questions to still miss the cache -- this page's own measurement showed genuine paraphrases only reaching 0.826-0.864 similarity, well under a 0.92+ threshold.",
        "The embeddings model must be poorly suited to this domain, since a well-trained embedding model should score two different phrasings of the same underlying question above 0.97 similarity by default.",
        "The cache size limit must have been reached, which is why no new entries are being matched."
      ],
      "correct": 1,
      "explanations": [
        "This page's own measurement shows the opposite: genuine paraphrases only reached 0.826-0.864 cosine similarity, comfortably below a 0.97 threshold -- a stricter threshold makes this specific problem worse, not better, since it's already causing real repeated questions to miss the cache.",
        "Correct. This page measured this exact effect directly: real paraphrases of the same question scored 0.826-0.864 cosine similarity with a real embedding model -- comfortably below a 0.92 threshold, which is why raising the threshold too high causes genuine repeated questions to still miss the cache.",
        "This page's own measured numbers show real, well-functioning paraphrase pairs landing at 0.826-0.864 with all-MiniLM-L6-v2 -- there's no stated expectation that a working embedding model should push genuine paraphrases above 0.97; the threshold, not the model, is what's mismatched to that real similarity range.",
        "This scenario describes a threshold problem, not a capacity problem -- nothing in the scenario suggests cache entries are being evicted due to a size limit."
      ]
    },
    {
      "scenario": "A production RAG system's uptime dashboard shows 99.9% availability, low error rates, and fast response times across the last month, yet a manual review of recent user conversations reveals the system has been giving factually wrong answers to a specific category of question for weeks.",
      "question": "Per this page, what does this scenario illustrate about production monitoring for RAG systems?",
      "options": [
        "The uptime dashboard's numbers reflect the same underlying problem, since a system serving factually wrong answers should also show elevated error rates once enough users are affected.",
        "Standard API/infrastructure monitoring (latency, error rate, uptime) cannot detect retrieval-quality or answer-quality degradation, since a RAG system serving wrong answers can still return fast, error-free 200 OK responses on every request -- quality monitoring has to be built as a separate, deliberate layer.",
        "This means the wrong answers are a minor, low-priority issue relative to the dashboard's numbers, since infrastructure health is what most directly determines whether users can successfully get a response at all.",
        "This specific failure mode can only be caught after the fact through user complaints, since no proactive monitoring technique can evaluate answer quality on live traffic before users are affected."
      ],
      "correct": 1,
      "explanations": [
        "This page's central point is exactly that this doesn't happen -- a RAG system serving wrong answers keeps returning fast, 200 OK responses, because nothing about generating an incorrect-but-well-formed answer trips an HTTP error or a latency spike; that's precisely why quality monitoring can't be inferred from infrastructure metrics.",
        "Correct. This page states this as the central framing of its monitoring section: a RAG system can be fast and error-free while serving wrong answers, because standard APM has no visibility into retrieval or answer correctness -- this requires deliberately running a fixed eval set against the live system on a schedule, not inferring quality from infrastructure metrics.",
        "This page treats retrieval/answer-quality degradation as a serious, first-class problem requiring its own deliberate monitoring layer, not a lesser concern subordinate to infrastructure health -- getting a fast response is not the same as getting a correct one, and this page's whole framing is that both need to be tracked.",
        "This page describes a real, proactive mitigation: running a fixed evaluation set against the live system on a recurring schedule and alerting when the score drops, plus periodic manual spot-checks of production output -- both catch degradation without waiting for user complaints to surface it first."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [Anthropic prompt caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Redis: What is semantic caching?](https://redis.io/blog/what-is-semantic-caching/)
- [GPTCache documentation](https://gptcache.readthedocs.io/en/latest/)
- [Portkey: Retries, fallbacks, and circuit breakers in LLM apps](https://portkey.ai/blog/retries-fallbacks-and-circuit-breakers-in-llm-apps/)
- [tenacity documentation](https://tenacity.readthedocs.io/en/latest/api.html)
- [pybreaker](https://github.com/danielfm/pybreaker)
- [Arize Phoenix: LLM tracing and observability](https://arize.com/blog/llm-tracing-and-observability-with-arize-phoenix/)
- [SkyPilot: Large-Scale AI Batch Inference — 9x Faster Embedding Generation](https://skypilot.ai/blog/large-scale-embedding/)
- [Introl: Embedding Infrastructure at Scale](https://introl.com/blog/embedding-infrastructure-scale-vector-generation-production-guide-2025)
- [Tiny-Critic RAG (arXiv)](https://arxiv.org/pdf/2603.00846) — small-model routing/grading for high-throughput RAG
- [Corrective RAG](corrective-rag.md) — the graceful-degradation architecture this page's reliability section references
- [RAGAS Deep Dive](ragas.md) and [Evaluation Metrics](evaluation-metrics.md) — the fixed-eval-set mechanism this page's monitoring section builds on
