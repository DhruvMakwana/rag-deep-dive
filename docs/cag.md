# CAG (Cache-Augmented Generation)

Every technique so far retrieves *something* per query — search a knowledge base, pull the relevant chunks, feed them into the prompt. CAG questions that assumption for a specific situation: if the entire knowledge base is small enough to fit in the model's context and doesn't change often, skip retrieval entirely. Load the whole thing into context, reuse the expensive part of processing it across every subsequent query, and let the model answer with the full document always "in mind" instead of whatever a similarity search happened to surface.

!!! note "Scope: what this page actually measures"
    The original CAG paper's mechanism is precomputing and reusing a model's raw KV-cache — only possible when you control the model's internals, which means a self-hosted, open-weight model. Hosted APIs (Claude, OpenAI) never expose that internal state for you to save and reload, and an 8B+ open-weight model isn't practical to run locally without a dedicated GPU. This page measures the closest real equivalent available via a hosted API: **Anthropic's `cache_control` prompt caching**, applied to an entire document as the system prompt. It's a genuine, honest substitution for the paper's actual mechanism, not the literal thing — see below for the exact distinction.

!!! example "Hands-on"
    The full comparison below is runnable, with real API calls against the same sample paper used across this site: [**CAG →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/cag) in the code repo. Needs an Anthropic API key; there's no local/free path here since the whole technique is about a hosted model's context and caching behavior.

??? abstract "TL;DR — quick revision"
    - **No retrieval at all**: the entire document goes into the system prompt on every call — there's no embedding step, no vector search, no chunk selection anywhere in this pipeline
    - **"Fresh" (plain long-context stuffing) vs. "cached" (CAG's actual claim) are different things**: sending the whole document every call is not CAG by itself — what makes it CAG is reusing the *same* processed prefix across queries instead of reprocessing it every time
    - **Cost reduction is dramatic and consistent, measured directly**: cache reads are billed at 0.1x the base input price vs. 1.25x to write the cache and 1.0x with no caching at all — reprocessing a ~15K-token document on every call costs roughly 6-7x what reading it from cache costs, measured at 84-86% total cost reduction across two full runs
    - **Latency reduction was NOT reliably observed at this document size**: the measured delta swung from +13% (cached actually slower) to -5% (cached faster) across two runs — noise-level, not a dependable win, even though Anthropic's own published figures claim up to 85% latency reduction at larger scale
    - **Caching can't change accuracy, and it didn't**: the model reads byte-identical content either way, so any answer differences between runs are ordinary LLM phrasing variance, not a caching effect
    - **It doesn't scale by using an even bigger context window** — a documented hybrid pattern ("RAG over CAG") exists for multi-document knowledge bases, with its own real trade-offs, covered below

## Fresh vs. cached: the distinction that actually matters

**Fresh** — the entire document is sent as the system prompt on every single call, with no caching. This alone is not CAG; it's just long-context stuffing, and it's expensive: the model reprocesses the full document from scratch every time, at full price, regardless of whether the document changed since the last call.

**Cached** — the same system prompt, but annotated with a cache breakpoint. The first call against a given document pays a real premium to write the cache. Every later call within the cache's TTL, against the byte-identical prefix, reads from the cache instead of reprocessing the document — this reuse across repeated queries is the entire mechanism CAG's name refers to. Conflating "long context" with "caching" is the most common mix-up in casual coverage of this topic: the size of what you put in context and whether that context gets reused are two separate levers.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/cag/cag_docs.py:load_document"
```

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/cag/cag_docs.py:generate_fresh"
```

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/cag/cag_docs.py:generate_cached"
```

Cost isn't a single flat "cached" rate — cache writes and cache reads are both multipliers on Sonnet 5's base input price ($2.00/MTok), computed directly from the API's own usage fields on each response:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/cag/cag_docs.py:cost_from_usage"
```

## Measured: cost drops sharply, latency doesn't — reliably

Two independent full runs, same document (~15,125 tokens), same 8 questions used across this repo:

```text
                  Fresh (total)   Cached (total)   Change
Cost, run 1       $0.2536         $0.0399          -84.3%
Cost, run 2       $0.2544         $0.0370          -85.5%
Latency, run 1    27.96s          31.63s           +13.1% (slower)
Latency, run 2    28.30s          26.93s            -4.8% (faster)
```

!!! success "Cost: dramatic, consistent, and easy to explain"
    Cache reads are billed at 0.1x the base input price; a fresh call costs 1.0x; the initial cache write costs 1.25x. Reprocessing a ~15K-token document on every call is genuinely expensive compared to reading it back from cache — this shows up as an 84-86% total cost reduction both times it was measured, a real and reliable effect at this document size.

!!! warning "Latency: not a reliable win here, despite the vendor's own marketing claim"
    Anthropic publishes "up to 85% latency reduction" for prompt caching. Measured directly across two full runs here, the actual delta ranged from cached being **13% slower** to cached being **5% faster** — essentially noise, not a dependable improvement. The likely reason: at ~15,000 tokens, network round-trip time and output generation dominate total wall-clock time far more than the prefill compute that caching actually saves. The vendor's claim almost certainly holds at a much larger cached-context size — hundreds of thousands of tokens, where prefill compute is a bigger share of total request time — but it doesn't reproduce at the scale this recipe's sample document sits at. Cost savings and latency savings are genuinely separate claims, and this page's own numbers show them diverging.

**Accuracy was identical either way, as expected** — 6-7 of 8 correct on both fresh and cached runs. Caching doesn't change what content the model reads, only how the unchanged prefix is billed and computed, so there's no mechanism by which it could affect correctness. The one recurring miss on this repo's standard eval set is a keyword-matching artifact, not a real error: the model correctly answers `d_model = 512`, but the eval's exact-match keyword is `dmodel = 512` (no underscore) — a formatting difference, not a wrong answer.

## Scaling beyond one document: "RAG over CAG"

CAG doesn't scale by reaching for an ever-bigger context window — computing one cached prefix over many documents together stops being practical well before you'd want it to, and a knowledge base that's actually large and growing needs something else. A documented hybrid pattern handles this: cache each document *separately*, then use a lightweight router — an embedding comparison or a small classifier, the same routing idea covered on the [Modular RAG](modular-rag.md) page, just applied to picking a cache instead of a retrieval path — to decide which document's precomputed cache to load for a given query. On a router miss, fall back to ordinary retrieval rather than forcing a wrong cache.

Concretely: a company with 5 separate policy handbooks could precompute and store one cache per handbook offline, keep a small per-handbook description or embedding in a lightweight index, and route each incoming question to the one handbook's cache that actually matches it — no retrieval *within* a handbook once it's selected, since the whole document is already in context.

This has a real, honest limitation: a question that genuinely needs two handbooks at once forces the router to either pick one (an incomplete answer) or load both caches together, which may not fit and isn't a cleanly solved problem in the literature this pattern comes from. The router is also a single point of failure in a way pure retrieval isn't — a misrouted query loads the wrong document's cache entirely, with none of the partial credit that fusing multiple retrieved sources can provide.

## When to use which

| | RAG | CAG |
|---|---|---|
| Knowledge base size | Scales to large, growing corpora | Limited to what fits in context |
| Cost per repeated query | Retrieval cost + a smaller generation context every time | Near-zero marginal cost once cached, measured directly above |
| Latency per repeated query | Retrieval latency + generation | Not reliably faster at moderate document sizes (measured above) — the win is compute-share-dependent, not automatic |
| Updates to the knowledge base | Re-embed/re-index just the changed document | Cache invalidates on any byte change to the cached prefix — no incremental update path |
| Retrieval failure risk | Real (wrong chunk retrieved, missed context) | None — the whole document is always present |

**Use CAG for:** a small-to-medium, relatively static knowledge base where retrieval errors are a bigger practical risk than the cost of holding it all in context — a single detailed product manual, one contract analyzed repeatedly, a static internal FAQ. **Use RAG for:** a genuinely large or fast-changing corpus, or workloads where per-query latency specifically (not just cost) needs to improve — this page's own measurement shows caching doesn't automatically deliver that. **Use "RAG over CAG" for:** a knowledge base made of several distinct, individually-small documents, where routing to the right one is easier than merging them into a single cached context.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: CAG">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team builds a system that sends an entire 15,000-token reference document as the system prompt on every single API call, with no cache_control or caching mechanism configured at all. They describe this as 'doing CAG.'",
      "question": "Based on this page's measured findings, is this an accurate description?",
      "options": [
        "Yes — CAG is defined by putting the entire document into the model's context instead of retrieving chunks, and that's exactly what this system does; whether the prefix gets cached afterward is a separate implementation detail, not part of the definition.",
        "No — this is plain long-context stuffing, the baseline CAG's caching layer is compared against. What makes a system CAG specifically is reusing the SAME processed prefix across repeated queries instead of reprocessing it from scratch every time, which requires an actual caching mechanism, not just a large context window.",
        "No — CAG's actual claim depends on precomputing and reusing the model's raw KV-cache, and a hosted API's cache_control prompt caching reprocesses the document's internal representations fresh on the server for each new cache write, so it's a different mechanism that only resembles CAG's cost profile by coincidence.",
        "Yes, but only if the caching is set up with a cache TTL of at least one hour — a shorter cache lifetime like 5 minutes means the system doesn't qualify as genuinely reusing the cache across queries."
      ],
      "correct": 1,
      "explanations": [
        "This is exactly the mix-up this page calls out directly: putting a whole document in context is necessary but not sufficient for CAG. Without reuse of that processed context across calls, it's just expensive long-context usage — the caching, not the context size, is what the page ties CAG's name to.",
        "Correct. This page measures this exact distinction directly: 'fresh' (full document resent every call, no caching) vs. 'cached' (same document, but with a cache_control breakpoint so later calls reuse the processed prefix). Only the second one demonstrates CAG's actual claim.",
        "This page treats hosted-API prompt caching as a genuine, documented substitution for the paper's literal KV-cache mechanism, not a coincidental resemblance — the underlying idea (reuse a processed prefix's computation across queries instead of redoing it) is the same one both mechanisms implement, just at different levels of access to the model's internals.",
        "TTL length affects how long a cache entry survives before it needs to be rewritten, but it doesn't determine whether the underlying mechanism counts as CAG — reuse of the same processed prefix within whatever TTL is configured is what matters, not a specific minimum duration. And this scenario has no caching configured at all, of any TTL, which is the actual problem."
      ]
    },
    {
      "scenario": "A team reads Anthropic's marketing claim of 'up to 85% latency reduction' from prompt caching and expects their own CAG-style system (a ~15,000-token cached document, queried repeatedly) to get noticeably faster once caching is enabled. After measuring it directly, they find the latency is about the same as without caching, sometimes even slightly slower.",
      "question": "What does this page's own measurement suggest is going on?",
      "options": [
        "Their caching implementation is misconfigured, since a correctly implemented cache should reproduce close to the vendor's published 85% latency reduction on any document, independent of its size.",
        "Cost savings and latency savings are separate claims that don't automatically move together — at a document size where network round-trip and output generation dominate total wall-clock time, the compute savings caching provides (skipping re-processing of the prefix) is a small fraction of total latency, so the improvement doesn't show up, even though cost savings (which scale directly with tokens billed) still do.",
        "Anthropic's published latency figure must have been measured under artificial conditions that don't reflect real usage, since a real production system should see the claimed improvement whenever caching is enabled.",
        "The team should switch to a 1-hour cache TTL instead of 5 minutes, since that would fix the latency issue."
      ],
      "correct": 1,
      "explanations": [
        "This page measured the identical caching pattern working correctly — cache writes and reads showing up properly in the usage fields, and the expected 84-86% COST reduction appearing reliably both times — while latency specifically didn't improve at this ~15K-token size. The finding is that latency reduction is size-dependent, not that a correct implementation guarantees the vendor's headline number at every document size.",
        "Correct. This is exactly this page's measured finding and its explanation: cost reduction (0.1x vs 1.0x-1.25x per-token pricing) scales reliably with any cached token count, but latency reduction depends on how much of total request time is actually prefill compute versus network/generation time — which shrinks in relative importance as documents get smaller.",
        "The page doesn't conclude the vendor's figure was measured artificially — it attributes the gap to a real, explainable factor: at ~15,000 tokens, prefill compute (what caching actually saves) is a small share of total latency compared to network round-trip and output generation, so the effect that IS real doesn't show up much at this scale. The claim likely reproduces at a larger cached-context size where prefill is a bigger share of total time.",
        "TTL length (5 minutes vs. 1 hour) controls how long a cache entry survives before needing to be rewritten — it has no bearing on why a cache HIT would or wouldn't be faster than no caching at all."
      ]
    },
    {
      "scenario": "A company implements 'RAG over CAG' for 5 separate policy handbooks: each handbook gets its own precomputed cache, and a lightweight router picks which handbook's cache to load based on the incoming question. A user asks a question that genuinely requires combining information from 2 of the 5 handbooks.",
      "question": "Based on this page's coverage of this pattern's real limitations, what happens?",
      "options": [
        "The system merges the two relevant caches into a single combined context automatically, since the router is specifically designed to detect when a question spans multiple handbooks and load all the relevant ones together before generating an answer.",
        "This is a genuine, acknowledged limitation of the pattern: the router either has to pick just one handbook's cache (risking an incomplete answer) or attempt to load both together, which may not fit and isn't a cleanly solved problem — this is a real trade-off of RAG-over-CAG, not an edge case it handles gracefully.",
        "The system falls back to running full RAG-style retrieval across the raw text of all 5 handbooks for this query, which fully resolves the multi-document question the same way any RAG system handles multi-document lookups.",
        "This scenario is unlikely to come up in a well-tuned system, since the router's embedding comparison would typically route a mixed question to whichever single handbook scores marginally higher in similarity, giving a mostly-correct answer."
      ],
      "correct": 1,
      "explanations": [
        "The page describes this as an unresolved trade-off, not a built-in capability — merging multiple precomputed caches into one combined context may not fit within the model's context window and isn't a cleanly solved mechanism in the pattern's own literature. The router's job is picking a cache to load, not detecting and stitching together multiple ones on demand.",
        "Correct. This is stated directly as a real, honest limitation of RAG-over-CAG: multi-document questions expose the pattern's weak point, since the router's whole job is picking ONE cache, and multi-cache merging isn't a solved mechanism.",
        "A fallback to broader retrieval is a reasonable engineering response to a router miss, but it isn't part of this pattern's design as described here — the page doesn't describe RAG-over-CAG as having a built-in retrieval fallback path, and treating the fallback as automatic and fully resolving skips over the actual trade-off the page identifies with multi-document questions.",
        "Routing to the higher-scoring handbook doesn't resolve a genuinely two-handbook question — it produces exactly the 'router picks just one, risking an incomplete answer' outcome the page describes as the real trade-off, not an escape from it. The page treats multi-handbook questions as a real, acknowledged limitation, not an edge case a good router avoids."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks](https://arxiv.org/abs/2412.15605) — the original CAG paper: precomputed KV-cache reuse, evaluated on SQuAD and HotpotQA
- [hhhuang/CAG](https://github.com/hhhuang/CAG) — the paper's reference implementation, showing the literal KV-cache manipulation (`past_key_values`, file-based persistence) this page's hosted-API approach substitutes for
- [Anthropic: Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) — `cache_control` mechanics, TTLs, pricing multipliers, and the exact `usage` field names measured on this page
- [Modular RAG](modular-rag.md) — the routing idea (embedding or small-LLM based) referenced above for picking which document's cache to load in "RAG over CAG"
