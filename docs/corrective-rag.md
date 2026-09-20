# Corrective RAG (CRAG)

Self-RAG asks a broad question at multiple points in the pipeline: "was this good?" CRAG asks one narrow, focused question — **"is what I retrieved actually good enough to answer from?"** — and, critically, pairs the answer with a specific corrective action rather than just a confidence label. A simple similarity threshold can tell you retrieval was weak; it can't tell you what to do about it. CRAG can.

!!! example "Hands-on"
    The full pipeline below is runnable, against real data: [**Corrective RAG →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/corrective-rag) in the code repo. Needs an LLM key, and a free [Tavily](https://tavily.com) key for the web-search fallback.

??? abstract "TL;DR — quick revision"
    - **One batched evaluator call scores every retrieved chunk** 1-5 for relevance, and the *maximum* score across the batch decides a 3-way verdict: `Correct`, `Incorrect`, or `Ambiguous`
    - **Each verdict triggers a genuinely different action**, not just a different label: `Correct` refines the retrieved text down to the sentences that actually hold up; `Incorrect` discards it entirely and falls back to a real web search; `Ambiguous` does both and combines them
    - **Verified against real data, all three verdicts**, including a live Tavily web search that returned actual current stock price data the sample paper obviously doesn't contain — and the model correctly cited it as an external source
    - **Realistic cost:** 2-4 small/cheap-model calls plus, on the `Incorrect`/`Ambiguous` paths, one real network call to a search API — and exactly one expensive generation call, regardless of which branch runs

## Following one real question through the pipeline

To make the mechanism concrete, three genuinely different questions — one for each possible verdict — are followed through the same corpus (the "Attention Is All You Need" paper) below, with the actual scores and output each one produced.

### 1. Retrieve, then score the whole batch at once

Naive retrieval pulls the usual top-k by similarity. CRAG's first real addition is what happens next: instead of trusting that top-k blindly, or checking relevance one document at a time, **one LLM call scores all of them together** — the same "batch it, don't loop it" principle used for [listwise reranking](advanced-rag-post-retrieval.md#llm-based-listwise-reranking).

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/corrective-rag/corrective_rag_docs.py:evaluator"
```

**Real scores, three different questions, same retrieval step:**

```text
"What optimizer did they use for training?"
  -> scores [1, 5, 2, 1, 1] -> max=5 -> Correct

"What is the current stock price of Nvidia?"
  -> scores [1, 1, 1, 1, 1] -> max=1 -> Incorrect

"What is the significance of this paper compared to more recent
 transformer variants?"
  -> scores [2, 1, 1, 3, 1] -> max=3 -> Ambiguous
```

The thresholds are simple — `max >= 4` is `Correct`, `max <= 2` is `Incorrect`, anything in between is `Ambiguous` — but the *outcome* of that classification genuinely changes what the pipeline does, which is the whole point of doing this at all instead of picking one fallback behavior for every weak retrieval.

### 2a. `Correct` → refine, don't just trust the whole chunk

A chunk can score well overall and still contain sentences that don't actually help — the retrieved 500-character chunk on the Adam optimizer question, for instance, also has a sentence about the learning-rate schedule that's adjacent but not what was asked. Refinement splits every chunk into sentence-level "knowledge strips," re-scores each one individually, and keeps only the strips that clear the bar on their own.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/corrective-rag/corrective_rag_docs.py:refine"
```

For the optimizer question, this is the path that ran, and the final answer came back fully grounded in the refined internal text: *"they used the Adam optimizer for training, with β1 = 0.9, β2 = 0.98, ϵ = 10^-9... They also employed a custom learning rate schedule..."* — no web search needed, because the internal corpus genuinely had the answer.

### 2b. `Incorrect` → discard everything, search the web instead

When the evaluator can't find anything worth keeping, refining bad chunks further wouldn't help — there's nothing there to refine. The corrective action is to stop trusting internal retrieval for this query entirely, rewrite the question into a short, keyword-style query the way a person would actually type it into a search engine, and fetch real external results.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/corrective-rag/corrective_rag_docs.py:websearch"
```

**Real, live trace for *"What is the current stock price of Nvidia?"*** — a question the 2017 Transformer paper obviously has no way to answer:

```text
Rewritten search query: "Nvidia stock price current"

Real Tavily result (excerpt): "NVIDIA(NVDA) stock is priced at $202.55...
As of 2026-07-18, NVIDIA(NVDA) stock has fluctuated between $197.97 and
$206.65... 52-week high of $236.54 and a 52-week low of $164.07."

Final answer: "Based on the context provided, there are multiple stock
prices listed for NVIDIA (NVDA) from different dates: $230.36 (as of
September 4)... $202.55 (as of July 18, 2026)... Note: The context
includes conflicting data points and dates... which suggests the data
may be from different sources or time periods."
```

This is a real network call against the live Tavily API, not a mock — the numbers above are genuine current search results, not fabricated. Worth noticing: the model correctly flagged the conflicting dates/prices across multiple search results as a real ambiguity in the *source data*, rather than silently picking one number and presenting it as certain.

### 2c. `Ambiguous` → do both, and let generation reconcile them

Sometimes the internal corpus has *something* relevant but not a confident, complete answer — refining it alone would leave gaps; discarding it and only searching the web would throw away a real partial answer. The `Ambiguous` branch keeps both: refined internal strips and web search results, concatenated before generation.

**Real trace for *"What is the significance of this paper compared to more recent transformer variants?"*** — the internal corpus has strong material on what the paper *itself* established, but nothing about how later variants compare to it:

```text
Source used: internal (refined) + web
Search query: "significance paper vs recent transformer variants"

Final answer (excerpt): "The 2017 Transformer paper established a
foundational architecture that remains highly influential... Recent
variants focus on addressing the quadratic complexity bottleneck of
the original attention mechanism: Sub-quadratic attention variants,
Linear attention mechanisms... While these newer approaches improve
computational efficiency... they are incremental improvements built
upon the core Transformer architecture rather than revolutionary
replacements."
```

The internal, refined strips supplied the paper's own claimed significance; the web search supplied what came after it — exactly the combination this branch exists for.

### 3. Generate — same call, regardless of branch

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/corrective-rag/corrective_rag_docs.py:pipeline"
```

Whichever branch ran, generation itself is the same single call over whatever context that branch produced — internal strips, web results, or both. The branching logic is entirely upstream of generation, which is what keeps this at "one expensive call regardless of verdict," not three different generation paths to maintain.

## Mechanics: call counts and batching

**Is evaluation per-document or batched?** Batched, not sequential — one LLM call scores every retrieved chunk together (or, in the original paper, one forward pass through a small fine-tuned classifier). Never k separate round-trips for k documents.

**Realistic cost per query:**

- `Correct`: 1 evaluator call + 1 refinement call + 1 generation call — **~2 small-model calls + 1 big-model call**
- `Incorrect`: 1 evaluator call + 1 query-rewrite call + 1 web search API call (network request, not an LLM call) + 1 generation call — **~1 small-model call + 1 API call + 1 big-model call**
- `Ambiguous`: both branches combined — **~3 small-model calls + 1 API call + 1 big-model call**

Never more than one expensive generation call, and never one LLM call per retrieved document.

## Use it for / skip it for

**Use it when both are true:** your knowledge base has real coverage gaps (a fast-moving domain, a still-being-built corpus, or queries legitimately outside what's indexed), and a wrong or unsupported answer is more costly than the extra latency of a fallback. Customer support over a KB that updates slower than the product, research tools where the internal corpus is necessarily incomplete, regulated domains where an ungrounded answer has real consequences — all good fits.

**Skip it when:** your KB is narrow and comprehensive enough that retrieval failure is rare (you're paying evaluator overhead for a problem that almost never occurs), latency is tight enough that added round-trips aren't affordable, or there's no sensible external fallback for the domain at all (querying private financial records, say — web search doesn't help there, and CRAG degrades into just an expensive relevance filter with nothing to fall back to).

## Relation to Self-RAG

Self-RAG is the broader framework — the generation model reflecting on retrieval relevance, its own output's groundedness, and usefulness, at multiple points. CRAG is narrower and more mechanical: one evaluator, one focused verdict, three concrete corrective actions. The two aren't mutually exclusive — a system could run CRAG's evaluator-and-fallback logic *and* Self-RAG's groundedness/usefulness checks on whatever CRAG hands to generation.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: Corrective RAG">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team's retrieval evaluator scores 5 retrieved chunks for a query as [1, 1, 2, 1, 1] — none particularly relevant, but one slightly higher than the rest.",
      "question": "Under the thresholds this page uses (max >= 4 -> Correct, max <= 2 -> Incorrect, else Ambiguous), what verdict does this get, and why does that matter?",
      "options": [
        "Ambiguous, since scores vary slightly across the 5 chunks rather than being perfectly uniform.",
        "Incorrect, since the MAXIMUM score across the batch (2) is at or below the lower threshold — meaning even the single best-scoring chunk isn't good enough to build an answer from, so the whole batch is treated as a retrieval failure and triggers web search.",
        "Correct, since at least one chunk scored higher than the others, indicating some signal was found.",
        "The verdict cannot be determined without knowing the individual chunk contents, only their scores."
      ],
      "correct": 1,
      "explanations": [
        "Score variance alone isn't what the threshold rule checks — it specifically looks at the MAXIMUM score in the batch, not how much the scores differ from each other.",
        "Correct. The decision rule is threshold-on-the-max, not threshold-on-variance or threshold-on-average: since even the best individual chunk (score 2) doesn't clear the bar, the whole retrieved set is judged Incorrect and the corrective action (web search) triggers.",
        "A relatively higher score among uniformly low scores is still a low score in absolute terms — 2 out of 5 doesn't indicate real signal was found, just that one chunk was marginally less irrelevant than the others.",
        "The verdict is fully determined by the max of the given scores under the stated threshold rule — no additional information about chunk content is needed to apply the rule itself."
      ]
    },
    {
      "scenario": "A team building a CRAG pipeline for querying a company's confidential internal financial records asks whether they should wire up a web-search fallback for the Incorrect verdict, the same way this page's demo uses Tavily.",
      "question": "Is a web-search fallback a good fit here?",
      "options": [
        "Yes — since the Incorrect verdict exists specifically to trigger a corrective action, every CRAG pipeline needs some fallback wired up for that branch, and web search is the default fallback this page's own implementation uses.",
        "No — for confidential/private data with no sensible external equivalent, a web-search fallback has nothing useful to contribute; CRAG in this domain degrades into just an expensive relevance filter, since there's no external source that could answer a question about private records.",
        "Yes — since the evaluator already showed internal retrieval scoring low on out-of-scope queries, a web-search fallback gives the pipeline a second, independent source to cross-check any Incorrect verdict against, regardless of the domain being queried.",
        "No — but only because Tavily's terms of service specifically prohibit querying it in connection with private company data, so switching to a different search API would resolve the concern."
      ],
      "correct": 1,
      "explanations": [
        "The Incorrect branch does need some corrective action, but that action doesn't have to be web search specifically — for domains with no sensible external equivalent, the page's own guidance is that CRAG just degrades into an expensive relevance filter, not that a public web search should be forced in anyway.",
        "Correct. This is explicitly the 'skip it for' case this page describes: querying private financial records has no meaningful external equivalent to fall back to, so the Incorrect branch has nowhere useful to go, and the whole apparatus reduces to a relevance filter with no real corrective action available.",
        "The 'independent cross-check' logic breaks down for domains with no external equivalent: a public web search has no way to verify or supply information about confidential internal records, so it isn't actually a usable second source there — cross-checking requires an external counterpart to exist, not just a desire for one.",
        "This isn't about any particular vendor's terms of service — no public web-search engine, whichever one is used, can contain or verify information about a company's confidential internal records. Swapping search providers wouldn't fix a fallback that has nothing relevant to find."
      ]
    },
    {
      "scenario": "A retrieved chunk about a company's optimizer choice ALSO contains an unrelated sentence about their learning-rate schedule details, and the evaluator scores the whole chunk 5/5 for a question specifically about the optimizer.",
      "question": "What does the refinement step (the Correct-verdict action) do that a simple 'keep the whole high-scoring chunk' approach wouldn't?",
      "options": [
        "Nothing different — refinement and just keeping the whole chunk produce identical results whenever the overall chunk score is high.",
        "It re-scores at the SENTENCE level, so even within a chunk that scored well overall, only the individual sentences that specifically hold up (the optimizer sentence) get kept — the tangential learning-rate-schedule sentence could be trimmed if it doesn't independently clear the bar, reducing padding sent to generation.",
        "It discards the entire chunk and replaces it with a web search result instead, since any chunk containing unrelated content is treated as unreliable.",
        "It merges the chunk with other retrieved chunks into one combined passage before scoring, rather than evaluating anything at the sentence level."
      ],
      "correct": 1,
      "explanations": [
        "A whole-chunk keep/discard approach can't distinguish between the genuinely relevant sentence and an adjacent-but-different one within the same chunk — refinement operates at a finer grain specifically to catch this case.",
        "Correct. This is exactly refinement's mechanism: split into sentence-level strips, score each independently, and keep only strips that individually pass — allowing a chunk to be 'refined down' to just its useful sentences even when the whole chunk's aggregate score was high enough to be judged Correct.",
        "Refinement is specifically the CORRECT-verdict action, which works with the existing internal chunk rather than discarding it for a web search — web search is the INCORRECT-verdict action, a different branch entirely.",
        "Refinement splits a chunk INTO smaller strips for finer-grained scoring — it doesn't merge multiple chunks together, which would move in the opposite direction of what this step is designed to do."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) — the original CRAG paper
- [Tavily](https://tavily.com) — search API built for LLM/agent use, used for the web-search fallback demoed above
- [Self-RAG](self-rag.md) — the broader self-reflection framework CRAG's evaluator complements
