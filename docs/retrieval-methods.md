# Retrieval Methods

Chunking decides what units exist to retrieve; the embedding model decides how "similar" gets measured. This page is about everything left over — the actual retrieval-time decisions that determine which of those units come back, in what order, and how much of the index gets touched. Six methods, each demonstrated with a real, run result rather than a description of what it's supposed to do.

!!! example "Hands-on"
    Every technique below is runnable: [**Retrieval Methods →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/retrieval-methods) in the code repo. Two methods run fully locally; four need an LLM key.

??? abstract "TL;DR — quick revision"
    - **MMR (Maximal Marginal Relevance)** diversifies top-k, trading some pure relevance for less redundancy — a real, tested `langchain_core` utility, not something to hand-roll
    - **Self-query retrieval** has an LLM split a query into a semantic part + structured metadata filters — but only works if the LLM is told the *valid values* for each field, not just the field names (measured: silently dropped a filter until the schema was explicit)
    - **Hypothetical Questions** generates plausible questions per chunk at *index* time, so query-time comparison is question-to-question, not question-to-answer — the mirror image of HyDE, which instead generates a hypothetical *answer* at query time
    - **Hierarchical index retrieval** narrows document → section → chunk in stages, instead of searching every chunk flat
    - **RAPTOR** clusters chunks and summarizes each cluster, so broad synthesis questions can match a summary node instead of failing to match any single leaf chunk
    - **Adaptive retrieval (FLARE-style)** only retrieves when the model's own self-reported confidence is low — measured: a too-permissive threshold let a wrong FLOPS figure ship ungrounded; a stricter one caught it, though even then, single-vector retrieval alone didn't find the actual answer

## Already covered elsewhere

A few retrieval-adjacent techniques got full treatment on other pages already — cross-referenced here instead of repeated:

| Technique | What it does | See |
|---|---|---|
| Sparse (BM25), dense, learned sparse (SPLADE), hybrid + RRF fusion | The three retrieval paradigms, measured head-to-head on the same corpus, and how to combine them | [Embedding Model Selection → BM25 vs. dense vs. SPLADE](embedding-models.md#a-real-worked-comparison-bm25-vs-dense-vs-splade) |
| Query rewriting, multi-query/RAG-Fusion, HyDE, step-back, decomposition | Fixing query quality before retrieval runs | [Advanced RAG — Pre-retrieval](advanced-rag-pre-retrieval.md) |
| Cross-encoder reranking, contextual compression | Fixing result quality after retrieval runs | [Advanced RAG — Post-retrieval](advanced-rag-post-retrieval.md) |
| Small-to-big / parent-document retrieval | Precise child-chunk matching, full-parent-chunk generation | [Chunking Strategies → Small-to-Big / Parent-Document Retrieval](chunking.md#small-to-big-parent-document-retrieval) |
| Graph-based retrieval (traversal) | Walking an entity/relationship graph instead of vector similarity | [Graph RAG → Querying the graph](graph-rag.md#querying-the-graph) |
| Cross-modal retrieval (search-both-and-fuse) | Searching per-modality indexes, fusing with RRF | [Multimodal RAG → Separate indexes + RRF fusion](multimodal-rag.md#separate-indexes-rrf-fusion) |

## Maximal Marginal Relevance (MMR)

Plain top-k retrieval has no concept of redundancy: if five near-duplicate chunks all happen to be the closest matches to a query, plain top-k happily returns all five, wasting four of your k slots on saying the same thing over and over while a genuinely different, also-relevant chunk gets pushed out. **MMR** fixes this by scoring each candidate on both relevance to the query *and* dissimilarity to whatever's already been selected, picking greedily: relevance minus redundancy, one slot at a time.

**Parameters:** `lambda_mult` — 1.0 is pure relevance (identical to plain top-k), 0.0 is pure diversity (ignores the query almost entirely once the first pick is made). 0.5-0.7 is a reasonable starting point; tune based on how much redundancy your actual corpus tends to produce.

**Measured, using the real `langchain_core.vectorstores.utils.maximal_marginal_relevance` utility** — not a hand-rolled reimplementation, since this is exactly the kind of well-established algorithm worth using a tested implementation for:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/retrieval-methods/retrieval_methods_docs.py:mmr"
```

At `lambda_mult=1.0` on the sample document, every one of the top-4 results is about multi-head attention specifically — relevant, but repetitive. At `lambda_mult=0.3`, the result set pulls in a training-cost footnote and an author-contributions passage alongside the attention content — genuinely different information, at some cost to pure topical focus.

**Use it for:** result sets prone to near-duplication — FAQ-style corpora with several similarly-worded entries, or any corpus where the top-k from plain similarity search tends to cluster tightly around one narrow restatement of the same point. **Skip it for:** corpora where the top matches are already naturally diverse, or latency-sensitive paths where the extra pairwise-similarity computation between candidates isn't worth the cost.

## Self-Query Retrieval

Most retrieval treats a query as pure semantic content. **Self-query retrieval** recognizes that a lot of real queries actually contain two different kinds of information mixed together — "what's the 2024 remote work policy" is a semantic search for "remote work policy" *plus* a hard filter, `year = 2024`. An LLM splits the query into these two parts; the filter gets applied first (a hard constraint, not a similarity signal), and only the surviving candidates get ranked by semantic similarity.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/retrieval-methods/retrieval_methods_docs.py:self_query"
```

!!! warning "A real failure this exact prompt produced, and the fix"
    The first version of this prompt told the LLM the metadata fields existed ("section", "year") but not what values "section" could actually take. Given the query *"What's the 2024 remote work policy?"*, the model extracted only `{"year": 2024}` — it treated "remote work" purely as semantic content, never recognizing it as a valid value for a categorical field it didn't know existed. The retrieved results correctly matched `year=2024` but silently included an unrelated `expenses` entry alongside the two real `remote_work` matches.

    The fix: tell the model the field's actual valid values — `"section": one of "remote_work", "expenses", "benefits"`. With that added, the same query correctly extracts both `{"section": "remote_work", "year": 2024}`, and the result set narrows to exactly the two matching entries. This is precisely why LangChain's real `SelfQueryRetriever` requires an explicit `AttributeInfo` schema per field, not just field names — a schema-free prompt has no way to know which words in a query are meant as filters versus just topic content.

**Use it for:** knowledge bases with real, structured metadata a user might naturally reference in plain language — dates, categories, authors, statuses — where hand-building a filter UI is more friction than letting the LLM infer it. **Skip it for:** unstructured corpora with no meaningful metadata to filter on, or metadata schemas complex enough that reliable extraction needs more than a short field-value schema (at that point, a structured search form is more reliable than hoping the LLM infers correctly every time).

## Hypothetical Questions

[HyDE](advanced-rag-pre-retrieval.md) fixes the question-vs-answer phrasing gap at *query time*, by generating a hypothetical answer and embedding that instead of the raw question. **Hypothetical Questions** fixes the exact same gap from the opposite direction, at *index* time: for each chunk, an LLM generates several plausible questions that chunk could answer, and those generated questions — not the chunk's own declarative text — get embedded and indexed. At query time, the user's real question is compared against other questions, not against answer-shaped text — same "shape" of text on both sides of the comparison.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/retrieval-methods/retrieval_methods_docs.py:hypothetical_questions"
```

**Trade-off vs. HyDE:** Hypothetical Questions pays its LLM cost once, upfront, at indexing time — query time is a normal, fast embedding lookup with no extra LLM call. HyDE pays an LLM call on *every single query*, with zero indexing overhead. Pick Hypothetical Questions when query volume is high relative to how often the corpus changes; pick HyDE when the corpus changes too often to justify re-generating questions for every chunk repeatedly.

**Use it for:** high query-volume systems built on a relatively stable knowledge base. **Skip it for:** a corpus that changes constantly, where the indexing cost would need to be paid over and over.

## Hierarchical Index Retrieval

Searching every chunk in a large corpus flat gets slower and noisier as the corpus grows — more opportunities for an irrelevant chunk from an unrelated document to score deceptively high purely by coincidence. **Hierarchical index retrieval** builds multiple independently-searchable levels — document-level, section-level, chunk-level — and searches them in stages: find the right document first, then the right section within it, then the right chunk within that section. The same instinct that makes a person go to the right book, then the right chapter, then the right paragraph, rather than scanning every page of every book in a library.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/retrieval-methods/retrieval_methods_docs.py:hierarchical"
```

**How this differs from [small-to-big/parent-document](chunking.md#small-to-big-parent-document-retrieval)** (easy to conflate — both involve parent-child structure): small-to-big is a 2-tier system built for one purpose — retrieve precisely on a small child chunk, then fetch a larger parent purely for generation context; only the child tier is ever searched. Hierarchical index retrieval is typically 3+ tiers, and **every tier is embedded and independently searchable** — the point is narrowing the search space itself at each stage, not just fetching more context after a match is already found.

**Use it for:** large-scale, many-document corpora where a flat search is measurably slow or noisy. **Skip it for:** a single small document's internal retrieval, where there's no meaningful "which document" stage to narrow first.

## RAPTOR

Flat retrieval — even hybrid, even reranked — struggles with questions that need synthesis across many parts of a document, not a single fact lookup: "what is this paper about overall" has no single chunk that answers it. **RAPTOR** builds a tree to fix this: cluster chunks by embedding similarity, summarize each cluster with an LLM into a higher-level node, then recurse — cluster the summaries themselves into a higher layer, summarize again — up toward a small number of broad summary nodes. At query time, search leaf chunks *and* summary nodes together: a narrow factual question tends to match best at leaf level; a broad synthesis question tends to match a summary node instead.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/retrieval-methods/retrieval_methods_docs.py:raptor"
```

Run against the sample paper, real clustering (`scikit-learn`'s `KMeans`) plus real LLM summarization produced genuinely coherent cluster summaries — one describing the shift away from recurrent networks toward pure attention, another describing the paper's authorship and permissions, another describing the BLEU-score results and training efficiency. Asked the broad question *"What is this paper about overall?"*, retrieval correctly surfaced a summary node rather than a single narrow leaf chunk that could only ever answer one piece of that question.

**Trade-off:** real offline cost — clustering plus an LLM summarization call per cluster, at every tree layer, done once at indexing time (or whenever the corpus changes). Query-time cost stays close to normal vector search, just against a richer, multi-level index.

**Use it for:** long documents or corpora where "what's the overall theme" questions are common, not just point lookups. **Skip it for:** corpora that are almost entirely single-fact lookups, where the extra indexing cost buys nothing.

## Adaptive Retrieval (a FLARE-style approach)

Instead of always retrieving a fixed top-k for every query regardless of whether it's actually needed, **adaptive retrieval** decides dynamically whether to retrieve at all. The real mechanism behind this in the research literature is **FLARE** (Forward-Looking Active REtrieval): monitor the model's own token-level confidence *during generation*, and only trigger a retrieval call when confidence drops on a specific claim — using the low-confidence sentence itself as the retrieval query.

Real token-level log-probabilities aren't exposed the same way across every hosted chat API, so the runnable version here uses a simpler, fully implementable stand-in: generate a draft answer, then ask the model to **self-report** its own confidence in that answer (1-5), and only retrieve if the self-reported score is low. Cruder than real FLARE's live token-probability signal, but the same "retrieve only when needed" shape, and it works against any chat-completions API.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/retrieval-methods/retrieval_methods_docs.py:adaptive"
```

!!! warning "A real failure, and what fixing it did and didn't solve"
    Asked *"What is the exact FLOPS count for training the big Transformer model?"* with `confidence_threshold=3`, the model self-reported confidence **3**, cleared the threshold, and confidently answered **3.3×10¹⁸ FLOPs**. The actual paper reports 3.3×10¹⁸ for the *base* model and **2.3×10¹⁹** for the *big* model — a full order of magnitude apart. Confidence 3 was exactly high enough to skip retrieval and exactly low enough that the claim was, in fact, wrong.

    Raising the threshold to **4** fixed the triggering problem: the same low confidence now correctly triggers a retrieval pass instead of shipping the answer directly. But it didn't fully fix the *answer* — plain top-1 dense retrieval on the sample document doesn't surface the actual FLOPS table row at all (it doesn't even rank in the top 5), because a numeric table row ("Transformer (big) 28.4 41.8 2.3· 10¹⁹") embeds poorly with a general-purpose dense model — exactly the dense-embeddings blind spot from the [embedding model selection page](embedding-models.md#a-real-worked-comparison-bm25-vs-dense-vs-splade). The system correctly said "not found in this context" rather than hallucinating a number — a real win — but adaptive retrieval only controls *whether* to retrieve; it can't fix what the underlying retrieval method is actually capable of finding.

**Use it for:** high-stakes factual QA where an ungrounded wrong answer is costly, and where retrieval cost per query genuinely needs to scale with question difficulty rather than being paid uniformly on every call. **Skip it for:** high-throughput, low-latency use cases where the extra self-assessment call isn't worth the cost, or where self-reported LLM confidence hasn't been validated to actually correlate with correctness on your own data — as this exact result shows, it doesn't always.

## Scenario Check

Six scenarios testing whether you can match a retrieval symptom to the right method.

<div class="quiz-widget" data-title="Scenario Check: Retrieval Methods">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A support-ticket knowledge base returns a top-5 result set where 4 of the 5 chunks are near-identical restatements of the same FAQ entry, and a genuinely different, also-relevant chunk gets pushed out of the top-5 entirely.",
      "question": "Which technique directly addresses this, and how?",
      "options": [
        "RAPTOR — clustering the chunks would prevent near-duplicates from existing in the first place.",
        "MMR — score candidates on relevance to the query MINUS similarity to chunks already selected, so near-duplicates stop crowding out genuinely different relevant results.",
        "Hierarchical index retrieval — narrowing to the right document first would eliminate the duplicates.",
        "Self-query retrieval — extracting metadata filters would remove the duplicate entries."
      ],
      "correct": 1,
      "explanations": [
        "RAPTOR clusters chunks for building higher-level summary nodes — it doesn't touch how the final top-k selection avoids redundancy at query time; near-duplicate leaf chunks would still all exist and could still all be selected.",
        "Correct. This is exactly MMR's mechanism — greedily picking candidates that are both relevant to the query and dissimilar to whatever's already been picked, which directly stops near-duplicate chunks from occupying multiple top-k slots.",
        "Hierarchical retrieval narrows by document/section, not by content similarity between individual chunks — near-duplicate FAQ entries could easily sit in the same section and still all get selected.",
        "Metadata filters constrain by structured fields (dates, categories) — they have no mechanism for detecting that two chunks say nearly the same thing content-wise."
      ]
    },
    {
      "scenario": "A team builds a self-query retriever whose prompt tells the LLM that a \"department\" metadata field exists, but never lists what departments actually exist in the data (\"Engineering\", \"Sales\", \"Legal\", etc.).",
      "question": "What's the most likely failure mode?",
      "options": [
        "The system will throw an error immediately, since the LLM can't extract a filter for a field it knows nothing about.",
        "The LLM may fail to recognize department names mentioned in a query as filter values at all, treating them as plain semantic content instead — silently under-filtering rather than erroring.",
        "The system generally still works, since LLMs can usually infer plausible categorical values for a common-sense field like department names directly from context.",
        "This only matters for numeric fields like dates, not string fields like department names."
      ],
      "correct": 1,
      "explanations": [
        "There's no validation step that would catch this and throw an error — the LLM just does its best without an error case, so a poor extraction happens silently, which is worse than a loud failure, not better.",
        "Correct. This is precisely the measured finding: without an explicit schema listing valid values, the LLM can miss that a term in the query IS meant as a categorical filter, treating it as ordinary semantic content instead — and the result is silently under-filtered, not obviously broken.",
        "This is exactly the failure the page measured — the LLM did NOT reliably treat the term as a filter value without an explicit schema; a word appearing in a query doesn't reliably signal \"treat me as a categorical filter\" rather than \"treat me as ordinary topic content\" without the schema spelling out the valid values, which is precisely why the documented fix required listing them explicitly.",
        "The same failure mode applies to any categorical field, string or otherwise — the LLM needs to know the field's actual valid values (or format, for dates) regardless of data type."
      ]
    },
    {
      "scenario": "A team is deciding between HyDE and Hypothetical Questions for a customer support bot with very high query volume against a knowledge base that only changes once a quarter.",
      "question": "Which is the better fit, and why?",
      "options": [
        "HyDE, because it requires no indexing changes at all.",
        "Hypothetical Questions, because its LLM cost is paid once at indexing time (rarely, given the quarterly change cadence), while HyDE would pay an LLM call on every single one of the high-volume queries.",
        "Neither — Hypothetical Questions' indexing cost eventually gets paid back through the corpus's ongoing update cycle, canceling out its advantage over HyDE at genuinely high query volume.",
        "HyDE, because generating a hypothetical answer captures a real answer's semantic content more directly than a generated question can, making it the generally stronger technique whenever it's an option."
      ],
      "correct": 1,
      "explanations": [
        "HyDE does require no indexing changes, but that's not the relevant trade-off here — its recurring per-query LLM cost is the real cost driver at high query volume, and that cost doesn't disappear just because indexing stays simple.",
        "Correct. Hypothetical Questions front-loads its LLM cost into indexing, which only happens quarterly here — a great trade against HyDE's cost structure of paying an LLM call on every one of the high-volume queries.",
        "This ignores the actual cadence given in the scenario — the corpus changes only quarterly, so Hypothetical Questions' indexing cost is paid rarely, not \"eventually canceled out.\" Both techniques are aimed at the same real problem (the question-vs-answer phrasing gap) regardless of volume; the given volume and change-cadence numbers are exactly what decides which ONE fits better here, not a reason to rule out both.",
        "Neither technique is inherently more accurate — they solve the identical phrasing gap from opposite directions (index-time question generation vs. query-time answer generation). The page's actual criterion for choosing between them is cost structure — query volume versus how often the corpus changes — not one method being intrinsically better at capturing meaning."
      ]
    },
    {
      "scenario": "An adaptive-retrieval system asks an LLM to self-report confidence (1-5) in its own draft answer before deciding whether to retrieve. On a specific factual question, the model reports confidence 3, and the system's threshold is set to trigger retrieval only when confidence is below 3.",
      "question": "Based on the measured finding on this exact page, what's the likely outcome?",
      "options": [
        "The system retrieves in this case, since a mid-scale score like 3 out of 5 is exactly the kind of borderline result a sensibly-set threshold should catch.",
        "The system does NOT retrieve (confidence 3 is not below the threshold of 3), and the resulting answer may ship ungrounded and wrong — exactly the measured failure with the FLOPS question.",
        "A self-reported confidence of 3 out of 5 corresponds to roughly a 60% chance the answer is factually correct, since the 1-5 scale is designed to map onto real-world accuracy.",
        "The system throws an error, since confidence exactly at the threshold is undefined behavior."
      ],
      "correct": 1,
      "explanations": [
        "The threshold as defined triggers retrieval only when confidence is strictly below 3 — a self-reported score of exactly 3 does not satisfy that condition, so the system does NOT retrieve here. That's precisely the boundary-value gap the page's measured FLOPS example demonstrates: a score sitting right at the edge of a threshold, not below it, ships without a retrieval check.",
        "Correct. This is the exact measured scenario: a boundary confidence score can sit just outside a retrieval trigger, letting an ungrounded (and in the measured case, factually wrong) answer ship — precisely why the threshold value itself matters, and needs validating against real behavior, not just picked arbitrarily.",
        "Self-reported LLM confidence isn't calibrated to a probability this way — the measured example on this page is direct evidence against it: confidence 3 accompanied an answer that was off by a full order of magnitude, not one that was \"roughly 60% likely correct.\" Nothing on this page validates a linear confidence-to-accuracy mapping.",
        "A simple numeric comparison (`>=` or `<`) handles a boundary value like this without erroring — the behavior is fully defined by whichever comparison operator the threshold check uses, just not necessarily the behavior you'd want."
      ]
    },
    {
      "scenario": "After raising an adaptive-retrieval confidence threshold so that retrieval correctly triggers for a difficult factual question, the system still cannot produce the correct answer — it now honestly says \"not found in this context\" instead of stating a wrong number.",
      "question": "What does this specific outcome demonstrate?",
      "options": [
        "Adaptive retrieval is broken and should not be used, since it failed to produce the right answer.",
        "Adaptive retrieval only controls WHETHER to retrieve — it cannot fix a genuine weakness in the underlying retrieval method itself (here, dense embeddings struggling to represent a numeric table row).",
        "The confidence threshold was set too high and should be lowered back down.",
        "This shows self-reported confidence should be replaced with real token-level log-probabilities before adaptive retrieval is used for anything beyond a demo."
      ],
      "correct": 1,
      "explanations": [
        "This is a meaningfully better outcome than the earlier failure (an honest \"not found\" instead of a confidently wrong number) — \"broken\" overstates the case; it's a partial win with a separate, identified limitation.",
        "Correct. Adaptive retrieval's job is deciding when to retrieve, not making retrieval itself smarter — once triggered, retrieval quality is entirely down to the underlying method, which in this case (single-vector dense retrieval, top-1) genuinely couldn't surface the right chunk. That's a retrieval-method limitation, not an adaptive-retrieval failure.",
        "Lowering the threshold back down is exactly what caused the original failure (shipping a wrong number ungrounded) — the higher threshold's behavior here (honest non-answer) is strictly safer than that.",
        "The page explicitly adopts the self-report specifically because real token-level log-probabilities aren't exposed uniformly across hosted chat APIs — the actual lesson from this outcome is to validate the self-report against real correctness on your own data, since it isn't always reliable, not that it must be swapped out before any further use; that same self-report is what correctly triggered the retrieval pass leading to this safer, honest non-answer in the first place."
      ]
    },
    {
      "scenario": "A 300-page technical manual gets a mix of narrow factual questions (\"what's the torque spec for bolt X\") and broad synthesis questions (\"summarize the maintenance philosophy of this manual\") from the same users.",
      "question": "Which technique is best suited to handling BOTH question types well within one system?",
      "options": [
        "MMR, since diversifying results naturally covers both narrow and broad questions.",
        "RAPTOR, since its multi-level tree lets narrow questions match at leaf level while broad synthesis questions match at a higher-level summary node — searching all levels together rather than picking one level for every query.",
        "Self-query retrieval, since metadata filters can distinguish narrow from broad questions.",
        "Hypothetical Questions, since matching a real question against index-time-generated questions works the same way whether the underlying content is a narrow fact or a broad theme."
      ],
      "correct": 1,
      "explanations": [
        "MMR diversifies a single flat result set for redundancy — it has no mechanism for synthesizing information across many chunks, which is specifically what broad synthesis questions need.",
        "Correct. This is exactly RAPTOR's design point: leaf chunks serve narrow factual lookups, and higher-level cluster summaries serve broad synthesis questions that no single chunk could answer — both searched together, letting each query naturally match whichever level actually fits it.",
        "Metadata filters constrain by structured fields, not by whether a question is narrow or broad in scope — there's no natural metadata field encoding \"question breadth.\"",
        "Hypothetical Questions bridges question-vs-answer phrasing at the level of a single chunk — it doesn't create any mechanism for synthesizing information scattered across many chunks. A broad \"summarize the whole manual\" question still has no single chunk (or single generated question tied to one chunk) that contains the synthesized answer — that scattered-information problem is what RAPTOR's tree specifically addresses, and Hypothetical Questions doesn't touch it."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [MMR — Maximal Marginal Relevance for search result diversification](https://www.elastic.co/search-labs/blog/maximum-marginal-relevance-diversify-results) — Elastic Search Labs
- [LangChain: Self-querying retrievers](https://python.langchain.com/docs/how_to/self_query/) — the `AttributeInfo` schema pattern
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059) — the original paper
- [Active Retrieval Augmented Generation (FLARE)](https://arxiv.org/abs/2305.06983) — the original paper introducing token-confidence-triggered retrieval
