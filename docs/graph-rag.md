# Graph RAG

Vector similarity search treats every chunk as isolated text — good at "find text similar to this text," bad at questions whose answer depends on how two things are *connected* rather than how similar they are to the query. Graph RAG builds an explicit knowledge graph — entities as nodes, relationships as edges — so a query can traverse connections a similarity score alone can't represent.

!!! example "Hands-on"
    The full pipeline below is runnable, against real data: [**Graph RAG →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/graph-rag) in the code repo. Needs an LLM key; graph construction and community summarization are genuinely expensive, so the built graph is cached after the first run.

??? abstract "TL;DR — quick revision"
    - **Building the graph is real work, measured honestly**: on a 92-chunk sample paper, extraction alone took 92 LLM calls (one per chunk) and produced 605 entities and 819 relationships — that's before community detection or summarization even start
    - **Two query modes answer genuinely different question types**: local search (entity-link the query, traverse the graph, generate from what's connected) for specific relational questions; global search (map-reduce over community summaries) for broad synthesis questions no single top-k retrieval can cover
    - **Both verified as real wins over naive retrieval, not asserted**: local search found a connection naive retrieval missed at k=1 and k=3 (only surfacing at k=5); global search's answer to a broad architecture question covered components naive top-5 completely missed
    - **A real, honest limitation, caught and reported**: entity deduplication by embedding similarity failed to merge "Encoder" and "the Encoder stack" — their actual similarity (0.833) fell just under the 0.88 threshold used
    - **Global search costs one LLM call per community, every single query** — genuinely expensive; a newer approach (LazyGraphRAG) exists specifically to avoid this recurring cost

## Building the graph

### 1. Extract entities and relationships — one LLM call per chunk

Each chunk gets read by an LLM and turned into `(entity, relationship, entity)` triples — the raw material the graph is built from.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/graph-rag/graph_rag_docs.py:extraction"
```

**Real output, from an actual chunk in the sample paper:**

```text
Chunk: "We used the Adam optimizer with beta1=0.9. The model has h=8
attention heads."

Extracted triples:
  (Adam Optimizer, has_parameter, beta1=0.9)
  (Model, has_component, Attention Heads)
  (Model, has_parameter, h=8)
  (Attention Heads, number, 8)
```

Run across all 92 chunks of the real sample paper, this produced **605 entities and 819 relationships** — genuinely substantial structure, but also 92 separate LLM calls before anything else happens. Naive RAG's entire indexing cost, by comparison, is zero LLM calls (just local embeddings).

### 2. Deduplicate entities — and a real limitation this surfaced

Different chunks describe the same real entity with different surface forms — "Encoder," "the Encoder stack," "encoder" all refer to the same thing. Left unmerged, these fragment the graph into disconnected near-duplicate nodes that should be one.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/graph-rag/graph_rag_docs.py:dedup"
```

!!! warning "A real, honest limitation, not a hypothetical one"
    This uses embedding similarity with a 0.88 cosine-similarity threshold to merge near-duplicate names. Checked directly: `"Encoder"` vs `"the Encoder stack"` scores **0.833** — just under the threshold — so these do *not* get merged, even though they're genuinely the same entity. Meanwhile `"Multi-Head Attention"` vs `"multi-head attention mechanism"` scores **0.927** and merges correctly. Similarity-based deduplication is a real, imperfect heuristic, not a solved problem — worth knowing before trusting a graph's entity count at face value.

### 3. Detect communities — a real graph algorithm, not an LLM call

Once the graph exists, clustering related entities together is purely structural — no LLM involved at this step.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/graph-rag/graph_rag_docs.py:communities"
```

Louvain community detection on the real 605-entity graph found **70 communities** — a large one centered on the Transformer model itself and its evaluation, down to many small, tightly-related clusters. Each community then gets one LLM call to produce a short summary of what it collectively represents — 70 more calls, on top of the 92 extraction calls above.

## Querying the graph

The graph built above supports two genuinely different ways of answering a question, and picking the right one matters — they're built for different question shapes, not interchangeable options.

### Local search — anchor on specific entities, then look at what's connected to them

Local search is for questions that are really about one or two specific things and how they relate — "what connects X to Y," "what does X depend on." Instead of hoping a single query embedding happens to rank the one right chunk highly among the whole corpus, it does something more direct: figure out which graph entities the question is actually about, then look at exactly what's connected to them.

Concretely, that means two steps: **entity-linking** the query to graph nodes by embedding similarity (the query says "label smoothing," the graph node might be canonicalized as `"Label Smoothing"` — close enough in embedding space to match, without needing an exact string match), and then **traversal** — walking outward from those matched nodes along graph edges, collecting every relationship (and the original source chunk it came from) encountered along the way.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/graph-rag/graph_rag_docs.py:local_search"
```

!!! success "Measured: a real win naive retrieval didn't have"
    Asked *"What connects label smoothing to the BLEU score improvements?"* — naive bi-encoder retrieval **misses the connecting chunk at both k=1 and k=3**, only surfacing it at k=5. Local search finds it directly: matched entities `['BLEU Score', 'Label Smoothing', 'Learning rate']`, traversed 11 triples across 4 source chunks, and generated: *"label smoothing... hurts perplexity... but improves accuracy and BLEU score"* — grounded in the exact sentence naive retrieval was slow to find. The entity-linking step found the right anchor points directly, instead of hoping a single query embedding would rank the connecting chunk highly enough.

### Global search — for questions no single entity or chunk can fully answer

Local search assumes the answer lives near a handful of specific entities. That assumption breaks for a genuinely broad question like "what are the main components of this system and how do they work together" — there's no single entity to anchor on, because the honest answer requires synthesizing across large parts of the *whole* corpus. This is where the community summaries built earlier get used: instead of anchoring on entities, global search asks *every* community, independently, whether it has something to contribute to this specific question, then combines whatever comes back.

That's a **map-reduce** pattern: the "map" step asks each community summary in isolation (parallelizable — and a community with nothing relevant just says so and gets filtered out), and the "reduce" step takes whatever partial answers survived and combines them into one final answer.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/graph-rag/graph_rag_docs.py:global_search"
```

!!! success "Measured: naive top-5 missed most of the architecture"
    Asked *"What are the main components of the Transformer architecture and how do they work together?"* — naive top-5 retrieval clustered around a single narrative thread (parallelization, the overall encoder/decoder split) and never surfaced positional encoding, residual connections, or the decoder's causal masking *at all*. Global search drew from 10 of the 70 communities and produced a genuinely comprehensive answer covering the encoder, the decoder, multi-head attention's actual mechanics, positional encoding, and residual connections together.

    This is the same underlying lesson [RAPTOR](retrieval-methods.md#raptor) already demonstrated elsewhere on this site — a flat top-k can't synthesize across an entire corpus, no matter how good the embeddings are — but arrived at through a genuinely different clustering basis: RAPTOR clusters chunks by *embedding similarity*; Graph RAG's communities are clusters of the *entity relationship graph itself*. Two different structures, same underlying fix for the same failure mode.

!!! warning "The real cost of global search"
    That answer required **10+ LLM calls just for the map step**, one per community, every single time this question is asked — not a one-time cost like the extraction/summarization above, but a recurring per-query cost. [LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) (Microsoft Research) is a real, more recent answer to exactly this problem — it skips upfront community summarization and does relevance filtering lazily at query time instead, trading some of global search's completeness for meaningfully lower cost. Not implemented here, but worth knowing as the actual answer to "this is too expensive to run in production as-is."

!!! note "Missing piece: choosing local vs. global isn't automatic here"
    In the code for this page, `local` vs. `global` is a flag *you* pass on the command line — the recipe doesn't decide for you. A real system shouldn't make a user choose that by hand: it should look at the query first and route it automatically, the same way [Adaptive RAG](tutorials/advanced-rag-from-scratch.md) routes each query to the right *pre-retrieval* technique before this graph is ever touched. Wiring an equivalent router in front of local/global search — narrow, entity-specific question → local; broad, whole-corpus synthesis question → global — is a natural next extension of this recipe, deliberately left out here to keep this page's scope to the graph mechanics themselves.

## Does all data become graph structure?

No, and this matters for what Graph RAG can and can't do:

- **Only text describing clear entities and relationships extracts cleanly.** Narrative or qualitative discussion without a clean entity-relationship structure often doesn't map into a triple at all — extraction is inherently lossy, compressing free text into `(entity, relationship, entity)` form.
- **Original text chunks are kept, not discarded.** Every edge in the graph above stores a reference back to the source chunk it came from — local search's answer above pulled in that original text alongside the graph facts, not just bare triples. Graph RAG is an additional structured layer on top of the same text, not a replacement for it.

**Practical implication:** most production systems run both — the vector index for plain similarity lookups, the graph for relational/multi-hop or broad synthesis questions — routing between them based on what kind of question came in, the same routing idea covered on the [pre-retrieval](advanced-rag-pre-retrieval.md) page.

## Use it for / skip it for

**Use it for:** domains where relationships between entities are the actual point of most queries — org structures, supply chains, regulatory dependency mapping, codebases (call/import graphs), citation networks, fraud-ring/network analysis (patterns of *connections* between accounts, not any single transaction alone). **Skip it for:** a narrow, well-curated corpus where multi-hop relational questions are rare — you'd be paying real extraction and summarization cost (measured above: ~162 LLM calls just to build the graph on a 92-chunk paper) for a problem that mostly doesn't come up; or any corpus that changes often enough that re-extraction becomes a constant tax rather than a one-time cost.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: Graph RAG">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team compares naive vector retrieval against Graph RAG's local search on the same question. Naive retrieval finds the right chunk only at k=5, missing it at k=1 and k=3. Local search finds the connecting fact directly through entity-linking and a 1-hop traversal.",
      "question": "What's the most accurate explanation for why local search succeeded where naive retrieval at low k struggled?",
      "options": [
        "Local search achieved this by re-embedding the query with a separate, higher-recall embedding model tuned specifically for entity names, distinct from the general-purpose embedding model naive retrieval used for whole chunks.",
        "Local search matched the query to specific graph ENTITIES by embedding similarity, then traversed the graph structure directly to the connected fact — rather than relying on a single query embedding to rank the one correct chunk highly enough among all chunks in the corpus.",
        "This particular question happened to have an unusually short, distinctive entity name ('BLEU Score') that made it easy to match, and most real questions wouldn't see the same benefit from local search.",
        "Naive retrieval's chunks were too large to contain the specific sentence needed, while local search's traversal step re-chunks the source text into smaller, more targeted pieces before generating the answer."
      ],
      "correct": 1,
      "explanations": [
        "Both naive retrieval and local search's entity-linking step use the same underlying embedding model (all-MiniLM-L6-v2) in this recipe — there's no separate, higher-recall model involved. The difference is what gets matched against what: entities and a graph traversal, versus whole chunks ranked by a single query embedding.",
        "Correct. Local search anchors on ENTITIES (matched by similarity to the query), then follows explicit graph edges to connected facts — a structural traversal, not a single ranked list competing across the whole corpus. This is exactly the mechanism that let it succeed where naive retrieval's single top-k ranking struggled at low k.",
        "The mechanism behind this result — entity-linking plus explicit graph traversal — applies to any question where the connecting fact exists as a graph relationship, not just ones with short or distinctive entity names. The page presents this as a repeatable structural advantage, not a property of this one question's particular wording.",
        "Local search doesn't re-chunk anything — it pulls in the same original source chunks the graph edges point back to. Both approaches operate over the same underlying chunks; the difference is in how the query gets matched to relevant content (single ranked list vs. entity match + traversal), not in chunk granularity."
      ]
    },
    {
      "scenario": "A team measures their Graph RAG entity deduplication step directly and finds that 'Encoder' and 'the Encoder stack' — which clearly refer to the same real-world entity — are NOT being merged into one graph node, even though their deduplication step uses embedding similarity.",
      "question": "What does this finding demonstrate?",
      "options": [
        "The deduplication code has a bug in how it computes cosine similarity, since two names that are obviously the same entity to a human reader should never score below a well-chosen threshold.",
        "Embedding-similarity-based deduplication is a real, imperfect heuristic with a measurable failure mode — these two names scored 0.833 similarity, just under the 0.88 threshold used, meaning genuinely-identical entities can still fail to merge depending on exact phrasing.",
        "This shows a lower similarity threshold, somewhere below 0.833, should be used instead, since that would have caught this specific merge without introducing new problems.",
        "This is specific to entity names that share a common root word like 'Encoder' — deduplication would reliably work for pairs of names that don't share any words in common."
      ],
      "correct": 1,
      "explanations": [
        "The similarity computation itself isn't the issue — 0.833 is a genuine cosine similarity between the two embeddings, correctly computed. The gap is between what a human immediately recognizes as the same entity and what a fixed numeric threshold captures; that's a limitation of the threshold-based approach, not a computation bug.",
        "Correct. This is precisely the honest finding: a real, measured similarity score (0.833) fell just under the chosen threshold (0.88) for two names that are genuinely the same entity — demonstrating that similarity-based deduplication has real edge cases, not that the mechanism is fundamentally broken.",
        "Lowering the threshold enough to catch 'Encoder' / 'the Encoder stack' risks merging genuinely different entities that happen to sit close together in embedding space — the page doesn't identify a better threshold as a clean fix, just notes the trade-off inherent in any single fixed cutoff. There's no threshold that catches every true match without also catching some false ones.",
        "The failure here isn't about shared root words — 'Multi-Head Attention' and 'multi-head attention mechanism' also share most of their words and merged successfully (0.927 similarity). The actual factor is how close the two full phrasings land in embedding space overall, which isn't reliably predicted by whether they share a root word."
      ]
    },
    {
      "scenario": "A team wants to answer both 'What optimizer did they use?' (a specific factual question) and 'What are the main components of the system and how do they interact?' (a broad synthesis question) using the same Graph RAG system they've built.",
      "question": "Which combination of query modes is the most appropriate choice, and why?",
      "options": [
        "Use global search for both questions, since drawing from more communities gives it strictly more information to work with than local search's smaller, entity-anchored traversal, making it the more thorough choice for any question.",
        "Use local search for the specific factual question (entity-link directly to the optimizer, traverse minimally) and global search for the broad synthesis question (map-reduce across many communities to cover ground a single traversal or single top-k retrieval couldn't) — matching each query MODE to what it was actually designed to answer.",
        "Use local search for both questions, since increasing the number of traversal hops from the matched entities would eventually reach the same breadth of coverage that global search's community summaries provide.",
        "Local and global search both ultimately run the same underlying LLM call over the same community summaries — the only difference is which one gets labeled 'local' or 'global' in the code."
      ],
      "correct": 1,
      "explanations": [
        "More communities searched isn't automatically better — global search's map-reduce answer for a narrow factual question would need to comb through community summaries that mostly aren't relevant, at the cost of one LLM call per community, while local search's direct entity-anchored traversal reaches the same fact more directly and far more cheaply. Thoroughness for a broad question doesn't transfer to being the better fit for a narrow one.",
        "Correct. This is exactly the distinction this page draws: local search (entity-link + traverse) is built for specific relational questions with a small number of natural anchor entities; global search (map-reduce over community summaries) is built for broad questions whose complete answer isn't reachable from one or two entities' immediate neighborhoods.",
        "Local search's traversal expands outward from a small number of matched entities hop by hop — more hops reach a wider local neighborhood, but it's still anchored to those starting entities, not a synthesis across the whole corpus the way global search's map-reduce over every community is built to produce. The two aren't the same operation at different traversal depths.",
        "The two modes use genuinely different mechanisms: local search does entity-linking followed by graph traversal over relationships, while global search runs a map-reduce over community summaries built earlier — different inputs, different LLM call patterns, and measurably different cost profiles, not just different labels on the same operation."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) — Microsoft's GraphRAG paper, local/global search and community detection
- [LazyGraphRAG: Setting a new standard for quality and cost](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) — the cost-reduction approach referenced above
- [RAPTOR](retrieval-methods.md#raptor) — the embedding-similarity-clustering counterpart to Graph RAG's community-structure clustering
