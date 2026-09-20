# Perplexity: Hybrid Search Infrastructure for an AI Answer Engine

Perplexity AI's product is an AI-answer engine, which places unusual demands on its retrieval backend that a conventional search product doesn't face in combination: the index has to be **exhaustive** enough to find an obscure page or an exact rare-term match, **fresh** enough to reflect near-real-time web content, and **fast** enough to hand an LLM usable evidence inside a sub-second budget. Perplexity's own engineering blog describes the architecture built to satisfy all three simultaneously, and it's a useful case study because it's one of the more detailed public disclosures of what a production hybrid retrieval pipeline actually looks like end-to-end, backed by both an operational blog post and a companion research paper with hard empirical numbers.

## The problem

Neither retrieval paradigm alone is sufficient for this workload. Pure vector (embedding) search is good at capturing conceptual and semantic similarity but is known to miss exact matches — rare terms, specific names, precise strings — because those don't reliably cluster in embedding space the way semantic meaning does. Pure lexical search (BM25) is the reverse: excellent at exact and rare-term matches, but blind to queries that are conceptually related without sharing vocabulary. An AI-answer engine that needs to reliably answer both "what is the capital of X" and "find the exact clause mentioning Y" can't pick one paradigm and drop the other — and it has to do the searching, filtering, and ranking fast enough that the result still fits inside the latency budget of a live LLM-backed answer.

## What they built, and why

Perplexity's pipeline runs in three stages. **Stage 1** is hybrid retrieval itself: a query hits the index through both lexical and semantic (embedding) modalities at the same time, and the two candidate sets are merged into one pool rather than run as separate, competing paths. **Stage 2** is prefiltering, where heuristics strip out stale or non-responsive content before any expensive ranking work happens on it — cheap filtering up front avoids wasting ranking budget on candidates that were never going to be used. **Stage 3** is progressive ranking: fast, cheap lexical and embedding scorers run first to narrow the field for speed, and only then do cross-encoder rerankers perform the expensive, fine-grained final scoring — and they do it at the **sub-document, passage level**, not on whole documents. That last detail matters architecturally: it means the LLM downstream is fed tight, granular evidence spans instead of entire pages, which both tightens relevance and reduces the amount of irrelevant context the model has to reason around.

Underneath the retrieval layer sits a content pipeline with dynamic, per-site parsing rules that are self-improving: an LLM grades the quality of a given site's parsed output, and that feedback loop refines the parsing rules for that site over time, rather than relying on a single static parser for the whole web.

Perplexity backs the architecture with its own open-sourced evaluation harness, `search_evals`, run across four benchmarks — SimpleQA, FRAMES, BrowseComp, and HLE — specifically to justify these design choices empirically rather than by assertion. A companion paper, "Q2D-Web" (arXiv 2609.08887), adds a sharper empirical point: across the retrievers tested, **BM25 alone has the lowest Recall@1000** of any of them, yet BM25 also contributes roughly **6x more unique correct hits** than the next-best method. In other words, BM25 looks like the weakest retriever by a naive recall metric, while simultaneously being the retriever most responsible for finding results nothing else found — direct evidence that sparse and dense retrieval are complementary rather than redundant.

## The numbers

Perplexity's index spans **200+ billion unique URLs**, serving **200 million+ queries per day**. Indexing throughput runs to **tens of thousands of index operations per second**, on infrastructure spanning tens of thousands of CPUs and hundreds of terabytes of RAM. Reported latency is **p50 = 358ms**, **p95 < 800ms**, which Perplexity claims is more than 150ms faster than the next-fastest competing search API. On the `search_evals` benchmark suite, Perplexity reports scores of **SimpleQA 0.930**, **FRAMES 0.453**, **BrowseComp 0.371**, and **HLE 0.288**.

!!! success "The lesson"
    Two concrete points generalize beyond Perplexity's own stack. First, reranking at the sub-document, passage level — not the whole-document level — is what actually makes retrieval usable as LLM context at web scale; document-level ranking leaves too much irrelevant material for the model to sort through. Second, BM25's value in a hybrid system isn't captured by its own recall number — it's captured by the unique correct hits it surfaces that dense retrieval systematically misses, like rare terms and exact names. The Q2D-Web result is a concrete, evidence-backed rebuttal to the common system-design shortcut of "just use embeddings and skip BM25": a component can look weak on its own headline metric while still being the reason a hybrid system finds things nothing else would have found.

## Sources

- [Perplexity: Architecting and Evaluating an AI-First Search API](https://www.perplexity.ai/hub/blog/architecting-and-evaluating-an-ai-first-search-api)
- [Perplexity: Q2D-Web](https://www.perplexity.ai/hub/blog/q2d-web)
- [Q2D-Web paper (arXiv 2609.08887)](https://arxiv.org/abs/2609.08887)
