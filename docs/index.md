---
title: Home
---

# RAG, Deep Dive

A working reference on Retrieval-Augmented Generation — from naive retrieve-then-generate up through agentic, graph-based, and cache-augmented architectures, plus the chunking, embedding, retrieval, evaluation, and production-engineering details that actually decide whether a RAG system works.

Every page follows the same standard: real, tested code where a technique can be demoed (not pseudocode), current research rather than recycled tutorial content, and an honest report of what actually happened when the code ran — including the times it surfaced something genuinely surprising. Where a claim is measured, the numbers on the page are from a real run, not an estimate.

!!! tip "How to use this site"
    Each page is self-contained: problem it solves → mechanism → worked example → trade-offs → an interview-ready one-liner summarizing it. Most pages end with a scenario-based quiz to check whether the concept actually stuck, not just whether you recognize the vocabulary. Start with **Naive RAG** if you're new to the topic, or jump directly to whatever you're debugging, designing, or interviewing for. Short on time before an interview? The [**Revision**](revision.md) page has every page's TL;DR, all 78 scenario questions, and 184 flashcards in one place.

## Quick-reference: which RAG architecture, when

| RAG Type | Core Idea | Use When | Avoid When |
|---|---|---|---|
| [**Naive RAG**](naive-rag.md) | Embed → similarity search → stuff top-k → generate | Simple, well-scoped Q&A, prototyping, low-stakes lookups | Any production system needing reliability — always upgrade at least to Advanced |
| [**Advanced RAG — pre-retrieval**](advanced-rag-pre-retrieval.md) (rewriting, HyDE, multi-query, step-back, decomposition) | Fix query quality *before* searching | Vocabulary mismatch, ambiguous/compound/comparative queries, zero-shot cross-domain retrieval | Simple factual lookups — adds latency for no benefit |
| [**Advanced RAG — post-retrieval**](advanced-rag-post-retrieval.md) (reranking, compression) | Fix result quality *after* searching | Any production system with real retrieval noise; multi-chunk contexts where "lost in the middle" is a risk | Single-chunk retrieval — skip both, feed raw chunk directly |
| [**Modular RAG**](modular-rag.md) | Treat pipeline stages as swappable, query-dependent modules | Mixed query complexity/types hitting the same system (support bots, multi-domain assistants) | Single, narrow use case where one fixed pipeline always suffices |
| [**Self-RAG**](self-rag.md) | Model/pipeline critiques its own retrieval relevance + generation faithfulness | High-stakes QA where hallucination is costly (legal, medical, financial); willing to pay for multiple calls | High-throughput, low-latency, simple-lookup use cases |
| [**Corrective RAG (CRAG)**](corrective-rag.md) | Grade retrieval as correct/incorrect/ambiguous, trigger web search or refinement accordingly | Knowledge base has real coverage gaps; wrong answers are costly; a sensible fallback source (web) exists | Narrow, comprehensive, well-curated KB where retrieval rarely fails; no sensible fallback source exists |
| [**Agentic RAG**](agentic-rag.md) | LLM itself decides retrieval/tool-use steps dynamically (ReAct loop) | Genuinely open-ended, multi-hop, multi-source questions where the strategy can't be known in advance | High-volume, well-understood query patterns — unpredictable cost/latency, harder to evaluate, not worth it |
| [**Graph RAG**](graph-rag.md) | Extract entities/relationships into a knowledge graph, retrieve via traversal | Relational/multi-hop questions ("how is X connected to Y"), fraud-ring/network analysis, org structures, codebases | Isolated-fact lookups with no relational structure to exploit; frequently-changing corpora (graph rebuild is costly) |
| [**Vision-RAG**](vision-rag.md) | Embed/retrieve whole page images, VLM reads pages directly | Documents where layout carries meaning — dense tables, charts, forms, multi-column PDFs; OCR/parsing keeps failing | Plain text documents — needless cost/latency over just embedding text |
| [**Multimodal RAG**](multimodal-rag.md) | Retrieve across genuinely different modalities (text/image), a unified shared embedding space or separate indexes fused with RRF | Knowledge base genuinely spans modalities and queries could be answered by any of them | Single-modality knowledge base — unnecessary complexity |
| [**Video-RAG**](video-rag.md) | Transcript- and frame-based retrieval keyed by timestamp, with full CLIP-based visual search as an option when narration doesn't cover the content | Long video corpora where events span time windows; need scene-level + moment-level retrieval | Short clips fitting in one chunk; or when interpretability/evidentiary defensibility matters more than elegance |
| [**CAG**](cag.md) (Cache-Augmented Generation) | Preload knowledge into context once, reuse precomputed KV cache, skip retrieval entirely | Small, static, well-scoped knowledge base (a handbook, one contract); retrieval errors are a bigger risk than context cost | Large/growing/frequently-updated corpora — hits hard token-limit failures at scale |
| [**PageIndex**](pageindex.md) | Vectorless — LLM reasons through a document's hierarchical structure tree instead of embedding similarity | Long, well-structured professional documents (contracts, filings, manuals) where section organization beats semantic similarity; explainability matters | Unstructured/short documents with no real hierarchy; queries needing synthesis across many scattered sections |

## Everything else — engineering the pipeline around the architecture

Picking an architecture is only half the system. These pages cover the decisions that determine whether any of the above actually performs well in production:

- [Chunking strategies](chunking.md) — fixed-size through agentic and adaptive chunking, with a full comparison table
- [Embedding model selection](embedding-models.md) — MTEB, fine-tuning, a real BM25-vs-dense-vs-SPLADE comparison, hybrid search with RRF fusion
- [Retrieval methods, consolidated](retrieval-methods.md) — MMR, self-query, hierarchical indexes, RAPTOR, hypothetical questions, FLARE-style adaptive retrieval
- [Response quality & safety](response-quality-safety.md) — hallucination mitigation, current guardrails tooling, RAG-specific prompt injection (including a real zero-click production exploit), PII/access control
- [Evaluation metrics](evaluation-metrics.md) — precision/recall/MRR/NDCG measured against real ground truth, and why BLEU/ROUGE miss factual errors that lexical overlap can't catch
- [RAGAS deep dive](ragas.md) — the four-metric diagnostic table plus four additional metrics and automatic eval-set generation, all measured on a real pipeline
- [Vector databases](vector-databases.md) — HNSW/IVF/PQ parameters explained from first principles, six real databases tested against the same corpus, and vector-DB choice as a system-design problem with real production case studies
- [Production considerations](production-considerations.md) — real retry/circuit-breaker code, semantic caching with a measured hit rate, monitoring, and real embedding-throughput scaling numbers

## System design, from real production systems

Eighteen case studies — Notion, Uber, Airbnb, Spotify, GitHub Copilot, LinkedIn, Dropbox, Sourcegraph, Klarna, Perplexity, Shopify, Anthropic's own eval postmortem, EchoLeak, and more — each built from a real, verifiable primary source, not a one-paragraph summary. See the [System Design & Case Studies](case-studies/index.md) section.

## New here?

- **Prepping for interviews?** Work through the architecture table top to bottom — each page ends with a scenario quiz, not just a vocabulary check.
- **Building something real?** Start from [Retrieval Methods](retrieval-methods.md) and [Vector Databases](vector-databases.md) for the engineering layer, then [RAGAS Deep Dive](ragas.md) and [Production Considerations](production-considerations.md) for how to know it's actually working.
- **Prepping for a system-design round specifically?** The [System Design & Case Studies](case-studies/index.md) section is real production decisions and real numbers, not hypotheticals.
- **Want to run the code yourself?** Every recipe referenced on this site lives in [rag-cookbook](https://github.com/DhruvMakwana/rag-cookbook), runnable standalone.

---

*Written by [Dhruv Makwana](about.md). Notes are being actively expanded with runnable code, tutorials, and worked examples — check back or follow the repo.*
