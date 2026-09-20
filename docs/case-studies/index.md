# System Design & Case Studies

Choosing a vector database, an evaluation strategy, or an agent architecture is a system-design question, and the strongest way to prepare for that question is seeing how real companies actually answered it — what constraint drove the decision, what the alternative would have cost them, and what broke before they got it right. Every case study here is built from a real, verifiable primary source (an engineering blog, a peer-reviewed paper, an official incident report), not marketing copy — and where a claimed number couldn't be verified, it's flagged rather than presented as fact.

## Retrieval infrastructure at scale

- [Notion → turbopuffer](notion-turbopuffer.md) — a cost- and multi-tenancy-driven migration at real SaaS scale, 600× growth in onboarding
- [Uber's billion-scale vector search](uber-vector-search.md) — what happens when an HNSW graph stops fitting in memory
- [Airbnb: choosing IVF over HNSW](airbnb-ivf-hnsw.md) — a real counter-example to reflexively picking the "better" algorithm
- [Spotify: Annoy → Voyager](spotify-voyager.md) — when deployment constraints, not benchmarks, decide the architecture

## Production RAG and retrieval architecture

- [GitHub Copilot's embedding model](github-copilot-embeddings.md) — why hard negatives, not easy ones, are the real retrieval bottleneck
- [LinkedIn's GraphRAG for customer support](linkedin-graphrag.md) — a peer-reviewed system with a rare real business-impact number
- [Dropbox Dash: RAG to agents](dropbox-dash.md) — an architecture that evolved in public, including a "more tools made it worse" finding
- [Sourcegraph Cody: abandoning embeddings](sourcegraph-cody.md) — when the theoretically-best retrieval method isn't the deployable one

## Agentic systems and hybrid search in production

- [Klarna's AI customer service agent](klarna-ai-assistant.md) — real numbers, and the honest walk-back a year later
- [Uber QueryGPT: multi-agent text-to-SQL](uber-querygpt.md) — decomposing one agentic task into narrow, evaluable sub-agents
- [Perplexity's hybrid search at web scale](perplexity-search.md) — hard evidence that sparse and dense retrieval are complementary, not redundant
- [Shopify's real-time product search](shopify-search.md) — why hybrid search needs a streaming embedding pipeline, not a static index

## Cost and evaluation in production

- [Dust: tiered prompt caching](dust-prompt-caching.md) — segmenting a cache by how often each layer actually changes
- [Anthropic's own eval postmortem](anthropic-eval-postmortem.md) — three real regressions that shipped past Claude Code's own eval suite
- [Notion's dual-track evaluation framework](notion-braintrust-evals.md) — separating "did we break anything" from "is the new model better"

## Security and document processing at scale

- [EchoLeak: a zero-click indirect injection](echoleak.md) — the full attack chain behind CVE-2025-32711
- [Anthropic's Constitutional Classifiers](constitutional-classifiers.md) — a jailbreak defense that got 40× cheaper without getting weaker
- [Harvey AI: document processing at legal scale](harvey-document-processing.md) — what breaks when ingestion volume grows 26× in a year
