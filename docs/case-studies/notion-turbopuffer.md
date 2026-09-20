# Notion → turbopuffer: Rebuilding Vector Search Twice in Two Years

Notion runs AI search and Q&A features across a workspace product used by millions of teams, which makes its vector search infrastructure a genuinely hard multi-tenancy problem: every workspace needs its own isolated, freshly-embedded slice of a much larger corpus, and the underlying data (pages, blocks, blocks inside blocks) changes constantly. Notion's own engineering blog walked through two years of rebuilding that infrastructure from scratch, twice, and the sequence is a useful case study precisely because the second rebuild wasn't triggered by an outage or a correctness bug — it was triggered by cost and growth curves that made the first "correct" architecture the wrong one.

## The problem

Notion launched its first dedicated vector search system in November 2023, built on a "pod" architecture: dedicated hardware, sharded by workspace ID, with range partitioning to keep each workspace's vectors together. That design made sense for the scale and requirements at launch. It did not survive contact with Notion's actual growth: daily onboarding volume grew **600x**, active workspaces using the feature grew **15x**, and the vector database's required capacity grew **8x**. A pod architecture sized and provisioned for one point on that curve becomes extremely expensive — and operationally painful — a year into that kind of growth, because dedicated hardware doesn't elastically track demand and idle pod capacity is capacity you're still paying for.

## What they built, and why

Notion attacked the problem in two stages rather than one. First, in May 2024, they moved off the pod architecture to a serverless model, which cut cost by half immediately by removing the need to provision fixed hardware ahead of demand. That bought time and cash, but it was a stopgap on the existing engine, not a rethink of the storage substrate itself.

The second and larger move, running from May 2024 to January 2025, was migrating the vector search engine itself to **turbopuffer**, a vector search engine built from the ground up on object storage rather than on locally-attached disk or in-memory indexes. Notion's stated reasoning for the choice centered on two things: object storage is inherently cheaper and more elastic than the storage tiers a traditional vector DB relies on, and turbopuffer's support for bulk modification of vector objects mapped directly onto Notion's re-embedding workflow, where large batches of vectors need to be rewritten whenever embedding models or chunking strategies change. A vector engine that treats bulk vector updates as a first-class, efficient operation — rather than something bolted on — is a materially different fit for a product that re-embeds constantly.

Notion followed this migration with a further optimization in July 2025: a "Page State" project that tracks per-span text and metadata hashes so that unchanged content is never re-embedded. Most edits inside a large workspace touch a small fraction of a page's content, so hashing at the span level lets Notion skip re-embedding everything else — turning "re-embed the page" into "re-embed the parts that actually changed."

## The numbers

The turbopuffer migration delivered a **60% reduction in search engine cost** and a **35% reduction in AWS EMR compute**, the batch processing layer feeding the embedding pipeline. Query latency improved from **70–100ms to 50–70ms**. The subsequent Page State hashing work cut embedding data volume by **70%** by eliminating redundant re-embedding of unchanged content.

!!! success "The lesson"
    Object storage as the substrate for vector search is a real, production-proven architecture choice at multi-tenant SaaS scale — not just a cost-cutting gimmick — because it decouples storage elasticity from compute provisioning in a way pod-based or in-memory designs can't match. But the second half of Notion's story matters just as much: after picking the right storage substrate, the next-biggest cost lever wasn't the index at all, it was refusing to re-embed content that hadn't changed. In a system that re-embeds continuously, avoiding unnecessary re-embedding work is as significant a lever as the architecture of the search engine itself.

## Sources

- [Notion Engineering: Two years of vector search at Notion](https://www.notion.com/blog/two-years-of-vector-search-at-notion)
