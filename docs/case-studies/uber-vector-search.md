# Uber: Powering Billion-Scale Vector Search with OpenSearch

Uber's engineering writeup describes a large-scale vector-search prototype developed in 2024 against more than 1.5 billion items. It provides concrete before-and-after indexing and query benchmarks at a scale relevant to production design, but does not establish that this particular corpus or its measured workload served live production traffic.

## The problem

Uber described a typical target of p99 latency at 100ms and 2,000 queries per second. Its prototype indexed roughly 1.5 billion items with approximately 400-dimensional vectors. At that scale, index build time, memory footprint, and reindexing become primary engineering concerns. Uber moved from Lucene's HNSW approach to OpenSearch partly for a wider choice of ANN algorithms and a potential path to GPU acceleration.

## What they built, and why

Uber's engineers moved off raw Lucene HNSW to an HNSW implementation inside OpenSearch that gave them room to evolve the algorithm layer independently — including, longer-term, a path toward FAISS and GPU-accelerated indexing. That architectural flexibility mattered because at this scale, algorithm-level tuning (graph construction parameters, storage layout) has an outsized effect on both cost and latency, and being locked into one library's implementation choices removes most of the levers available to fix problems as they emerge.

The single hard constraint Uber calls out explicitly is memory: the HNSW graph has to fit entirely in memory, or query latency degrades catastrophically — from roughly 100ms into "tens of seconds," in Uber's own words. This isn't a soft preference to be optimized later; it's a binary threshold that determines the entire shape of the system, because it dictates node count, shard count, and how much of the index can even be considered for a given deployment budget. Uber cut their index size from 11TB to 4TB specifically to make this constraint more affordable to satisfy, by disabling `_source` storage and `doc_values` — both of which store extra per-document data that isn't needed for pure ANN retrieval but is kept by default.

In its tests, Uber found the best performance for this workload when shard count equaled node count, and observed better query performance as it added replicas. To avoid disrupting queries while building a new index, Uber used blue/green clusters: build on a separate cluster, then reroute traffic.

## The numbers

The optimized prototype took p99 latency from **250ms to under 120ms**, a **52% reduction** as reported by Uber. Index build time dropped from **12.5 hours to 2.5 hours**, about an **80% reduction**. Index size fell from **11TB to 4TB** through the storage-field changes described above. These are substantial improvements, but the reported *under 120ms* result does not demonstrate that Uber met its typical **100ms p99 target at 2,000 QPS** for this prototype.

!!! success "The lesson"
    At billion-item scale, memory isn't one tuning parameter among many — it's the binding constraint that determines the whole architecture, from shard count to which index-layer flexibility you need to have preserved from the start. Systems designed at smaller scale can treat "does the graph fit in memory" as an implementation detail to revisit later; at billion-scale, it has to be a first-class design constraint from day one. And because reindexing at this scale is unavoidable and expensive, a zero-downtime pattern like blue/green deployment needs to be built into the architecture up front, not retrofitted after the first painful reindex.

## Sources

- [Uber Engineering: Powering Billion-Scale Vector Search with OpenSearch](https://www.uber.com/blog/powering-billion-scale-vector-search-with-opensearch/)
