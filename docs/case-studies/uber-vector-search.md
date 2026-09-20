# Uber: Powering Billion-Scale Vector Search with OpenSearch

Uber operates vector search against a corpus of over 1.5 billion items, used across recommendation and matching use cases where a query needs an approximate nearest-neighbor lookup returned fast enough to sit in a live serving path. This is one of the few public writeups that documents vector search at genuine billion-item scale with concrete before/after infrastructure numbers, which makes it a valuable reference for what actually changes when an ANN system stops being a research prototype and has to serve production traffic at that size.

## The problem

Uber's target was a p99 latency of 100ms at 2,000 queries per second, over roughly 1.5 billion items with approximately 400-dimensional vectors. At that scale, the constraints that matter in a small proof-of-concept — index build time, memory footprint, how gracefully the system reindexes — stop being background concerns and become the primary engineering problem. Uber built this on OpenSearch, but their initial approach used OpenSearch's underlying Lucene HNSW implementation directly, which limited their flexibility to swap or tune the ANN algorithm and closed off a path to hardware acceleration.

## What they built, and why

Uber's engineers moved off raw Lucene HNSW to an HNSW implementation inside OpenSearch that gave them room to evolve the algorithm layer independently — including, longer-term, a path toward FAISS and GPU-accelerated indexing. That architectural flexibility mattered because at this scale, algorithm-level tuning (graph construction parameters, storage layout) has an outsized effect on both cost and latency, and being locked into one library's implementation choices removes most of the levers available to fix problems as they emerge.

The single hard constraint Uber calls out explicitly is memory: the HNSW graph has to fit entirely in memory, or query latency degrades catastrophically — from roughly 100ms into "tens of seconds," in Uber's own words. This isn't a soft preference to be optimized later; it's a binary threshold that determines the entire shape of the system, because it dictates node count, shard count, and how much of the index can even be considered for a given deployment budget. Uber cut their index size from 11TB to 4TB specifically to make this constraint more affordable to satisfy, by disabling `_source` storage and `doc_values` — both of which store extra per-document data that isn't needed for pure ANN retrieval but is kept by default.

On sharding, Uber found that setting shard count equal to node count was their production optimum, and that adding more replicas monotonically improved query performance (more replicas means more parallel capacity to serve the same query load). For reindexing — an unavoidable, recurring operation at this scale, since embeddings and data both change — Uber used blue/green clusters: build the new index on a separate cluster, validate it, then cut traffic over, so reindexing a billion-item graph never means degrading or dropping live queries.

## The numbers

The rebuilt system took p99 latency from **250ms to under 120ms**, a **52% reduction**. Index build time dropped from **12.5 hours to 2.5 hours**, a **79% reduction**. Index size fell from **11TB to 4TB** through the storage-field changes described above. These three numbers compound: a smaller index is cheaper to hold entirely in memory, a faster build means reindexing is less disruptive to run frequently, and both together are what got latency under the 100ms target at 2,000 QPS.

!!! success "The lesson"
    At billion-item scale, memory isn't one tuning parameter among many — it's the binding constraint that determines the whole architecture, from shard count to which index-layer flexibility you need to have preserved from the start. Systems designed at smaller scale can treat "does the graph fit in memory" as an implementation detail to revisit later; at billion-scale, it has to be a first-class design constraint from day one. And because reindexing at this scale is unavoidable and expensive, a zero-downtime pattern like blue/green deployment needs to be built into the architecture up front, not retrofitted after the first painful reindex.

## Sources

- [Uber Engineering: Powering Billion-Scale Vector Search with OpenSearch](https://www.uber.com/blog/powering-billion-scale-vector-search-with-opensearch/)
