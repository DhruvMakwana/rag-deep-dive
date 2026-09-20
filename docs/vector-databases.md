# Vector Databases

Everything upstream of this page — chunking, embedding, hybrid retrieval — assumes there's somewhere to actually store and search the vectors. This page covers that storage layer in full: the ANN algorithms and every parameter they expose, real tested code against six real databases sharing one corpus and one metadata scheme, and the system-design reasoning interviewers actually probe when they ask "how would you choose a vector database for this."

!!! example "Hands-on"
    The full pipeline below is runnable: [**Vector Databases →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/vector-databases) in the code repo. No API key needed — embeddings run locally.

??? abstract "TL;DR — quick revision"
    - **The core problem is approximate, not exact, nearest-neighbor search**, and there are exactly three algorithm families worth knowing in depth: HNSW (graph-based, the default), IVF (cluster-based, needs training), and PQ (compresses the vectors themselves, usually paired with IVF)
    - **Every tunable parameter trades off recall against speed, memory, or build time** — `M`/`ef_construction`/`ef_search` for HNSW, `nlist`/`nprobe` for IVF, `m`/`nbits` for PQ — this page explains what each one literally does, with real current defaults from each database's own source or docs
    - **Six real databases, same corpus, same metadata, both HNSW and IVF shown as runnable code** — FAISS, ChromaDB, Qdrant, Milvus Lite, pgvector, and Weaviate, all indexed and queried against the same chunked PDF with each chunk's real page number as metadata
    - **Choosing a vector database is a system design question**, not a feature-comparison exercise — the standard senior-level answer starts with "do you need one at all," reasons through a recall/latency/memory triangle (plus quantization as a fourth lever), and only escalates to a dedicated store once a concrete constraint demands it

## Why vector databases exist

The core problem: **efficient similarity search over millions to billions of high-dimensional vectors.** Brute-force comparison — the query vector against every stored vector — is O(n) per query, impractical at any real scale. Vector databases solve this with **Approximate Nearest Neighbor (ANN)** search: trading a small amount of accuracy (occasionally missing the true top-1) for a large speed gain, since "very close to the true top-k" is what RAG retrieval actually needs.

There are three ANN algorithm families worth knowing in real depth, because nearly every vector database's index configuration is just a different naming of the same underlying knobs.

## HNSW: the graph-based default

**Mechanism.** HNSW (Hierarchical Navigable Small World) builds a multi-layer graph. Every vector lives in layer 0; each vector is also inserted into some number of layers above it with exponentially decaying probability, so the top layer has very few nodes with long-range connections and lower layers get progressively denser. A search starts at a single entry point in the top layer, greedily moves toward the query's region, descends layer by layer — a "zoom-in" phase similar in spirit to a skip list — and only does the expensive, wide candidate search once it reaches layer 0. Each node's neighbor list is capped: `M` connections per layer above 0, and a separate, larger cap (`2M`) at layer 0 specifically, since every vector lives there and needs more connectivity to stay well-linked.

**Parameters, what they actually control, and how to choose them** (per the original paper — Malkov & Yashunin, IEEE TPAMI 2018, [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) — and current vendor documentation):

| Parameter | What it controls | Trade-off | Tunable after build? |
|---|---|---|---|
| **M** (`max_neighbors`, `max_connections`, `maxConnections` depending on library) | Max bidirectional edges per node per layer | Higher M → better recall, especially at high dimensionality or high target recall, at the cost of more memory (roughly linear in M) and slower builds. The paper's own guidance: *"a reasonable range of M is from 5 to 48... smaller M generally produces better results for lower recalls and/or lower dimensional data, while bigger M is better for high recall and/or high dimensional data"* | No — baked into the graph structure at build time |
| **ef_construction** | Size of the dynamic candidate list used while choosing which neighbors to actually link, during insertion | Higher → better graph quality → better recall ceiling for every future search, at the cost of slower index builds. The paper found diminishing returns fast: *"a reasonable quality index can be constructed for efConstruction = 100... in just 3 minutes. Further increase... leads to little extra performance but in exchange for significantly longer construction time"* | No — also baked in at build time |
| **ef_search** (`ef`, `efSearch`, `hnsw_ef`) | Size of the dynamic candidate list at query time | Higher → more of the graph explored → better recall, higher latency. Must be ≥ the number of results requested. This is the knob to sweep first when tuning recall, since it needs no rebuild | Yes — a pure query-time setting |

Real current defaults, pulled directly from each library's source or docs (not blog posts, which are often stale):

| Library | M default | ef_construction default | ef_search default |
|---|---|---|---|
| FAISS (`IndexHNSWFlat`) | 32 | 40 | 16 |
| ChromaDB (`max_neighbors`) | 16 | 100 | 100 |
| Qdrant (`m`) | 16 | 100 | = ef_construct unless overridden per-query |
| pgvector | 16 | 64 | 40 |
| Weaviate (`max_connections`) | 32 | 128 | -1 (dynamic — see below) |
| Milvus | no fixed default (valid range 2–2048; leans on AUTOINDEX) | no fixed default (range 1–INT_MAX) | must be ≥ `top_k` |

Every demo on this page sets `M=32, ef_construction=200, ef_search=64` uniformly across all six databases (above most of their defaults, since this corpus is small enough that build cost is irrelevant and the point is showing higher-quality search).

!!! note "Weaviate's dynamic ef mode"
    Weaviate's `ef` defaults to **-1**, meaning "dynamic" — it computes an effective `ef` per query as `dynamic_ef_factor × limit`, clamped between `dynamic_ef_min` (100) and `dynamic_ef_max` (500), rather than using one fixed value for every query regardless of how many results were actually requested.

## IVF: cluster-then-search

**Mechanism.** Run k-means over a training sample of the dataset to produce `nlist` centroids, partitioning the vector space into Voronoi cells. Every vector is then assigned to (and stored in) the inverted list of its nearest centroid — hence "inverted file index." At query time, only the `nprobe` cells closest to the query are actually scanned, instead of the whole dataset.

**`nlist`** — the number of clusters. FAISS's own sizing guidance ([faiss wiki: Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)): roughly `4·sqrt(N)` to `16·sqrt(N)` for N up to about a million vectors; past that, FAISS's own recipes switch to fixed power-of-2 sizes paired with an HNSW coarse quantizer for fast centroid lookup (`IVF65536_HNSW32` for 1M–10M vectors, `IVF262144_HNSW32` for 10M–100M, `IVF1048576_HNSW32` for 100M–1B). Training needs a real, sufficiently large sample: FAISS's own guidance is **30× to 256× `nlist` training vectors** — too few and the centroids come out unbalanced, starving some clusters and overloading others.

**`nprobe`** — how many of the `nlist` clusters get searched per query, the direct IVF equivalent of `ef_search`: default 1, tunable per-query with no rebuild, and search time scales roughly linearly with it. Setting `nprobe = nlist` is mathematically equivalent to brute-force search over the whole dataset — correct, but with none of IVF's speed benefit left.

pgvector supports IVFFlat directly (`lists` = `nlist`, `probes` = `nprobe`), with its own sizing rule in the [README](https://github.com/pgvector/pgvector/blob/master/README.md): `rows / 1000` for under a million rows, `sqrt(rows)` above that.

### FAISS IVF and pgvector IVFFlat, as code

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/vector-databases/vector_databases_docs.py:faiss_ivf_demo"
```

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/vector-databases/vector_databases_docs.py:pgvector_ivfflat_demo"
```

!!! note "IVF needs real scale to show its actual behavior"
    Run against this page's 121-chunk corpus, FAISS's own training step prints `WARNING clustering 121 points to 8 centroids: please provide at least 312 training points` — a direct, real instance of the 30×–256× training-data rule above (8 clusters calling for roughly 240–2,048 training points, far more than this corpus has). At this scale `nprobe=1` and `nprobe=8` (exhaustive) return identical results, and pgvector's `lists` formula rounds down to 1 — a single cluster containing every row, which makes IVFFlat behave identically to a full scan. None of this is a bug in the code; it's IVF's clustering having nothing meaningful to do below the thousands-of-vectors range it's actually built for.

## PQ: compressing the vectors themselves

**Mechanism.** Product Quantization splits each vector into `m` equal-length sub-vectors, then runs k-means independently within each of the `m` subspaces to learn a codebook of `2^nbits` centroids. Each sub-vector is stored as just the ID of its nearest sub-centroid — the full vector compresses from D floats down to `m` small integer codes. Distances are approximated at query time via precomputed lookup tables between the query's sub-vectors and each subspace's centroids, without ever reconstructing the full-precision vector.

- **`m`** (number of sub-quantizers): the vector dimension must be a multiple of `m`. More sub-quantizers generally means better reconstruction accuracy at a fixed `nbits`, at the cost of more codes to store per vector.
- **`nbits`** (bits per sub-quantizer code): controls centroid count per subspace (`2^nbits`). **8 is the standard default** specifically because 256 centroids fit in exactly one byte per sub-vector — clean byte alignment that's both memory-efficient and fast for SIMD-based lookup.

**Concrete compression math** (verified against [Pinecone's product quantization writeup](https://www.pinecone.io/learn/series/faiss/product-quantization/)): a 128-dimensional float32 vector is 512 bytes uncompressed. Under PQ with `m=8, nbits=8`, it compresses to 8 sub-codes × 8 bits = 8 bytes — a **64× compression ratio**. On the standard SIFT1M benchmark (1M × 128-dim vectors), this takes a FlatL2 index from 256MB down to 6.5MB. PQ is usually combined with IVF as "IVF-PQ" — IVF narrows the search to a few clusters, PQ makes each stored vector cheap enough that billion-scale corpora fit in RAM at all.

None of this page's own demos use PQ — at 121 vectors, compression has no benefit and would only add approximation error. It matters once memory, not latency, becomes the binding constraint (see [the scale bands below](#choosing-a-vector-database-is-a-system-design-question)).

## Beyond RAM: disk-based ANN

Both HNSW and IVF-PQ assume the index lives in memory. Past a certain scale, it doesn't fit — [Uber's own engineering writeup](https://www.uber.com/blog/powering-billion-scale-vector-search-with-opensearch/) on their 1.5-billion-item HNSW deployment states the constraint directly: the graph has to be fully RAM-resident, or query latency degrades to "tens of seconds." Two disk-based designs exist specifically for this: **DiskANN** (Microsoft Research) keeps a compressed, PQ-style summary of every vector in RAM for fast candidate filtering, and fetches full-precision vectors from SSD only as needed — its own published SIFT-1B benchmark reports roughly 5,000 QPS at 95%+ recall on a single 64GB-RAM machine. **SPANN** (also Microsoft) takes the inverted-file idea further: centroids stay in RAM, the (much larger) posting lists live on disk — reported running in production at Bing, at hundreds-of-billions scale, using roughly a tenth of the RAM a comparable fully in-memory system would need. Milvus exposes DiskANN as a selectable index type directly; ChromaDB's distributed/cloud tier uses SPANN internally.

## Similarity metrics

- **Cosine similarity**: angle between two vectors, ignoring magnitude — the common default for text embeddings, since magnitude rarely carries meaningful signal there.
- **Dot product**: factors in magnitude — used when a model was specifically trained with it in mind (some models are normalized so dot product and cosine become numerically equivalent).
- **Euclidean (L2) distance**: less common for text, more standard in other embedding domains.
- The metric isn't usually a free choice: it should match whatever the embedding model was actually trained with, since that determined what "close" meant during training. Mismatching it silently degrades retrieval quality with no obvious error.

## Six databases, one corpus

Every demo below indexes the same 121 chunks from the standard sample paper used throughout this site, each tagged with the real PDF page number it came from — so a metadata filter like "only pages 7–9" or "page ≤ 10" is filtering on something genuine, not a synthetic label. Every demo uses `M=32, ef_construction=200, ef_search=64` (or that database's equivalent names) unless noted.

### FAISS

A library, not a server or a full database — no persistence, metadata store, or network API of its own; `IndexHNSWFlat` and `IndexIVFFlat` are data structures you embed directly in a process. That makes it the fastest option to prototype with and the one with no operational surface at all, at the cost of having to build metadata filtering, persistence, and multi-tenancy yourself if you need them (the `IndexIDMap2` + external `{id: page}` dict pattern below is exactly that DIY layer). [Official docs & wiki →](https://github.com/facebookresearch/faiss/wiki)

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/vector-databases/vector_databases_docs.py:build_corpus"
```

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/vector-databases/vector_databases_docs.py:faiss_demo"
```

Measured, unfiltered top-3 for *"What optimizer and learning rate schedule were used for training?"*: all three hits land on page 7 (`0.592`, `0.438`, `0.385`) — restricting the same search to pages 7–9 returns the identical three results, since they were already inside that range.

!!! note "Import order matters on macOS"
    `faiss` needs to be imported before `torch` (pulled in by `sentence-transformers`) in the same process — the reverse order loads two conflicting OpenMP runtimes and segfaults on the first FAISS call.

### ChromaDB

A lightweight, developer-friendly database with HNSW as its only local index type (its distributed/cloud tier switches to SPANN internally, currently non-configurable). Metadata is native — a `where` filter travels in the same call as the vector query — and every HNSW setting lives in one structured `configuration={"hnsw": {...}}` dict at collection-creation time. `space`, `max_neighbors`, and `ef_construction` are fixed for the life of the collection; `ef_search`, `num_threads`, `batch_size`, `sync_threshold`, and `resize_factor` can all be changed later without a rebuild. [Official docs →](https://docs.trychroma.com/docs/collections/configure)

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/vector-databases/vector_databases_docs.py:chromadb_demo"
```

Measured, unfiltered top-3 for *"What BLEU score did the model achieve on English-to-German translation?"*: page 8 (`0.248`), page **1** (`0.278`), page 8 (`0.301`) — the abstract on page 1 mentions the same BLEU number in summary form, so it's a genuine near-match, not noise. Filtering to `where page in [8, 9]` replaces that page-1 result with the next real page-8 chunk (`0.342`), keeping the result set anchored to where the actual number is derived rather than just restated.

!!! note "Current field name: `max_neighbors`, not `M`"
    ChromaDB's `configuration={"hnsw": {...}}` dict uses `max_neighbors` for what FAISS, Qdrant, and Milvus all call `M` — passing `M` raises `InvalidArgumentError` in the current release.

### Qdrant

A full-featured, filter-first database — Qdrant's own architecture applies metadata filters *during* graph traversal rather than strictly before or after the ANN search, which is part of why it's frequently cited for the strongest filtering performance among open-source options. Below a configurable size (`full_scan_threshold`, default 10,000 KB) it skips the graph entirely and does an exact scan, since HNSW's overhead isn't worth it for a small enough collection. Embedded mode (`QdrantClient(":memory:")` or `path=...`) needs no server at all. [Official docs →](https://qdrant.tech/documentation/concepts/indexing/)

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/vector-databases/vector_databases_docs.py:qdrant_demo"
```

Measured, unfiltered top-3 for *"How many attention heads does the base model use?"*: page **15** (`0.632`), page 9 (`0.597`), page 7 (`0.568`) — page 15 is the appendix's attention-visualization figures, a real but tangential match. Filtering to `page <= 10` excludes it and surfaces page 5 instead (`0.568`, tied with page 7), which is the actual model-architecture section defining the head count.

!!! note "`.search()` was removed, not just deprecated"
    `qdrant-client` v1.16 fully removed `.search()`, `.search_batch()`, `.recommend()`, and several other methods — `query_points()` is the only search entry point on the current client. Per-query recall tuning goes through `search_params={"hnsw_ef": ...}` (falls back to `ef_construct` if unset), and `search_params={"exact": True}` forces brute-force search — useful for computing a local recall baseline to check ANN parameters against.

### Milvus Lite

Milvus is the option built for genuinely large, multi-tenant production deployments, with by far the widest index-type catalog of the six: FLAT, IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW and its quantized variants (HNSW_SQ/PQ/PRQ), DiskANN, GPU-accelerated IVF variants, binary-vector indexes, a sparse-vector index, and AUTOINDEX (which picks parameters for you). Milvus Lite is the same API running embedded and pure-Python, for local development that matches production code one-to-one. [Official docs →](https://milvus.io/docs/index.md)

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/vector-databases/vector_databases_docs.py:milvus_demo"
```

Measured results match FAISS exactly (same cosine metric, same embeddings): unfiltered top-3 all on page 7 (`0.592`, `0.438`, `0.385`), unchanged after filtering to `page >= 7 and page <= 9`.

!!! note "The `.db` path is a directory, not a file"
    Despite the `.db` suffix suggesting a single file, `MilvusClient("./file.db")` creates a directory (WAL + storage) — clearing it between runs needs `shutil.rmtree`, not `Path.unlink`, which raises `PermissionError` on a directory. Local, serverless use also needs the `milvus_lite` extra: `pip install "pymilvus[milvus_lite]"` — a plain `pip install pymilvus` doesn't pull it in. Milvus's own docs give parameter *ranges* for HNSW (`M`: 2–2048, `efConstruction`: 1–INT_MAX) rather than one fixed default — AUTOINDEX is the intended default path if you don't want to pick values yourself.

### pgvector

Not a database of its own — a Postgres extension, so vectors and metadata live as ordinary table columns and filtering is just SQL `WHERE` combined with `ORDER BY ... <=> ... LIMIT`, no separate filter API to learn. Supports both HNSW and IVFFlat as index types, plus a feature none of the other five have: **iterative index scans** (`SET hnsw.iterative_scan = strict_order`), which keep expanding the search when a highly selective `WHERE` filter would otherwise starve a fixed-size ANN candidate list down to too few results. [Official docs (README) →](https://github.com/pgvector/pgvector/blob/master/README.md)

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/vector-databases/vector_databases_docs.py:pgvector_demo"
```

Measured results match ChromaDB exactly, same embeddings and metric: unfiltered top-3 of page 8 (`0.248`), page 1 (`0.278`), page 8 (`0.301`); `WHERE page BETWEEN 8 AND 9` produces the same page-8-only result set ChromaDB's `where` filter did.

### Weaviate

The option with the deepest built-in tuning surface of the six: beyond `ef`/`ef_construction`/`max_connections`, Weaviate exposes `dynamic_ef_min`/`dynamic_ef_max`/`dynamic_ef_factor` (dynamic per-query `ef`, on by default), `flat_search_cutoff` (falls back to brute-force below 40,000 objects, the same small-collection philosophy as Qdrant's `full_scan_threshold`), `vector_cache_max_objects`, and a `filter_strategy` choice between `acorn` and `sweeping` for filtered search. It also ships a genuine standalone flat/brute-force index type (`Configure.VectorIndex.flat`), recommended specifically for small collections and multi-tenant setups where each tenant's own index is small enough that HNSW's graph overhead isn't worth paying. Vectors are supplied directly (`self_provided`, since embeddings already exist here) and the HNSW config nests inside `Configure.Vectors.self_provided(...)` as of client ≥ 4.16 — not a bare top-level kwarg, a common source of bugs against older tutorials. [Official docs →](https://docs.weaviate.io/weaviate/config-refs/indexing/vector-index)

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/vector-databases/vector_databases_docs.py:weaviate_demo"
```

Measured results match Qdrant's pattern exactly (same relative distances, cosine-consistent): unfiltered top-3 of page 15 (`0.368`), page 9 (`0.403`), page 7 (`0.432`); filtering to `page <= 10` again drops the page-15 appendix hit for page 5.

### Pinecone: cloud-only, documented here but not run

Pinecone has no self-hosted path at all, so it isn't part of this page's tested corpus — everything above needed nothing but a local process or a local Docker container. Its current API (the package is now `pinecone`, not `pinecone-client` — that name stopped working as of v6.0.0):

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="...")
pc.create_index(
    name="attention-paper",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
index = pc.Index("attention-paper")
index.upsert(vectors=[{"id": "0", "values": [...], "metadata": {"page": 7}}])
index.query(vector=[...], filter={"page": {"$gte": 7, "$lte": 9}}, top_k=3, include_metadata=True)
```

Metadata filtering uses a Mongo-style operator set (`$eq`, `$in`, `$gte`, `$and`, `$or`, and others) in the same call as the vector search. For multi-tenant use, [Pinecone's own guidance](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/vector-database-multi-tenancy/) recommends one **namespace per tenant** by default — noisy-neighbor isolation and a single delete for tenant offboarding — reserving a shared index with metadata filtering for when cross-tenant queries are genuinely required (at the cost of losing per-tenant cost tracking), and a fully separate index per tenant only for compliance-tier isolation needs.

## Choosing a vector database is a system design question

**The core trade-off is a triangle, not a single axis**: recall, latency, and memory — pick any two. Want high recall and low latency? Pay in memory (a denser HNSW graph, or brute force for small enough data). Want low memory and high recall? Pay in latency (smaller, cheaper index, more candidates scanned). Want low memory and low latency? Pay in recall. **Quantization is the senior-level fourth lever on top of that triangle**: int8 or product quantization commonly buys 4–16× memory reduction for only 1–3% recall loss, which is often a better trade than accepting worse latency or recall directly — worth naming explicitly rather than treating the trade-off as strictly three-way.

**The default-then-escalate pattern is what separates a senior answer from a name-drop.** Naming Pinecone or Milvus first, before establishing why the workload needs a dedicated system, reads as reaching for a tool rather than reasoning about the problem. The stronger opening: most workloads under roughly 10–50M vectors run cleanly in Postgres with pgvector alongside the primary data — one fewer system, SQL joins between vectors and application data for free — and a dedicated vector database earns its place only once a concrete constraint demands it: scale past what a single Postgres instance handles well, a hard sub-20ms p99 under real concurrent load, or a specific capability (native hybrid search, best-in-class filtering) the simpler option lacks.

**Scale bands, synthesized from FAISS's own guidance plus multiple vendors' independent framing** (treat the exact cutoffs as commonly-cited ranges, not one universal law):

| Scale | What actually works | Why |
|---|---|---|
| Under ~10K–100K vectors | Flat / brute-force, no ANN index at all | Simpler to build and debug, and fast enough that approximation buys nothing — reaching for a dedicated vector database here is itself a system design mistake |
| ~50K up to 1M–50M | HNSW (the default for most workloads) | Best recall/latency balance at moderate scale; memory cost is usually still affordable |
| 1M–100M | IVF or IVF-PQ becomes competitive with HNSW | Lower memory overhead than a graph; [Airbnb's own case study](https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/) explicitly chose IVF over HNSW here for their production constraints — a real counter-example to reflexively assuming HNSW always wins |
| 100M–1B+ | Memory becomes the binding constraint, not an implementation detail | [Uber's own writeup](https://www.uber.com/blog/powering-billion-scale-vector-search-with-opensearch/) states their 1.5B-item HNSW graph must be fully RAM-resident or latency degrades to "tens of seconds" — this is the regime where PQ compression or a disk-based index (DiskANN, SPANN) stops being optional |

**One important, honest caveat**: not every interviewer is grading vendor knowledge at all. A frequently echoed sentiment from real ML system design interview experiences is that some interviewers explicitly aren't interested in "DB choice, NoSQL trade-offs" and instead focus entirely on embedding serving, the ANN/recall/latency reasoning, and the retrieval funnel — meaning the reasoning about scale → index type → tuning matters more than which product name gets picked.

### Real production trade-offs, not hypothetical ones

**Notion → turbopuffer**: a cost- and multi-tenancy-driven migration at real SaaS scale — 600× growth in daily onboarding, 15× in active workspaces. Moved from a dedicated-hardware "pod" architecture to a serverless, object-storage-backed one (turbopuffer), cutting search infrastructure cost by 60% and improving query latency from 70–100ms down to 50–70ms. [Full case study →](case-studies/notion-turbopuffer.md)

**Uber → OpenSearch, billion-scale**: 1.5B+ items, ~400-dim vectors, targeting 100ms p99 at 2,000 QPS on HNSW. The stated hard constraint — the HNSW graph must be fully RAM-resident or query latency explodes to "tens of seconds" — is the concrete, sourced version of the memory-vs-latency trade-off above. [Full case study →](case-studies/uber-vector-search.md)

**Airbnb: choosing IVF over HNSW**: explicitly evaluated both for serving listing-embedding lookups and chose IVF as the better speed/accuracy trade-off for their actual production constraints — useful as a direct counter-example to treating HNSW as unconditionally the right default. [Full case study →](case-studies/airbnb-ivf-hnsw.md)

**Spotify: Annoy → Voyager**: a reminder that non-algorithmic constraints can dominate the decision. Voyager (built on hnswlib) benchmarked at 10× faster than their older Annoy library at equal accuracy, using 4× less memory via 8-bit quantization — but the two explicitly named non-performance drivers for the rewrite were **statelessness** and **language support**, not benchmark numbers. [Full case study →](case-studies/spotify-voyager.md)

More real production case studies — GitHub Copilot's embedding model, LinkedIn's GraphRAG deployment, Dropbox Dash's RAG-to-agent evolution, Sourcegraph Cody's decision to abandon embeddings, and others spanning agentic systems, cost optimization, and security incidents — are in the dedicated [System Design & Case Studies](case-studies/index.md) section.

### Follow-up questions interviewers actually probe

These map directly to the real engineering trade-offs above, so even framed generically they're grounded in real system behavior, not trivia:

- **"What happens when the index no longer fits in memory?"** — the direct answer is disk-based ANN (DiskANN/SPANN, [above](#beyond-ram-disk-based-ann)), or PQ compression if a moderate recall loss is acceptable.
- **"How do you handle a highly selective filter versus a broad one?"** — a filter that only passes a tiny fraction of the corpus (well under 0.1% at million-scale is a commonly cited threshold) favors pre-filtering, since post-filtering a fixed-size ANN candidate list would return too few results. A broad, low-selectivity filter favors post-filtering or Qdrant/pgvector's approach of filtering *during* graph traversal rather than strictly before or after it.
- **"How would you re-index without downtime?"** — shadow indexing (build the new index in the background, promote it once caught up — the pattern Google BigQuery documents for its own vector indexes), or a blue/green cluster swap, the approach Uber used for their own billion-scale reindex.
- **"How do you handle multi-tenant isolation?"** — Pinecone's own three-tier framework (above) generalizes across databases: namespace-per-tenant by default, shared index with metadata filtering only when cross-tenant queries are required, dedicated index per tenant only for compliance-tier isolation.
- **"How would this change at 10× scale?"** — usually quantization first (a 4–16× memory win for a small recall cost is the highest-leverage single change), then sharding strategy, then whether an in-memory index needs to become disk-based.

### A practical decision path

1. **Already on Postgres, and under roughly 10–50M vectors?** Use pgvector — one fewer system to operate, and SQL joins between vectors and the rest of the application's data in the same transactional database.
2. **Need zero-ops and cost isn't the binding constraint?** Pinecone — fully managed, no infrastructure to run, worth it when developer velocity matters more than infrastructure cost or self-hosting isn't otherwise required.
3. **Past pgvector's comfortable range and need to self-host?** Weaviate for hybrid search and filtering built in, Qdrant for the strongest filtering performance and native sparse/hybrid support, Milvus for genuinely billion-scale horizontal production with the widest index-type catalog.
4. **Just prototyping, no production stakes yet?** ChromaDB for the fastest local setup, or FAISS specifically when the goal is raw speed as an embedded library with no server at all.

"Open source" doesn't mean "free forever regardless of scale" — Milvus, Weaviate, and Qdrant all also sell a managed cloud tier on top of the free self-hosted software. The real choice is self-hosting (free software, paying for infrastructure and operational time) versus paying someone else to run it — a managed tier of an open-source database, or Pinecone, the one option in this comparison with no self-hosted path at all.

| Database | Type | License | Self-hostable? | Best for |
|---|---|---|---|---|
| **FAISS** | Library, not a full DB | MIT | Yes — just a library, no server | Research, prototyping, embedding in an application; build persistence, filtering, and multi-tenancy yourself |
| **Milvus** | Full-featured DB | Apache 2.0 | Yes (or Zilliz Cloud) | Billion-scale production needing metadata filtering, multi-tenancy, and the widest index-type selection |
| **Weaviate** | Full-featured DB | BSD-3 | Yes (or Weaviate Cloud) | Native hybrid search and the deepest built-in HNSW tuning surface of the six |
| **Qdrant** | Full-featured DB | Apache 2.0 | Yes (or Qdrant Cloud) | Strongest filtering performance; native sparse/hybrid support; frequently the most generous free tier |
| **ChromaDB** | Lightweight, developer-friendly | Apache 2.0 | Yes | Prototyping and quick local development |
| **pgvector** | PostgreSQL extension | PostgreSQL license | Yes — it's your own Postgres | Teams already on Postgres wanting one fewer system, comfortable up to roughly tens of millions of vectors |
| **Pinecone** | Managed/hosted | Proprietary | No — cloud-only | Zero-ops teams willing to pay for it |

**Beyond index choice**, three more considerations that come up in both production and interviews: **pre- vs. post-filtering** (can the database combine a metadata filter with the ANN search itself, or does it filter a fixed result set afterward — the selectivity question above); **update and deletion support** (some HNSW implementations aren't naturally efficient for high-churn corpora, since removing a node can require partial graph rebuilding); and **horizontal scaling** (does the system shard across machines as vector count grows into the billions — a concern at Milvus/Uber scale that's irrelevant at FAISS-prototype scale).

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: Vector Databases">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team is tuning an HNSW index and wants to improve recall without rebuilding the entire index, which would take hours on their large corpus.",
      "question": "Based on this page, which parameter can be changed without a full rebuild, and why?",
      "options": [
        "M, since it only affects memory usage, not the graph structure itself.",
        "ef_construction, since it governs candidate-list size the same way ef_search does, and is designed to be tunable post-build for exactly this kind of recall improvement.",
        "ef_search (ef/hnsw_ef), since it's a query-time candidate-list size, not something baked into the graph structure during index construction.",
        "HNSW doesn't expose any query-time-only tuning knob -- every parameter, including what's sometimes called ef_search, is fixed at insertion for a given build."
      ],
      "correct": 2,
      "explanations": [
        "M directly determines the graph's connectivity (how many edges per node) and is fixed once the graph is built -- changing it requires reconstructing the graph, not just an in-place update.",
        "ef_construction controls the candidate list used only during insertion/build, and is baked into the resulting graph structure -- it's ef_search, not ef_construction, that's the post-build, query-time tunable one; the two are easy to conflate since both are 'candidate list size' knobs, but only one applies after the index already exists.",
        "Correct. This page states directly that ef_search is a pure query-time setting, tunable per-query with no rebuild required -- unlike M and ef_construction, which are fixed once the graph is built.",
        "This page identifies ef_search specifically as the counter-example: it's a pure query-time setting that can be swept without touching the built graph at all -- unlike M and ef_construction, it isn't fixed at insertion time."
      ]
    },
    {
      "scenario": "A team runs FAISS's IndexIVFFlat against a small internal test corpus of a few hundred vectors and sees a warning about insufficient training points, plus no measurable difference between nprobe=1 and an exhaustive nprobe setting.",
      "question": "What does this page identify as the correct interpretation of this behavior?",
      "options": [
        "This indicates a bug in FAISS's IVF implementation that only appears at small scale.",
        "IVF's clustering doesn't have meaningful work to do at this scale -- the training-data warning and the lack of nprobe difference are both expected, real consequences of IVF being a large-scale technique, not a code defect.",
        "FAISS's own documentation sets a hard minimum of one million vectors before IndexIVFFlat is considered usable at all, based on the training-data-to-nlist ratio guidance.",
        "The warning means the index's search results at this scale can't be trusted, since undertrained clusters produce centroids too unbalanced to support correct nearest-neighbor search."
      ],
      "correct": 1,
      "explanations": [
        "This page explicitly frames this as expected behavior tied to scale, not a bug -- the same code and training step are used identically in production at real scale.",
        "Correct. This page's own measured example shows exactly this: a real FAISS training-data warning firing because there aren't enough training vectors for the requested nlist, and nprobe=1 matching the exhaustive result because the clusters are too small and sparse to differ meaningfully -- both real, honest consequences of testing IVF below the scale it's designed for.",
        "This page's own IVF sizing guidance is a set of scale-dependent formulas and training-ratio guidelines (30x-256x nlist training vectors, index variants for different vector-count ranges), not a single hard 'unusable below one million' rule -- the actual constraint is having enough training data relative to nlist, not an arbitrary total-vector-count floor.",
        "This page states the opposite: the index still returns correct results at this scale -- what the warning flags is that IVF's speed/recall trade-off has nothing meaningful to show below the scale it's built for, not that results become untrustworthy or wrong."
      ]
    },
    {
      "scenario": "In a system design interview, a candidate is asked how they would choose a vector database for a new RAG feature, and immediately answers 'I'd use Pinecone because it's the most popular managed vector database.'",
      "question": "Per this page's framing of vector database choice as a system design question, what's the issue with opening the answer this way?",
      "options": [
        "There is no issue -- naming a well-known, managed option upfront demonstrates strong tool familiarity, which is what this page frames as the strongest opening for a system design answer.",
        "It skips the reasoning about actual constraints (scale, existing infrastructure, cost, ops burden) that should come first -- this page frames naming a specific product before establishing why a dedicated system is even needed as reaching for a tool rather than reasoning about the problem.",
        "Pinecone specifically should be avoided in a system design answer, since this page recommends pgvector or a self-hosted option as the only genuinely senior-level choices.",
        "The correct opening is to always lead by naming pgvector specifically, since this page treats it as the universally correct default answer regardless of the workload's actual scale."
      ],
      "correct": 1,
      "explanations": [
        "This page states the opposite pattern is what actually reads as strong: reasoning through scale, existing infrastructure, and concrete constraints before naming a specific product -- naming a product first, however well-known, reads as reaching for a tool rather than reasoning about the problem.",
        "Correct. This page states this pattern directly: a senior answer establishes why a dedicated system is needed at all (scale past what a single Postgres instance handles, a hard latency SLA under real concurrency, or a specific missing capability) before naming a specific database -- Pinecone included.",
        "This page explicitly treats Pinecone as a legitimate choice once the reasoning establishes a genuine need for zero-ops managed infrastructure -- it's one of four options in this page's own decision path, not something to avoid outright; the issue in the scenario is skipping the reasoning, not the product named.",
        "This page frames pgvector as a strong default starting point for the reasoning, not a name to lead with regardless of scale -- past roughly 10-50M vectors, or under a hard low-latency/high-concurrency constraint, this page explicitly says pgvector is no longer the recommended choice."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- Malkov, Y.A. & Yashunin, D.A., ["Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs"](https://arxiv.org/abs/1603.09320) — the original HNSW paper, source for the M/ef_construction/ef_search guidance above
- FAISS GitHub wiki, [Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) and [Faiss indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) — IVF sizing formulas, training-data guidance, PQ parameter constraints
- [ChromaDB documentation](https://docs.trychroma.com/docs/collections/configure) — current HNSW configuration fields and mutability rules
- [Qdrant indexing](https://qdrant.tech/documentation/concepts/indexing/) and [search](https://qdrant.tech/documentation/concepts/search/) documentation
- [Milvus documentation](https://milvus.io/docs/index.md) — full current index-type catalog
- [pgvector GitHub README](https://github.com/pgvector/pgvector/blob/master/README.md) — HNSW and IVFFlat parameter reference, iterative index scans
- [Weaviate vector index configuration](https://docs.weaviate.io/weaviate/config-refs/indexing/vector-index) and [vector indexing deep dive](https://docs.weaviate.io/weaviate/tutorials/vector-indexing-deep-dive)
- [Pinecone: Product Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/) and [multi-tenancy in vector databases](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/vector-database-multi-tenancy/)
- Microsoft Research, [DiskANN](https://github.com/microsoft/DiskANN) and [SPANN](https://arxiv.org/pdf/2111.08566) — disk-based ANN for billion-scale-beyond-RAM
- Notion Engineering, [Two years of vector search at Notion](https://www.notion.com/blog/two-years-of-vector-search-at-notion)
- Uber Engineering, [Powering Billion-Scale Vector Search with OpenSearch](https://www.uber.com/blog/powering-billion-scale-vector-search-with-opensearch/)
- Airbnb, [Embedding-Based Retrieval for Airbnb Search](https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/) and the underlying [KDD 2018 paper](https://dl.acm.org/doi/10.1145/3219819.3219885)
- Spotify Engineering, [Introducing Voyager](https://engineering.atspotify.com/2023/10/introducing-voyager-spotifys-new-nearest-neighbor-search-library)
- [Embedding Model Selection](embedding-models.md) — the dense embeddings this page's databases all index and search are produced the same way described there
