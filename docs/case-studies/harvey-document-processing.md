# Harvey AI: Document Processing at Legal Scale

Harvey is a legal AI platform used by a large share of Am Law 100 firms, and its document-processing pipeline is the thing nearly every lawyer query touches — a single query might reference one file or several thousand at once. Over the course of a year, the volume flowing through that pipeline grew roughly 26-fold, and the engineering choices Harvey made to survive that growth without falling over are a concrete, specific case study in how ingestion infrastructure actually breaks at scale, well past the point where generic chunking-strategy advice stops being useful.

## The problem

A document pipeline that works fine at modest volume tends to be built as one conceptual pipeline: extract, chunk, embed, index, done. The trouble is that this framing quietly assumes every stage fails the same way and scales the same way, which stops being true well before you reach real production load. Harvey needed its pipeline to survive roughly 25x growth in a single year without a rewrite, and the failure modes that growth exposed were not about any single stage being slow — they were about the stages no longer behaving like parts of one system.

## What they built, and why

Harvey's pipeline runs in three stages. **Extraction and metadata** downloads the file, detects its type, extracts text and structure, runs OCR on scanned content, and captures metadata. **Embedding** splits the extracted content into retrievable chunks and generates embeddings per chunk. **Indexing** writes the chunks, metadata, and embeddings into the retrieval layer. At low volume this reads as a straightforward linear pipeline. At high volume, Harvey found that extraction, chunking, embedding, indexing, storage, and retrieval each behave like independent systems that break in structurally different ways — and the engineering response had to treat them that way.

The first fix was a **Unified Document Format (UDF)**: a versioned internal schema shared across every stage of the pipeline, replacing an ad hoc, monolithic document object that had grown organically. Once volume is high enough, an implicit, undocumented contract between stages becomes a source of silent cross-stage breakage — one stage's informal assumption about what a "document" looks like drifts out of sync with another's, and failures show up far from their actual cause. A shared, versioned schema stopped that drift and gave every stage the same contract to code against.

The second fix targeted the handoff between the embedding stage and the indexing stage specifically: Harvey moved that handoff from JSON to **Arrow IPC**. At tens of millions of chunks a day, JSON serialization and deserialization overhead stops being a rounding error and starts being a real, measurable cost in both time and memory — Arrow's columnar, binary format cut serialization/deserialization time, payload size, and memory footprint on that path.

The third fix was a **durable job and workflow framework**. Below a certain volume, a worker crash or a mid-deploy restart just means restarting the batch — mildly annoying, not expensive. At Harvey's volume, restart-from-zero on every transient failure becomes unacceptably wasteful, so the framework was rebuilt to resume an in-flight batch from where it left off instead of discarding the work already done.

Harvey also had to migrate its vector database without downtime, which it did through **dual writes to the old and new systems in parallel**, combined with shadow reads against the new system and backfills to bring it fully in sync before cutover — a pattern that avoids a hard stop-the-world migration on infrastructure that's serving live queries continuously.

Perhaps the most structurally important decision was **rejecting a single shared resource pool across stages** in favor of stage-specific capacity control: separate task queues, separate worker pools, per-stage memory sizing, per-stage rate limits, task-slot monitors, and per-activity retry budgets. The reasoning follows directly from the observation that each stage fails differently — OCR is CPU/GPU-bound, embedding is rate-limited by an external API, indexing is I/O-bound against the vector store. A shared pool means one stage's failure mode (an API rate limit, say) starves capacity from a completely unrelated stage that was working fine. Isolating capacity per stage means each stage's failures stay contained to that stage.

Finally, Harvey built **ordered fallback chains for extraction** — a ladder of extraction strategies attempted in sequence when the primary one fails on a given document — along with explicit backoff and retry accounting specifically tuned for vector-database rate limits, rather than a single generic retry policy applied uniformly across every kind of failure.

## The numbers

Over twelve months, documents processed grew from **0.94 million to 24.8 million** — a **26× increase**. Data volume grew from **1.44TB to 56TB**, a **39× increase**. The pipeline sustained an average throughput of roughly **3.5 million documents per day**.

!!! success "The lesson"
    At sufficient scale, a document-ingestion pipeline stops being "one pipeline" and becomes N independently-failing systems that happen to be glued together — the fix is a shared, strongly-typed contract between stages (Harvey's UDF) plus genuine per-stage isolation of compute, queues, and retries, not a single shared resource pool that assumes every stage fails alike. Serialization format is a real cost lever at this volume, not a style preference — the JSON-to-Arrow switch on the embed-to-index handoff mattered because tens of millions of chunks a day make even small per-chunk overhead compound into a measurable systems cost. This is a specific, concrete answer to the system-design question of how chunking and indexing infrastructure actually breaks under real production load, well beyond generic advice about chunk size or overlap.

## Sources

- [Harvey: Scaling Document Processing](https://www.harvey.ai/blog/scaling-document-processing-across-harvey)
