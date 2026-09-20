# Shopify: Real-Time Embeddings for Hybrid Product Search

Shopify's product search has to work across millions of independent merchants and billions of queries, including sustained peak load during events like Black Friday and Cyber Monday, while the underlying inventory, pricing, and product content are changing continuously. Shopify's own engineering blog has published two posts on this system — one on the ranking architecture, one on the real-time embedding pipeline behind it — and together they're a useful case study because they name the specific failure mode of treating hybrid search as a static-index problem, and describe the streaming infrastructure built to avoid it.

## The problem

A search index that's rebuilt in batch, on some periodic schedule, is fundamentally mismatched to a marketplace where merchants are changing prices, updating descriptions, and adding or removing inventory constantly. A batch-only index means search results can point at stock that's gone or prices that are wrong, and the staleness gets worse the longer the batch interval is. At the same time, Shopify's own engineering blog makes an unusually direct statement of the classic hybrid-retrieval tradeoff: pure vector search misses exact matches and wastes ranking budget, while BM25 alone loses semantic intent. Solving for relevance and solving for freshness are usually treated as separate problems, but at Shopify's scale, an architecture that only solved one of them wouldn't actually be usable in production.

## What they built, and why

Shopify's ranking layer combines two categories of signal deliberately, rather than betting entirely on either one. Classical information-retrieval features do the exact-match and structural work: typo correction, synonym handling, faceting, and roughly **80 handcrafted text-similarity features** — things like TF-IDF scoring on the title field and exact-match checks on brand. Layered on top of that are ML and semantic signals — transformer-based neural rankers and embeddings — that catch conceptual and intent-based matches the classical features can't. Neither layer is treated as a replacement for the other; the architecture is explicitly hybrid because each layer catches failure modes the other one has.

The half of the system that makes this hybrid design workable under constantly-changing inventory is a **real-time embedding pipeline**, built on **Google Cloud Dataflow and Apache Beam** with **GPU-accelerated (T4)** inference, generating text and image embeddings continuously and publishing them to downstream search-index consumers as a stream rather than a periodic batch job. This is the piece that keeps the semantic half of the hybrid system from going stale the moment a merchant changes a listing.

Because shipping new ranking features or models normally means writing performance-critical C++, and that's a skillset most of Shopify's data scientists don't have, Shopify also built custom tooling to close that gap: **RankFlow**, a Python-like domain-specific language that compiles down to C++, paired with a **TurboDSL** execution engine. The point of this tooling isn't a nicety — it's what let data scientists iterate on ranking features and models directly, without a C++ engineer as a bottleneck in the loop, while still getting compiled-code performance out the other end.

## The numbers

The embedding pipeline generates roughly **2,500 embeddings per second**, or about **216 million embeddings per day**. Tuning thread configuration cut per-worker memory footprint by roughly **2.6x**, which let Shopify use cheaper machine types while keeping GPU utilization saturated. The TurboDSL execution engine delivered a **48% speedup** in ranking feature computation. The system as a whole serves billions of queries during Black Friday and Cyber Monday peak load.

It's worth being precise about what Shopify has *not* published: neither engineering post discloses a head-to-head relevance or conversion number comparing the hybrid architecture against a single-method baseline — there's no "+X% relevance" or "+Y% GMV" figure in either source. The rationale for going hybrid is stated clearly and qualitatively; the business-impact quantification of that choice isn't public.

!!! success "The lesson"
    The differentiator in this case study isn't the hybrid ranking architecture itself — combining classical IR signals with ML signals is a familiar pattern. It's that Shopify built a **streaming**, not batch, embedding pipeline as a precondition for that hybrid architecture to actually work under fast-changing inventory. A hybrid search design that assumes a static or rarely-updated corpus will quietly degrade in exactly the semantic-relevance half of the system as soon as the underlying catalog starts changing faster than the reindex schedule — which, for a multi-merchant marketplace, is always. Real-time embedding generation is what keeps the semantic layer of a hybrid system honest.

## Sources

- [Shopify Engineering: World-Class Product Search](https://shopify.engineering/world-class-product-search)
- [Shopify Engineering: How Shopify Improved Consumer Search Intent with Real-Time ML](https://shopify.engineering/how-shopify-improved-consumer-search-intent-with-real-time-ml)
