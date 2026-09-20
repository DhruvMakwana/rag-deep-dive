# Airbnb: Choosing IVF Over HNSW for Real-Time Personalization

Airbnb's search and recommendation systems use embedding-based retrieval to power two related features: similar-listing recommendations and real-time personalization within search ranking, both described in Airbnb's KDD 2018 paper "Real-time Personalization using Embeddings for Search Ranking at Airbnb" and later engineering writeups. Buried inside that work is a smaller but genuinely useful decision most RAG and retrieval discussions skip past: when Airbnb had to choose an ANN index to serve real-time nearest-neighbor lookups against listing embeddings, they evaluated both IVF and HNSW and chose **IVF**. That choice runs directly against the reflexive industry habit of reaching for HNSW as the default "good" ANN algorithm, which is exactly what makes it worth studying.

## The problem

Airbnb needed to serve nearest-neighbor lookups against listing embeddings in real time, as part of both the similar-listings feature and live personalization signals injected into search ranking. Any production ANN deployment forces a choice among index types, and each comes with a different profile of trade-offs across query latency, recall, memory footprint, index build/update cost, and how gracefully the index handles new or changing data. HNSW and IVF are two of the most common choices, and they sit in genuinely different places on that trade-off surface: HNSW is graph-based and tends to offer strong recall-per-latency, while IVF partitions the vector space into clusters and searches only the nearest ones, trading some recall for a different cost profile around memory and update behavior.

## What they built, and why

Airbnb explicitly evaluated both algorithms against their actual production constraints rather than against abstract ANN benchmarks, and picked IVF as the better speed/accuracy trade-off for their system. The public writeups don't reduce this to a single benchmark number, but the decision itself is the substantive finding: a team with real production traffic, real latency budgets, and a real update pattern for its data ran the comparison themselves rather than assuming the algorithm with the best reputation in ANN benchmarking papers was automatically the right one for their deployment.

The retrieval system this index choice served — embedding-based similar-listings and real-time personalization — was itself a significant product win, which is part of why the underlying index decision is worth taking seriously rather than treating as an incidental implementation detail.

## The numbers

The business results reported for the embedding-based retrieval system this index served: a **21% increase in carousel click-through rate** and **4.9% more bookings** driven by discovered listings. These numbers describe the value of the retrieval system as a whole rather than isolating IVF's individual contribution, but they establish that this was production infrastructure serving real revenue-impacting traffic — not a research exercise — which is exactly the setting where the IVF-over-HNSW decision was made.

!!! success "The lesson"
    The "better" ANN algorithm on paper isn't always the right one for a specific production constraint set. HNSW's strong reputation in generic recall/latency benchmarks doesn't automatically transfer to every deployment — write throughput, memory budget, update frequency, and how the index needs to behave as underlying data changes can all tip the decision toward an index type that looks worse in a benchmark table. The transferable habit here isn't "prefer IVF" or "prefer HNSW" — it's Airbnb's actual process: ask what your system specifically needs before defaulting to whichever algorithm benchmarks best in the abstract, and be willing to run your own comparison rather than inherit someone else's.

## Sources

- [Grbovic & Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb," KDD 2018](https://dl.acm.org/doi/10.1145/3219819.3219885)
- [Airbnb Tech Blog: Embedding-Based Retrieval for Airbnb Search](https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/)
- [Airbnb Engineering (Medium): Listing Embeddings for Similar Listing Recommendations and Real-time Personalization in Search](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e)
