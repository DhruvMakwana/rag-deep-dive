# Spotify: Retiring Annoy for Voyager

Spotify's recommendation features, including Discover Weekly, had run on Annoy — an approximate nearest-neighbor library Spotify itself originally built and open-sourced — since 2013. In 2023, Spotify's engineering team retired Annoy in favor of Voyager, a new nearest-neighbor search library built on top of hnswlib. The writeup Spotify published about the switch is a useful case study specifically because it doesn't lead with benchmark numbers as the whole story — it names organizational and deployment constraints as decisive alongside the performance gains, which is a distinction easy to miss when comparing ANN libraries purely on paper.

## The problem

Annoy had served Spotify well for a decade, but a decade is a long time in ANN research. Spotify's own assessment was blunt: Annoy was "now solidly in the middle of the pack" against newer libraries that had incorporated a decade of subsequent algorithmic progress. That alone would justify evaluating alternatives. But Spotify's writeup makes clear that raw speed and accuracy weren't the only forces pushing them to rewrite a decade-old, widely-depended-on piece of infrastructure — two structural constraints mattered just as much.

## What they built, and why

Voyager is built on hnswlib, giving it the benefit of a well-established HNSW implementation rather than Annoy's older tree-based approach. Spotify's benchmarks against Annoy showed Voyager as **10x faster at equivalent accuracy**, and **up to 50% more accurate at equivalent speed** — a meaningful jump reflecting how far ANN algorithm design had moved since 2013. Voyager also used **8-bit float quantization** to cut memory footprint **4x** relative to Annoy, and needed **16x less memory during index build** compared to using raw hnswlib directly, meaning Spotify got hnswlib's algorithmic quality without inheriting its build-time memory cost.

But Spotify explicitly named two non-performance drivers that mattered as much as those benchmark numbers. First, **statelessness**: Spotify needed to deploy nearest-neighbor search via Kubernetes without requiring a stateful cluster, and Annoy's architecture didn't cleanly support that deployment model. A library that can't run cleanly in the infrastructure pattern your platform team has standardized on creates ongoing operational friction no benchmark number captures. Second, **language support**: Spotify's backend teams worked in JVM languages (Java and Scala) while ML teams worked in Python, and Spotify found that many ANN libraries only support one ecosystem well. Voyager needed to serve both audiences natively rather than forcing one team to work around bindings built for the other.

## The numbers

Relative to Annoy, Voyager delivered **10x faster search at the same accuracy**, **up to 50% higher accuracy at the same speed**, and **4x lower memory usage** via 8-bit float quantization. Relative to using hnswlib directly, Voyager needed **16x less memory during index construction**. These numbers describe a clear win on the axes ANN libraries are usually compared on — but per Spotify's own writeup, they weren't the whole justification for the rewrite.

!!! success "The lesson"
    Benchmark numbers like "10x faster" or "4x less memory" are necessary but not sufficient to justify replacing infrastructure a decade of production traffic depends on. Spotify's own account puts statelessness and cross-language support on equal footing with the performance case — organizational and deployment constraints that determine whether a library actually fits how your teams build and ship, not just how fast it runs in isolation. When comparing retrieval or ANN libraries on paper, it's easy to underweight exactly these fit questions because they don't show up in a benchmark table, and easy to overweight the numbers that do.

## Sources

- [Spotify Engineering: Introducing Voyager, Spotify's New Nearest-Neighbor Search Library](https://engineering.atspotify.com/2023/10/introducing-voyager-spotifys-new-nearest-neighbor-search-library)
