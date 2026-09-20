# Multimodal RAG

Every technique so far retrieves text. But a real document rarely *is* only text — the same paper this whole site uses as a running example has a genuine architecture diagram, an attention-mechanism diagram, a multi-head attention diagram, sitting right alongside the prose. Multimodal RAG is retrieval across genuinely different modalities — text, images, audio, video, or any combination — recognizing that the right answer to a question sometimes isn't a paragraph, it's a picture.

!!! note "Scope of this page"
    Multimodal RAG covers *any* modality mix — text+audio, text+video, all of the above at once. This page (and its code) scopes down to just **text + images**, using CLIP-family models, since that's the combination with the most established, testable tooling and the clearest real corpus to demonstrate against (the sample paper's own diagrams). [Video-RAG](video-rag.md) covers the temporal/video-specific case separately.

!!! example "Hands-on"
    The full comparison below is runnable, against the sample paper's own real embedded figures — not stock images: [**Multimodal RAG →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/multimodal-rag) in the code repo. Both indexing approaches are fully local; only the final answer generation needs an LLM key.

??? abstract "TL;DR — quick revision"
    - **Two architectural approaches exist**: a unified embedding space (a CLIP-family model puts text and images in the same vector space, so one similarity search covers both) or separate per-modality indexes combined by fusion
    - **Newer isn't automatically better — measure it**: 3 candidate embedding models tested on the same corpus and questions. SigLIP (often cited as CLIP's successor) scored *worse* than base CLIP on both text and image retrieval. `jina-clip-v2` doubled base CLIP's text Recall@3 (0.375 → 0.75) at the same image accuracy, and is the model used throughout this page
    - **The unified approach has a real cost, even with the best model tested**: routing text through `jina-clip-v2`'s text encoder instead of a dedicated text embedding model costs **25 points of Recall@3** on 8 factual questions, shown in full below
    - **Separate indexes + RRF fusion is the better default**, verified two ways: it correctly retrieves both a diagram and its explanatory text for an image-appropriate question, and correctly returns *only* text for a plain factual question
    - **RRF is rank-based, not relevance-based**: with a small image pool, the top-ranked image can win fusion on pure noise unless gated by a minimum-similarity threshold; and RRF produces an exact tie at rank 0 between modalities on every query — a structural property of rank fusion that no tiebreaker resolves

## Two ways to combine modalities

**Approach 1 — unified embedding space.** A model like CLIP embeds text and images into the *same* shared vector space, so a text query and a relevant image end up close together despite being different modalities entirely. One similarity search covers everything.

**Approach 2 — modality-specific indexes + fusion.** Keep separate indexes per modality — text through a dedicated text embedder, images through CLIP — retrieve from each independently, then combine the results. The same idea as [hybrid sparse+dense retrieval](embedding-models.md#a-real-worked-comparison-bm25-vs-dense-vs-splade) — combining two differently-strengthed retrieval signals rather than trusting one to cover everything — just across modalities here instead of retrieval methods.

The obvious question is which one to actually reach for. Testing both against the same real corpus gives a clear, measured answer.

## Is there something better than CLIP?

CLIP (2021) is genuinely old by ML standards. Rather than guess at a replacement, the [MTEB multimodal retrieval leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is the real place to check — its top entries (Qwen3-VL-Embedding-8B, Gemini Embedding 2, Voyage Multimodal 3.5) are either too large for an 8GB machine to run locally or API-only, so 2 realistically-sized, locally-runnable candidates got measured head-to-head against base CLIP, on the exact same corpus and questions:

- **SigLIP** (Google, often cited as the strongest current open-weights image-text model) — loaded fine, but scored **worse on both axes**: 0.25 Recall@3 on plain text (vs. CLIP's 0.375) and correctly matched only 1 of 4 test queries to its diagram (vs. CLIP's 2 of 4), with near-zero, uncalibrated raw similarity scores throughout. SigLIP is trained with a different loss (sigmoid, not contrastive softmax) and a much shorter text context window — likely a genuine mismatch for this use case, not a config error.
- **Jina CLIP v2** (built specifically for long-context text+image retrieval, including charts/tables) — needs `transformers==4.46.3` pinned, since newer `transformers` releases dropped a function its custom model code depends on, and `trust_remote_code=True`, since it ships custom modeling code from its own HuggingFace repo. Its `sentence-transformers` integration doesn't route image inputs correctly, so this recipe calls the model's own `AutoModel.encode_text` / `encode_image` methods directly instead of the generic `.encode()` wrapper used everywhere else on this site.

Recall@3 and image-matching accuracy for all three, on the same corpus and questions:

```text
Model                                Text Recall@3   Image queries correct (of 4)
---------------------------------------------------------------------------------
clip-ViT-B-32 (base CLIP)            0.375           2/4
google/siglip-base-patch16-224       0.25            1/4
jinaai/jina-clip-v2                  0.75            2/4   <- winner
```

`jina-clip-v2` is what this page's code and every example below use — it doubles base CLIP's text-retrieval quality at the same image-matching accuracy. It isn't flawless: it still gets 2 of the 4 image-specific test queries wrong — asked about scaled dot-product attention specifically, it retrieves the Multi-Head Attention diagram instead (the two diagrams are visually and conceptually close). The broader lesson: a model being newer or more cited as SOTA (SigLIP) doesn't automatically mean it's better for a given use case — only measuring each candidate settles it.

!!! tip "More memory available? Try a bigger model"
    `jina-clip-v2` was chosen to fit an 8GB machine. With more RAM/VRAM, `Qwen3-VL-Embedding-8B` (top of the [MTEB multimodal leaderboard](https://huggingface.co/spaces/mteb/leaderboard) at the time of writing, but needing well over 8GB) is worth measuring the same way. Gemini Embedding 2 and Voyage Multimodal 3.5 are API-only alternatives — no local memory cost, but they need `MultimodalEncoder` rewritten to call the provider's endpoint instead of a local `transformers` model.

## Building both, against the paper's own real figures

The sample "Attention Is All You Need" PDF has 3 embedded diagrams — the overall encoder-decoder architecture, the scaled dot-product attention mechanism, and multi-head attention — extracted directly, not substituted with stock images:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/multimodal-rag/multimodal_rag_docs.py:extraction"
```

### The unified approach

Everything — text chunks and images alike — goes into one shared embedding space (`jina-clip-v2`, the winner measured above), so a single similarity ranking can return either kind of result:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/multimodal-rag/multimodal_rag_docs.py:unified"
```

!!! warning "Measured: a real cost to plain-text retrieval, even with the best model tested"
    Using a CLIP-family text encoder (required if text and images share one space) means giving up the dedicated text embedding model used everywhere else on this site. Measured directly, on 8 real factual questions against this same corpus, using `jina-clip-v2` specifically — not the weaker base CLIP:

    ```text
    Question                                          Keyword         Naive   jina-clip-v2
    ------------------------------------------------------------------------------------
    How many attention heads did they use?            "h = 8"         ✅      ✅
    What is the model's embedding dimension?          "dmodel = 512"  ✅      ❌
    How many layers are in the encoder?               "N = 6"         ✅      ✅
    What optimizer was used for training?             "Adam"          ✅      ✅
    What BLEU score did they get (En-De translation)? "28.4"          ✅      ✅
    What GPUs was the model trained on?                "P100"         ✅      ✅
    How long did the base model train for?            "12 hours"      ✅      ✅
    What dropout rate did they use?                   "Pdrop = 0.1"   ✅      ❌

    Recall@3:                                                         1.000   0.750
    ```

    That's 25 points of Recall@3 lost, on plain factual questions with no image relevance at all — smaller than base CLIP's 62-point gap, but not zero, even with the model actually built for long-context text+image retrieval. The cost isn't a training-data mismatch specific to one model; it's structural: any text encoder good enough to sit in the *same* space as image embeddings is trading away some text-specific retrieval quality to do it. That's the real reason to reach for separate indexes whenever text quality matters and images don't need to directly outrank text.

### Separate indexes + RRF fusion

Text goes through the dedicated embedder (full retrieval quality preserved); images go through CLIP (the only place CLIP is actually needed); the two independently-ranked lists get combined with Reciprocal Rank Fusion — the same RRF used for [multi-query/RAG-Fusion](advanced-rag-pre-retrieval.md#multi-query-retrieval-rag-fusion) elsewhere on this site:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/multimodal-rag/multimodal_rag_docs.py:separate_fusion"
```

!!! success "Measured: correctly retrieves the right modality for each question type"
    **Image + text, mixed correctly** — asked *"How are multiple attention heads concatenated together?"* (deliberately not the scaled-dot-product query — `jina-clip-v2` gets that one wrong, see above; this is the one verified correct):

    ```text
    [TEXT]  "...Figure 5: Many of the attention heads exhibit behaviour
             that seems related to the structure of the sentence..."
    [IMAGE] Multi-Head Attention diagram (multiple parallel attention
             heads, Linear, Concat)
    [TEXT]  "...output values. These are concatenated and once again
             projected, resulting in the final values, as depicted in
             Figure 2. Multi-head attention allows the model to jointly
             attend to information from diff[erent subspaces]..."

    Answer: "...The outputs from each individual attention head (head_1,
             head_2, ..., head_h) are concatenated and then projected
             through a linear transformation using a weight matrix W^O.
             MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O..." —
             grounded in both the diagram and the surrounding text.
    ```

    **Text only, correctly, for a plain factual question** — asked *"What optimizer did they use for training?"*:

    ```text
    [TEXT]  "warmup_steps = 4000. 5.4 Regularization..."
    [TEXT]  "(3.5 days). 5.3 Optimizer We used the Adam optimizer [20]
             with β1 = 0.9, β2 = 0.98 and ϵ = 10−9..."
    [TEXT]  "target tokens. 5.2 Hardware and Schedule We trained our
             models on one machine with 8 NVIDIA P100 GPUs..."

    Answer: "...they used the Adam optimizer for training, with
             β1 = 0.9, β2 = 0.98, ϵ = 10^-9." — no image pulled in,
             because none is actually relevant here.
    ```

!!! warning "RRF has no relevance floor"
    RRF only knows RANK within a list — it has no concept of whether "rank 1" is actually relevant or just the least-irrelevant option available. With only 3 images in the pool, all three can score a nearly-identical, low similarity against a plain text question — genuine noise, not signal — yet the highest of the three still becomes "rank 1," earns a real RRF fusion score, and gets pulled into the final top-3 ahead of a relevant text chunk. A minimum similarity threshold fixes this: a modality's results don't contribute to fusion at all unless its best candidate clears a real relevance bar. This is the same effect [multi-query/RAG-Fusion](advanced-rag-pre-retrieval.md#multi-query-retrieval-rag-fusion) demonstrates elsewhere on this site — RRF diluting a good match with weak competition — showing up here as a small-candidate-pool edge case instead.

## Why there's no "image-only" example on this page

It's a fair question to ask for one. Against several deliberately image-specific queries (*"a diagram showing MatMul Scale Mask SoftMax boxes"*, *"Linear Concat multi-head attention diagram"*), neither approach — with base CLIP or with `jina-clip-v2` — returns an image-only top-3 result on this corpus. There are two structural reasons why:

**Reason 1 — RRF's rank-0 tie (separate + fusion approach).** `1/(k+0+1)` is mathematically identical whether a text chunk or an image occupies rank 1 of its own list — a genuine tie, not a near-miss. Normalizing each candidate's raw similarity within its own list, then breaking ties by that, doesn't help either: the top item in *any* list normalizes to exactly 1.0 by construction, so the tie persists regardless. RRF is fundamentally a consensus mechanism across ranked lists, not a magnitude comparator, so it has no principled way to let one modality's confident pick outright beat another's merely-present one.

**Reason 2 — the modality gap (unified approach), present but smaller with a better model.** With base CLIP, all 3 images land at the *very bottom* of the combined 95-item ranking (positions 92-94) for every image-specific query — a stark, uniform floor. With `jina-clip-v2`, that gap narrows substantially for at least one image: for *"a diagram showing MatMul Scale Mask SoftMax boxes"*, the Scaled Dot-Product Attention diagram ranks **4th of 95** overall. That's still not the same as landing in `unified_search`'s literal top-3, though — in practice it lands at position 5, just behind four text chunks whose wording overlaps the query more tightly ("queries and keys of dimension dk...", "Attention(Q,K,V) = softmax(QK^T/√dk)V"). Even a well-positioned image doesn't out-rank text passages that are near-verbatim matches to the query's own phrasing.

This matches a real, published, well-studied phenomenon: [CLIP-family text and image embeddings occupy structurally separate regions of the shared vector space](https://arxiv.org/abs/2203.02053) — an artifact of how contrastive vision-language models are initialized and trained, not specific to this corpus or model. `jina-clip-v2` narrows that gap compared to base CLIP but doesn't close it — cross-modal similarity (text query vs. image) is still systematically weaker than intra-modal similarity (text query vs. other text), even when the image is the genuinely correct answer.

**Takeaway:** a reliable image-only result on a corpus this size, with either approach here, would need a much larger, more distinctly-imaged corpus, a model with an even smaller modality gap than `jina-clip-v2`, or an explicit modality-routing step upstream (the same routing idea covered on the [Modular RAG](modular-rag.md) page) rather than relying on fusion or a shared embedding space alone.

## When to use which approach

**Use separate indexes + fusion for:** the general case — it preserves full text-retrieval quality and only pays the CLIP-family model's cost where it's actually needed (images), and correctly returns text-only results for plain factual questions. This should be your default. **Use a unified embedding space for:** specifically when you need images to be able to outrank text outright in a genuinely open, single ranking — RRF's rank-0 tie structurally prevents that in the fused approach, so if your use case depends on an image sometimes winning cleanly over mediocre text matches, unified is the only one of the two that can do it (at the real text-quality cost measured above, 25 points of Recall@3 even with the best model tested). **Skip multimodal retrieval entirely for:** a corpus that's genuinely text-only — the added complexity (a second embedding model, a fusion or unification step) buys nothing if there's no second modality to retrieve.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: Multimodal RAG">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team builds a multimodal RAG system by embedding both text chunks and images into a single shared CLIP space, reasoning that this is simpler than maintaining two separate indexes. They then notice that plain factual text questions are answered less reliably than before they added images.",
      "question": "What's the most likely explanation, based on this page's measured findings?",
      "options": [
        "CLIP's shared embedding space causes images to systematically outrank text results at retrieval time, so the drop in factual-question accuracy comes from images crowding out the correct text chunks in the unified ranking.",
        "Routing text through a CLIP-family text encoder (necessary to keep everything in one shared space) sacrifices real text-retrieval quality compared to a dedicated text embedding model \u2014 measured directly at a 25-point Recall@3 drop on the same factual questions, even with the best CLIP-family model tested \u2014 a real structural trade-off, not a bug.",
        "The drop would disappear if the text were re-embedded with base CLIP instead of jina-clip-v2, since base CLIP was measured with a smaller modality gap on this corpus.",
        "This cost is specific to jina-clip-v2's particular training recipe, and a different CLIP-family model chosen instead would let text retrieval keep its full dedicated-embedder-level accuracy while still sharing a space with images."
      ],
      "correct": 1,
      "explanations": [
        "This is backwards -- this same page's modality-gap findings show images tend to rank far BELOW text in a unified index (near the very bottom with base CLIP), not above it. The text-accuracy drop comes from the CLIP-family text encoder itself being a weaker text retriever, not from images displacing correct text chunks.",
        "Correct. This is exactly the measured, quantified cost this page demonstrates: unifying text and images into one shared space means text retrieval quality drops (1.00 -> 0.75 Recall@3 measured directly, even with jina-clip-v2, the best model tested) because a CLIP-family text encoder is a weaker text retriever than a dedicated embedding model \u2014 a real structural trade-off of the unified approach.",
        "This reverses the actual measurement -- base CLIP's plain-text Recall@3 (0.375) is far worse than jina-clip-v2's (0.75). Switching back to base CLIP would widen the text-quality gap, not close it; jina-clip-v2 is the stronger of the two CLIP-family text encoders measured here.",
        "The page states this as a structural cost, not a model-specific quirk -- any text encoder good enough to sit in the same space as image embeddings trades away some text-only retrieval quality to do it. Choosing a different CLIP-family model could change the SIZE of that cost but wouldn't eliminate it."
      ]
    },
    {
      "scenario": "A team's separate-indexes-plus-RRF-fusion multimodal system occasionally returns an image in its top-3 results even for questions that have nothing to do with any image in the corpus, when the image pool is very small (e.g. only 3 images total).",
      "question": "What causes this specific failure mode, based on what this page's testing surfaced?",
      "options": [
        "RRF is double-counting the same image across the fused ranking, since each image gets independently embedded by both the CLIP branch and, redundantly, by the text branch before fusion runs.",
        "RRF combines ranked lists based purely on RANK POSITION, with no built-in concept of absolute relevance \u2014 with only a few images in the pool, whichever one scores even marginally highest still becomes 'rank 1' in its list and earns a real fusion score, even if every image is actually irrelevant to the query.",
        "This only happens because RRF's fusion weights favor images over text by default, and reversing that weighting so text always wins ties would fix it without needing a relevance threshold.",
        "This only happens when the image and text indexes were built at different points in time, so the two rankings are being fused against a stale version of one of the indexes."
      ],
      "correct": 1,
      "explanations": [
        "Images aren't embedded twice in this setup -- in the separate-indexes approach, each image is embedded once, only through CLIP, and text is embedded once, only through the dedicated text embedder. The noise problem here comes from how few candidates exist in the image list, not from duplicate embedding.",
        "Correct. RRF only reasons about relative rank, not absolute relevance \u2014 with a small image pool, all 3 images can score near-identical, low, uncalibrated similarity (genuine noise, not signal) for an unrelated question, yet the highest-scoring one still gets a real fusion boost as 'rank 1' unless a minimum similarity threshold gates participation in fusion at all.",
        "Standard RRF has no per-modality weighting to begin with -- it fuses purely by each item's own rank within its own list. The fix this page actually describes is a minimum similarity threshold that keeps a modality's results out of fusion entirely unless they clear a real relevance bar, not adjusting weights between modalities.",
        "Index staleness isn't the mechanism here -- the same, freshly built 3-image index still shows this behavior. The cause is RRF's rank-only logic applied to a very small candidate pool with no relevance floor, not any mismatch from outdated data."
      ]
    },
    {
      "scenario": "A team tries several deliberately image-specific queries against a unified CLIP-family index. With base CLIP, every image ranks near the very BOTTOM of the combined ranking for every such query. After switching to a newer, better-measured model (jina-clip-v2), one image jumps to rank 4 out of 95 for one query \u2014 but still doesn't crack the actual top-3 returned by search, because a handful of text chunks happen to share near-verbatim wording with the query.",
      "question": "What does this pattern most likely indicate?",
      "options": [
        "Rank 4 out of 95 means jina-clip-v2 has essentially closed the modality gap for this query, and the only reason the image doesn't appear in the literal top-3 is an unrelated tie-breaking quirk in how the ranking list gets truncated.",
        "This matches CLIP-family models' well-documented 'modality gap' \u2014 text and image embeddings occupy structurally separate regions of the shared vector space, an artifact of how these models are initialized and trained. A better model can measurably narrow that gap (rank 92 -> rank 4 for this image), but doesn't close it, so cross-modal similarity still tends to lose to strong intra-modal (text-vs-text) matches even when the image is genuinely relevant.",
        "This specific query was a rare exception -- across the rest of this page's image-specific test queries, jina-clip-v2 otherwise fully eliminates the base-CLIP modality gap.",
        "Because a better model produced such a large rank improvement, the modality gap must be entirely an artifact of which embedding model is used, rather than a structural property of how contrastive vision-language models are trained."
      ],
      "correct": 1,
      "explanations": [
        "No tie-breaking or truncation quirk is involved -- the image is genuinely outranked by several text chunks whose wording matches the query more closely (near-verbatim phrases like 'queries and keys of dimension dk'). Rank 4 of 95 is real, measurable narrowing of the gap, not closure of it.",
        "Correct. This is exactly what re-measuring the modality gap with a second, better model showed on this page: the gap is real and reproducible, but not fixed-size \u2014 a stronger model can shrink it substantially (92nd -> 4th out of 95 for this specific image and query) without eliminating it, since text-vs-text similarity still tends to edge out even a well-ranked cross-modal match.",
        "The page is explicit that jina-clip-v2 narrows the gap but doesn't close it -- it still gets 2 of the 4 image-specific test queries wrong overall. This result isn't an isolated exception to an otherwise-solved problem; it's representative of a persisting, just-smaller gap.",
        "The page cites this as a well-documented structural property of how contrastive vision-language models are initialized and trained, not merely an artifact of one model's quality. A better model can measurably shrink the gap, exactly as shown here, without that meaning the gap itself is just a fixable modeling flaw."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — the original CLIP paper
- [Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning](https://arxiv.org/abs/2203.02053) — the real, measured phenomenon behind why images still rank below strong text matches in the unified approach above, even with a model that narrows the gap substantially
- [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665) — extends the shared-embedding-space idea to audio, depth, and thermal alongside text and images
- [MTEB Multimodal Retrieval Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — the real leaderboard checked before picking `jina-clip-v2` as the third candidate to measure
- [Advanced RAG — Pre-retrieval](advanced-rag-pre-retrieval.md#multi-query-retrieval-rag-fusion) — the RRF mechanism reused here for cross-modal fusion
