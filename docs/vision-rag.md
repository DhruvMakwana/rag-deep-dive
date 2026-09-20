# Vision RAG

Every RAG technique on this site so far starts the same way: extract text from a document, then chunk it. That extraction step is itself a failure point — a table's numbers get flattened into a sequence that loses which value belongs to which row and column; a multi-column layout gets read in the wrong order; a chart's legend ends up disconnected from its data. Vision RAG skips extraction entirely: render each page as an image, retrieve over the images directly, and let a vision-capable LLM read the retrieved page the way a person would — by looking at it.

!!! note "How this differs from Multimodal RAG"
    [Multimodal RAG](multimodal-rag.md) extracts individual figures out of a PDF and retrieves text chunks plus those figures — text is still extracted and chunked either way. Vision RAG never extracts text at all: a whole page is the retrieval unit, embedded directly as pixels, and the page image itself is what the LLM reads at generation time.

!!! example "Hands-on"
    The full comparison below is runnable, against the same sample paper used across this site: [**Vision RAG →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/vision-rag) in the code repo. Retrieval is fully local; generation needs a vision-capable LLM key (Anthropic or OpenAI).

??? abstract "TL;DR — quick revision"
    - **No text extraction anywhere**: each PDF page is rendered as an image, embedded directly, and — when retrieved — read by the LLM as an image, not OCR'd text
    - **Late interaction (MaxSim), not a single vector**: a page is embedded as a grid of patch vectors, not pooled into one summary vector, so a page can win retrieval on the strength of one region — a single table cell — even if the rest of the page is unrelated to the query
    - **Not a universal upgrade over naive text RAG**: measured Recall@3 on 8 plain factual questions, naive chunk-based text RAG actually scores higher (1.000 vs. 0.875) — page-level retrieval is a coarser unit than a precise chunk, and on a mostly plain-text document, chunking's precision has room to win
    - **Where it genuinely wins**: asked a question whose answer is a specific table cell, naive text RAG retrieves a passage that only *mentions* the table without containing its data, and the LLM correctly declines to guess; Vision RAG retrieves the table's page image and reads the correct value straight off it
    - **The smallest usable model in the ColPali family still needs a real fix to load**: its packaged checkpoint config points at a training-time-only local path; the practical fix is a documented, reusable one — not a workaround to avoid mentioning

## How a page becomes patches

Before any embedding happens, a page image goes through a fixed preprocessing pipeline, and each stage's output feeds the next. Verified directly on a real page from this recipe's sample PDF, rendered at 1275×1650px:

| Stage | What happens | The arithmetic | Result |
|---|---|---|---|
| 1. Tiling | The page is cut into a grid of tiles, each resized to the model's fixed 512×512px input size. This page splits into a 4×4 grid (16 local tiles), plus 1 extra tile holding a downscaled thumbnail of the *whole* page — so the model retains some sense of global layout even though each of the other 16 tiles only sees a local crop. | 16 local tiles + 1 thumbnail tile | **17 tiles total** |
| 2. Patchifying one tile | Each 512×512 tile is cut into non-overlapping 16×16px patches (a patch-embedding layer is a convolution with stride equal to its kernel size, so every pixel belongs to exactly one patch — never overlapping). How many patches fit along one edge? | 512 ÷ 16 = 32 patches/edge | A **32×32 grid = 1,024 raw patches**, in *one* tile |
| 3. Merging (pixel shuffle) | Every non-overlapping 4×4 block of adjacent raw patches is squashed into a single token — the "4" here is the model's `scale_factor` setting, a real compute/quality trade-off (4× fewer tokens to run late interaction over, at the cost of coarser localization within a tile). How many 4-patch blocks fit along one edge of the 32×32 grid? | 32 ÷ 4 = 8 blocks/edge | An **8×8 grid = 64 merged tokens**, per tile (down from 1,024) |
| 4. Totaling across the page | Every one of the 17 tiles goes through steps 2-3 independently, each producing its own 64 tokens. | 17 tiles × 64 tokens/tile | **1,088 image tokens for the whole page** |

The 1,088 figure is confirmed directly by counting the actual image-placeholder tokens the processor emits for this page — it isn't a fixed constant of the model, it falls out of the page's pixel dimensions relative to the fixed 512×512 tile size, so a differently-shaped or differently-rendered page image would tile into a different number of tiles and produce a different count.

What one merged token actually "sees": each of the 64 tokens per tile corresponds to a 4×4 block of the tile's 32×32 raw patches, i.e. roughly a 64×64-pixel region of that 512×512 tile — about an eighth of the tile's width and height. Mapped back to the original page crop that tile came from, that's enough area to cover a short table cell, a few words, or a small piece of a diagram, but not a whole paragraph or a whole table — which is exactly the granularity late interaction needs to let one strong local match stand out.

## Scoring: late interaction (MaxSim), not a single vector

Once a page is a grid of tokens (1,088 of them here, each a 128-dim vector after the model's projection) and a query is a short sequence of token vectors (typically 15-25 for a plain-language question, also 128-dim), scoring sums the best match each query token finds anywhere on the page:

```
score(query, page) = Σ (for each query token) max (over all page patches) dot_product
```

Concretely, with a 3-token query and a 4-token page (real runs use ~20 and ~1,088, but the mechanism is identical at any size): compute every query-token-vs-page-token dot product, giving a 3×4 grid of similarity scores. For each of the 3 query tokens (each row of that grid), keep only the single highest value — its best match anywhere on the page. Sum those 3 per-token maxima to get the page's final score. A different page gets scored the same way, independently, and the page with the higher summed score ranks first.

This is the same mechanism ColBERT uses for text retrieval, with image patches standing in for document tokens. The practical consequence: a page can win because ONE region of it matches strongly — one table cell, one line of a chart's legend — even if the rest of the page has nothing to do with the query. A single pooled vector per page couldn't do this; averaging a strong local match together with everything else on the page would dilute it away.

The trade-off is index size: storing ~1,088 vectors per page is far heavier than one vector per text chunk. For a handful of pages this doesn't matter; at real document-collection scale it's the main practical cost of this approach, and production setups typically use a vector store with native multi-vector support (Qdrant, Vespa) plus compression rather than storing raw patch vectors.

## The model, and a checkpoint requirement worth knowing about

`vidore/ColSmolVLM-256M-base` — the lightest model in the ColPali/ColVision family with a usable [ViDoRe benchmark](https://huggingface.co/blog/manu/colpali) score (~80 NDCG@5, vs. ~89 for the 2-3B leaders like ColQwen2.5). It's the one ColVision checkpoint that's plausibly runnable on an 8GB machine without a dedicated GPU — the larger models in this family realistically need one.

This specific checkpoint's `adapter_config.json` sets `base_model_name_or_path` to `./models/ColSmolVLM-256M-Base`, a local path from its own training run that doesn't exist anywhere else. `transformers` auto-detects the adapter config on load and tries to resolve that path as a Hugging Face repo id, which fails. The checkpoint's full merged weights are already present in its `model.safetensors`, so the practical fix is to download the snapshot, patch a local copy of `adapter_config.json` to point at the real base model (`HuggingFaceTB/SmolVLM-256M-Base`, listed on the model card), and load from that local copy. The code below does this automatically.

## Rendering pages, and loading the model

Rendering is the entire "extraction" step this approach needs — no OCR, no text parsing:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/vision-rag/vision_rag_docs.py:render_pages"
```

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/vision-rag/vision_rag_docs.py:vision_model"
```

## Indexing and searching page images

Every page is embedded as a full patch grid, and search scores every page against the query with MaxSim:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/vision-rag/vision_rag_docs.py:vision_index_search"
```

Compared directly against the naive text RAG baseline used across this site — extract text, chunk it, embed with a dedicated text embedder:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/vision-rag/vision_rag_docs.py:naive_text_rag"
```

!!! warning "Measured: naive text RAG wins on plain factual Recall@3"
    On the same 8 factual questions used across this repo, against the same "Attention Is All You Need" PDF:

    ```text
                                     Recall@3
    Naive text RAG (chunk-level)    1.000
    Vision RAG (page-level)         0.875
    ```

    The one question Vision RAG misses: *"How many attention heads did they use?"* retrieves the paper's attention-visualization figure pages — which literally show per-head attention diagrams — ahead of the page with the actual answer (`h = 8`, in a results table). The query's wording matches those figures visually and semantically more strongly than it matches a small table cell. Naming the table directly (*"...per Table 3?"*) retrieves the right page. This is a real, structural property of page-level retrieval on a mostly plain-text document: a page is a coarser unit than a precise 500-character chunk, so a chunker's precision has room to win outright when the document has little layout complexity to begin with.

## Where Vision RAG actually wins: table-layout integrity

Asked *"What BLEU score did the base model get on the newstest2013 development set, per Table 3?"* (the correct value is a specific table cell: 25.8):

```text
=== Naive text RAG ===
Top retrieved chunk: "...establishing a new state-of-the-art BLEU score
of 28.4. The configuration of this model is listed in the bottom line
of Table 3. Training took 3.5 days on 8 P100 GPUs..."

Answer: I cannot determine what BLEU score the base model achieved on
the newstest2013 development set. [...] The context only explicitly
states that their best model achieved a BLEU score of 28.4, but does
not provide the specific score for the base model.

=== Vision RAG ===
Top retrieved page: the full Table 3 page image

Answer: According to Table 3, the base model achieved a BLEU score of
25.8 on the newstest2013 development set.
```

Naive text RAG retrieves a passage that *mentions* Table 3 and *a* BLEU score in the same breath, without containing Table 3's actual data — 28.4 is a different model's headline test-set score from a separate table entirely. The LLM, working only from that chunk, correctly declines to guess rather than repeating the nearby number as if it answered the question. Vision RAG retrieves the actual table's page image and reads the real cell value directly off it. This is the failure mode Vision RAG exists to fix: text chunking can separate a table's meaning from its data long before retrieval even runs, and no amount of clever prompting recovers what the chunk boundary already destroyed.

## When to use which approach

**Use naive text RAG for:** plain-text-heavy documents — narrative prose, prose-style reports, most academic papers' body text. Chunk-level retrieval is a finer-grained, cheaper unit, and this page's own measurement shows it can out-recall page-level retrieval when there's little layout complexity to gain from. **Use Vision RAG for:** documents where a value's meaning depends on its position — financial tables, forms, multi-column layouts, charts, infographics. The measured example above is exactly this case: a number is meaningless without knowing which row and column it came from, and only reading the actual layout gets that right. **Consider both, routed by document type:** a production system indexing a mixed corpus (plain reports alongside dense tables) doesn't have to pick one globally — this is the same modality-routing idea covered on the [Modular RAG](modular-rag.md) page, applied to text-vs-layout instead of retrieval strategy.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: Vision RAG">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team builds a Vision RAG system and measures its Recall@3 against their existing naive chunk-based text RAG system, on a corpus of plain narrative-text documents with almost no tables or complex layout. They're surprised to find naive text RAG actually scores HIGHER Recall@3 than Vision RAG.",
      "question": "What's the most likely explanation, based on this page's measured findings?",
      "options": [
        "Because MaxSim scoring depends on finding one especially strong localized patch match, it produces unreliable similarity scores on pages that are mostly uniform prose with no distinct visual region to anchor to.",
        "Page-level retrieval is a coarser unit than a precise text chunk \u2014 on a document with little layout complexity, there's nothing for Vision RAG's core advantage (preserving table/layout structure) to win on, while naive chunking's finer granularity can out-recall it on plain factual questions.",
        "The ColSmolVLM checkpoint's documented loading fix (patching adapter_config.json to point at the real base model) must not have applied correctly, silently degrading every page embedding in this run.",
        "This is a top-k cutoff artifact -- measuring Recall@5 or Recall@10 instead of Recall@3 would show Vision RAG clearly ahead of naive text RAG on this same corpus."
      ],
      "correct": 1,
      "explanations": [
        "MaxSim still computes a valid, well-defined score on any page, plain-text or not -- every query token finds its single best-matching patch regardless of content. The issue this page measures isn't scoring reliability breaking down on prose, it's that a whole page is a coarser retrieval unit than a precise text chunk, so chunk-level precision can win when there's little layout complexity to exploit.",
        "Correct. This is exactly the measured finding on this page: Vision RAG's real advantage is preserving layout/table structure, which a plain-text document has little of. A whole page is a coarser retrieval unit than a 500-character chunk, so on plain text, chunk-level precision has room to win outright.",
        "That fix only determines whether the model loads at all -- once loaded, it produces embeddings normally. This page's measured Recall@3 gap is attributed directly to retrieval-unit granularity (a page vs. a chunk), not to a checkpoint-loading problem, and nothing in the setup here suggests the fix failed silently.",
        "The page's explanation isn't about metric choice -- a page is inherently a coarser retrieval unit than a 500-character chunk regardless of how wide k is set. Widening k doesn't change why plain, low-layout-complexity text favors fine-grained chunking; it would need to be re-measured to know, and nothing here suggests it reverses the result."
      ]
    },
    {
      "scenario": "Asked a question whose correct answer is a specific number inside a table, a naive text-RAG system retrieves a text passage that mentions the table's name and a DIFFERENT number from a nearby part of the document, in the same sentence. The LLM, given only that passage, correctly says it cannot determine the answer from the given context, rather than guessing the nearby number.",
      "question": "What does this scenario demonstrate about where Vision RAG provides real value, based on this page's findings?",
      "options": [
        "The retrieval step should be re-ranked with a cross-encoder that reads the full chunk text, since a re-ranker would have surfaced the sentence actually containing Table 3's data before generation ran.",
        "Text chunking can separate a table's meaning from its actual data before retrieval even runs \u2014 the chunk that gets retrieved can talk ABOUT a table without containing its values, a failure no amount of clever prompting recovers from, which is exactly the failure mode Vision RAG's page-image retrieval avoids.",
        "This demonstrates that the retrieved chunk's text directly contradicted the correct table value, which is why the LLM refused to answer -- it had encountered conflicting information rather than merely incomplete information.",
        "The LLM should have combined the 28.4 BLEU figure mentioned in the retrieved chunk with general knowledge of typical base-vs-large model score gaps to estimate a plausible base-model value."
      ],
      "correct": 1,
      "explanations": [
        "A re-ranker can only reorder candidates that were already captured by chunking -- it can't inject a value that no chunk ever contained. Here, the chunk that got retrieved genuinely doesn't contain the base model's BLEU score at all (28.4 belongs to a different result entirely), so no re-ranking of the same chunk set recovers it; only retrieving the actual table's page image does.",
        "Correct. This matches this page's measured table-BLEU example directly: the retrieved chunk referenced the table without containing its data, and Vision RAG's page-image retrieval got the actual page with the table intact, reading the correct value straight off it.",
        "There's no contradiction in the retrieved chunk -- it simply never contains the base model's BLEU score at all; 28.4 is a different model's result mentioned nearby. The LLM's refusal is about missing information, not conflicting information, which is exactly the distinction this scenario is testing.",
        "Estimating a number from a nearby figure plus general priors is still fabrication -- it isn't grounded in anything the document actually states about the base model's newstest2013 score. That's the exact overconfident behavior the page's correct-refusal example is contrasted against."
      ]
    },
    {
      "scenario": "A team implementing Vision RAG considers using a single pooled embedding vector per page (mean-pooling all patch embeddings together) instead of keeping every patch's embedding separately for late-interaction scoring, to save on index storage.",
      "question": "Based on how MaxSim/late-interaction scoring works, what would this team most likely lose by switching to single-vector pooling?",
      "options": [
        "Nothing meaningful -- since each patch vector already encodes cross-patch context from the vision transformer's self-attention layers, mean-pooling just aggregates that already-contextualized information into one summary without losing anything late interaction relies on.",
        "A page's ability to win retrieval on the strength of ONE strong local match (e.g. one table cell or chart legend) \u2014 pooling averages that strong local signal together with the rest of the page's unrelated content, diluting exactly the kind of precise match late interaction is designed to preserve.",
        "Retrieval speed at query time, since scoring a query against a page's single pooled vector requires more floating-point operations than scoring against the full multi-vector patch grid.",
        "Index storage size, since storing one embedding per patch is actually smaller in aggregate than storing a single higher-dimensional pooled vector once compression is applied."
      ],
      "correct": 1,
      "explanations": [
        "Even though self-attention lets each patch embedding incorporate context from other patches beforehand, MaxSim needs the INDIVIDUAL patch vectors to survive to query time so each query token can find its own best match. Pooling collapses them into one vector before that comparison ever happens, destroying the localized signal regardless of how contextualized each patch was beforehand.",
        "Correct. This is the direct consequence of how MaxSim works: it takes each query token's SINGLE best match across all page patches. A pooled single vector can't represent 'this one small region matches strongly' \u2014 it can only represent an average over the whole page, which is exactly why single-vector bi-encoders score measurably lower than late-interaction models on layout-heavy retrieval benchmarks.",
        "This has it backwards -- a single dot product per page is cheaper than summing many query-token-vs-patch dot products across ~1,088 vectors. Pooling is actually faster at query time, not slower; the real costs this page discusses are index storage size and loss of localized-match precision, not query-time compute.",
        "This reverses the page's own stated trade-off -- storing ~1,088 vectors per page is described directly as far heavier than one vector per chunk. Native multi-vector compression (Qdrant, Vespa) reduces that overhead but doesn't make it smaller than a single pooled vector; storage is precisely what pooling would save, at the cost of losing localized-match precision instead."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449) — the paper introducing the late-interaction, page-image retrieval approach this page builds on, including the ViDoRe benchmark
- [vidore/ColSmolVLM-256M-base](https://huggingface.co/vidore/ColSmolVLM-256M-base) — the specific checkpoint used here
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) — the original late-interaction/MaxSim mechanism this approach adapts from text to image patches
- [Multimodal RAG](multimodal-rag.md) — the text+extracted-figures approach this page contrasts with directly
- [Modular RAG](modular-rag.md) — the routing idea referenced above for mixing text-only and layout-aware retrieval by document type
