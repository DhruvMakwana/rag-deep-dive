# Embedding Model Selection

The embedding model determines *everything* about what "similar" means in vector search. Get chunking right, get reranking right, get the generation prompt right — none of it matters if the embedding model can't tell a relevant chunk from an irrelevant one in the first place. And unlike most RAG decisions, this one is annoyingly easy to get wrong silently: a bad model doesn't crash, it just quietly returns worse rankings, which is exactly the kind of failure that's hard to notice until someone asks a question and gets a bad answer.

!!! example "Hands-on"
    Every claim on this page is backed by a measured Recall@k number, not just a description: [**Embedding Model Selection →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/embedding-model-selection) in the code repo. No LLM API key needed — every model runs fully locally.

??? abstract "TL;DR — quick revision"
    - **Hard rule:** same embedding model for indexing and querying, always — mismatched models produce silently wrong rankings, not a crash (measured: 0.88 → 0.75 Recall@3 in a same-dimension mismatch)
    - **Matryoshka Representation Learning (MRL):** one model, truncate the output vector to any smaller size with `truncate_dim` — no retraining, and quality often survives surprisingly well (OpenAI `text-embedding-3`, Gemini Embedding, Qwen3-Embedding, `nomic-embed-text-v1.5` all support this)
    - **Quantization** (int8/binary) is a separate, stackable lever from MRL — same dimension, smaller number format; binary is 32x smaller than float32 but can cost real recall (measured: 0.88 → 0.38 on a small eval set)
    - **Instruction-tuned embeddings** expect a task-specific prefix at inference time — get the *specific model's* convention right (some prefix only the query, some prefix both sides differently) or silently lose quality
    - **Multi-vector / late-interaction (ColBERT)** keeps one embedding per *token*, not per chunk, scored via MaxSim — a fundamentally different retrieval paradigm from every single-vector model
    - **Three retrieval paradigms, not one:** BM25 (word counts, no model), dense embeddings (learned meaning, one vector per chunk), and **SPLADE** (learned meaning, shaped like BM25) each fail where the others succeed — worked comparison on a real 6-document example shows BM25 scoring the actual best answer a flat 0.000 while dense scores it highest
    - **BGE-M3** produces dense, sparse, *and* multi-vector representations from one model and one `encode()` call
    - **On MTEB, check the retrieval sub-score specifically**, not overall rank — a model can top the leaderboard on classification while being mediocre at retrieval
    - **Fine-tune only after cheaper fixes are exhausted** — hybrid search, reranking, and a better off-the-shelf model usually close more of the gap than fine-tuning does, for far less effort

## The one rule that isn't optional

Index with one embedding model, query with a different one, and nothing crashes — the vectors are the same shape, the math runs fine, the numbers just stop meaning anything. Two different models put "similar" text in different, incomparable regions of vector space, so a cosine similarity between them is comparing apples to a completely unrelated fruit.

Measuring this directly: indexing with `all-MiniLM-L6-v2` and querying with a *different* same-dimension model (`multi-qa-MiniLM-L6-cos-v1`) dropped Recall@3 from 0.88 to 0.75 on an 8-question eval set. That's real degradation, not the "retrieval completely breaks" some write-ups claim — the two models still share some structural similarity from similar training objectives. The more dangerous version of this bug is a genuine **dimension mismatch** (say, switching from a 384-dim to a 768-dim model): that one crashes the similarity computation outright, which is actually the easier failure to catch. Same-dimension-but-wrong-space is the one that ships to production silently.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/embedding-model-selection/embedding_models_docs.py:mismatched"
```

## Matryoshka Representation Learning (MRL)

Most embedding models bake in one fixed output size — 384, 768, 1024 dimensions, whatever it was trained to produce, and that's what you're stuck storing and searching against. **Matryoshka Representation Learning** trains a model differently: the *most important* information is concentrated in the first few dimensions, with each additional dimension adding progressively finer detail — like a nesting doll, hence the name. This means you can truncate the output vector to any smaller size after the fact, with no retraining, and it degrades gracefully instead of falling apart.

**Why this matters practically:** storage and search cost scale directly with dimension count, multiplied across every chunk in your index. A model that lets you drop from 1024 to 256 dimensions with a small, controlled accuracy cost is a direct infrastructure cost lever — cutting storage 4x and speeding up every similarity computation, without switching models or re-embedding from scratch at a different size (you can even keep the same stored full-size vectors and truncate at query time to experiment).

**Measured trade-off** (`nomic-embed-text-v1.5`, Matryoshka-trained down to 128 dims):

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/embedding-model-selection/embedding_models_docs.py:matryoshka"
```

**Who supports this today:** OpenAI's `text-embedding-3-small`/`-large`, Google's Gemini Embedding, Alibaba's Qwen3-Embedding family (configurable 32 to 1024+ dims), and open models like `nomic-embed-text-v1.5`. Check the specific model's documentation for which exact sizes it was trained/validated at — truncating to an arbitrary size the model wasn't designed for works less predictably than the officially supported cut points (for `nomic-embed-text-v1.5`: 768, 512, 256, 128).

## Quantization

A separate, *stackable* lever from Matryoshka — instead of storing fewer dimensions, store each dimension in fewer bits. A `float32` embedding uses 4 bytes per dimension; `int8` quantization uses 1 byte (4x smaller); binary quantization packs 8 dimensions into a single byte (32x smaller than float32). None of this touches dimensionality — a 768-dim float32 vector and a 768-dim binary vector cover the same semantic space, just represented with different precision.

**Measured trade-off** (same model and dimension, precision only):

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/embedding-model-selection/embedding_models_docs.py:quantization"
```

`float32 → int8` cost almost nothing here (0.88 → 0.75 Recall@3 for a 4x storage win); `binary` cost considerably more (down to 0.38) — the sharpest cut in the whole precision spectrum, since 1-bit-per-dimension throws away far more signal than 8-bit does. In production, binary quantization is typically paired with a **rescoring step**: retrieve a larger candidate set cheaply with binary vectors, then re-rank the shortlist with the original float32 vectors — getting most of the storage win without eating the full accuracy cost on the final ranking.

## Instruction-tuned embeddings

Many current embedding models expect a short task-specific string prepended to the input at inference time — not a stylistic nicety, but part of how the model was trained, meaning skipping it or using the wrong convention costs real retrieval quality. Critically, **the exact convention differs per model** and getting it wrong is easy to do silently:

- **Qwen3-Embedding** expects an instruction only on the **query** side (e.g. `"Instruct: Given a question, retrieve passages that answer it\nQuery: {question}"`) — passages are embedded plain. The model's own documentation notes skipping this costs roughly 1-5% retrieval quality.
- **`nomic-embed-text-v1.5`** expects a **different prefix on each side** — `"search_query: "` on questions, `"search_document: "` on passages — not the same string on both, and not query-only like Qwen3.

**Measured** (`nomic-embed-text-v1.5`, correct dual-prefix convention vs. none):

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/embedding-model-selection/embedding_models_docs.py:instruction"
```

!!! warning "An honest negative result"
    On this specific 8-question eval set, prefixed and unprefixed queries scored **identically** — 0.88 both ways. That's not evidence prefixes don't matter; it's evidence that an 8-question eval set doesn't have the statistical power to reliably surface every real effect, including ones the model's own documentation confirms exist at benchmark scale. Small eval sets are useful for catching large, obvious breaks (like the mismatched-model and quantization results above) — they're not reliable for detecting small effects, and reporting a null result honestly here matters more than forcing a tidy narrative.

## Multi-vector / late-interaction retrieval (ColBERT)

Every technique above still produces **one vector per chunk** — the entire chunk's meaning compressed into a single point. ColBERT-style **late-interaction** retrieval does something structurally different: it keeps **one embedding per token**, and scores a query against a document using **MaxSim** — for each query token, find its single best-matching document token, then sum those best-match scores across all query tokens. The interaction between query and document happens *late*, at scoring time, token-by-token, rather than being collapsed into one similarity number upfront.

**Why this can outperform single-vector retrieval:** a chunk's single vector is necessarily an average/compromise across everything the chunk discusses, which can dilute the specific signal a narrow query is looking for. Multi-vector retrieval keeps every token's signal intact, so a query matching one precise detail buried in an otherwise-unrelated chunk can still score well — the cost is more storage (many vectors instead of one per chunk) and more compute at query time (comparing token-by-token instead of one dot product).

### Worked example: how MaxSim actually scores a query against a document

Query: *"Who won the World Cup?"* Document chunk: *"The 2022 FIFA World Cup was won by Argentina in Qatar."*

A late-interaction model embeds **every token of both**, contextually — so each token's vector already reflects the tokens around it, not just the word in isolation. MaxSim then does this, one query token at a time: compare it against **every** document token vector, keep only the single **best** match, and move to the next query token. These are real cosine similarities, computed from `all-MiniLM-L6-v2`'s actual per-token embeddings (`model.encode(text, output_value="token_embeddings")`) — not invented for illustration:

| Query token | Best-matching document token | Cosine similarity |
|---|---|---|
| `who` | `by` | 0.427 |
| `won` | `won` | 0.822 |
| `the` | `the` | 0.569 |
| `world` | `world` | 0.769 |
| `cup` | `cup` | 0.806 |
| `?` | `.` | 0.405 |

**MaxSim score = sum of the best-match column = 3.797.** That single number is the query-document relevance score — but unlike a single cosine similarity, you can see exactly *which* tokens drove it: `won`, `world`, and `cup` matched strongly (0.77-0.82) since those words appear literally in both, while `who`, `the`, and `?` contributed far less.

**An honest caveat about this specific example:** `who` matched `by` here, not `argentina` — which would be the more semantically satisfying match, since "who" is really asking for the entity name. That's `all-MiniLM-L6-v2` showing its limits for this task: it's a general-purpose *sentence* embedding model, not a model specifically trained with the MaxSim objective the way real ColBERT/BGE-M3 multi-vector models are — its individual token embeddings weren't optimized for this kind of fine-grained token-to-token matching. The **mechanism** shown here (compare against every token, keep only the best match, sum the best matches) is exactly what a real late-interaction model does; a model actually trained for it would be expected to produce sharper, more semantically correct token alignments than this quick illustration does.

**Contrast with single-vector retrieval:** a normal embedding model would average the whole query into one vector and the whole document into one vector, then compute one cosine similarity between those two points — a single number with no way to see that `won`/`world`/`cup` carried the match while `who`/`the`/`?` barely contributed. That averaging is exactly what late-interaction avoids — and exactly why it costs more storage (12 document vectors instead of 1) and more compute (roughly 6×12 = 72 pairwise comparisons instead of 1) to get it.

**BGE-M3 packages this as one of three outputs from a single model call:**

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/embedding-model-selection/embedding_models_docs.py:bge_m3"
```

One `encode()` call, three genuinely different retrieval representations: a dense vector (semantic similarity, same as everything above), sparse lexical weights (covered next), and per-token ColBERT-style vectors — letting one model serve all three retrieval "lanes" a production hybrid system might otherwise need three separate models for.

## Learned sparse retrieval (SPLADE)

Every technique so far on this page — Matryoshka, quantization, instructions, multi-vector/ColBERT — has been about **dense embeddings**: a model that reads a whole passage and compresses it into one vector of continuous numbers, trained so **similar meaning** lands close together in that vector space, whether or not the actual words match. SPLADE takes a genuinely different approach, one best understood by contrast with the classic non-neural alternative it's trying to improve on: **BM25**.

**BM25 (classic sparse / lexical retrieval):** represents a document as **word counts** — no model, no training, just a statistical formula. It scores a query-document pair based on whether the query's actual words appear in the document, how often (term frequency), and how rare that word is across the whole corpus (inverse document frequency — "the" appearing means nothing, a rare technical term appearing means a lot). The representation has one slot per vocabulary word, and almost all slots are zero for any given document — hence **sparse**. Its blind spot: pure vocabulary matching, with no notion that "automobile" and "car" mean the same thing.

**SPLADE (learned sparse retrieval):** trained like a dense model, shaped like a BM25 vector. Instead of computing word-count statistics, a neural network **predicts which vocabulary words should be weighted highly** for a passage — including words that never literally appear in it, if the model learned they're related (a form of learned query/document expansion BM25 has no mechanism for). The output stays sparse — mostly zeros, searchable with the same fast inverted-index infrastructure as BM25 — but the weights come from learned meaning, not from counting.

### A real worked comparison: BM25 vs. dense vs. SPLADE

Query: **"car problems"**. A small 6-document corpus, including the one document that actually answers the query and five distractors:

```text
Doc A: "My automobile has engine trouble"        ← the actual answer, shares zero words with the query
Doc B: "Car insurance policy renewal"             ← shares the word "car", but isn't about car problems
Doc C: "Best restaurants in the city"             ← unrelated
Doc D: "How to bake sourdough bread"              ← unrelated
Doc E: "Weather forecast for next week"           ← unrelated
Doc F: "Car maintenance schedule and tips"        ← shares "car", tangentially related
```

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/embedding-model-selection/embedding_models_docs.py:bm25_vs_dense"
```

**BM25** (`rank_bm25`'s `BM25Okapi`, no model, pure statistics):

| Document | BM25 score |
|---|---|
| Doc A (automobile / engine trouble) | **0.000** |
| Doc B (car insurance) | 0.637 |
| Doc C, D, E (unrelated) | 0.000 |
| Doc F (car maintenance) | 0.579 |

Doc A — the one document that's actually about car problems — scores a flat **zero**. It shares no words with the query at all, and BM25 has no way to know "automobile" means "car." Doc B and Doc F outrank it purely because they contain the literal word "car."

**Dense embeddings** (`all-MiniLM-L6-v2`, cosine similarity — the same paradigm as everything earlier on this page):

| Document | Cosine similarity |
|---|---|
| Doc A (automobile / engine trouble) | **0.593** |
| Doc F (car maintenance) | 0.341 |
| Doc B (car insurance) | 0.340 |
| Doc C, D, E (unrelated) | ~0.00 to -0.03 |

Exactly the opposite pattern from BM25. Doc A now ranks clearly first, because the model learned that "automobile" and "engine trouble" are semantically about the same thing as "car problems" — no shared vocabulary required.

**SPLADE**, using a real `naver/splade-cocondenser-ensembledistil` model via `sentence-transformers`' `SparseEncoder`:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/embedding-model-selection/embedding_models_docs.py:splade"
```

| Document | SPLADE score |
|---|---|
| Doc A (automobile / engine trouble) | **25.825** |
| Doc B (car insurance) | 11.750 |
| Doc F (car maintenance) | 11.528 |
| Doc C, D, E (unrelated) | 0.000 |

Doc A wins clearly, same as dense — but the *mechanism* is directly inspectable in a way dense cosine similarity never is. Decoding Doc A's own sparse vector back to vocabulary terms shows exactly why: `engine` (2.81), `trouble` (2.25), `engines` (2.07), `problems` (2.00), and — the interesting one — **`car` (1.96)**, a real, non-zero learned weight for a word that never appears anywhere in "My automobile has engine trouble." That's SPLADE's learned query/document expansion working exactly as described: closing BM25's vocabulary gap while staying in the same sparse, inverted-index-friendly, inspectable shape as BM25 — you can point at the exact term and weight that drove the match, which cosine similarity on a dense vector never lets you do.

### When SPLADE actually earns its place

**Over plain BM25:** whenever queries and documents describe the same thing with genuinely different words — a support ticket saying "the app keeps freezing" needs to match a bug report saying "application becomes unresponsive," which share zero terms in common, exactly like Doc A above. BM25's term-overlap scoring can't bridge that gap at all; SPLADE's learned expansion can, while still being fast, inverted-index-searchable sparse retrieval rather than paying the full cost of a dense vector search.

**When to skip it, even over dense:** if you're already running dense retrieval, SPLADE's main advantage over dense is *interpretability and exact-term precision* — you can see exactly which vocabulary terms scored a match, and rare/exact tokens like product SKUs or error codes are handled naturally. If neither of those is a real pain point in your system, plain hybrid dense+BM25 covers most of the same ground with one less moving part to run and maintain.

### Hybrid search: combining BM25 and dense with RRF

BM25 and dense embeddings have close-to-opposite blind spots — BM25 misses paraphrases entirely, dense sometimes under-weights exact rare terms. **Hybrid search** runs both independently, then fuses the two ranked lists with **Reciprocal Rank Fusion (RRF)**: convert each list's raw scores to ranks, then sum `1/(k + rank + 1)` for each document across both lists — a document ranked highly by either signal gets a real boost, without needing BM25's statistics and cosine similarity to be on comparable numeric scales at all.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/embedding-model-selection/embedding_models_docs.py:hybrid_rrf"
```

Fusing the exact BM25 and dense scores measured above:

```text
1. Doc A  fused_score=0.03227
2. Doc B  fused_score=0.03227
3. Doc F  fused_score=0.03226
4. Doc D  fused_score=0.03101
5. Doc C  fused_score=0.03078
6. Doc E  fused_score=0.03054
```

Fusion correctly pulls Doc A up from dead-last-tied-at-zero (BM25's ranking) into a shared first place — a real, measured win over BM25 alone. But look closely at the actual numbers: **Doc A and Doc B are tied at 0.03227**, five decimal places deep, not just close. That's not a coincidence specific to this corpus — it's the same structural property of RRF documented on the [Multimodal RAG](multimodal-rag.md#separate-indexes-rrf-fusion) page: whichever document sits at rank 0 in *any* input list scores `1/(k+0+1)` from that list, identically, regardless of how much better or worse it actually is than the next-best candidate. Doc A is rank 0 in the dense list; Doc B is rank 0 in the BM25 list — different lists, same tied contribution. Fusion fixed the *ordering* problem (Doc A no longer loses to Doc B and Doc F outright) but didn't — and structurally can't — express that Doc A is a dramatically better match than Doc B, only that each is "the best" according to one signal.

**Use hybrid + RRF for:** the general default when you have both a lexical and a semantic retrieval signal available and don't want to hand-tune a weighted blend between two different score scales. **Know its limit:** RRF is a consensus mechanism across rankings, not a magnitude comparator — it can't let one signal's high-confidence pick outright outrank another signal's merely-present one, a limitation worth remembering anywhere fusion is used, not just here.

## MTEB and current benchmarks — read the sub-score, not the badge

The [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is the standard place to compare embedding models, but it scores across many unrelated tasks at once — retrieval, classification, clustering, bitext mining, and more — and reports one blended overall rank. A model's overall position can be dragged up by strong classification or clustering performance while its retrieval sub-score, the only one that actually matters for RAG, is mediocre. The reverse happens too: a model tuned hard for retrieval can rank lower overall while being the better choice for your actual task.

**Concretely:** Qwen3-Embedding-8B topped MTEB's multilingual leaderboard overall score in mid-2025 — a genuinely strong result, worth taking seriously. But "topped the overall leaderboard" and "best retrieval model for my RAG system" are different claims. Before picking a model off the leaderboard, click into its retrieval sub-score specifically, and ideally run it against a small eval set built from your own domain — the same kind of Recall@k measurement every demo on this page uses, just pointed at your data instead of a sample PDF.

## The current landscape, roughly

Exact pricing and rankings shift constantly — check each provider's own documentation before committing — but the rough shape of the market as of this writing:

| Tier | Examples | Notes |
|---|---|---|
| **Proprietary, hosted API** | OpenAI `text-embedding-3-small`/`-large`, Cohere `embed-v4`, Voyage AI's domain-specialized models, Google Gemini Embedding | Zero infra to run; per-token pricing; several support Matryoshka truncation natively |
| **Open-source, self-hostable, general** | Qwen3-Embedding (0.6B/4B/8B, Matryoshka + instructions), `nomic-embed-text-v1.5` (Matryoshka, small enough for modest hardware), BGE-M3 (dense+sparse+multi-vector) | No per-token cost, full data control, need your own GPU/CPU inference |
| **Domain-specialized** | Voyage's code/legal/finance models, various fine-tuned BGE/E5 variants | Meaningfully better than general models on jargon-heavy text, at the cost of narrower applicability |

**Practical starting point:** for a self-hosted system with no strong domain-specialization need, `nomic-embed-text-v1.5` or a Qwen3-Embedding size matched to your hardware are reasonable open defaults — both Matryoshka-capable, both instruction-aware. For a hosted, zero-ops option, the current-generation OpenAI/Cohere/Google models are all reasonable general-purpose choices; benchmark the retrieval sub-score on your own data before committing either way, per the MTEB caveat above.

## Fine-tuning: why, when, and how

**Why it's sometimes necessary:** every embedding model on this page was trained on broad, general text. A general model has learned a broad notion of "similar," but hasn't specifically learned that, say, "consideration" in a contract means something narrow and legal — not its everyday meaning — or that two support tickets describing the same underlying bug in completely different words are actually the same issue. When retrieval keeps missing in a way that traces back to domain vocabulary the model was never trained to distinguish, that's the specific signal fine-tuning addresses — not vague "the answers aren't great" dissatisfaction, but a demonstrated, domain-specific gap.

**When to reach for it — and the order that actually matters:** fine-tuning is one of the most expensive levers on this entire page. It needs labeled query-passage relevance data (real user clicks/feedback, domain experts hand-labeling pairs, or an LLM prompted to generate realistic queries for known passages), a training pipeline, and ongoing upkeep as the corpus evolves. Reach for it only after cheaper fixes are exhausted, in this order:

1. **Try a better off-the-shelf model first** — check its MTEB retrieval sub-score, ideally against a domain-relevant benchmark subset if one exists.
2. **Add hybrid search** (dense + BM25 or SPLADE) — often closes a large chunk of the "domain terminology" gap for free, since the lexical side catches exact-term matches the dense model never learned to generalize to.
3. **Add reranking** — a cross-encoder reranking the retrieved candidates corrects a lot of embedding imprecision without touching the embedding model at all.
4. **Only fine-tune once 1-3 are in place and your own eval set still shows a real gap** — and only if you can realistically get labeled data. Fine-tuning without labeled data to train *or evaluate* against is guessing twice.

**How, concretely, using `sentence-transformers`:**

1. Collect labeled pairs — at minimum (query, relevant_passage) positives; negatives can often be left implicit (see step 3).
2. Pick an open model you can actually train — a closed API model (OpenAI, Cohere) can't be fine-tuned this way; start from something like `nomic-embed-text-v1.5` or a small Qwen3-Embedding size.
3. Pick a loss. **`MultipleNegativesRankingLoss`** is the common practical default: it needs only positive pairs — every other example in the same training batch is treated as an implicit negative, which is far more data-efficient than hand-labeling explicit negatives. Contrastive and triplet losses (pulling known-relevant pairs together, pushing known-irrelevant pairs apart — the same pairwise-loss idea used to train [cross-encoder rerankers](advanced-rag-post-retrieval.md), a different kind of model that re-scores a shortlist of candidates by reading the query and each document together for precision, applied here to the embedding model instead) are the alternative when you do have labeled negatives.
4. Fine-tune briefly — starting from a pretrained checkpoint, a small number of epochs on a modest labeled set typically goes further than expected; overtraining on a small dataset risks overfitting to it.
5. Evaluate the fine-tuned model against the *same* Recall@k / retrieval-sub-score methodology used to justify picking up fine-tuning in the first place, on a held-out slice of your labeled data — never ship a fine-tuned model on faith that training "probably helped."

```python
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# Real labeled pairs go here — this is illustrative, not run against real data.
train_data = Dataset.from_dict({
    "anchor": ["What's the maximum liability cap in a standard indemnification clause?"],
    "positive": ["Indemnification clauses typically cap liability at the contract's total value..."],
})

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_data,
    loss=MultipleNegativesRankingLoss(model),
    args=SentenceTransformerTrainingArguments(output_dir="./fine-tuned-model", num_train_epochs=1),
)
trainer.train()
```

!!! note "Verified API, not a verified result"
    The class names and import paths above are confirmed real and current against the installed `sentence-transformers` version — unlike every other code block on this page, this one hasn't been run against real labeled data or measured for a before/after Recall@k, since building a legitimate labeled training set was out of scope for this page. Treat it as a correct starting point for the training loop shape, not as evidence fine-tuning helped anything.

## Ensembling multiple embedding models

**Why:** a single embedding model is a single trade-off — general-purpose models cover broad phrasing well but miss domain nuance; domain-fine-tuned models nail the jargon but can be worse at everyday phrasing outside their training focus. Ensembling runs **both** and combines the results, rather than forcing one model to be good at everything.

**How:** embed and retrieve with each model independently, producing two separate ranked lists over the same chunks, then fuse those lists with **RRF (Reciprocal Rank Fusion)** — each chunk's fused score is the sum of `1/(k + rank)` across every list it appears in, so a chunk ranked highly by multiple sources wins over one that only one source liked. This is the identical fusion pattern used for [multi-query retrieval](advanced-rag-pre-retrieval.md#multi-query-retrieval-rag-fusion) (merging results from several rephrasings of the same query) and [hybrid sparse+dense search](embedding-models.md#hybrid-search-combining-bm25-and-dense-with-rrf) earlier on this page (merging BM25 and embedding results), just applied along a different axis here — multiple embedding models, instead of multiple query phrasings or multiple retrieval methods.

**When it's worth the extra embedding cost:** queries genuinely span both general and domain-specific phrasing in the same system (a support bot fielding both "how do I reset my password" and jargon-heavy enterprise-integration questions, say), and the cost of embedding every chunk twice is acceptable. **When to skip it:** a single well-chosen model already covers the query distribution well — ensembling adds real cost (double the embedding calls, double the index storage) for a benefit that only shows up when one model's blind spot is actually being hit in practice. Measure the gap on your own eval set before adding this complexity, same principle as fine-tuning above.

## Scenario Check

Six scenarios testing whether you can pick the right lever for a given symptom — the way this actually comes up in system design interviews and real incident postmortems.

<div class="quiz-widget" data-title="Scenario Check: Embedding Model Selection">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team migrates their embedding provider and re-indexes their entire knowledge base with the new model. Two weeks later, a hotfix accidentally reverts the query-embedding service to the old model, while the vector index still holds embeddings from the new model. No errors appear in any logs.",
      "question": "What's the most likely symptom users actually experience?",
      "options": [
        "The system throws a dimension-mismatch error and stops serving queries until the mismatch is fixed.",
        "Retrieval quality degrades — not necessarily to zero, but noticeably — because the query vectors and indexed vectors now live in different, only partially comparable spaces.",
        "Nothing changes, since both models were trained on similar internet-scale text corpora.",
        "The vector database automatically detects the version mismatch and blocks the query."
      ],
      "correct": 1,
      "explanations": [
        "This is what happens with a genuine dimension mismatch (say, 384 vs. 768 dims) — a real and even easier-to-catch failure mode. But many embedding model upgrades keep the same output dimension, so a dimension mismatch isn't guaranteed here — and the measured result in this exact same-dimension scenario showed degraded, not crashed, recall.",
        "Correct. This is exactly the measured finding: a same-dimension model mismatch dropped Recall@3 from 0.88 to 0.75 — real, noticeable degradation, but not a total collapse, and with no crash or error to flag it. That silence is what makes this bug dangerous in production.",
        "Different models produce different, generally incomparable vector spaces even when trained on similar data — similarity in training data doesn't imply similarity in the resulting embedding geometry.",
        "No mainstream vector database inspects embeddings for \"which model produced this\" — it has no way to know; this is a real gap teams have to guard against operationally (e.g., embedding model version tags in metadata, deploy-time checks), not something infrastructure catches automatically."
      ]
    },
    {
      "scenario": "A production RAG system stores 50 million chunk embeddings at 1024 dimensions in float32. Storage cost has become a real budget line item, and query latency is also creeping up as the index grows.",
      "question": "Which single change addresses both problems at once, assuming the embedding model supports it?",
      "options": [
        "Switch to a different, smaller embedding model and re-embed everything from scratch.",
        "Use Matryoshka truncation to drop to a smaller dimension (e.g. 256) — smaller vectors mean less storage AND faster similarity computation, without retraining or re-embedding from different model weights.",
        "Fine-tune the existing model on the company's own data to make it more accurate.",
        "Switch from cosine similarity to Euclidean distance for faster computation."
      ],
      "correct": 1,
      "explanations": [
        "This would require a full re-embedding pass and carries real migration risk and cost — a much heavier lever than truncation, and not needed if the current model already supports Matryoshka.",
        "Correct. If the model was Matryoshka-trained, truncating from 1024 to 256 dimensions cuts storage 4x and speeds up every similarity computation directly, without touching the model itself or re-embedding from a different architecture.",
        "Fine-tuning targets accuracy, not storage or latency — it doesn't address either problem in the scenario, and doesn't reduce vector size at all.",
        "The similarity metric choice doesn't change vector size or storage footprint — and using a metric the model wasn't optimized for can quietly hurt retrieval quality instead of helping."
      ]
    },
    {
      "scenario": "A team wants to cut their vector storage costs by 32x and considers binary quantization. Someone on the team argues this is basically the same lever as Matryoshka truncation, just \"more aggressive.\"",
      "question": "Is that framing correct?",
      "options": [
        "Yes — both techniques reduce the amount of data stored per chunk, so they're interchangeable levers at different strengths.",
        "No — Matryoshka reduces the number of dimensions; quantization reduces the number of bits used per dimension. They're separate, stackable levers, not the same technique at different intensities.",
        "No — quantization is only usable with sparse embedding models, not dense ones.",
        "Yes, but only for models that were NOT Matryoshka-trained."
      ],
      "correct": 1,
      "explanations": [
        "They both save storage, but through mechanically different means — conflating them risks missing that you can (and often should) apply both together for compounding savings, not pick one instead of the other.",
        "Correct. Matryoshka changes how many dimensions exist; quantization changes how many bits represent each dimension that remains. A 768-dim binary vector and a 256-dim float32 vector both save storage, via completely different mechanisms — and you can combine them (truncate AND quantize) for even larger combined savings.",
        "Quantization (as demonstrated via sentence-transformers' `precision` parameter) works directly on dense embeddings — int8 and binary precision are standard options for ordinary dense vectors, not sparse-only.",
        "Quantization support has nothing to do with whether a model is Matryoshka-trained — they're independent, unrelated model capabilities that happen to compose well together."
      ]
    },
    {
      "scenario": "A team adopts Qwen3-Embedding and, to save one code path, decides to reuse the exact same instruction-prefix logic they already had working for nomic-embed-text-v1.5 in another project.",
      "question": "What's wrong with this plan?",
      "options": [
        "Nothing — since both models are instruction-tuned, they should read instruction text the same general way, so reusing one convention on the other risks at most a minor wording mismatch.",
        "The two models use genuinely different conventions: Qwen3-Embedding expects an instruction only on the query side, while nomic-embed-text-v1.5 expects a different short prefix on BOTH the query and the document side — reusing one model's convention for the other silently costs retrieval quality on whichever model gets the wrong treatment.",
        "Qwen3-Embedding's instruction just needs to appear somewhere in the input, not specifically on the query side, so prefixing the passage instead would work just as well.",
        "nomic-embed-text-v1.5 treats \"search_query: \" and \"search_document: \" as interchangeable labels for the same generic instruction, so using either one on either side works the same."
      ],
      "correct": 1,
      "explanations": [
        "Being \"instruction-tuned\" doesn't imply a shared format — the two models differ in where the instruction goes, not just its wording. Reusing nomic's dual-prefix convention on Qwen3 adds an instruction to passages that were trained to stay plain, and reusing Qwen3's query-only convention on nomic means passages never get the \"search_document: \" prefix nomic expects — that's a structural mismatch, not a minor phrasing variance.",
        "Correct. Qwen3-Embedding's documented convention is query-side-only instruction text; nomic-embed-text-v1.5's is a distinct prefix on each side (\"search_query: \" / \"search_document: \"). Applying the wrong one doesn't error — it just quietly underperforms, the same silent-failure pattern as the mismatched-model case.",
        "Qwen3-Embedding's documented convention places the instruction specifically on the query side, with passages embedded plain — moving it to the passage side isn't an equivalent placement, it's outside the convention the model was actually trained and validated on.",
        "nomic's convention assigns each prefix to a specific side on purpose — query text gets \"search_query: \", passage text gets \"search_document: \". Swapping them, or using only one for both sides, doesn't reproduce the representation the model was trained to expect on each side."
      ]
    },
    {
      "scenario": "A support team needs retrieval that handles both \"what's your refund policy\" (semantic match needed) and \"order #ORD-88213-A\" (exact lexical match needed, and this specific order ID never appeared anywhere in training data for any embedding model). Someone proposes using only a strong dense embedding model for everything.",
      "question": "What's the likely failure mode, and what addresses it?",
      "options": [
        "No failure mode — modern dense embedding models handle arbitrary alphanumeric strings just as well as natural language semantic queries.",
        "Dense embeddings are typically weak at exact, rare, alphanumeric matches like order IDs, since these carry little reusable semantic signal the model can generalize from — a learned sparse method (SPLADE) or classic lexical search (BM25) alongside dense retrieval covers this gap.",
        "The fix is to fine-tune the dense model on every possible order ID before deployment.",
        "This is only a problem for Matryoshka-truncated embeddings — full-dimension embeddings handle exact IDs fine."
      ],
      "correct": 1,
      "explanations": [
        "Dense embeddings are trained to capture semantic/conceptual similarity — an arbitrary, rare identifier like an order number has essentially no semantic content for the model to generalize from, which is a real, well-documented weak spot, not a solved problem.",
        "Correct. This is precisely the case for hybrid retrieval — combining dense (semantic) with sparse lexical methods (BM25's statistics, or SPLADE's learned term weighting) so exact/rare-term queries are covered by the lexical side while semantic queries are covered by the dense side.",
        "Order IDs are generated continuously and can't be exhaustively enumerated for training — this doesn't scale and misses the structural point that dense embeddings aren't the right tool for this kind of exact match at all.",
        "Truncation reduces dimensionality but doesn't change what the model was trained to represent in the first place — a full-dimension dense embedding has the same fundamental weakness on rare exact-match strings as a truncated one."
      ]
    },
    {
      "scenario": "A demo measures Recall@3 with and without a documented instruction prefix on a tiny 8-question eval set and finds no difference — 0.88 both ways — even though the model's own official documentation reports the prefix improves retrieval quality at benchmark scale.",
      "question": "What's the most defensible conclusion?",
      "options": [
        "The documentation must be wrong, since the measured result contradicts it.",
        "The prefix convention is genuinely real, but it's likely specific to the exact benchmark corpus nomic validated it on — an eval set built from different documents and questions, like this one, wouldn't be expected to show the same effect.",
        "An 8-question eval set likely lacks the statistical power to reliably detect a real but modest effect — a null result on a tiny eval set doesn't disprove an effect measured at much larger benchmark scale.",
        "The eval set is small, but eight questions is still enough data to trust; the more likely explanation is the prefix strings themselves were implemented slightly wrong, silently no-oping the convention."
      ],
      "correct": 2,
      "explanations": [
        "A single small demo contradicting a benchmark-scale finding is far more likely to be a sample-size limitation than proof the vendor's larger-scale evaluation is simply incorrect.",
        "The prefix convention comes from how the model was trained on the input format itself, not from properties of one specific benchmark corpus — it isn't restricted to the corpus it was validated on. The real reason this eval shows no difference is the same one that applies everywhere else on this page: eight questions is too small a sample to reliably detect a modest, real effect, not a domain mismatch between corpora.",
        "Correct. This is exactly the honest interpretation: small eval sets are good at catching large, obvious effects (a mismatched model, aggressive quantization) but unreliable for detecting smaller effects — reporting the null result plainly, rather than forcing a narrative, is the right call here.",
        "Eight questions is genuinely too small to reliably detect a modest effect — the kind of statistical-power limitation the page calls out explicitly — so there's no need to reach for an implementation bug to explain a null result here. A coding error is possible in principle, but it isn't the more likely or more defensible explanation given how little power a sample this size has in the first place."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — the standard embedding benchmark; check the retrieval sub-score specifically
- [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://qwenlm.github.io/blog/qwen3-embedding/) — instruction format, Matryoshka support, model sizes
- [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) — Matryoshka truncation levels, `search_query`/`search_document` prefix convention
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — dense + sparse + multi-vector (ColBERT-style) from one model
- [Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever](https://arxiv.org/pdf/2408.16672) — late-interaction / multi-vector retrieval background
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://dl.acm.org/doi/10.1145/3404835.3463098) — the original learned sparse retrieval paper
- [Sentence Transformers: Quantization](https://sbert.net/examples/sentence_transformer/applications/embedding-quantization/README.html) — `precision` parameter, int8/binary embeddings, rescoring pattern
