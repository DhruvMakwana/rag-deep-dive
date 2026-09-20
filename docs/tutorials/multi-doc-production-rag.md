# Build a Multi-Document Production RAG Pipeline

Every other recipe on this site tests one technique against one small, single-document corpus, on purpose — isolating a variable is the right way to measure what a specific technique actually does. This tutorial does the opposite: it combines chunking-strategy choice, retrieval-method choice, a real vector database, and RAGAS evaluation into one pipeline, run against three real papers at once (not one), so the comparisons and failure modes below are the ones that only show up once several pieces are stacked together.

The code lives in [`rag-cookbook/multi-doc-production-rag/`](https://github.com/DhruvMakwana/rag-cookbook/tree/main/multi-doc-production-rag); every block below is pulled live from those exact files at build time.

!!! note "Read these first"
    This tutorial assumes you've read [Chunking Strategies](../chunking.md), [Retrieval Methods](../retrieval-methods.md), [Vector Databases](../vector-databases.md), and [RAGAS Deep Dive](../ragas.md) — it reuses their techniques directly and doesn't re-explain how each one works internally.

## The corpus

Three real, closely-related papers — "Attention Is All You Need," "BERT," and "GPT-3" — roughly 230 pages combined, producing 700 to 1,000+ chunks depending on strategy. Being closely related (all about language model pretraining and architecture) makes cross-document confusion a real, testable risk: a question like "how many layers does the model have" is genuinely ambiguous across all three, each with a real, different, correct answer.

A 9-question eval set spans all three documents, with a real keyword from that paper's actual text backing each question — the same keyword-derived ground truth pattern used on the [Evaluation Metrics](../evaluation-metrics.md) page, extended here to also check that a retrieved chunk came from the *right document*, not just that it contains the right keyword.

## 1. Install

```bash
git clone https://github.com/DhruvMakwana/rag-cookbook.git
cd rag-cookbook/multi-doc-production-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/multi-doc-production-rag/requirements.txt"
```

## 2. Configure

```bash
cp .env.example .env
```

The chunking and retrieval comparisons need no API key at all. The final RAGAS evaluation step needs an LLM — defaults to a local Ollama model (`ollama serve`, `ollama pull qwen3:4b`), or set `LLM_PROVIDER=anthropic`/`openai` in `.env`.

## 3. Chunking strategies, compared on the same multi-document corpus

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/multi-doc-production-rag/multi_doc_rag_docs.py:chunkers"
```

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/multi-doc-production-rag/multi_doc_rag_docs.py:chunking_eval"
```

Measured Recall@3 / Hit Rate@3 across the 9-question eval set:

```text
fixed      chunks=1011  recall@3=0.139  hit_rate@3=0.222
recursive  chunks=1017  recall@3=0.181  hit_rate@3=0.444
sentence   chunks= 743  recall@3=0.333  hit_rate@3=0.667
```

Sentence-window chunking — grouping complete sentences instead of splitting at a fixed character count — clearly outperforms both fixed-size and recursive chunking here, roughly tripling hit rate over fixed-size despite producing *fewer*, not more, chunks. On a corpus with dense technical claims packed into single sentences (parameter counts, layer counts, specific scores), keeping each sentence whole mattered more than chunk-size tuning.

## 4. Dense-only vs. hybrid retrieval, on the winning chunking strategy

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/multi-doc-production-rag/multi_doc_rag_docs.py:hybrid_rrf"
```

Measured against the same 9 questions, sentence-window chunks:

```text
dense-only  recall@3=0.333  hit_rate@3=0.667
hybrid+RRF  recall@3=0.159  hit_rate@3=0.444
```

!!! success "Hybrid search made retrieval worse here, and that's a real result"
    This isn't the usual outcome — [Embedding Model Selection](../embedding-models.md) has a real case where hybrid RRF fusion clearly helps. Checking why hybrid underperforms on *this* eval set: none of the 9 questions have exact-match value (rare terms, product codes, IDs) that BM25 is actually good at finding. For a natural-language question like "how many attention heads does the base Transformer model use," BM25's own top-ranked result often isn't the chunk that actually answers it — its top hit for that exact question came from a different page entirely, one with high keyword overlap but the wrong content. RRF has no mechanism to detect that BM25 is contributing noise for a specific query; it sums both rankers' reciprocal-rank scores unconditionally, so a bad BM25 ranking can pull a correct dense-retrieval top-1 result out of the fused top-3. The lesson isn't "don't use hybrid search" — it's that hybrid search's benefit is conditional on the corpus and query style actually having exact-match value to contribute, which this particular eval set doesn't.

## 5. Indexing into a real vector database, with metadata filtering across documents

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/multi-doc-production-rag/multi_doc_rag_docs.py:qdrant_multi_doc"
```

Measured against the genuinely ambiguous question "How many layers does the model have?":

```text
unfiltered:            attention_is_all_you_need.pdf p3 (0.444), attention_is_all_you_need.pdf p3 (0.420), gpt3.pdf p20 (0.380)
filtered to bert.pdf:  bert.pdf p3 (0.337), bert.pdf p5 (0.309), bert.pdf p8 (0.288)
```

Unfiltered, BERT's own answer (12 layers) never makes the top-3 at all — the Transformer and GPT-3 papers' own layer-count mentions score higher, since "how many layers" matches their phrasing at least as well. This is exactly the scenario metadata filtering exists for: when the calling application knows which document a question is actually about (a user viewing a specific paper, a chatbot scoped to one product's docs), a `doc=` filter turns an ambiguous multi-document retrieval into an unambiguous one — not a hypothetical use case, a real fix for a real miss measured above.

## 6. RAGAS evaluation of the assembled pipeline

Running naive RAG (sentence chunking, dense retrieval, top-3) over the full 9-question eval set and scoring with [RAGAS's core four metrics](../ragas.md#the-four-metrics-precise-mechanics):

```text
faithfulness:                          0.8889
answer_relevancy:                      0.6067
llm_context_precision_with_reference:  0.6111
context_recall:                        0.6667
```

Per-question, 3 of 9 questions show `context_recall = 0.0` — a real retrieval miss, not a generation problem: two of them (the Transformer's optimizer, GPT-3's context window) got an honest "the context doesn't specify" from the model rather than a hallucinated guess, which is the correct behavior when retrieval genuinely fails to surface the answer.

!!! success "The same grounding failure mode measured on RAGAS Deep Dive, reproduced here"
    The third miss is the interesting one: asked "What are BERT's two pretraining tasks called?", the naive RAG pipeline answered *"BERT is pre-trained using masked language modeling (MLM) and next sentence prediction (NSP)"* — completely correct. It still scored **faithfulness = 0.0, context_precision = 0.0, context_recall = 0.0**. The retrieved top-3 chunks for that specific question didn't actually contain "Masked LM" or "NSP" — the model answered from its own training knowledge about a well-known paper, not from what retrieval surfaced. This is the identical failure mode [RAGAS Deep Dive](../ragas.md#putting-the-four-metrics-together-diagnosing-a-rag-system) measured on a single-document corpus, reproducing here on a genuinely different, multi-document one — real evidence that the pattern isn't specific to one paper or one eval set.

## Run it yourself

```bash
python multi_doc_rag.py --part chunking
python multi_doc_rag.py --part retrieval
python multi_doc_rag.py --part qdrant
python multi_doc_rag.py --part ragas
python multi_doc_rag.py --part all
```

## Where this still breaks

- **The chunking and retrieval winners here are corpus-specific, not universal.** Sentence-window chunking won on this eval set because its questions target single-sentence factual claims — a corpus with longer, multi-sentence arguments to retrieve might favor a different strategy entirely. Re-run the comparison on your own corpus before assuming these results transfer.
- **The eval set is small (9 questions).** Real production systems need eval sets in the dozens-to-hundreds range before trusting a metric difference as more than noise — this tutorial's numbers are illustrative of the *method*, not a claim that sentence-window chunking universally beats recursive chunking.
- **This pipeline has no reranking, no query rewriting, and no retrieval-quality gate before generation.** See [Advanced RAG — Post-retrieval](../advanced-rag-post-retrieval.md) and [Corrective RAG](../corrective-rag.md) for the pieces this capstone deliberately leaves out to keep the comparison focused.

## Sources & further reading

- [Chunking Strategies](../chunking.md) — the fixed-size and recursive strategies compared here
- [Retrieval Methods](../retrieval-methods.md) and [Embedding Model Selection](../embedding-models.md#hybrid-search-combining-bm25-and-dense-with-rrf) — the hybrid RRF fusion mechanism used here
- [Vector Databases](../vector-databases.md) — Qdrant's real current API, used here for indexing and filtering
- [RAGAS Deep Dive](../ragas.md) — the four-metric diagnostic table this tutorial's final evaluation step uses, including the same grounding-failure pattern reproduced here
