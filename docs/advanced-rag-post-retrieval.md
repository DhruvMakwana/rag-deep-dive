# Advanced RAG — Post-retrieval Techniques

Retrieval doesn't end when you have your top-k chunks. Naive retrieval's ranking is only ever approximate, and even a genuinely relevant chunk is usually padded with sentences that don't answer the question. The techniques on this page all operate *after* the initial search — re-scoring what came back, trimming it down, or both — before any of it reaches the LLM.

!!! example "Hands-on"
    Every technique below is runnable, and every number is measured, not asserted: [**Post-retrieval →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/post-retrieval) in the code repo. Cross-encoder reranking needs no LLM key (fully local); contextual compression and listwise reranking both need one.

??? abstract "TL;DR — quick revision"
    - **Cross-encoder reranking fixes a real miss**: naive bi-encoder retrieval finds the right chunk *somewhere* in a wider candidate pool but ranks it outside the top-3; reranking with a cross-encoder correctly promotes it — because it reads query and chunk *together*, instead of comparing two independently-computed vectors
    - **Contextual compression cuts ~94% of the retrieved text** on a typical multi-chunk retrieval, keeping only the sentences that actually answer the question — same underlying fact, far less padding sent to the final LLM call
    - **LLM-based listwise reranking (RankGPT-style) fixes the same miss cross-encoder reranking does** — but at roughly **230x the latency** (one full LLM round-trip vs. one local forward pass), since it sends the whole candidate list to the LLM in a single call instead of scoring each candidate independently
    - **A real prompting lesson worth knowing:** telling the model to "return ONLY the list" doesn't work — it explains its reasoning anyway. Fighting that after the fact (parsing around the prose) is fragile; asking it to wrap just the final answer in an unambiguous `<ranking>...</ranking>` tag and letting it reason freely everywhere else is the fix that actually holds up
    - **The overarching lesson:** reranking's value depends on naive retrieval actually having a *ranking* problem — recall the chunk, rank it wrong. On a small, well-organized corpus, naive bi-encoder retrieval is often already good enough that there's nothing to fix; reranking earns its keep on messier, larger corpora with more distractor chunks

## Cross-Encoder Reranking

**The problem it fixes:** bi-encoder retrieval (what naive RAG does) embeds the query and every chunk *independently*, then compares vectors with cosine similarity. That independence is exactly what makes it fast and scalable — chunk embeddings are pre-computed once, offline, and search is just nearest-neighbor lookup. But it also means the model never looks at the query and a chunk *together*. It can retrieve the right chunk into a wide candidate pool and still rank it below several chunks that are only superficially similar.

A cross-encoder fixes this by feeding query and chunk into the model **together** — `[CLS] query [SEP] chunk [SEP]` — so bidirectional attention lets every query token attend to every chunk token and vice versa in one forward pass. It outputs a single relevance score per pair. This is far more accurate, but nothing can be pre-computed: every query-chunk pair needs a fresh forward pass, which is too slow to run over an entire corpus. The standard production pattern is therefore two-stage: bi-encoder retrieval casts a wide net (e.g. top-50 to top-150, optimizing for recall), then a cross-encoder rescoring only that shortlist (optimizing for precision). This is exactly Anthropic's Contextual Retrieval recommendation: hybrid retrieval → top-150 → cross-encoder rerank → top-20 → LLM.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/post-retrieval/post_retrieval_docs.py:cross_encoder"
```

**Input → output, from an actual run:**

```text
Input question:
  What is the complexity per layer for a recurrent layer compared to
  self-attention?

Naive bi-encoder retrieval (15-candidate pool), target chunk ranked #4
— outside the top-3:
  [rank 0] "...As noted in Table 1, a self-attention layer connects all
            positions with a constant number of sequentially executed
            operations, whereas a recurrent layer requires O(n)..."
  [rank 1] "...recurrent layers, by a factor of k. Separable convolutions
            [6], however, decrease the complexity considerably..."
  [rank 2] "...layer in a typical sequence transduction encoder or
            decoder. Motivating our use of self-attention we consider
            three desiderata..."
  [rank 3] "Table 1: Maximum path lengths, per-layer complexity and
            minimum number of sequential operations..." <- the actual
            table containing O(n·d2) for recurrent layers, MISSED by
            naive top-3

Cross-encoder rescoring the same 15 candidates (relevance scores, not
ranks — notice the big drop after rank 2):
  rank 0: score=7.07
  rank 1: score=6.30
  rank 2: score=4.55  <- target chunk, now correctly in the top-3
  rank 3: score=4.23
  rank 4: score=3.21
  ...
  rank 14: score=-6.38
```

!!! success "Measured: fixes a real miss"
    Naive top-3 never surfaces the chunk containing "O(n·d2)" (the recurrent-layer complexity from Table 1) — it's present in the wider 15-candidate pool, just ranked 4th. Reranking that same pool with a cross-encoder correctly promotes it into the top-3. The cross-encoder's own scores show why: it gives the target chunk a clearly positive score (4.55) while everything past rank 5 or so scores negative — it isn't subtly reordering close calls, it's confidently separating genuinely relevant chunks from ones that only *looked* relevant to the bi-encoder.

**A caveat worth taking seriously:** the [ARAGOG benchmark](https://arxiv.org/pdf/2404.01037) found reranking doesn't show a clear advantage over naive retrieval on every dataset. That tracks with what's demonstrable here too — most already-clear questions against this small, well-organized paper already have naive bi-encoder retrieval finding the right chunk in the top-3, leaving nothing for a reranker to fix. Reranking's value shows up specifically when naive retrieval's *recall* is fine but its *ranking* isn't — a messier, larger corpus with more genuinely similar-looking distractor chunks will show this more often than a single 15-page paper does.

**Use it for:** production pipelines where you already do wide bi-encoder retrieval and want to sharpen the final top-n before generation — cheap once you have a candidate pool, since scoring 15 candidates takes well under a tenth of a second with the model already loaded. **Skip it for:** corpora where naive retrieval's top-k is already reliable — measure first, same principle as every other technique on this site.

??? note "How a cross-encoder actually produces that single score"
    It's not a repurposed next-token-prediction model. Most everyday LLMs (GPT, LLaMA) are decoder-only, trained via next-token prediction with *causal* attention — each token only attends to previous tokens, which is what makes next-token prediction work at all. A cross-encoder is usually a different architecture family entirely: **encoder-only, BERT-style**, with *bidirectional* attention — every token attends to every other token, in both directions, with no next-token constraint. That's what lets it deeply model the full relationship between every query token and every chunk token in one pass.

    The input is formatted as `[CLS] query tokens [SEP] chunk tokens [SEP]`. The `[CLS]` token's final hidden state is treated as a pooled summary of the whole pair (standard BERT practice), and a small classification head (1-2 linear layers) turns that vector into a single relevance score — often passed through a sigmoid for a 0-1 "how relevant" probability.

    Training is supervised, not next-token prediction: labeled `(query, chunk, relevance)` triples, usually via **pointwise loss** (binary cross-entropy — relevant or not) or **pairwise loss** (given a relevant and an irrelevant chunk for the same query, train `score(relevant) > score(irrelevant)` by a margin — common, since ranking is fundamentally about relative order). Training data commonly comes from **MS MARCO** (real search-engine query-relevance pairs, hundreds of thousands of examples), fine-tuned on top of a pretrained BERT-style checkpoint.

    You don't train your own — pretrained rerankers already exist and are what production systems actually use: `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers, open-source, the one used in this page's demo), **Cohere Rerank** (hosted API, the one ARAGOG benchmarked), **BGE-reranker** (BAAI, open-source, multiple sizes), and **Jina Reranker** (hosted/open). Newer rerankers are increasingly built on decoder-only LLMs directly (prompted for a relevance score), but classic BERT-style cross-encoders remain the production default since they're far cheaper per-pair than a full generative call — a cost/latency trade-off that comes up again below, when listwise LLM reranking is measured directly against this same cross-encoder.

## Contextual Compression

**The problem it fixes:** even a genuinely relevant retrieved chunk is usually mostly padding relative to the question — a 400-500 token chunk might have one sentence that actually answers it. That padding costs tokens, and it can actively hurt quality via the **"lost in the middle" effect**: LLMs pay less attention to information buried in the middle of a long context than to what's near the beginning or end, so stuffing several long, mostly-irrelevant chunks into the context window can degrade the final answer, not just its cost.

Contextual compression runs each retrieved chunk through a cheap LLM call that extracts (or rewrites) only the sentences relevant to the question, discarding the rest — after retrieval (and optionally after reranking), before the final generation call.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/post-retrieval/post_retrieval_docs.py:compression"
```

**Input → output, from an actual run:**

```text
Input question: What optimizer did they use for training?

5 chunks retrieved (98-484 chars each, ~2000 chars total) — only 2 of
the 5 actually contain anything about the optimizer:

  chunk 0 (98 chars):  "warmup_steps = 4000. 5.4 Regularization We employ
                        three types of regularization during training..."
  chunk 1 (484 chars): "(3.5 days). 5.3 Optimizer We used the Adam
                        optimizer [20] with β1 = 0.9, β2 = 0.98 and
                        ϵ = 10−9. We varied the learning rate..."
  chunk 2 (453 chars): "target tokens. 5.2 Hardware and Schedule We
                        trained our models on one machine with 8 NVIDIA
                        P100 GPUs. For our base models..."
  chunk 3 (484 chars): "Table 2: The Transformer achieves better BLEU
                        scores than previous state-of-the-art models..."
  chunk 4 (461 chars): "were written at 10-minute intervals. For the big
                        models, we averaged the last 20 checkpoints..."

After compression — chunks 0, 3, 4 compressed to "NONE" and dropped
entirely; chunks 1 and 2 kept only their relevant sentence:

  "We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10−9."
  "We trained our models on one machine with 8 NVIDIA P100 GPUs."
```

!!! success "Measured: ~94% reduction, target fact survives"
    The 5 retrieved chunks total ~1980 characters; compression brings that down to ~128 characters across 2 surviving extracts — a 94% reduction — and the actual answer ("Adam") is still in there. Notice this isn't just trimming: 3 of the 5 chunks compress to nothing at all, because they were retrieved (correctly, by similarity) as being on-topic but don't actually answer *this specific question*. Reranking would have kept or dropped these chunks wholesale; compression operates at the sentence level, within chunks reranking would have kept.

**When it's actually worth the extra call:** for a *single* retrieved chunk, skip it — compress-then-generate is two calls where generate-directly is one, for no benefit. It earns its keep specifically on multi-chunk retrieval (the common case, top-n = 5 to 20), for three real reasons: the per-chunk compression calls run in parallel using a cheap model, so added latency is closer to "one parallel batch + one final call" than pure sequential addition; the expensive high-quality model then runs once on a much smaller context instead of paying premium prices for every token of every raw chunk; and a smaller, higher-signal context directly mitigates "lost in the middle."

**Use it for:** multi-chunk retrieval where individual chunks are long relative to what actually answers the question. **Skip it for:** single-chunk retrieval, or chunks that are already tightly scoped (small, focused chunking strategies leave less padding for compression to trim).

## LLM-Based Listwise Reranking

**The problem it fixes — or rather, a different way of fixing the same problem:** a cross-encoder scores each candidate chunk *independently* against the query (pointwise). An LLM can instead see the **entire candidate list at once** and reason about candidates relative to each other in a single call — this is the RankGPT approach: number the candidates, ask the LLM to return them reordered by relevance. It's a genuinely more modern technique than classic BERT-style cross-encoders, and worth knowing as an alternative, not a strict upgrade.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/post-retrieval/post_retrieval_docs.py:listwise"
```

**Input → output, from an actual run — same scenario as cross-encoder reranking, for direct comparison:**

```text
Input question:
  What is the complexity per layer for a recurrent layer compared to
  self-attention?

The LLM's raw response (it reasons freely, then wraps only the final
answer in the requested tag):

  "Let me analyze each passage:

   Passage 4: This is key! Contains Table 1 showing:
   - Self-Attention: O(n²·d) complexity per layer
   - Recurrent: O(n·d²) complexity per layer
   This directly answers the question.

   Passage 1: Discusses sequential operations but mentions computational
   complexity, noting 'self-attention layers are faster than recurrent
   layers when the sequence...'
   [... more reasoning, several more passages, several more stray digits
   like 'Table 1' and '[4]' along the way ...]

   <ranking>4,1,9,3,5,11,2,6,13,14,7,10,15,8,12</ranking>"

Parsed ranking (from the tag only): passage 4 first — the correct
target — matching what cross-encoder reranking also found.
```

!!! warning "A prompting lesson this demo surfaced: 'return ONLY the list' doesn't work"
    The first version of this prompt instructed the model to return *only* a comma-separated list, nothing else. It didn't comply — it wrote several paragraphs of reasoning first, including plenty of stray digits ("Table 1", "[4]", "O(n²·d)") that have nothing to do with the actual ranking. A first attempt at parsing around this split the *entire* response on commas and grabbed the first few numeric-looking tokens — which pulled digits straight out of the reasoning prose instead of the real ranking, and silently returned the wrong top-3 with no error thrown anywhere.

    The fix that actually holds up isn't a smarter parser fighting the model's behavior — it's a different prompt. Instead of demanding the model suppress reasoning it clearly wants to produce, the prompt above lets it reason freely and asks it to wrap *only* the final answer in an unambiguous `<ranking>...</ranking>` tag. Verified across several runs: 100% tag compliance, regardless of how much reasoning prose comes before it. Extraction is then one simple, unambiguous regex instead of a heuristic guessing which line is "the real answer." **Don't fight a model's tendency to explain itself — give it an unambiguous place to put the part you actually need.**

!!! success "Measured: fixes the same miss, at real cost"
    On the identical scenario used for cross-encoder reranking, listwise reranking also correctly promotes the target chunk into the top-3. But the cost difference is stark and directly measured: scoring all 15 candidates with a pre-loaded cross-encoder took **0.036 seconds**; the single listwise LLM call took **8.32 seconds** — roughly **230x slower**, for the same outcome on this scenario. That tracks with the broader research picture: LLM-based reranking can be modestly more accurate than cross-encoders in some settings, but at meaningfully higher latency and cost per query — cross-encoders reportedly get close to LLM-level accuracy at a fraction of the time.

**Use it for:** small candidate lists where you want the LLM reasoning about relevance holistically rather than scoring in isolation — useful when relevance genuinely depends on comparing candidates against each other, not just against the query alone. **Skip it for:** latency-sensitive paths, or any candidate set large enough that a cross-encoder gets you comparable results for a fraction of the cost — which, on the evidence here, is most of the time.

## Disambiguation: RRF vs. Cross-Encoder Reranking

Easy to conflate — both involve "ranking," but they solve different problems at different stages.

**RRF (Reciprocal Rank Fusion)** — covered on the [pre-retrieval page](advanced-rag-pre-retrieval.md#multi-query-retrieval-rag-fusion) — *merges* several separate ranked lists (e.g. from multi-query paraphrases, or hybrid sparse+dense search) into one combined list. It scores purely on rank position — `score = Σ 1/(k + rank)` — no model inference, no re-reading of chunk or query, just arithmetic on ranks. Cheap and deterministic.

**Cross-encoder reranking** *re-scores* a single list of candidates by actually reading query and chunk together for a real relevance judgment — genuine model inference, not rank arithmetic.

They commonly run back-to-back, not as alternatives:

```
Query → [Multi-query generates 3 variants] → [3 separate retrievals]
      → [RRF merges the 3 ranked lists into 1 candidate list]
      → [Cross-encoder reranks that merged list for real relevance]
      → [Contextual compression trims the top-n] → [Final generation]
```

## Scenario Check

Four scenarios testing whether you can match a post-retrieval symptom to the right technique.

<div class="quiz-widget" data-title="Scenario Check: Advanced RAG — Post-retrieval">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team retrieves the top-15 candidates with a bi-encoder for every query. Manual inspection shows the truly relevant chunk is almost always SOMEWHERE in that top-15 — but it's frequently ranked 5th-10th rather than 1st-3rd, so the final top-3 sent to the LLM often misses it.",
      "question": "Which technique most directly addresses this specific symptom?",
      "options": [
        "Contextual compression, since trimming the top-3 chunks down to their most relevant sentences would let the LLM notice the answer more easily even if it's slightly buried.",
        "Cross-encoder reranking, since the problem is a RANKING failure (the chunk is retrieved, just mis-ordered) rather than a RECALL failure — a cross-encoder reads query and chunk together to score real relevance, which is exactly what's needed to fix ordering within an already-retrieved candidate set.",
        "Query rewriting, since a clearer, more specific version of the query would shift the bi-encoder's embedding closer to the target chunk and improve its rank.",
        "Multi-query/RAG-Fusion, since fusing several paraphrased query results with RRF would push the correct chunk higher by rewarding chunks that rank well across multiple lists."
      ],
      "correct": 1,
      "explanations": [
        "Compression only trims sentences within chunks that already made it into the selected top-n — it never changes which chunks get selected in the first place. If the relevant chunk is sitting at rank 5-10 and only the top-3 are sent onward, compression never even sees it.",
        "Correct. The symptom described — right chunk present, wrong rank — is precisely what cross-encoder reranking targets. It's a stage that runs on an already-retrieved candidate set, using real query-chunk joint attention to fix relative ordering, not a fix for retrieval recall.",
        "The scenario describes a retrieval that's already succeeding at finding the chunk (it's consistently in the top-15) — there's no vocabulary-mismatch symptom here for rewriting to fix. Rewriting changes what gets embedded before the first retrieval pass; it doesn't re-score an already-retrieved candidate set the way reranking does.",
        "RRF fuses multiple ranked lists from different query phrasings — it's a recall-oriented fix for a chunk that a single query might miss, not a re-scoring of a chunk using real query-chunk relevance. The scenario describes a recall success but ranking failure, which needs a technique that reads query and chunk together, like cross-encoder reranking, not more parallel retrieval lists."
      ]
    },
    {
      "scenario": "A team adds contextual compression to a RAG pipeline that always retrieves exactly ONE chunk per query and sends it straight to a large, expensive model for generation.",
      "question": "Is this a good application of contextual compression, and why?",
      "options": [
        "Yes, because even for a single chunk, extracting only the relevant sentence still reduces the tokens sent to the expensive generation model, so total cost per query goes down.",
        "No — for a single retrieved chunk, compress-then-generate is two LLM calls where generate-directly-from-the-raw-chunk is one, with no parallelization benefit (nothing to parallelize with just one chunk) and no 'lost in the middle' problem to mitigate (a single chunk isn't a long, noisy multi-chunk context).",
        "Yes, because the compression call gives the generation model a shorter, cleaner context than the raw chunk, which should make its answer at least marginally more accurate.",
        "No, but the real problem is that compression's LLM extraction step needs to see the full candidate set from reranking to correctly judge relevance, so applying it to an unranked single chunk is invalid input."
      ],
      "correct": 1,
      "explanations": [
        "This only counts the tokens saved on the generation call — it ignores that the compression step is itself a full LLM call added in front of it. For one chunk, that's two calls total instead of one, and none of compression's real payoffs (parallelizing many chunks' calls, shrinking a genuinely long multi-chunk context) apply to offset the extra call.",
        "Correct. Compression earns its keep specifically on MULTI-chunk retrieval, where the compression calls run in parallel and the final generation call operates on a much smaller, higher-signal context. None of that applies with just one chunk — it's strictly two calls instead of one for no benefit.",
        "The compression step's extraction is itself an LLM call with its own error modes (it can over-trim or under-trim), so it isn't guaranteed to help — and for a single chunk there's no multi-chunk 'lost in the middle' risk or padding-heavy context for it to be fixing, so the extra call has little upside to offset its cost.",
        "Compression's extraction call only needs a chunk and the question — it never requires the full candidate set or a prior reranking pass to correctly judge relevance within a single chunk. It's independent of reranking, operating at the sentence level within whatever chunks are already selected. The real reason single-chunk compression is a poor fit is the two-calls-for-one-benefit cost math, not an input-validity requirement."
      ]
    },
    {
      "scenario": "A team implements RankGPT-style listwise LLM reranking. The prompt says: 'Return ONLY a comma-separated list of numbers, nothing else.' In production, the model still writes several paragraphs of reasoning before the list — including plenty of stray digits from citations and table references — and the parsing code sometimes extracts a completely wrong ranking with no error thrown.",
      "question": "What's the most robust fix, based on the measured finding on this page?",
      "options": [
        "Rephrase the instruction to explicitly forbid preamble text, since the original prompt's wording ('nothing else') may not have been specific enough about what counts as extra text.",
        "Redesign the prompt to let the model reason freely, but ask it to wrap ONLY the final ranking in an unambiguous tag like <ranking>3,1,4,2</ranking> — then parse specifically for that tag's contents instead of trying to suppress reasoning the model clearly wants to produce.",
        "Switch to a smaller, cheaper model, since a less capable model is less likely to generate extended reasoning before the final answer.",
        "Increase the number of candidates in the list, so the model spends proportionally more of its response on the ranking itself and less on reasoning prose."
      ],
      "correct": 1,
      "explanations": [
        "The original prompt already said to return nothing else, and the model still explained its reasoning — the measured finding on this page is that fighting the model's tendency to explain itself with more precise or forceful wording doesn't reliably work; a differently-worded suppression instruction isn't a demonstrated fix, giving the model an explicit place to put the answer instead is.",
        "Correct. This is exactly the fix that held up when actually tested: instead of fighting the model's tendency to explain itself, give it an explicit, unambiguous place to put the part that matters. Verified across multiple runs to produce 100% tag compliance regardless of how much reasoning precedes it — a simple, robust regex extraction instead of a fragile heuristic.",
        "Model capability isn't the documented lever here — smaller models are not reliably less prone to explaining themselves, and the measured fix (redesigning the prompt to give reasoning an accepted outlet) worked regardless of which model was used; swapping models doesn't address the underlying prompting mismatch.",
        "A longer candidate list gives the model more to reason about, not less room to do so — if anything it invites more explanatory text, not less. This doesn't address why the model adds reasoning prose in the first place, which is unrelated to list length."
      ]
    },
    {
      "scenario": "A team compares cross-encoder reranking against LLM-based listwise reranking on the same candidate set and finds both correctly promote the truly relevant chunk to the top of the results. But the cross-encoder version responds in well under a second, while the listwise LLM version takes several seconds per query.",
      "question": "Given this outcome, which technique should the team default to for a latency-sensitive production path, and why?",
      "options": [
        "Listwise LLM reranking, since seeing the whole candidate list at once lets it reason about relevance more holistically than a cross-encoder's independent per-pair scoring.",
        "Cross-encoder reranking, since it achieved the same outcome on this scenario at a small fraction of the latency (a local forward pass vs. a full LLM round-trip) — listwise reranking should be reserved for cases where its holistic, whole-list reasoning provides a real accuracy benefit that specifically justifies the added cost and latency.",
        "Neither — since both techniques converged on the same result here, that's a sign naive bi-encoder ranking was already going to be sufficient without any reranking step at all.",
        "It doesn't matter much for accuracy, since both reached the same top result, so the choice can be made on factors like team familiarity with either approach."
      ],
      "correct": 1,
      "explanations": [
        "Holistic, whole-list reasoning is listwise reranking's real theoretical advantage, but the measured scenario shows it didn't translate into a better outcome here — both techniques promoted the identical correct chunk to the top. For a latency-sensitive path, paying ~230x the latency for an advantage that isn't demonstrated on the actual data isn't the defensible default.",
        "Correct. This is exactly the trade-off measured on this page: matching outcomes at a roughly 230x latency difference means cross-encoder reranking is the more defensible default for latency-sensitive paths, with listwise reranking reserved for situations where relative, whole-list reasoning demonstrably earns its much higher cost.",
        "This scenario compares two reranking approaches against each other precisely because naive bi-encoder ranking was already shown elsewhere on this page to miss this chunk in the top-3 despite it being retrievable. Both rerankers fixed that; dropping reranking entirely would reintroduce the exact ranking problem this page measured, not avoid it.",
        "The accuracy outcome was tied here, but the scenario is explicitly about a latency-sensitive path — a roughly 230x latency gap isn't a minor detail to be settled by preference or familiarity, it's the central deciding factor the question is asking about."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [ARAGOG: Advanced RAG Output Grading](https://arxiv.org/pdf/2404.01037) — benchmark finding that reranking doesn't always show a clear advantage
- [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents](https://arxiv.org/abs/2304.09542) — the RankGPT paper, listwise LLM reranking
- [Contextual Retrieval (Anthropic)](https://www.anthropic.com/news/contextual-retrieval) — the hybrid retrieval → rerank → top-n production pattern referenced above
