# Evaluation Metrics

RAGAS packages a specific, opinionated set of metrics behind an LLM judge. This page covers the metrics underneath and around that: standard information-retrieval metrics for the retrieval half of a RAG system, and the classic lexical-overlap metrics for the generation half — computed with real code against a real corpus and real generated answers, not described in the abstract.

!!! example "Hands-on"
    The full pipeline below is runnable: [**Evaluation Metrics →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/evaluation-metrics) in the code repo. No paid API needed — generation runs through a local Ollama model.

??? abstract "TL;DR — quick revision"
    - **Retrieval metrics need ground truth** — Precision@k, Recall@k, MRR, NDCG, and Hit Rate all require knowing which chunks are actually relevant to a query; this page builds that ground truth from a real substring match, not hand-picked page numbers
    - **Measured on a real system**: dense retrieval at k=3 scored Precision@3 = 0.58, Recall@3 = 0.62, MRR = 0.81, NDCG@3 = 0.74, Hit Rate@3 = 0.88 across 8 questions — one question (attention head count) missed its relevant chunk entirely at k=3
    - **Classic generation metrics (BLEU, ROUGE-L, METEOR) measure word overlap, not correctness** — this page's own measured run shows a factually wrong answer (41.0 instead of the correct 28.4 BLEU score) scoring BLEU=32.9, ROUGE-L=0.68, METEOR=0.76, and semantic similarity=0.89, because the wrong number sits inside an otherwise well-phrased, structurally similar sentence
    - **This page covers retrieval metrics and classic/embedding generation metrics; the LLM-judged groundedness diagnostic table (Faithfulness, Context Recall, etc.) is a separate, more involved measurement** already covered with real code in [RAGAS Deep Dive](ragas.md)

## Retrieval metrics

All five metrics below need the same input: for a given query, a ranked list of retrieved chunk IDs, and the *real* set of chunks that are actually relevant to it. Getting that ground truth right matters more than the metric formulas — a wrong relevance set produces confident, meaningless numbers.

**Precision@k.** Of the top-k retrieved chunks, what fraction are actually relevant? `relevant_retrieved / k`. Penalizes retrieving noise.

**Recall@k.** Of all relevant chunks that exist in the corpus for a query, what fraction did the top-k retrieval actually catch? `relevant_retrieved / total_relevant`. Penalizes missing relevant content entirely, independent of how much noise came with it.

**MRR (Mean Reciprocal Rank).** For each query, find the rank of the *first* relevant result — rank 3 gives a reciprocal rank of 1/3 — then average across queries. Rewards a relevant result appearing near the top; doesn't care how many relevant results exist in total, so it's a good proxy for "does the user find something useful quickly."

**NDCG (Normalized Discounted Cumulative Gain).** Rewards relevant results ranked higher with a logarithmic rank discount, normalized against the ideal ordering. With graded relevance labels (highly relevant / somewhat relevant / irrelevant) it's strictly more informative than MRR; the code below uses the simpler binary-relevance case, since that's what the substring-based ground truth actually supports.

**Hit Rate@k.** The coarsest of the five: for what fraction of queries was *at least one* relevant chunk found anywhere in the top-k? Easy to compute and communicate, but doesn't distinguish "one relevant chunk barely made it in" from "every retrieved chunk was relevant."

### Building real ground truth from the data itself

The standard 8-question eval set used throughout this site carries a `keyword` field — a substring that only appears on the page(s) that actually answer that question. That's enough to build a real, verifiable relevance set per query, rather than hand-labeling "relevant" pages by eye:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/evaluation-metrics/evaluation_metrics_docs.py:build_relevance_from_keyword"
```

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/evaluation-metrics/evaluation_metrics_docs.py:retrieval_metrics"
```

Measured across the standard 8-question eval set, dense retrieval at k=3:

```text
Precision@3: 0.583
Recall@3:    0.620
MRR:         0.812
NDCG@3:      0.740
Hit Rate@3:  0.875
```

!!! success "One real miss explains most of the gap from a perfect score"
    The "How many attention heads did they use?" question scored 0.0 on Precision@3, Recall@3, and Reciprocal Rank. The chunk that actually contains "h = 8" sits on page 5 — but dense retrieval's top-3 for that query pulls in a page from the appendix's attention-visualization figures instead, a real but tangential semantic match that outranks the literal answer. This is the same retrieval miss the [Vector Databases](vector-databases.md#qdrant) page's Qdrant and Weaviate demos hit on the identical question — a metadata filter fixed it there; here it shows up directly in the retrieval-metric numbers instead of being fixed.

## Generation metrics: classic lexical overlap

These predate LLM-based evaluation, from machine translation and summarization research, and are worth knowing specifically for their limitations.

**BLEU.** Measures n-gram overlap between generated text and a reference — what fraction of generated word sequences (unigrams, bigrams, etc.) also appear in the reference. Originally built for machine translation.

**ROUGE.** Similar n-gram overlap, but recall-oriented — what fraction of the *reference's* n-grams appear in the generated text. Originally built for summarization. ROUGE-L specifically uses longest-common-subsequence overlap rather than fixed-size n-grams.

**METEOR.** Improves on BLEU by accounting for synonyms and stemming rather than requiring exact word matches, and balances precision and recall — generally more forgiving than raw BLEU on paraphrased but correct text.

**Embedding-based semantic similarity.** Embed both the generated answer and the reference, then compute cosine similarity — captures meaning-level closeness rather than surface wording, at the cost of not directly checking whether the answer is actually grounded in retrieved context.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/evaluation-metrics/evaluation_metrics_docs.py:generation_metrics"
```

Measured against 8 real answers generated locally (no paid API):

```text
BLEU:        33.4
ROUGE-L:     0.661
METEOR:      0.629
SemanticSim: 0.825
```

!!! success "A factually wrong answer that still scores well on every metric here"
    Asked *"What BLEU score did they get on English-to-German translation?"*, the naive RAG pipeline answered *"...their big model achieved a BLEU score of 41.0 on the WMT 2014 English-to-German translation task"* — factually wrong; 41.0 is the paper's English-to-**French** score, not English-to-German (28.4). That wrong answer still scored **BLEU=32.9, ROUGE-L=0.68, METEOR=0.76, semantic similarity=0.89** — all comfortably high, because the wrong number sits inside an otherwise well-formed, structurally similar sentence. None of these four metrics has any mechanism for checking whether the specific number is correct; they all measure closeness in wording or meaning to the reference, not truth. This is the concrete version of the abstract warning every metrics tutorial gives about BLEU/ROUGE: a factually wrong answer phrased similarly to the reference can score just as well as a right one.

## Beyond lexical overlap: groundedness, completeness, utilization

A RAG-specific framing, complementary to the metrics above:

- **Groundedness (faithfulness):** is every claim in the answer actually supported by retrieved context? The core hallucination check — an unsupported claim is either drawn from the model's own parametric memory (risky) or fabricated outright.
- **Completeness:** does the answer use *all* the relevant information available in retrieved context, or leave out something important that was actually retrieved? Groundedness alone doesn't catch this — an answer can be perfectly faithful and still incomplete.
- **Utilization:** of everything retrieved, how much actually shows up in the generated answer? Low utilization can mean the model is ignoring good context (a generation-side problem) or that retrieval brought back irrelevant padding the model correctly ignored — which one it is matters more than the raw number.

Checking these properly needs an LLM judge — decomposing an answer into individual claims and checking each one against retrieved context, the same way BLEU/ROUGE can't. That's exactly what RAGAS's Faithfulness and Context Recall metrics do, and [RAGAS Deep Dive](ragas.md) has real, measured numbers from running them against this same corpus — including a case where a factually correct answer still scored near-zero on groundedness, the mirror image of this page's BLEU/ROUGE finding above.

## Human evaluation

Every automated metric on this page and on [RAGAS Deep Dive](ragas.md) is a proxy for what actually matters: a correct, useful answer to a real question. A response can look fine on every automated number while still being subtly unhelpful, tonally wrong, or wrong in a way none of these metrics were built to catch.

**Direct scoring.** Raters score responses on defined dimensions (correctness, helpfulness, groundedness) against a written rubric, for consistency across raters.

**Pairwise comparison.** Show raters two candidate responses and ask which is better — often more reliable than absolute scoring, since relative judgments are cognitively easier than absolute ones.

**Inter-rater agreement.** With multiple raters, measuring agreement (e.g. Cohen's kappa) surfaces whether a rubric is ambiguous or the task is genuinely subjective — low agreement means the resulting scores need lower confidence attached to them.

In practice, human evaluation is expensive enough that it's usually reserved for validating that an automated metric (RAGAS, an LLM judge, or the metrics on this page) actually correlates with human judgment on a sample, plus periodic spot-checks of production output, since automated judges can drift or carry blind spots that only surface over time.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: Evaluation Metrics">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team's naive RAG pipeline generates the answer \"the model achieved a BLEU score of 41.0\" when the correct answer is 28.4 -- a real number from a different, nearby part of the source document.",
      "question": "Based on this page's own measured example, what happens when this wrong answer is scored with BLEU, ROUGE-L, and METEOR against the correct reference answer?",
      "options": [
        "All three metrics score near zero, since the numeric answer is factually wrong.",
        "The metrics can still score comparatively high, since they measure word/phrase overlap with the reference, not whether the specific factual claim is correct.",
        "BLEU and ROUGE will catch the error but METEOR will not, since METEOR is specifically designed to check numeric facts.",
        "The metrics will report a computation error, since BLEU, ROUGE, and METEOR all require validating that any numeric tokens in the candidate match the reference before scoring the rest of the sentence."
      ],
      "correct": 1,
      "explanations": [
        "This page's own measured run shows the opposite: the wrong-BLEU-number answer scored BLEU=32.9, ROUGE-L=0.68, and METEOR=0.76 -- all comfortably high, not near zero.",
        "Correct. This page's own measured example demonstrates exactly this: a wrong number embedded in an otherwise well-phrased, structurally similar sentence scored high on all three lexical-overlap metrics, because none of them check factual correctness -- only wording/meaning closeness to the reference.",
        "None of the three metrics has a mechanism for checking numeric facts specifically -- all three scored this wrong answer comparably high in this page's own measured example, METEOR included.",
        "This page's own measured run shows the opposite: all three metrics computed cleanly and produced ordinary scores (BLEU=32.9, ROUGE-L=0.68, METEOR=0.76) -- none of them contains any special numeric-token validation step; they treat a wrong number exactly like any other mismatched word or n-gram."
      ]
    },
    {
      "scenario": "A team wants to compute Precision@k and Recall@k for their retrieval system but only has a set of test questions -- no pre-labeled list of which document chunks are 'relevant' to each one.",
      "question": "Per this page's approach, what's a real, verifiable way to build that missing ground truth without manually hand-labeling every chunk?",
      "options": [
        "Precision@k and Recall@k require running a separate, LLM-judged relevance pass over every chunk in the corpus before the metrics can be computed, the same way RAGAS's Context Precision works.",
        "Use each question's expected-answer keyword (a substring known to only appear in the chunk(s) that actually answer it) to build the relevant set programmatically -- any chunk containing that keyword is ground-truth relevant.",
        "Treat every chunk in the corpus as equally relevant to every question, and compute the metrics as an average over the whole corpus.",
        "Use the top-1 result from a single retrieval run as the only relevant chunk for every query, regardless of the query's actual content."
      ],
      "correct": 1,
      "explanations": [
        "This page's ground truth isn't LLM-judged at all -- it's a much simpler, real substring/keyword match: a chunk counts as relevant if it contains the keyword known to answer that specific question. That LLM-judged approach is closer to how RAGAS's Context Precision works (covered on the RAGAS page), not what this page's Precision@k/Recall@k ground truth uses.",
        "Correct. This page's own build_relevance_set function does exactly this -- a chunk counts as relevant if it actually contains the keyword that answers the question, which is a real, checkable criterion rather than a hand-picked label.",
        "Treating every chunk as relevant would make Precision@k collapse toward k/corpus-size for every query regardless of what was actually retrieved -- it removes exactly the signal (which chunks are truly relevant) the metric needs to be informative.",
        "Using the retrieval system's own top-1 result as ground truth would make the retrieval system trivially 'perfect' at evaluating itself -- the relevance set needs to be independent of the system being measured."
      ]
    },
    {
      "scenario": "A team observes that one specific question in their eval set scores 0.0 on Precision@3, Recall@3, and Reciprocal Rank, while every other question scores reasonably well.",
      "question": "What does this pattern most directly indicate, per this page's own measured example of exactly this situation?",
      "options": [
        "This indicates ground truth was built incorrectly for that question, since a correctly-built relevance set should guarantee the retrieval system finds at least one relevant chunk in the top-3.",
        "The retrieval system's top-k results for that specific query contained none of the chunks that are actually relevant to it -- a genuine retrieval miss for that query, not a general system failure.",
        "A score of 0.0 means the query's answer requires information spread across multiple chunks that individually don't fully answer it, which is why no single top-3 chunk registers as relevant.",
        "This pattern only occurs with keyword-based ground truth and would not occur with human-labeled relevance judgments."
      ],
      "correct": 1,
      "explanations": [
        "This page's own measured example shows the ground truth was built correctly (the keyword-based relevance set correctly identifies the chunk with 'h = 8' as the answer) -- the 0.0 score reflects a real retrieval miss, not a flaw in how the relevant set was constructed.",
        "Correct. This page's own example (the attention-head-count question) shows precisely this: the truly relevant chunk existed and could be located, but the top-3 dense-retrieval results didn't include it -- a real retrieval miss for that specific query, not a corpus-wide failure.",
        "This page's own example shows a different, more specific cause: the answer lives entirely in one chunk (page 5, containing 'h = 8'), but a different, tangential chunk (the appendix's attention-visualization figures) ranked higher by semantic similarity and pushed the real answer out of the top-3 -- a ranking miss, not an answer fragmented across chunks.",
        "The same kind of miss can occur regardless of how ground truth was built -- a retrieval system can fail to surface a truly relevant chunk whether that chunk was identified by a keyword match or a human labeler."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [sacrebleu](https://github.com/mjpost/sacrebleu) — the BLEU implementation used above
- [rouge-score](https://github.com/google-research/google-research/tree/master/rouge) — Google's ROUGE implementation used above
- [NLTK METEOR](https://www.nltk.org/api/nltk.translate.meteor_score.html) — the METEOR implementation used above
- [RAGAS Deep Dive](ragas.md) — the LLM-judged groundedness/faithfulness diagnostic table this page's classic metrics complement, with real measured numbers on the same corpus
- [Vector Databases](vector-databases.md#qdrant) — the same retrieval miss this page's Precision@3=0 example shows up in directly, fixed there with a metadata filter
