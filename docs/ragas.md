# RAGAS Deep Dive

An open-source evaluation framework, not a single metric — a Python library computing a specific set of RAG-tailored metrics, most using an LLM as the underlying judge. RAGAS is a structured, purpose-built application of the LLM-as-judge pattern, specifically for RAG pipelines. This page runs it against a real pipeline's real output, not a curated example, to see whether its headline pitch — that the four core metrics together tell you exactly where a RAG system is failing — actually holds up.

!!! note "Scope: ragas ships two parallel evaluation APIs, this page uses both"
    Current `ragas` (0.4.x) has a legacy `evaluate()` + `EvaluationDataset` + named-metric-class API — what every existing tutorial shows — and a newer per-sample `ragas.metrics.collections` + `.ascore()` API the current docs actually lead with. The legacy path still works (every import just fires a `DeprecationWarning`), and it's the only one that produces the four-metrics-together diagnostic table this page opens with — the modern collections classes can't be passed to `evaluate()` at all, since they don't share a base class with what `evaluate()` expects. Further down, this page also uses the modern collections API for four additional metrics and for synthetic eval-set generation, since neither is available through the legacy path.

!!! example "Hands-on"
    The full pipeline below is runnable: [**RAGAS Deep Dive →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/ragas-deep-dive) in the code repo. Needs an Anthropic API key, used both for the naive RAG pipeline's own generation and as RAGAS's judge LLM.

??? abstract "TL;DR — quick revision"
    - **Four metrics, four different pipeline stages**: Faithfulness and Answer Relevancy are generation-side, Context Precision is a ranking signal, Context Recall is a retrieval/indexing signal — the pattern across all four is meant to tell you exactly *where* to intervene, not just that quality dropped somewhere
    - **Needs a few version pins and one non-default wrapper flag** — `langchain-community==0.3.31`, a `LangchainEmbeddingsWrapper` around a plain HuggingFace embeddings object instead of ragas's own, and `bypass_temperature=True` on the judge LLM wrapper, since current Claude models don't accept a `temperature` parameter
    - **Measured on a real naive RAG pipeline**: aggregate scores of 0.91 faithfulness, 0.86 answer relevancy, 0.63 context precision, 0.88 context recall across the standard 8-question eval set
    - **The most useful result was the anomaly, not the average**: one question got a factually correct answer, but scored near-zero on faithfulness and context recall — the model answered from its own training data after retrieval missed the actual table row, exactly the failure mode a plain "is the final answer right" check would miss entirely, and the pattern reproduced across separate runs
    - **Four more metrics are covered beyond the core diagnostic table** — AnswerCorrectness, SemanticSimilarity, FactualCorrectness, and NoiseSensitivity from ragas's modern collections API, plus automatic synthetic eval-set generation via `TestsetGenerator`

## The four metrics — precise mechanics

**Faithfulness.** Does the generated answer only contain claims actually supported by retrieved context? RAGAS breaks the answer into individual atomic claims (via LLM decomposition), then checks each claim against retrieved context individually. Score = supported claims ÷ total claims. Decomposing into claims rather than judging the whole answer at once means a 5-correct-1-unsupported answer gets a graded score, not one binary judgment that loses which specific claim is the problem.

**Answer Relevancy.** Does the generated answer actually address the question asked — independent of whether it's factually correct? Computed by generating several hypothetical questions the answer would be a good response to, then comparing those (by embedding cosine similarity) against the original question. This is deliberately the mirror image of HyDE: comparing question-to-question, not question-to-answer, avoids the same phrasing-asymmetry problem HyDE exists to fix on the retrieval side.

**Context Precision.** Of the chunks actually retrieved, are the relevant ones ranked near the top rather than buried lower? An LLM judges each retrieved chunk's relevance individually, and the metric rewards relevant chunks appearing early in the ranked list — mathematically similar in spirit to how NDCG rewards early-ranked relevance in general information retrieval. A low score with genuinely-relevant chunks somewhere in the results points at reranking, not retrieval recall.

**Context Recall.** Did retrieval capture everything actually needed to answer correctly? This is the one metric that needs a ground-truth reference answer: the reference gets decomposed into claims the same way Faithfulness decomposes the generated answer, and each reference claim is checked against the *retrieved context* (not the generated answer). A low score means retrieval genuinely failed to find necessary information — a retrieval/indexing problem, not a generation problem, since the information was never in the retrieved set to begin with.

## Running it: naive RAG pipeline + RAGAS scoring

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/ragas-deep-dive/ragas_deep_dive_docs.py:naive_rag"
```

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/ragas-deep-dive/ragas_deep_dive_docs.py:ragas_eval"
```

!!! note "Setup requirements for this ragas release"
    `langchain-community==0.3.31` needs pinning — `ragas`'s own code imports `langchain_community.chat_models.vertexai`, a module later `langchain-community` releases removed (moved to a separate `langchain-google-vertexai` package), and `ragas` carries no upper pin on it.

    Embeddings go through `LangchainEmbeddingsWrapper` around a plain `langchain-huggingface` embeddings object, the same pattern used for the judge LLM via `LangchainLLMWrapper` — `ragas`'s own `HuggingfaceEmbeddings` class fails pydantic validation on direct instantiation in this release.

    The judge LLM wrapper needs `bypass_temperature=True`. `LangchainLLMWrapper` sets a `temperature` value on the underlying model before every call by default, and current Claude models no longer accept that parameter (they use adaptive thinking instead of sampling controls) — `bypass_temperature` is the wrapper's own documented flag for LLMs like this.

## Measured: does the diagnostic table hold up on a real system?

Aggregate scores across the standard 8-question eval set used throughout this site:

```text
faithfulness:                          0.9062
answer_relevancy:                      0.8596
llm_context_precision_with_reference:  0.6250
context_recall:                        0.8750
```

## Putting the four metrics together — diagnosing a RAG system

The four metrics map to different pipeline stages, so their pattern tells you where to intervene:

| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Diagnosis |
|---|---|---|---|---|
| Low | — | — | High | Generation is hallucinating or ignoring good context — a generation/prompting problem |
| — | Low | — | — | Answer doesn't address the question asked — a generation/prompting problem (wrong sub-question, drifting) |
| High | High | Low | High | Right info retrieved but ranked poorly — a reranking problem |
| High | High | — | Low | Necessary info not retrieved at all — a retrieval/indexing problem |
| High | High | High | High | System working well across the whole pipeline |

!!! success "A factually correct answer with near-zero grounding"
    Asked *"What BLEU score did they get on English-to-German translation?"*, the naive RAG pipeline answered *"the Transformer model achieved a BLEU score of 28.4"* — factually correct, 28.4 is the real number. But that question has repeatedly scored near-zero on faithfulness, context precision, and context recall across separate runs of this pipeline (0.167/0.0/0.0 in one run, 0.25/0.0/0.0 in another). A `context_recall` of 0.0 means the retrieved context didn't actually support the reference claim; in the first run, checking the actual top-3 retrieved chunks confirmed exactly why — one chunk was the results table cut off right before the Transformer's own rows appear, and the other two both discussed the *English-to-French* numbers (41.0, 41.8) from a different part of the paper. None of the three retrieved chunks contained "28.4" at all. Retrieval genuinely missed the right table row, both times. The model answered correctly anyway — from its own training data on this well-known paper, not from what was actually retrieved.

    This is the diagnostic table's "High-looking answer, but check faithfulness and recall before trusting it" case in the wild, and it isn't a one-off fluke: a factually right answer with near-zero grounding in what was actually retrieved, which a simple "is the final answer correct" check would have missed completely both times. It's also a real, measured argument for why Faithfulness and Context Recall exist as separate metrics from plain answer accuracy — accuracy alone would have marked this a pass on both runs.

## Beyond the core four: additional metrics

`ragas.metrics.collections` — the modern per-sample API the scope note above describes — ships a much larger catalog than the four-metric diagnostic table: classic NLP metrics (`BleuScore`, `RougeScore`), semantic and factual checks (`SemanticSimilarity`, `FactualCorrectness`, `AnswerCorrectness`), `NoiseSensitivity`, `ContextEntityRecall`, rubric-based custom metrics, and agentic-workflow metrics (`ToolCallAccuracy`, `TopicAdherence`, goal-accuracy variants) for systems this page doesn't cover. Four of these — the ones with the most direct overlap with the core four — are measured here against the same 8-question eval set:

- **AnswerCorrectness** — a weighted blend of factual correctness and semantic similarity against the reference answer.
- **SemanticSimilarity** — embedding cosine similarity between generated answer and reference, on its own.
- **FactualCorrectness** — decomposes both answer and reference into claims and checks them against each other via NLI, independent of retrieved context.
- **NoiseSensitivity** — how much an answer's correctness degrades when the retrieved context contains irrelevant ("noisy") chunks alongside relevant ones.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/ragas-deep-dive/ragas_deep_dive_docs.py:extended_metrics"
```

Measured aggregate scores, same 8-question eval set:

```text
answer_correctness:   0.7238
semantic_similarity:  0.7899
factual_correctness:  0.4600
noise_sensitivity:    0.5089
```

!!! note "FactualCorrectness is measurably stricter than semantic similarity"
    On the "What GPUs was the model trained on?" question, the naive RAG pipeline answered *"The model was trained on 8 NVIDIA P100 GPUs"* against a reference of *"The models were trained on NVIDIA P100 GPUs"* — the same fact, phrased slightly differently. `SemanticSimilarity` scored this pair 0.877, but `FactualCorrectness` scored it **0.0**. `FactualCorrectness`'s claim-decomposition-plus-NLI check is stricter about scope and phrasing than embedding similarity, penalizing differences (singular vs. plural framing, an added specific number) a human reader would likely still call factually equivalent. Worth knowing before treating a low `FactualCorrectness` score alone as proof of an actual factual error, rather than a stricter check flagging a phrasing mismatch.

!!! note "Setup for the modern collections API"
    This API builds its LLM via `llm_factory` + the `instructor` library rather than LangChain, so `LangchainLLMWrapper`'s `bypass_temperature` flag doesn't apply. Current Claude models reject both `temperature` and `top_p`, and this API has no equivalent constructor flag — both need removing directly from the constructed LLM's own `model_args` dict, as `make_modern_judge_llm()` above does. Separately, `instructor>=1.17` is required against Anthropic models specifically — older versions fail to parse a response when the model returns a `ThinkingBlock` before its text content.

## Generating an eval set automatically

Every measurement on this page needed a hand-written eval set with real reference answers. `ragas.testset.TestsetGenerator` builds one automatically from source documents instead — useful when a reference eval set doesn't exist yet.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/ragas-deep-dive/ragas_deep_dive_docs.py:testset_generation"
```

Run against the full sample paper, real generated questions include both clean, formal phrasing and a genuine typo-laden variant — `TestsetGenerator` deliberately varies persona and query style, not just topic:

```text
"As I examine the foundational contributions behind the Transformer architecture, I am curious to
understand precisely what specific innovations Noam Shazeer proposed..."

"wat did noam shazer propse in teh transformr paper"
```

Both are real generated questions from the same run, about the same underlying content — a genuine argument for synthetic generation producing a more realistic eval set than a hand-written one, which rarely includes deliberately messy phrasing.

!!! note "The multi-hop synthesizer needs more cross-document structure than one paper provides"
    Every run against this page's sample paper skips the multi-hop synthesizer entirely (`No relationships match the provided condition. Cannot form clusters.`), even against the full ~40K-character document — a single technical paper doesn't have enough thematic or entity overlap *between different sections* for the relationship-builder to find valid multi-hop clusters on. All generated questions come from the single-hop synthesizers instead. This isn't a bug; multi-hop generation needs a corpus with genuine cross-document relationships (e.g. multiple related papers or a multi-file knowledge base) to have anything to connect.

## Practical considerations for using RAGAS in production

**LLM choice for the judge matters and costs money.** Most RAGAS metrics use an LLM internally for claim extraction and relevance judgments — cost scales with the number of claims and chunks evaluated, for every eval-set item. Typically run against a curated eval set of dozens to a few hundred examples, not on every live production query in real time.

**Context Recall specifically requires ground-truth reference answers** — a labeled eval set with correct answers, more work to build than the reference-free metrics (Faithfulness, Answer Relevancy, and Context Precision can all run without ground truth, using only the question, retrieved context, and generated answer).

**RAGAS scores are still LLM judgments, not ground truth themselves** — the same caveats as any LLM-as-judge approach (verbosity bias, judge-model quality ceiling) apply. Worth validating against a human-eval sample before trusting scores at scale, same as any other automated metric.

**Use RAGAS for regression testing, not just one-time scoring.** Run against a fixed eval set every time chunking, embedding model, reranker, or prompt changes — turns RAGAS into a CI-style regression test for RAG quality, catching a change that improves one metric while quietly regressing another.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: RAGAS">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team runs RAGAS on their RAG system and finds a specific question scored Faithfulness = 0.17 and Context Recall = 0.0, even though the final generated answer happens to be factually correct.",
      "question": "Based on this page's own measured example, what does this pattern most likely indicate?",
      "options": [
        "Faithfulness and Context Recall must have used a different, weaker judge LLM for this particular question than for the rest of the eval set, causing an anomalously low score unrelated to what was actually retrieved.",
        "The retrieved context didn't actually contain the information needed to answer the question -- the model likely answered from its own training data/parametric memory rather than from what retrieval actually surfaced, which a plain answer-correctness check would completely miss.",
        "This combination means the reference answer itself must be wrong, since Context Recall checks the reference's claims against retrieved context and a 0.0 score means those claims aren't actually true.",
        "Low Context Recall here means the source document itself never contained the specific number needed, since Context Recall measures whether the underlying corpus, not just what was retrieved, has the necessary information."
      ],
      "correct": 1,
      "explanations": [
        "This page attributes the low scores to a specific, identified cause -- the retrieved chunks genuinely didn't contain the answer (one was cut off right before the relevant table row, the other two covered a different, nearby number) -- not to any inconsistency in which judge model was used; the same judge LLM scores every question in the eval set.",
        "Correct. This is exactly the real example measured on this page: the top retrieved chunks didn't contain the needed number at all, yet the model still produced the right answer -- almost certainly from prior knowledge about a well-known paper, not from grounding in retrieved context. Faithfulness and Context Recall exist specifically to catch this gap that plain accuracy checking cannot.",
        "This inverts what Context Recall checks -- it verifies whether the (already correct) reference's claims are supported by what was actually retrieved, not whether the reference itself is true. Here the reference was correct; retrieval simply failed to surface the chunk that would have supported it.",
        "This page's own example shows the information was present in the corpus the whole time (the correct table row exists on the page) -- what failed was retrieval surfacing it in the top-3, not the corpus lacking it. Context Recall measures what retrieval actually returned, not what the corpus contains overall."
      ]
    },
    {
      "scenario": "A team's RAGAS results show High Faithfulness, High Answer Relevancy, High Context Recall, but Low Context Precision.",
      "question": "Per the diagnostic table on this page, what should the team investigate first?",
      "options": [
        "The embedding model used for indexing, since a Low Context Precision score with everything else High most directly points at embeddings placing the wrong content close together in vector space.",
        "Reranking -- the relevant chunks ARE being retrieved (high recall) and the generation is faithful and relevant, but the relevant chunks aren't being ranked near the top, which is specifically what Context Precision measures.",
        "The generation prompt, since Faithfulness and Answer Relevancy already being high doesn't rule out the prompt as the root cause of the one remaining low score.",
        "Nothing needs fixing, since 3 of 4 metrics are high."
      ],
      "correct": 1,
      "explanations": [
        "High Context Recall already establishes the relevant content was found somewhere in the retrieved set -- that rules out an embedding-quality problem (which would show up as content not being retrieved at all) and points specifically at ranking, which is what Context Precision measures.",
        "Correct. This is exactly the table's own mapping: High Faithfulness + High Answer Relevancy rules out a generation-side problem, High Context Recall confirms the needed information WAS retrieved somewhere, and Low Context Precision specifically means it wasn't ranked near the top -- a reranking fix, not a retrieval-recall fix.",
        "Faithfulness and Answer Relevancy being high specifically does rule out a generation-side cause here -- both metrics are generation-stage signals, and both are already high, meaning generation is working correctly on the context it received; the low score is isolated to Context Precision, a ranking-stage metric.",
        "A low score on any one of the four metrics is diagnostic information pointing at a specific stage -- ignoring it because the average looks fine defeats the entire purpose of using four separable metrics instead of one aggregate score."
      ]
    },
    {
      "scenario": "A team wants to measure Context Recall on their RAG system but has no labeled reference answers for their evaluation questions -- only the questions themselves.",
      "question": "Based on this page, what's the correct conclusion?",
      "options": [
        "Context Recall can be computed the same way as the other three metrics, using only the question, retrieved context, and generated answer.",
        "Context Recall specifically requires a ground-truth reference answer to check retrieved context against -- without one, the team can still compute Faithfulness, Answer Relevancy, and Context Precision, but not Context Recall.",
        "Answer Relevancy also needs a labeled reference answer, since it's comparing the generated answer's quality against a known correct response the same way Context Recall does.",
        "Context Recall can be approximated by using the generated answer itself as a substitute for the reference answer, with no real difference in what's being measured."
      ],
      "correct": 1,
      "explanations": [
        "This page states directly that Context Recall is the one metric among the four that needs a ground-truth reference -- it isn't computed the same way as the reference-free three.",
        "Correct. This page states this explicitly: Faithfulness, Answer Relevancy, and Context Precision can all run without ground truth, using only question, retrieved context, and generated answer -- Context Recall specifically needs a reference answer, since it checks whether the reference's own claims are supported by retrieved context.",
        "This page states Answer Relevancy is explicitly reference-free -- it works by generating hypothetical questions from the answer itself and comparing them to the original question, with no reference answer involved at all. Only Context Recall among the four requires one.",
        "Using the generated answer in place of a reference would defeat the metric's actual purpose -- Context Recall checks whether retrieval found what the REFERENCE (independently correct) answer needed, not whether it supports whatever the system happened to generate, which could itself be wrong or incomplete."
      ]
    },
    {
      "scenario": "A team measures both SemanticSimilarity and FactualCorrectness on the same answer/reference pair and gets a high SemanticSimilarity score (0.877) but a FactualCorrectness score of 0.0, even though both a human reviewer and the team agree the answer states the same fact as the reference, just phrased slightly differently.",
      "question": "Based on this page's own measured example, what's the correct interpretation of this discrepancy?",
      "options": [
        "This means FactualCorrectness specifically penalizes any answer that includes a more precise number or detail than the reference, treating added specificity itself as an automatic scoring violation regardless of accuracy.",
        "FactualCorrectness's claim-decomposition-plus-NLI check is measurably stricter about scope and phrasing than embedding similarity, so a low FactualCorrectness score alone isn't proof of an actual factual error -- it can also reflect a phrasing/scope mismatch a human would still consider equivalent.",
        "SemanticSimilarity must be the less trustworthy signal here, since its embedding-based comparison is more prone to being fooled by superficially similar wording than FactualCorrectness's claim-level check.",
        "This combination reflects a data-entry mismatch -- the reference and generated answers must refer to different training runs (different actual GPU counts), which explains why FactualCorrectness treats them as contradictory facts."
      ],
      "correct": 1,
      "explanations": [
        "This page doesn't establish a blanket rule that added specificity is always penalized -- what it shows is that FactualCorrectness's claim-decomposition-plus-NLI check is stricter about phrasing and scope in general (this example being singular-vs-plural framing plus an added number), not that any more-specific answer automatically fails; the two metrics aren't in a 'broken vs. reliable' relationship, they're just checking different things.",
        "Correct. This page states this directly with a real measured example (the GPU question): FactualCorrectness's stricter claim-decomposition-plus-NLI check can penalize phrasing/scope differences that embedding similarity treats as equivalent -- a low score there needs a second look, not automatic treatment as a factual error.",
        "This page frames it the opposite way for this specific case: SemanticSimilarity correctly identified genuine factual equivalence (0.877), while FactualCorrectness's stricter check flagged a phrasing/scope difference a human reader called equivalent -- 'stricter' here doesn't mean 'more trustworthy for this judgment,' it means differently scoped, and a low FactualCorrectness score needs a second look rather than being trusted by default.",
        "This page states directly that this is the same fact, phrased slightly differently (singular 'model' vs. plural 'models', and an added specific count) -- not two different underlying facts about different training runs. The discrepancy is about how strictly the metric compares phrasing/scope, not a genuine factual conflict between two different claims."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [RAGAS documentation](https://docs.ragas.io) — official docs, including the current per-sample `collections` API this page's scope note describes
- [RAGAS GitHub (vibrantlabsai/ragas)](https://github.com/vibrantlabsai/ragas) — the project moved orgs from `explodinggradients/ragas`; old links redirect here
- [RAGAS testset generation documentation](https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/) — `TestsetGenerator`'s persona/query-style synthesis and multi-hop/single-hop synthesizer split
- [Advanced RAG — Pre-retrieval](advanced-rag-pre-retrieval.md#hyde-hypothetical-document-embeddings) — HyDE, the technique Answer Relevancy's question-to-question comparison deliberately mirrors
- [Embedding Model Selection](embedding-models.md) — the dense retrieval covered here is the exact retrieval mechanism this page's naive RAG pipeline uses and RAGAS evaluates
