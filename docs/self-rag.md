# Self-RAG

Even with good retrieval, reranking, and compression, a generation LLM can still: generate an answer when it should have retrieved but didn't, use irrelevant retrieved content just because it's sitting in the prompt, or produce an answer that isn't actually supported by the retrieved context — hallucination despite having the right documents right there. Every earlier technique on this site improves *what gets retrieved*. Self-RAG is about making the model **critique its own process and output** instead of just consuming whatever it's handed.

!!! example "Hands-on"
    The full loop below is runnable: [**Self-RAG →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/self-rag) in the code repo. Relevance grading, generation, and usefulness grading all need an LLM key; the groundedness check is fully local (no key needed for that specific stage).

??? abstract "TL;DR — quick revision"
    - **Two implementation paths exist, and they're easy to conflate:** the original paper fine-tunes a model to emit special reflection tokens; almost everyone in production instead uses a **prompted** version — a pipeline of separate LLM calls doing the same grading jobs, no fine-tuning at all. Both are legitimately "Self-RAG."
    - **The prompted loop, in order:** retrieve → grade each chunk's relevance → generate from only the relevant ones → grade groundedness (is the answer actually supported?) → grade usefulness (does it actually address the question?) → return, regenerate once, or fall back honestly
    - **The genuinely new piece here:** groundedness grading uses a local **NLI (Natural Language Inference) model** instead of another LLM call — meaningfully cheaper (~200-500ms locally vs. a full LLM round-trip) and, per recent benchmarking, comparably accurate for this specific kind of check
    - **A real, honest finding from building this:** across several deliberate attempts to elicit a hallucination (incomplete context, no explicit refusal instruction, topically-adjacent-but-wrong context), Claude Sonnet 5 consistently declined rather than guessed — a genuinely useful data point about when this whole apparatus earns its keep with a frontier model, versus older/smaller/open models more prone to filling gaps
    - **Realistic cost: 4-6+ LLM calls per query**, more if it loops — use a small/cheap model for the grading calls and reserve the expensive one for generation

## Two implementation paths — trained vs. prompted

**Path 1 — trained (the original 2023 Self-RAG paper):** the target LLM is fine-tuned to emit special **reflection tokens** as literal tokens in its own output — `[Retrieve: Yes/No]`, `[IsRel: Relevant/Irrelevant]`, `[IsSup: Fully/Partially/No Support]`, `[IsUse: 1-5]` — interleaved with generated text. A separate "critic" model (e.g. GPT-4, prompted with a rubric) labels a large corpus first; the target model is then fine-tuned with standard next-token-prediction loss over sequences that include these tokens at the right points. Nothing exotic about the training objective — the special part is that the tokens exist at all.

**Path 2 — prompted (what most real implementations use):** no fine-tuning. A pipeline of separate LLM calls, each doing one grading job via structured output, standing in for what the trained tokens would decide. This is the dominant approach in practice — the standard LangGraph reference implementation, and most reproductions on models with no trained/reflection-token version available, use this pattern. It's legitimately Self-RAG, not a lesser substitute — and it's what this page's code builds.

## The prompted loop, piece by piece

Naive RAG does four things blindly: it retrieves a fixed top-k with no check on whether any of it is actually useful, it hands the LLM everything it retrieved even if some of it is off-topic, it generates without confirming the answer is actually backed by what was retrieved, and it returns whatever comes out even if that answer doesn't really address the question. Self-RAG inserts an explicit check at each of those four blind spots. To make this concrete, one real question — *"What optimizer did they use for training?"* — is followed through all five stages below, with the actual intermediate output at each one.

### 1. Grade relevance — don't hand the LLM everything you retrieved

Bi-encoder retrieval returns the top-k by similarity, but "similar enough to rank in the top-k" and "actually useful for answering this question" aren't the same thing — a chunk can rank highly because it shares vocabulary or topic with the query without actually containing the answer. This step checks each retrieved chunk individually, before generation ever sees it, and throws out the ones that don't hold up.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/self-rag/self_rag_docs.py:relevance"
```

**What actually happened, for the running example** — naive retrieval pulled 5 chunks; relevance grading kept 2 and discarded 3:

```text
chunk 0 (relevant=False): "warmup_steps = 4000. 5.4 Regularization We employ
  three types of regularization during training..."
chunk 1 (relevant=True):  "(3.5 days). 5.3 Optimizer We used the Adam optimizer
  [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10−9..."
chunk 2 (relevant=True):  "target tokens. 5.2 Hardware and Schedule We trained
  our models on one machine with 8 NVIDIA P100 GPUs..."
chunk 3 (relevant=False): "Table 2: The Transformer achieves better BLEU
  scores than previous state-of-the-art models..."
chunk 4 (relevant=False): "were written at 10-minute intervals. For the big
  models, we averaged the last 20 checkpoints..."
```

Worth noting honestly: chunk 2 (GPU hardware, not the optimizer) got graded relevant too — a judgment call, not a wrong one, since it's from the same training-setup section and could plausibly help contextualize the answer. This step is a real LLM judgment, not a keyword filter, and it won't always draw the line exactly where you would.

### 2. Generate — but only from what survived grading

The retrieved-and-filtered chunks (chunk 1 and chunk 2 from above) become the *only* context the model sees for this question — chunks 0, 3, and 4 are gone, so nothing about regularization, BLEU scores, or checkpoint averaging can leak into or distract this answer.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/self-rag/self_rag_docs.py:generate"
```

For the running example, this produces: *"According to the context, they used the Adam optimizer with β1 = 0.9, β2 = 0.98 and ϵ = 10^-9."* Note the prompt explicitly tells the model to say so rather than guess when the context is thin — the same discipline used in [naive RAG's prompt](naive-rag.md). That instruction is what's doing the work in the fallback trace further down this page, where the model correctly declines instead of guessing.

### 3. Grade groundedness — is the answer actually backed by that context, or did the model add something?

This is the check for the specific failure naive RAG can't catch: an answer that *sounds* right and uses the right vocabulary, but says something the retrieved text doesn't actually support — a number changed, a claim added, a detail from the model's own general knowledge slipped in instead of what was in front of it. Most write-ups implement this as another LLM call asking "is this supported?" — this recipe uses a local **NLI (Natural Language Inference) cross-encoder** instead, treating the retrieved context as a *premise* and the generated answer as a *hypothesis*, and asking whether the premise entails it.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/self-rag/self_rag_docs.py:groundedness"
```

**For the running example:** context = chunks 1 and 2 above, answer = the Adam-optimizer sentence from step 2 → the model outputs `entailment`, so `grounded=True`. To see the check actually distinguishing something, here's a controlled test against the same context with a deliberately wrong claim swapped in:

```text
Context: "We used the Adam optimizer with beta1=0.9, beta2=0.98."

Answer: "The model was trained using the Adam optimizer."
-> entailment (grounded=True)

Answer: "The model was trained using stochastic gradient descent (SGD)."
-> contradiction (grounded=False)
```

Only `entailment` counts as grounded — both `contradiction` (the context actively disagrees) and `neutral` (the context just doesn't say enough either way) mean the answer isn't backed by what was retrieved, and the pipeline treats them the same way: not good enough to return as-is.

### 4. Grade usefulness — grounded isn't the same as helpful

An answer can pass the groundedness check and still fail the user: *"I cannot be completely certain about training details without more information"* is not contradicted by anything, but it also doesn't answer the question — grounded, yet useless. This step catches that gap.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/self-rag/self_rag_docs.py:usefulness"
```

For the running example, the Adam-optimizer answer from step 2 grades `useful=True` — it directly answers what was asked. The hedge sentence above, tested directly against this same grader, comes back `False`. Groundedness and usefulness are deliberately separate checks because they can fail independently: grounded-but-useless (the hedge), or — in principle — useful-but-ungrounded (a confident, on-topic answer that's simply wrong).

### 5. The full loop — return, retry, or fall back honestly

All four stages above are wired together here: if an answer passes both groundedness and usefulness, it's returned immediately (that's what happens for the running example — one pass, no retry needed). If either check fails, the pipeline regenerates once and re-checks. If it's *still* not grounded and useful after that, it stops trying and returns an explicit "I don't have enough reliably-supported information" message instead of a second silent failure — the same "flag rather than guess again forever" pattern real production Self-RAG systems use.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/self-rag/self_rag_docs.py:pipeline"
```

!!! success "A real, unforced end-to-end trace — the fallback working as intended"
    Asked *"What company is the author who proposed replacing RNNs with self-attention affiliated with?"* — a fact already known, from the [pre-retrieval page's IRCoT testing](advanced-rag-pre-retrieval.md#multi-hop-query-planning-ircot), to be genuinely unfindable by any query phrasing on this corpus:

    ```text
    Retrieved: 5, graded relevant: 0

    --- attempt 1 ---
    Answer: I don't have enough context to answer this question...
    Groundedness: no_context (grounded=False)
    Useful: False

    --- attempt 2 ---
    Answer: I cannot answer this question because no relevant context was provided...
    Groundedness: no_context (grounded=False)
    Useful: False

    Status: fell_back
    Final answer: I don't have enough reliably-supported information in the
    retrieved context to answer this confidently.
    ```

    Every stage did its job correctly on a genuine retrieval failure, not a manufactured one: relevance grading discarded all 5 candidates as irrelevant, generation honestly declined rather than guessing, both checks correctly flagged the result, the loop retried once, and the pipeline surfaced an honest final message instead of a confident wrong answer. This is the actual value proposition of the whole apparatus, working exactly as intended.

!!! note "A real, honest finding: this model didn't want to hallucinate"
    Building the groundedness demo required deliberately trying to elicit an ungrounded answer — several constructions were tested: context that omits the requested fact entirely, prompts with no explicit "say so if missing" instruction, and topically-related-but-non-answering context. **Claude Sonnet 5 declined every single time**, across repeated trials, rather than filling in a plausible-sounding answer from its own pretraining knowledge (which, for a paper this famous, it almost certainly has memorized). That's a genuinely useful data point, not a failed experiment: it suggests the marginal value of a separate groundedness-checking stage is smaller with a frontier, safety-tuned model like this one than it would be with an older, smaller, or less-aligned model more prone to filling gaps confidently. The checks were still verified directly — see the controlled tests above — and remain valuable for less-aligned models, adversarial inputs, or domains where you can't risk even a rare failure.

## Comparison to LLM-as-judge

LLM-as-judge is typically a **separate model/call scoring output after the fact**, outside the generation pipeline — used for offline evaluation, not something the system itself acts on. Self-RAG integrates the same kind of judgment **into the pipeline itself**, at generation time, so the system can act on it: discard irrelevant chunks, regenerate, or fall back — rather than just measuring quality after the answer has already gone out.

## Use it for / skip it for

**Use it for:** high-stakes QA where a wrong or unsupported answer is costly (legal, medical, financial, compliance) and the extra latency/cost of multiple calls per query is acceptable. **Skip it for:** high-throughput, low-latency, simple-lookup use cases where naive RAG or a single reranking pass is already good enough — realistic cost here is 4-6+ LLM calls per query, more on a retry, and that's not free.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: Self-RAG">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team implements Self-RAG using GPT-4o, a model with no publicly available fine-tuned reflection-token checkpoint. They build a pipeline of separate structured LLM calls for relevance, groundedness, and usefulness grading instead.",
      "question": "Is this a legitimate implementation of Self-RAG?",
      "options": [
        "No — without the trained reflection tokens from the original paper, this is a different technique that shouldn't be called Self-RAG.",
        "Yes — this is the prompted implementation path, which is what most real production systems (including the standard LangGraph reference pattern) actually use, precisely because fine-tuned reflection-token checkpoints aren't available for most models people want to use.",
        "No — GPT-4o would first need to be fine-tuned for structured-output/function-calling before it could reliably emit the relevance, groundedness, and usefulness labels this pipeline depends on.",
        "Yes, but only if the structured LLM calls are run using the exact same model architecture as the original paper's fine-tuned checkpoint."
      ],
      "correct": 1,
      "explanations": [
        "Both paths are legitimately Self-RAG — they implement the same core idea (explicit relevance/groundedness/usefulness checks instead of blind consumption) through different mechanisms. The prompted path isn't a lesser substitute.",
        "Correct. The prompted, multi-call pipeline is the dominant real-world implementation specifically because most models people actually want to use (GPT-4o, Claude, Llama3-8B without a Self-RAG fine-tune) have no trained reflection-token version available.",
        "GPT-4o (and most frontier chat models) already supports structured-output-style responses out of the box — that's precisely why the prompted path works across many different models without any model-specific fine-tuning; the paper's fine-tuning step produces reflection tokens, it isn't what grants structured-output capability itself.",
        "Model architecture has nothing to do with which implementation path is being used — the prompted path works with any model capable of following structured-output instructions, regardless of its architecture."
      ]
    },
    {
      "scenario": "A team builds a groundedness-checking stage using an LLM-judge call for every generated answer. Someone suggests replacing it with a local NLI (Natural Language Inference) cross-encoder model instead, for a specific class of questions.",
      "question": "For which case does this page's evidence suggest the NLI-based swap is well-justified?",
      "options": [
        "For open-ended creative writing tasks with no single retrieved source document to check against.",
        "For QA-over-retrieved-context tasks with a clear source document — checking whether a generated answer is entailed by specific retrieved passages, where NLI models are both meaningfully cheaper (local, ~200-500ms) and comparably accurate to an LLM judge per recent benchmarking.",
        "Whenever the check needs to run with low latency, regardless of whether there's a single retrieved passage to check the answer against — since NLI models need no LLM call at all, the speed advantage alone should decide it.",
        "Only when the retrieved context is longer than the NLI model's maximum input length."
      ],
      "correct": 1,
      "explanations": [
        "NLI models work best specifically when there's a clear source document/premise to check entailment against — open-ended tasks without a single clear source are explicitly noted as a WORSE fit for this approach, not a better one.",
        "Correct. This matches exactly what's documented: NLI models are a strong fit for QA-over-retrieved-context faithfulness checking specifically — cheaper and comparably accurate there — while being a weaker fit for open-ended generation without a clear source document.",
        "Speed isn't the deciding factor — NLI's premise/hypothesis structure requires a clear source passage to check the answer against; without one, as in open-ended generation, there's nothing to use as the premise, so the latency advantage doesn't transfer to that class of check no matter how much speed matters.",
        "Context length isn't the deciding factor described here — the deciding factor is whether there's a clear premise (source document) to check the hypothesis (answer) against, which is a structural property of the task, not simply its length."
      ]
    },
    {
      "scenario": "While testing a Self-RAG pipeline, a developer tries several deliberate constructions meant to make the LLM hallucinate an unsupported answer — incomplete context, no explicit refusal instruction, topically-adjacent-but-wrong context. In every single trial, the model declines to answer rather than guessing.",
      "question": "What's the most defensible conclusion to draw from this outcome?",
      "options": [
        "The groundedness-checking stage adds no measurable value with a model this well-aligned, so it should be dropped for any deployment that only ever uses Claude Sonnet 5 or comparably safety-tuned frontier models.",
        "This specific, well-aligned frontier model shows strong resistance to hallucinating in this particular setup — a genuinely useful finding about when the checking stage's marginal value is smaller, though the checks remain independently verifiable (and valuable for less-aligned models, adversarial inputs, or high-stakes domains) via controlled tests.",
        "The experiment failed and should be redesigned until a hallucination is successfully produced, since a Self-RAG demo requires showing the check catching a real failure.",
        "This suggests hallucination has effectively been solved as a general LLM problem, since even deliberately adversarial prompts couldn't produce one in these trials."
      ],
      "correct": 1,
      "explanations": [
        "The page explicitly cautions the opposite: even with this well-aligned model, adversarial inputs or higher-stakes domains still warrant keeping the check — resisting hallucination in a handful of deliberate trials on one corpus doesn't rule out the rarer cases where it's the only thing standing between a miss and a confident wrong answer.",
        "Correct. This is exactly the honest reasoning this page applies: a real, repeated finding (this model resists hallucinating in this setup) is reported as a genuine data point, not hidden or spun as either a total win or a total failure — and the underlying checks are still verified to work correctly via controlled tests regardless.",
        "Treating a genuine, repeatable finding as a failed experiment to be tuned away until it produces the 'expected' result is precisely the kind of manufactured-result thinking this whole site argues against — the honest finding IS the result.",
        "A handful of trials against one model on one narrow corpus is far short of evidence that hallucination is solved generally — the page explicitly scopes its claim to this specific model in this specific setup, and still recommends keeping the check for other models, adversarial conditions, or high-stakes domains."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) — the original paper, trained reflection tokens
- [LangGraph's Self-RAG reference implementation](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb) — the standard prompted pattern this recipe follows
