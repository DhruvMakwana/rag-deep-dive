# Advanced RAG — Pre-retrieval Techniques

Naive RAG embeds the raw query, as-is, and searches. Every technique on this page intervenes *before* that search happens — rewriting, expanding, or reformulating the query itself, on the theory that a better-shaped query finds better chunks. That theory is correct, but only *conditionally*: each technique below fixes one specific, narrow failure mode, and does nothing (or actively adds noise) when that failure mode isn't present. The demos on this page are built around that idea directly — each technique is tested against a scenario that actually matches its own "when to use" case, so what you're seeing below is each technique doing the thing it was designed to do, not a technique judged against an unrelated question it was never meant to answer. **The single most important skill on this page isn't memorizing which technique does what — it's recognizing which failure mode your own queries actually have, so you reach for the one technique that fixes it instead of applying all six as a blanket "best practice."**

!!! example "Hands-on"
    Every technique below is runnable, and every number is measured, not asserted: [**Query Transformation →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/query-transformation) in the code repo. All 6 need an LLM key — query transformation is inherently LLM-driven.

??? abstract "TL;DR — quick revision"
    - **Query rewriting fixes a real miss** on a genuinely vague, colloquial query — naive retrieval never finds the target passage; the rewritten, technically-phrased query reliably does. It has nothing to fix on an already-clear query, though, and blindly applying it everywhere just adds latency and occasional noise
    - **Multi-query/RAG-Fusion also fixes a real miss**, on a broad, multi-faceted question where naive retrieval never surfaces one relevant aspect of the answer — but only in ~3 of 4 runs, since it depends on a fresh, unseeded LLM call generating paraphrases that happen to land near the target vocabulary. Real, but not guaranteed
    - **HyDE is unreliable in a specific, disclosed way**: it mostly ties naive on numeric-fact questions (naive already does fine there), but can occasionally *underperform* when a safety-tuned LLM refuses to hallucinate a specific number instead of confidently fabricating one — which defeats the technique's actual mechanism
    - **Step-back prompting showed no measurable effect**, even tested against a scenario built specifically to match its intended use case (a narrow fact with real general context nearby) — this single, well-organized paper doesn't separate general context from specific facts enough for the technique to add anything
    - **Query decomposition is a clean, real win**: naive retrieval misses one of two facts in a compound question even at k=6; decomposition reliably finds both
    - **IRCoT (interleaved retrieval + reasoning)** succeeds cleanly when the needed facts are each individually findable somewhere — but honestly reports "not found" rather than hallucinating when a fact is buried in a way no query phrasing can surface
    - **The overarching lesson:** each technique fixes one *specific* failure mode. If your corpus and query distribution don't actually exhibit that failure mode, the technique adds cost for nothing — occasionally it actively hurts. Diagnose your actual failure mode first, then reach for the one technique built for it

## Already covered elsewhere

| Technique | What it does | See |
|---|---|---|
| Contextual Retrieval (Anthropic) | Prepend an LLM-generated context blurb to each chunk before embedding | [Chunking Strategies](chunking.md#late-chunking) |
| RAPTOR | Cluster chunks, summarize each cluster, search leaf + summary levels together | [Retrieval Methods](retrieval-methods.md#raptor) |
| Adaptive / FLARE-style retrieval | Only retrieve when the model's own confidence is low | [Retrieval Methods](retrieval-methods.md#adaptive-retrieval-a-flare-style-approach) |

## Query Rewriting

**The problem it targets:** users phrase queries badly — too short, too vague, colloquial, or using different words than the source document ever uses ("how do I get my money back" vs. a document that only ever says "refund"). Query rewriting passes the raw query through an LLM before retrieval, asking it to produce a clearer, more specific version — the rewritten query gets embedded and searched, not the original.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/query-transformation/query_transformation_docs.py:rewrite"
```

**Input → output, from an actual run:**

```text
Input query:      why dont they just use RNNs like everyone else did before
Rewritten query:  What are the advantages of the proposed architecture over
                  traditional RNN-based approaches?

Naive top-1 chunk (wrong section):
  "...because it may allow the model to extrapolate to sequence lengths
  longer than the ones encountered during training. 4 Why Self-Attention
  In this section we comp..."

Rewrite top-1 chunk (right section):
  "...Recurrent models typically factor computation along the symbol
  positions of the input and output sequences. Aligning the positions to
  steps in computation time, they generate a sequence of hidden states ht...
  This inherently sequential nature precludes parallelization within
  training examples, which becomes critical at longer..."
```

!!! success "Measured: fixes a real miss on a genuinely vague query"
    Asked the vague, colloquial question *"why dont they just use RNNs like everyone else did before"* — the kind of underspecified, non-technical phrasing rewriting exists to fix — naive retrieval **never finds** the passage explaining the model "precludes parallelization," at k=1, 2, or 3. Rewriting reliably finds it instead, at every k tested. (The exact rewritten wording varies run to run — it's a live LLM call — but it consistently lands on the model's technical vocabulary instead of the original colloquial phrasing.)

    The mechanism is visible directly: the raw colloquial phrasing shares almost no vocabulary with the paper's own technical language ("precludes parallelization," "sequential computation"), so its embedding lands far from the answer. Rewriting bridges exactly that gap. But this only works because the query genuinely had a vocabulary-mismatch problem to begin with — rewrite an already-clear, well-formed question (*"How many attention heads did they use?"*) and there's nothing to fix; the rewritten version is just a more formal restatement, and can even add noise (a rewrite might list plausible-but-wrong alternatives alongside the right answer, diluting the embedding's signal).

**Where it has genuinely proven value — conversational rewriting:** in multi-turn chat, a follow-up like *"what about for enterprise?"* is meaningless retrieved on its own — resolving it against conversation history into *"What is the refund policy for enterprise customers?"* is close to essential for any chatbot-style RAG; without it, retrieval on follow-ups silently fails, not just underperforms.

**Use it for:** vague/colloquial queries with real vocabulary mismatch (the case measured above), and conversational follow-ups (near-essential). **Skip it for:** queries that are already clear and specific relative to your corpus's own vocabulary — measure first, don't apply as a blanket "best practice."

## Multi-Query Retrieval & RAG-Fusion

Different from rewriting — rewriting produces *one* improved query; multi-query generates *several* different phrasings of the same question, retrieves each independently, and merges the results. The theory: a single query embedding, however well-phrased, only points to one region of vector space — different phrasings can surface different relevant chunks a single query would miss.

**Merging with RRF (Reciprocal Rank Fusion):** each chunk's fused score is `Σ 1/(k + rank)`, summed across every query variation's ranked list it appears in (`k` is a damping constant, typically 60). A chunk that ranks well across *multiple* phrasings outscores one that only a single phrasing liked — the intuition being that agreement across independent queries is a stronger relevance signal than one strong vote.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/query-transformation/query_transformation_docs.py:multi_query"
```

**Input → output, from an actual run where multi-query found what naive missed:**

```text
Input query:
  What lets this model connect words that are far apart in a sentence
  without losing track over long distances?

Generated paraphrases:
  1. What mechanism allows the model to maintain relationships between
     distant words across long sequences?
  2. How does the model preserve context when processing words that are
     separated by many tokens?
  3. What architectural feature enables the model to avoid degradation
     when linking words with large positional gaps between them?

Naive retrieval at k=5: none of the top 5 chunks mention path length at
all — the raw query's embedding never lands near that passage.

RRF-fused retrieval at k=5, rank 2 (right section, found via paraphrase #2):
  "...dependencies is a key challenge in many sequence transduction tasks.
  One key factor affecting the ability to learn such dependencies is the
  length of the paths forward and backward signals have to traverse in
  the network. The shorter these paths between any combination of
  positions in the input and output sequences, the easier it is to learn
  long-r[ange dependencies]..."
```

!!! success "Measured: fixes a real miss on a broad question — most of the time"
    Asked *"What lets this model connect words that are far apart in a sentence without losing track over long distances?"* — a broad question whose answer could reasonably be phrased several different ways — naive retrieval **never surfaces** the passage explaining "maximum path length" (the mechanism that actually answers this) at k=5, across every run tested. Generating 3 paraphrases and fusing them with RRF found it in 3 of 4 runs.

    That "3 of 4," not "4 of 4," is the honest number: multi-query makes a fresh, unseeded LLM call for its paraphrases every run, so the exact set of phrasings generated — and therefore whether one of them happens to land near the target passage's vocabulary — varies. This is real, disclosed non-determinism, not a bug; it means the fix is meaningfully more likely, not guaranteed. Note that this only helps because the question was genuinely broad and multi-faceted — on a narrow factual question with one unambiguous best-matching chunk, RRF has nothing to add: it can only dilute an already-optimal single match with weaker competing matches from the other paraphrases, never improve on it.

**RAG-Fusion** is the name for this specific combination — multi-query generation plus RRF-based merging — and it's a standard production pattern, but its value proposition is specifically "recall you'd otherwise miss with one query," not "better than the best single query on every question."

**Use it for:** ambiguous or multi-faceted questions where you have real evidence single-query retrieval misses relevant chunks. **Skip it for:** narrow factual lookups the corpus already answers cleanly with the raw query — the N extra retrieval calls (N=3-5 typically) buy nothing there.

## HyDE (Hypothetical Document Embeddings)

The problem: a question is short and interrogative ("What optimizer did they use?"); the actual answer in the document reads as a statement ("We used the Adam optimizer..."). Embedding models sometimes struggle because a question's embedding isn't always closest in vector space to its answer's embedding, even when topically related. HyDE's fix: ask an LLM to write a plausible hypothetical answer — even if it's factually wrong — and embed *that* instead of the raw question. The hallucination is a scaffold, discarded after retrieval; only the real retrieved chunks make it into the final answer.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/query-transformation/query_transformation_docs.py:hyde"
```

**Input → output, from an actual run:**

```text
Input question:
  What BLEU score did they get on English-to-German translation?

Hypothetical passage the LLM generated (this is what actually gets
embedded and searched — not the question above):
  "# BLEU Score for English-to-German Translation
   The neural machine translation system achieved a BLEU score of 27.8
   on the English-to-German translation task using the WMT 2014 dataset.
   This result represented a significant improvement over previous
   state-of-the-art systems at the time..."

Note the hallucinated number (27.8) is WRONG — the paper's real answer is
28.4. That doesn't matter for retrieval: HyDE only needs the passage to
be answer-SHAPED, not answer-correct. It gets discarded after retrieval;
only the real retrieved chunk feeds the final answer.
```

On a different run of the exact same question, the model instead refused to hallucinate:

```text
"I don't have enough context to answer what BLEU score was achieved on
 English-to-German translation. To provide an accurate answer, I would
 need to know..."
```

Embedding that refusal text — which shares no real vocabulary with the paper's own reporting style — missed the target passage at k=1, where naive retrieval succeeded.

!!! note "Measured: unreliable in a specific, disclosed way"
    Asked the numeric-fact question *"What BLEU score did they get on English-to-German translation?"*, repeated across several runs: HyDE mostly **tied** naive retrieval — both find the "28.4" passage at k=1-3, since naive already does well here on the cleaned-up corpus. But in one run, asked to write a hypothetical answer to that same question, the LLM responded *"I don't have enough context to answer this accurately..."* instead of hallucinating a plausible number — and embedding that refusal text actually **missed** the target passage at k=1, where naive succeeded.

    That's the real, underdiscussed tension worth flagging: HyDE's mechanism specifically wants a confident, answer-shaped hallucination as a retrieval scaffold — but well-aligned modern LLMs increasingly refuse to speculate on specific factual claims, which quietly undermines the technique exactly on the crisp numeric-fact questions it's often reached for. On a broader conceptual question instead — *"how does the model track word order without recurrence?"* — the LLM readily wrote a detailed hypothetical explanation every time, no refusal; but naive retrieval was already succeeding there too post-cleanup, so there was no real gap left for HyDE to close either way on this corpus.

**Use it for:** zero-shot/cross-domain retrieval where query style differs a lot from document style, and conceptual/explanatory questions the model will confidently answer rather than refuse. **Skip it for:** crisp numeric-fact questions, where a safety-tuned model may refuse to hallucinate a specific number — defeating the mechanism outright — or corpora where question and answer phrasing are already close, where HyDE adds an LLM call for a gap that isn't there.

## Step-Back Prompting

Many questions are specific instances of a broader principle; retrieving narrowly for the specific question can miss the general background a reasoning-heavy answer actually needs. Step-back prompting asks an LLM to generate a more abstract "step-back" version of the query, retrieves for *both* the general and the specific question, and feeds both sets of context into generation — the only one of these techniques that deliberately broadens scope rather than narrowing or reformulating it.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/query-transformation/query_transformation_docs.py:step_back"
```

**Input → output, from an actual run:**

```text
Input question:         What value did they use for Pdrop during training?
Step-back question:     What hyperparameters were used during training?

Naive retrieval (specific question alone) finds "Pdrop = 0.1" at k=1.
Step-back's combined set (specific + general question, deduped) finds it
at k=1 too — the general question added nothing the specific one hadn't
already found.
```

!!! note "Measured: no effect, even in a scenario built to fit its exact use case"
    Tested against a specific-fact-with-real-general-context question — *"What value did they use for Pdrop during training?"* (step-back generates the broader question *"What hyperparameters were used during training?"* for this one) — retrieval tied naive exactly at every k tested. The same held across several other candidate questions built the same deliberate way (a training-schedule question, an ablation-results question): naive, on its own, already found the specific fact directly every time, leaving nothing for the broader retrieval pass to add.

    The likely reason: this page's sample document is a single, short, well-organized technical paper — general principles and specific facts sit close together throughout (a sentence about a hyperparameter and the paragraph explaining why it was chosen are typically adjacent). Step-back's real value case — think of a general policy rule and a specific year's numbers living in genuinely separate documents or sections far apart, like a company's tax brackets across separate annual filings — doesn't exist in a 15-page paper. A null result here says more about this corpus's structure than about the technique's real-world usefulness: it's a genuine negative result, not a sign the technique doesn't work.

**Use it for:** multi-step reasoning, comparative questions across time/context, or corpora where general principles and specific facts are genuinely far apart (long manuals, regulatory documents, anything spanning multiple sources). **Skip it for:** simple factual lookups, or any corpus small/coherent enough that context and fact already sit together.

## Query Decomposition

Naive RAG embeds the whole query as one vector — if that query is actually a **compound question** bundling multiple distinct information needs, a single embedding represents a blurry average of all of them, and retrieval often surfaces chunks partially matching one part while missing the rest entirely. Decomposition detects this and splits the query into independent sub-questions, retrieving separately for each — different in kind from multi-query, which generates paraphrases of the *same* question rather than splitting into genuinely different ones.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/query-transformation/query_transformation_docs.py:decompose"
```

**Input → output, from an actual run:**

```text
Input question:
  What optimizer did they use, and how many attention heads did they use?

Sub-questions generated:
  1. What optimizer did they use?
  2. How many attention heads did they use?

Naive retrieval (single embedding, k=6): finds "h = 8" (attention heads),
never finds "Adam" (the optimizer) — the blurred embedding favors one
sub-topic and starves the other regardless of budget.

Decomposition (3 chunks retrieved per sub-question, k=6 total): finds
BOTH "Adam" and "h = 8" — each sub-question gets its own dedicated
retrieval pass instead of competing for the same embedding.
```

!!! success "Measured: a clean, real win"
    Asked *"What optimizer did they use, and how many attention heads did they use?"* — a genuinely compound question — naive retrieval misses **one of the two facts even at k=6**: the single blurred embedding keeps favoring one sub-topic (attention heads) over the other (the optimizer) regardless of how much budget it's given. Decomposition, retrieving separately per sub-question, reliably surfaces **both** facts within that same k=6 budget. This is the one technique on this page where a single-fact question (the kind the rest of this page's questions are built from) couldn't demonstrate the real value case at all — decomposition correctly detects a non-compound question and leaves it alone, tying naive exactly — so a genuinely compound question was necessary to show what it's actually for.

**Use it for:** comparison questions, multi-part "and" questions, anything needing information from genuinely different sections the single embedding would never combine well. **Skip it for:** single-fact lookups — the decompose-detection call plus N separate retrieval passes cost real latency for a question that wasn't compound in the first place.

## Multi-Hop Query Planning (IRCoT)

Decomposition splits a compound question upfront, all at once. **IRCoT** (Interleaved Retrieval with Chain-of-Thought) does something more dynamic: generate one reasoning step, use *that step itself* as the next retrieval query, retrieve, and repeat — each hop informed by what's actually been found so far, rather than a fixed plan made before any retrieval has happened. This targets genuinely multi-hop questions where the second lookup depends on what the first one turns up, not just "two unrelated facts asked in one sentence."

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/query-transformation/query_transformation_docs.py:ircot"
```

**Getting this right took two real fixes, not just a clean write-up:** the first version only showed the model its own prior reasoning sentences at each hop, never the actual retrieved chunk text — with no real evidence to look at, the model had no way to judge whether it actually knew enough, and just kept generating near-identical restated queries forever. The fix was mechanical but essential: feed the real retrieved passages back into context at every hop, and add a forced final-synthesis call at the end so the loop always commits to a real answer instead of trailing off in a hedge.

!!! success "A real success — sequential hops correctly identifying what's still missing"
    Asked *"What optimizer did they use, and how many attention heads did they use?"*, one run went: **Hop 1** — "What optimizer was used in the model training?" (finds Adam). **Hop 2** — "I need to find information about the number of attention heads" (finds h=8). **Hop 3** — "ANSWER: They used the Adam optimizer... and h=8 parallel attention heads." Final synthesis correctly combined both, fully grounded. Each hop's query was shaped by what the *previous* hop had (and hadn't) turned up — genuinely adaptive, not a fixed script.

!!! warning "A real failure — and it isn't IRCoT's fault"
    Asked to find which company an author (credited in an informally-written footnote) is affiliated with, IRCoT failed after 3 hops of near-identical restated queries, and — correctly — reported it couldn't find the answer rather than guessing. Checking why: **a single, perfectly-phrased direct query for that exact fact also fails to find it in the top-5**, with no query transformation involved at all. The footnote text embeds poorly with a general-purpose dense model regardless of how the query is phrased. No amount of reasoning-and-retrieval interleaving can chain together a step that individually isn't findable — this is the same lesson [FLARE-style adaptive retrieval](retrieval-methods.md#adaptive-retrieval-a-flare-style-approach) already surfaced: query strategy controls *how* and *when* you search, not what the underlying embedding model is fundamentally capable of representing.

**Use it for:** genuinely multi-hop questions where a later lookup depends on an earlier one's answer, and where the individual facts involved are realistically findable by *some* phrasing. **Skip it for:** questions that are really just several independent facts (decomposition is cheaper and equally effective there), or corpora where the specific fact needed is known to be poorly represented by the embedding model regardless of query strategy — no amount of reasoning fixes that ceiling.

## The pattern across all six

| Technique | Measured effect, on a scenario matching its use case | Real value case |
|---|---|---|
| Query rewriting | Fixes a real miss (vague query) | Vague/colloquial queries, conversational follow-ups |
| Multi-query/RAG-Fusion | Fixes a real miss, in ~3 of 4 runs | Ambiguous queries where single-query retrieval demonstrably misses chunks |
| HyDE | Unreliable — ties normally, can hurt on numeric facts | Cross-domain retrieval, question/answer phrasing mismatch |
| Step-back prompting | No effect, even scenario-matched | Reasoning across genuinely separated general/specific content |
| Query decomposition | Clean win (on a compound question) | Genuinely compound, multi-part questions |
| IRCoT | Succeeds when facts are individually findable | Multi-hop questions with real sequential dependency |

Two clean wins, one technique whose unreliability is itself the honest finding, one genuine null result, and one technique whose success is entirely conditional on the underlying fact being findable at all — out of six techniques that are all, individually, real and well-established. The thread connecting all six: every one of them targets a *specific* failure mode in how a query relates to your corpus. Before reaching for any of them, ask which failure mode your own queries actually have — vague phrasing, a broad multi-faceted question, a question/answer style mismatch, missing general context, a compound question, or a genuinely multi-hop one. Apply the technique that matches; skip the rest. The same discipline applies to reranking and chunking strategy elsewhere on this site — measure against your own data before adding any technique to a pipeline.

## Scenario Check

Six scenarios testing whether you can match a query symptom to the right technique — and whether you know when a technique's absence of effect is itself informative.

<div class="quiz-widget" data-title="Scenario Check: Advanced RAG — Pre-retrieval">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team adds LLM-based query rewriting to their RAG pipeline as a blanket \"best practice\" applied to every incoming query, without first checking whether their actual queries suffer from vocabulary mismatch. Measured end-to-end, retrieval quality gets slightly WORSE, not better.",
      "question": "What's the most likely explanation, based on the measured finding on this page?",
      "options": [
        "The rewritten queries likely picked up synonyms and phrasing not present anywhere in the corpus, pulling the embeddings away from the document's actual vocabulary rather than toward it.",
        "Their queries were likely already clear and well-phrased, so rewriting had no real vocabulary-mismatch problem to fix and only added verbosity/noise to embeddings that were already good.",
        "The rewriting step probably used a different embedding model than the one indexing the corpus, so the rewritten query's vector landed in a different space than the stored chunk vectors.",
        "Since the rewrite comes from a live, unseeded LLM call, the regression is most likely just run-to-run noise rather than a pattern tied to the query type."
      ],
      "correct": 1,
      "explanations": [
        "This inverts the actual mechanism shown on this page: rewriting's demonstrated failure mode is adding noise to queries that were already well-matched to the corpus's vocabulary, not the rewrite drifting toward unrelated vocabulary — in the worked example here, rewriting moves toward the document's own technical phrasing, not away from it.",
        "Correct. This is exactly the measured pattern: rewriting an already-clear query has nothing to fix and can only add noise (more verbose phrasing, occasionally even introducing wrong candidate terms). Blanket application without checking for the actual target failure mode is the root cause.",
        "This describes a real but different failure mode (an embedding-space mismatch from using inconsistent models) — nothing on this page indicates that happened here. The measured cause is specific to applying rewriting where there was no vocabulary-mismatch problem to fix in the first place, not a model-mismatch bug.",
        "Run-to-run LLM variance is real elsewhere on this page (multi-query, HyDE), but the rewriting regression described here is diagnosed as a reproducible pattern tied to which queries get rewritten (already-clear ones), not noise from one unlucky generation — inspecting the actual rewrites confirms the same verbose, unhelpful pattern recurs on already-clear queries specifically."
      ]
    },
    {
      "scenario": "A team generates 3 paraphrases of each query and fuses results with RRF, expecting this to always outperform or match single-query retrieval. On one specific question, the original single query already retrieved the perfect chunk at rank 1 — after adding multi-query/RRF, that chunk drops out of the top-3 entirely.",
      "question": "What causes this specific kind of regression?",
      "options": [
        "RRF recomputes each chunk's relevance from scratch using its text content, so a chunk's original rank position doesn't actually factor into its fused score.",
        "RRF rewards a chunk's CONSISTENCY across multiple ranked lists, not its PEAK rank in any single list — so when the original phrasing was already optimal, paraphrase variants can only dilute that peak match with their own weaker competing matches, never improve on it.",
        "The paraphrases likely drifted semantically far enough from the original question that RRF fused in chunks about a different topic entirely.",
        "Generating more paraphrases than 3 would fix this, since more independent votes make the fused ranking converge back to the single best chunk."
      ],
      "correct": 1,
      "explanations": [
        "RRF's formula (Σ 1/(k + rank)) is purely a function of rank position across lists — it never re-reads chunk content at all; the real inherent property is that it sums across lists, rewarding cross-list consistency over a single best position.",
        "Correct. This is precisely the measured mechanism: RRF's score is a sum across lists, rewarding chunks that rank reasonably well everywhere over one chunk that ranked #1 in just one list — which can actively work against you when the single original query was already the best possible phrasing.",
        "The measured case involved on-topic, reasonable paraphrases — the regression is a structural property of how RRF aggregates ranks across lists (diluting a peak match), not a symptom of the paraphrases being off-topic.",
        "More paraphrases add more competing ranked lists to sum over, which can dilute the peak match further rather than restoring it — the dilution is inherent to how RRF sums across lists, not a symptom of too few paraphrases."
      ]
    },
    {
      "scenario": "A HyDE-based retrieval system is asked a specific factual question. On one run, the LLM generates a hypothetical answer and retrieval finds the right chunk. On a second run of the EXACT SAME question, the LLM instead responds \"I don't have enough context to answer this accurately\" and retrieval performs differently.",
      "question": "What does this demonstrate about HyDE specifically?",
      "options": [
        "The vector database must be returning stale cached results from an earlier index update, since the same underlying document didn't change between runs.",
        "HyDE's result isn't perfectly reproducible run-to-run, because the hypothetical answer comes from a live, unseeded LLM call each time — and a well-aligned model refusing to speculate on a specific fact can undermine HyDE's actual mechanism, which specifically depends on getting a confident, answer-shaped hallucination.",
        "Since HyDE's output depends on a fresh LLM call, it should be replaced with a fixed template hypothetical answer to guarantee reproducible retrieval.",
        "Since HyDE discards the hypothetical answer after retrieval and never shows it to the user, its exact wording has no bearing on which chunk actually gets found."
      ],
      "correct": 1,
      "explanations": [
        "The variability here comes from the LLM's hypothetical-answer generation being a fresh, unseeded call each run — chunk embeddings and the vector index are static and weren't touched between runs, so there's no cache/staleness bug to diagnose.",
        "Correct. This is exactly the measured, disclosed finding: HyDE's hypothetical-answer generation is a fresh, unseeded LLM call every time, so results vary run to run — and a model that refuses to hallucinate a plausible-but-possibly-wrong answer (increasingly common, safe behavior) directly undermines the technique's core mechanism, which needs that confident hallucination as a retrieval scaffold.",
        "A fixed template would remove exactly the mechanism that makes HyDE work — an LLM inferring a plausible, topically-specific hypothetical answer close to the document's real phrasing. The real, disclosed trade-off on this page is that non-determinism is a property to be aware of, not something to engineer away by discarding the LLM step itself.",
        "It's true the hypothetical text is discarded after retrieval and never reaches the final answer — but that's irrelevant to this question: the hypothetical's wording IS what gets embedded and searched, so a refusal instead of a confident hallucination changes the embedding entirely, which is exactly why retrieval performance differed between the two runs."
      ]
    },
    {
      "scenario": "A team tests step-back prompting against their RAG system and finds it produces identical retrieval results to naive RAG on every test question, across a fairly short, well-organized internal wiki page.",
      "question": "What's the most defensible interpretation of this null result?",
      "options": [
        "Step-back prompting fixes the same class of problem as query decomposition — splitting a question into parts — so this null result just means the internal wiki page didn't contain any compound questions.",
        "A short, well-organized document likely keeps general context and specific facts close together already, so there's no genuine gap between \"broad principle\" and \"specific fact\" for step-back's broader retrieval pass to bridge — the null result reflects this corpus's structure, not necessarily the technique's real-world value.",
        "The step-back question generation likely produced overly broad questions that retrieved a completely different, off-topic set of chunks each time, diluting the results.",
        "Step-back prompting's real value case requires the general context and the specific fact to live in literally different source documents, not just different sections of the same document."
      ],
      "correct": 1,
      "explanations": [
        "This conflates step-back prompting with query decomposition. Step-back retrieves for a broader, more general version of the same question to surface missing context — it doesn't split a question into separate sub-parts the way decomposition does. Nothing about this scenario's compound-question status is at issue; it's about whether general context sits separately from the specific fact.",
        "Correct. This is exactly the reasoning applied on this page: a short, coherent document naturally keeps related general and specific content close together, so there's nothing for step-back's broader retrieval pass to uniquely surface that narrow retrieval wouldn't already find — the null result is informative about the corpus, not proof the technique lacks value elsewhere.",
        "Retrieval tied naive exactly — the same specific-fact chunk was found either way — which is inconsistent with the step-back questions pulling in an unrelated, diluting set of chunks. The step-back question generation worked as intended (as shown in the worked example: \"What hyperparameters were used during training?\"); it simply had no unique gap left to fill on this corpus.",
        "The page's real value case includes content \"far apart\" within one large document (e.g., long manuals or regulatory documents), not only content split across literally separate documents — restricting step-back's usefulness to the multi-document case only is narrower than what the page actually describes."
      ]
    },
    {
      "scenario": "A team has a genuinely compound question — \"What's the refund policy AND what's the warranty period?\" — and naive RAG's single embedding retrieves chunks mostly about refunds, missing warranty information even when given a generous top-8 results budget.",
      "question": "Which technique directly and reliably fixes this specific failure mode, and why?",
      "options": [
        "HyDE, since the hypothetical answer it generates would be a fuller, more detailed passage that's more likely to overlap with both the refund and warranty sections.",
        "Query decomposition, since splitting into two independent sub-questions and retrieving separately for each guarantees both topics get a dedicated, independent retrieval pass rather than competing for space in one blurred embedding.",
        "Step-back prompting, since asking a broader version of the question — like \"What are this company's customer policies?\" — would retrieve more general chunks likely to cover both topics.",
        "Multi-query/RAG-Fusion, since generating 3-5 paraphrased versions of the compound question and fusing the results with RRF would surface chunks from both topics across the different phrasings."
      ],
      "correct": 1,
      "explanations": [
        "A single hypothetical passage is still one embedding representing whatever blend of topics the LLM chose to write about — a longer, more detailed hypothetical doesn't structurally guarantee both refund and warranty content are represented any more than the original compound query does; it may still lean toward the topic it happened to elaborate on.",
        "Correct. This is precisely decomposition's mechanism and exactly the measured clean-win case on this page: splitting into \"what's the refund policy\" and \"what's the warranty period\" as independent sub-questions guarantees each gets its own dedicated retrieval pass, rather than both competing for representation within one averaged embedding.",
        "Even a broader step-back question retrieves for one additional query, still not guaranteeing both distinct topics are separately represented — it's aimed at surfacing missing general context around a single specific fact, not at guaranteeing dedicated retrieval for each of several genuinely separate topics the way decomposition's explicit splitting does.",
        "Multi-query generates different phrasings of the same combined question — each paraphrase still bundles both refund and warranty together in one embedding, so RRF is fusing several blended embeddings, not independent, dedicated retrieval passes for each topic the way decomposition provides."
      ]
    },
    {
      "scenario": "An IRCoT-style multi-hop retrieval system is asked a question requiring two sequential lookups. After 3 hops of reasoning and retrieval, the system honestly reports it cannot find the answer, rather than generating a plausible-sounding but unverified response. Investigation reveals that even a single, perfectly-phrased direct query for the needed fact also fails to retrieve it in the top-5 results.",
      "question": "What does this specific outcome demonstrate about IRCoT's real limitation?",
      "options": [
        "Decomposition would have succeeded here where IRCoT failed, since decomposition retrieves for each sub-question independently instead of depending on prior hops.",
        "IRCoT's reasoning-and-retrieval loop can only chain together retrieval steps that are individually findable by the underlying embedding model — if a specific fact is poorly represented in the embedding space regardless of query phrasing, no amount of query reformulation or reasoning can retrieve it, the same ceiling FLARE-style adaptive retrieval already demonstrated.",
        "Running a few more hops would likely have eventually surfaced the fact, since each additional hop gives the reasoning loop another chance to phrase the query differently.",
        "Since this specific fact could not be retrieved even after 3 hops of multi-hop reasoning, multi-hop techniques are less reliable than direct single-query retrieval for facts embedded in informally-written text."
      ],
      "correct": 1,
      "explanations": [
        "Decomposition still retrieves using a direct query for each sub-question — and the page confirms even a single, perfectly-phrased direct query for this exact fact fails in the top-5. Decomposition would face the identical retrieval ceiling here; the failure isn't about IRCoT's interleaving design, it's about what the embedding model can represent at all.",
        "Correct. This is exactly the demonstrated finding, and it directly parallels the FLARE lesson elsewhere on this site: any query-transformation or reasoning technique built on top of a retrieval method is capped by what that retrieval method can actually find — no amount of clever querying fixes a fact the embedding model fundamentally represents poorly.",
        "Since a single, perfectly-phrased direct query for this fact also fails to retrieve it in the top-5, the bottleneck is the embedding representation of that chunk itself, not an insufficient number of phrasing attempts — more hops generating more query phrasings wouldn't change what the embedding model can represent in the first place.",
        "Both approaches failed on this exact fact — the single, perfectly-phrased direct query failed too, so this case doesn't show single-query retrieval outperforming multi-hop reasoning here. It's one diagnosed failure tied to one poorly-embedded fact, not evidence against multi-hop techniques generally — this same page shows IRCoT succeeding cleanly when the needed facts are each individually findable."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [RAG-Fusion: a New Take on Retrieval-Augmented Generation](https://arxiv.org/abs/2402.03367) — multi-query + RRF
- [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) — the original HyDE paper
- [Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models](https://arxiv.org/abs/2310.06117) — the original step-back prompting paper
- [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://arxiv.org/abs/2212.10509) — the original IRCoT paper
