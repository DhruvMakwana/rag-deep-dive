# Modular RAG

Every technique on the [pre-retrieval](advanced-rag-pre-retrieval.md) and [post-retrieval](advanced-rag-post-retrieval.md) pages is an individual fix bolted onto a fixed retrieve → generate pipeline. Modular RAG is a step up in abstraction: treat indexing, pre-retrieval, retrieval, post-retrieval, and generation as **independent, swappable modules** that different queries can move through differently, rather than one path everyone takes. [Adaptive RAG](tutorials/advanced-rag-from-scratch.md), [Self-RAG](self-rag.md), [Corrective RAG](corrective-rag.md), and [Agentic RAG](agentic-rag.md) are all specific instances of this idea — each one is "modular RAG plus a particular way of deciding which module runs."

Every one of those routers, though, makes its decision with an LLM call. This page asks the question none of them tested: **does routing actually need one?**

!!! example "Hands-on"
    The full 3-way comparison below is runnable, against real data: [**Modular RAG →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/modular-rag) in the code repo. Two of the three routers need no LLM key at all.

??? abstract "TL;DR — quick revision"
    - **The same routing decision, built 3 ways**: keyword heuristics (no model call), embedding-based semantic routing (one local embedding call, no LLM), and the exact LLM router already built in [`adaptive-rag/`](https://github.com/DhruvMakwana/rag-cookbook/tree/main/adaptive-rag)
    - **Measured on two separate sets, deliberately**: a "design set" (the queries the heuristic rules were written looking at) and a genuinely held-out set (written *without* looking at those rules) — because a router's accuracy on the queries it was tuned against tells you almost nothing about how it'll do on the next query someone actually asks
    - **The heuristic router's accuracy collapses from 1.00 to 0.40** moving from the design set to held-out phrasing — real, measured brittleness, not a hypothetical caveat about rule-based systems
    - **The embedding-based router is the more honest number**: 0.67 → 0.80 across the same two sets — roughly consistent, because it was never fit to either one
    - **The LLM router is the only one both accurate and consistent everywhere** — 1.00 on both sets — at roughly 30-1500x the latency of the other two, and a real per-call cost they don't have

## What "modular" actually means here

A modular RAG system separates the pipeline into pieces that can each be swapped independently:

- **Indexing** — chunking/embedding strategy, potentially different per document type
- **Pre-retrieval** — rewriting, HyDE, decomposition, step-back — chosen per query, not applied blindly to every query (the entire lesson of the [pre-retrieval page](advanced-rag-pre-retrieval.md))
- **Routing** — deciding which retrieval path or index to use at all
- **Retrieval** — sparse, dense, or hybrid, against whichever index routing picked
- **Post-retrieval** — reranking, compression, applied conditionally
- **Generation** — a cheap model for simple lookups, an expensive one for reasoning-heavy questions

The **routing module** is the piece this page focuses on, because it's the one every other advanced technique on this site quietly assumes works, without ever measuring the cheaper alternatives to the LLM call each one reaches for by default.

## The same decision, three ways

All three routers solve the identical problem: classify a query into one of 5 categories (`CLEAR`, `VAGUE`, `COMPOUND`, `BROAD`, `NEEDS_CONTEXT`) — the same categories [Adaptive RAG](tutorials/advanced-rag-from-scratch.md#3-the-router) uses to pick a pre-retrieval technique.

### Heuristic routing — pattern-matching, zero cost

The cheapest possible option: a handful of keyword and pattern rules, evaluated in plain Python with no model call at all. Fast by construction — there's no inference step to be slow.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/modular-rag/modular_rag_docs.py:heuristic"
```

### Embedding-based semantic routing — one local embedding call, no LLM

A middle ground: write a few example utterances per category once, embed them (offline, one-time cost), and average them into a centroid vector per category. At query time, embed the incoming query and route to whichever centroid it's closest to by cosine similarity — no LLM call anywhere in this path, just the same embedding model already used for retrieval.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/modular-rag/modular_rag_docs.py:embedding"
```

### LLM-based routing — the router already built in adaptive-rag/

The most expensive, most flexible option, reused directly rather than rebuilt: the exact same prompt and parsing logic from [`adaptive-rag/`](tutorials/advanced-rag-from-scratch.md#3-the-router), included here specifically as the accuracy baseline the other two get measured against.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/modular-rag/modular_rag_docs.py:llm_router"
```

## Measuring all three honestly — and why two test sets, not one

Testing a router only on the queries used to design it answers the wrong question. A heuristic router's keyword rules are, almost by definition, going to match the exact phrasings someone had in mind while writing them — a 100% score there proves the rules were written correctly, not that they generalize. So this comparison uses **two separate sets**: a 9-query "design set" (the same queries [`adaptive-rag/`](tutorials/advanced-rag-from-scratch.md#3-the-router)'s router was originally validated on) and a 5-query **held-out set, written without looking at the heuristic router's actual keyword list** — the fair test of whether a router generalizes, not just fits.

| Router | Design-set accuracy | Held-out accuracy | Avg latency |
|---|---|---|---|
| Heuristic | 1.00 | **0.40** | ~0ms |
| Embedding | 0.67 | 0.80 | ~50ms |
| LLM | 1.00 | 1.00 | ~1500ms |

!!! warning "The headline number to distrust is the heuristic's 1.00"
    On the design set, the heuristic router gets every single query right — a keyword like `"dont"` or `"and"` matching exactly the patterns it was built around. Tested on the held-out set — *"What is the batch size, plus what hardware did they train on?"*, *"What accounts for this model generalizing so well to new domains?"*, *"What dropout probability was applied?"* — accuracy collapses to **2 out of 5**. Real questions people actually ask don't reliably contain the exact keywords a rule-writer anticipated; this is measured brittleness, not a hypothetical caveat about heuristics in general.

!!! success "The embedding router's number is the more trustworthy one"
    0.67 on the design set, 0.80 on held-out — genuinely similar, because the embedding router was never fit to either set; its example utterances were written independently of both. Its actual confusion pattern is consistent and diagnosable too: it repeatedly misroutes distinctly-phrased `COMPOUND` questions to `NEEDS_CONTEXT` — a real, specific weakness in how compound-question phrasing sits in embedding space relative to the example utterances chosen for `NEEDS_CONTEXT`, not a mysterious black box.

!!! note "The LLM router costs what you'd expect for being right everywhere"
    ~1500ms average per classification call, against ~0ms and ~50ms for the other two, plus a real per-call API cost neither of the others has at all. It earns that cost here by being the only router that's both accurate *and* consistent across both test sets — but that's specifically the trade-off worth measuring before defaulting to an LLM call for every routing decision in a pipeline, the same way every earlier page on this site has argued against applying any technique unconditionally.

## Concrete example — a customer support system

A modular pipeline doesn't have to use the same router (or even the same routing *mechanism*) for every decision it needs to make:

- *"What's your refund policy?"* → cheap, high-confidence category (a simple FAQ lookup) → **heuristic or embedding routing** is fine here, since the cost of an occasional misroute is low and the volume is high
- *"Compare enterprise vs. individual plan refund timelines across regions"* → a genuinely ambiguous, high-stakes routing decision (get this wrong and the whole downstream pipeline — decomposition, hybrid retrieval, reranking — runs on the wrong plan) → **the LLM router's cost is worth it here**

Same system, two different routing mechanisms, chosen per decision based on how much a misroute actually costs — the same idea as choosing generation model size per query, applied one level upstream.

## Connection to everything else on this site

Self-RAG and Corrective RAG are modular RAG systems where the orchestration module includes a self-evaluation/correction loop. Agentic RAG goes one step further — an LLM *is* the orchestration module, making the routing/looping decision itself at runtime rather than following a rule someone wrote in advance, whether that rule is a keyword, a cosine-similarity threshold, or a fixed if/else branch. All three are the same underlying idea (modules, chosen dynamically) with a different mechanism deciding which module runs.

## Use it for / skip it for

**Use LLM-based routing for:** decisions where being wrong is expensive and query phrasing is genuinely unpredictable — the accuracy/consistency this page measured is worth the latency there. **Use heuristic or embedding-based routing for:** high-volume, well-defined categories where the cost of an occasional misroute is low, and where you can actually validate on a real held-out set before trusting the numbers — not just the queries used to build the rules.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: Modular RAG">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team builds a keyword-based heuristic router and tests it only on the same 9 example queries they used while writing its rules. It scores 100% and they conclude the router is production-ready.",
      "question": "What's the most important gap in this validation, based on what this page measured?",
      "options": [
        "There is no gap — 100% accuracy is 100% accuracy, and a larger test set would just be more of the same confirmation.",
        "Testing only on the same queries the rules were written against can't distinguish 'this router generalizes well' from 'these rules were fit to match these exact examples' — a genuinely held-out set (written without looking at the rules) is needed to measure real-world accuracy, and this page's own held-out test showed accuracy collapsing from 1.00 to 0.40.",
        "The router should have been tested with a larger LLM model, since the size of the test set does not affect what a heuristic router can measure.",
        "The gap is that a 9-query test set is simply too small to draw conclusions from — the fix is running the same design-set queries at a much larger scale before trusting the 100% score."
      ],
      "correct": 1,
      "explanations": [
        "A 100% score on the exact queries used to design the rules is close to circular — it mostly confirms the rules were written correctly for those queries, not that they'll handle a query nobody anticipated while writing them.",
        "Correct. This is exactly the distinction this page's methodology draws, and the measured collapse (1.00 -> 0.40 on held-out phrasing) is the concrete evidence for why it matters — a design-set score alone is not a reliable estimate of real-world accuracy.",
        "This isn't about LLM model size at all — heuristic routing involves no model call whatsoever; the issue is entirely about test set design (design-set-only vs. genuinely held-out).",
        "Running more queries drawn from the same design process wouldn't fix this — the flaw isn't sample size, it's that every query in that set still comes from the same source as the rules themselves. A larger design set would still score artificially high for the same reason; only a genuinely independent held-out set (like this page's 5-query one) tests generalization."
      ]
    },
    {
      "scenario": "A team needs to route two different kinds of decisions in their RAG pipeline: (1) whether an incoming FAQ question is about billing, shipping, or returns (high volume, low cost of an occasional misroute) and (2) whether a complex enterprise query needs a multi-step decomposition-and-verification pipeline before generation (low volume, high cost if misrouted).",
      "question": "Based on this page's measured trade-offs, which routing approach fits each decision best?",
      "options": [
        "Use the LLM router for both decisions, since the page's own results show it as the most accurate router on both test sets — once a router is proven accurate, latency and per-call cost become implementation details to optimize separately, not factors that should change which router gets used.",
        "Use heuristic or embedding-based routing for the high-volume FAQ categorization (low cost of an occasional error, needs speed/cost efficiency at scale) and the LLM router for the complex enterprise query (low volume, high cost of a wrong routing decision, worth paying the accuracy/consistency premium).",
        "Use heuristic routing for both, since the held-out set that showed its accuracy collapsing to 0.40 was only 5 queries — too small a sample to outweigh heuristic routing's zero-cost, zero-latency advantage for either decision.",
        "Routing mechanism choice is irrelevant to cost of error or query volume — the only consideration is which mechanism is easiest to implement."
      ],
      "correct": 1,
      "explanations": [
        "The page frames accuracy, latency, and cost as a trade-off to weigh per decision, not a hierarchy where accuracy always wins and the rest are secondary — for the high-volume FAQ decision, the ~30-1500x latency difference is exactly why a less-accurate-but-cheaper router is the better fit despite being less accurate.",
        "Correct. This mirrors the customer-support example on this page directly: match the routing mechanism to how much a misroute actually costs and how much accuracy the specific decision genuinely needs, rather than using one mechanism for everything.",
        "The page doesn't treat that held-out result as sample noise — a collapse from 1.00 to 0.40 is a large, directionally consistent effect, and the same brittleness would apply directly to the complex enterprise query, where a wrong route is far more costly than for FAQ categorization.",
        "This page's central methodology is explicitly measuring cost of error, volume, and accuracy/latency trade-offs as the deciding factors — implementation ease is not what's being evaluated here."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks](https://arxiv.org/abs/2407.21059) — the paper formalizing this modular framing
- [Adaptive RAG](tutorials/advanced-rag-from-scratch.md) — the LLM router this page measures against, and where the 9-query design set comes from
- [semantic-router](https://github.com/aurelio-labs/semantic-router) — an established open-source library implementing embedding-based routing, if you want a maintained implementation rather than the from-scratch version on this page
