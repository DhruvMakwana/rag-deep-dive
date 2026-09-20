# PageIndex

Every technique on this site so far relies on either embeddings and vector similarity search, or precomputed context caching. PageIndex rejects the vector-similarity premise entirely, for a specific class of documents: instead of chunking and embedding, it turns a document's actual structure — headers, sections, table of contents — into a tree, and has an LLM navigate that tree to find the right section. The idea is closer to how a person finds an answer in a long report: flip to the right chapter based on the document's own organization, not scan every page for text that looks similar to the question.

!!! note "Scope: what this page actually measures"
    Vector RAG's edge on plain, shallow documents (this site's usual "Attention Is All You Need" sample) is real — that paper has ~7 top-level sections and barely any sub-numbering, which gives a tree-navigation approach almost nothing to navigate. This page uses a different sample document on purpose: NIST's **"Artificial Intelligence Risk Management Framework (AI RMF 1.0)"**, a real, free, 48-page regulatory document with genuine 4-level structure (Part → Section → Subsection → numbered category). It's the kind of document this technique is actually built for, and using it here is what makes the comparison below meaningful rather than favorable.

!!! example "Hands-on"
    The full comparison below is runnable, with real API calls: [**PageIndex →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/pageindex) in the code repo, using the real open-source `pageindex` package. Needs an Anthropic API key; retrieval and generation both run through it.

??? abstract "TL;DR — quick revision"
    - **No embeddings, no vector database**: a document's real structure becomes a tree; an LLM reasons over section titles and summaries to decide where to look, then pulls the actual text from just those sections
    - **A different sample document than every other page on this site, deliberately**: the usual paper's shallow structure wouldn't exercise this technique's actual advantage, so this page uses a genuinely deep, 4-level-structured regulatory PDF instead
    - **Measured directly: PageIndex 5/5 correct with accurate page citations, naive text RAG 3/5**, on a 5-question eval set built specifically for this document
    - **The two naive-RAG misses have a specific, checked cause**: embedding similarity retrieved generically similar-sounding prose elsewhere in the document instead of the exact numbered subsection or exact appendix text — precisely the failure mode this technique targets
    - **The real cost**: each PageIndex query here took 6-12 seconds (an actual reasoning loop over the tree), against naive RAG's well under a tenth of a second (one cosine-similarity lookup) — a real latency trade-off for the accuracy gain, not a free win

## How it works

**Tree construction.** A layout parser extracts the document's actual structure — headings, table of contents — directly, without an LLM, where possible. An LLM is used for two smaller things: confirming a detected heading really starts where the parser thinks it does, and writing each section's summary. For the 48-page sample document here, indexing took about a minute and cost a few cents.

**Retrieval and generation, together.** Given a question, the chat model is handed the tree — section titles and summaries, not full text — reasons about which section(s) look relevant, pulls the actual text for just those sections, and answers, looping back for more if the first ones weren't enough. This is an agentic tool-calling loop under the hood, not a single classification call: the model can go deeper into a promising branch, back out of an unpromising one, and only reads full section text once it has a specific place to look.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/pageindex/pageindex_rag_docs.py:index_document"
```

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/pageindex/pageindex_rag_docs.py:pageindex_search"
```

Compared directly against the naive text RAG baseline used elsewhere on this site — extract text, chunk it, embed with a dedicated text embedder, retrieve by cosine similarity:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/pageindex/pageindex_rag_docs.py:naive_text_search"
```

## Measured: PageIndex 5/5, naive text RAG 3/5

A 5-question eval set specific to this document — each keyword checked directly against the real page text, not guessed:

```text
                                                              Naive   PageIndex (pages cited)
What is Risk Tolerance adapted from?                          PASS    PASS  [12]
How is GOVERN described relative to other functions?          PASS    PASS  [25,26,27]
What does MEASURE 2.7 evaluate?                                FAIL    PASS  [35]
Per Appendix B, are AI risks comprehensively addressed?        FAIL    PASS  [43,44]
Is it easy to tell if an opaque AI system is fair/secure?      PASS    PASS  [21]
```

Every PageIndex page citation was checked against the document's real tree structure, and each one is accurate — it navigated to the actual section containing the answer, not just a plausible-sounding one.

!!! success "Why the two naive-RAG misses happened, checked directly"
    For *"What does MEASURE 2.7 evaluate?"*, the top-3 retrieved chunks were a nearby MAP-function table and generic test/evaluation/validation process text — semantically close to the query's own vocabulary ("measure", "evaluate"), but not the actual MEASURE 2.7 entry. The same pattern held for the Appendix B question: the retrieved chunks were generic risk-management prose from elsewhere in the document, not Appendix B's specific language. In both cases, embedding similarity pulled toward text that *sounds* like the query rather than the document's own structural answer. PageIndex doesn't have this failure mode by construction — it isn't comparing the query's wording against candidate text at all, it's reasoning about which section of the document's own outline should contain the answer.

!!! warning "The real cost: latency, not measured here in dollars"
    Retrieval and generation happen as one agentic call in PageIndex, and the library doesn't expose a clean per-call token/cost breakdown the way a raw API call does (see the [CAG](cag.md) page for that kind of accounting on a different technique). What is directly measured: each PageIndex query here took 6-12 seconds, against naive RAG's well under a tenth of a second — a real difference, since PageIndex is running an actual multi-step reasoning loop over the tree rather than one embedding lookup. That latency (and the real, if unmeasured, extra token cost of the agentic loop) is the genuine price of the accuracy gain above, not a side effect to ignore.

## Scaling beyond one document

PageIndex's method targets long, individually-structured documents — but a knowledge base of many such documents (a library of contracts, a shelf of regulatory filings) doesn't automatically extend to "build one giant tree over everything." Each document keeps its own tree; picking which document to search within first is a separate, coarser routing problem — the same idea as choosing which retrieval path to take, covered on the [Modular RAG](modular-rag.md) page, or which cached document to load, covered on the [CAG](cag.md#scaling-beyond-one-document-rag-over-cag) page. This page's own comparison is deliberately about retrieval *within* one document, where PageIndex's structural-navigation advantage is most directly testable.

## When to use which

| | Naive vector RAG | PageIndex |
|---|---|---|
| Best fit | Loosely structured or short documents — chat logs, tickets, memos with no real hierarchy | Long, well-structured professional documents — contracts, filings, regulatory/technical manuals with genuine section structure |
| What it needs | An embedding model, a vector index | Just the document's own structure (headers, TOC) plus an LLM |
| Per-query cost | One embedding + one similarity search — fast and cheap | An LLM reasoning loop over the tree — measured here at 6-12s per query, real token cost beyond that |
| Failure mode | Retrieves text that *sounds* like the query, missing the document's own exact answer (measured directly above) | Needs genuine exploitable structure — a structureless document gives it nothing to navigate |
| Explainability | An opaque similarity score | A page-citation trail showing exactly which sections were used, checkable against the document's real structure |

**Use PageIndex for:** long, genuinely structured professional documents, especially where "why did the system think the answer was here" needs a real answer for compliance or trust reasons, not just a similarity score. **Use vector RAG for:** large corpora of short or loosely-organized documents, or anywhere per-query latency and cost need to stay minimal. **Neither cleanly wins** at synthesis questions spanning many unrelated sections at once — that's a closer fit for the relational, multi-source problem [Graph RAG](graph-rag.md) is built for.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: PageIndex">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team builds a PageIndex-based retrieval system, testing it against the 'Attention Is All You Need' paper (roughly 7 top-level sections, at most one level of sub-numbering). They find it performs about the same as their existing vector RAG system and conclude PageIndex offers no real benefit.",
      "question": "Based on this page's own scoping choice, is this a fair test of PageIndex?",
      "options": [
        "Yes -- since this page's own scoping note admits vector RAG's edge is real on plain, shallow documents, that means vector RAG is the generally stronger technique and PageIndex's advantage is a narrow special case that doesn't generalize much beyond regulatory filings.",
        "No -- this page explicitly avoided that exact paper as a demo document for this reason: PageIndex's advantage depends on genuine, exploitable document structure, and a shallow ~7-section paper gives a tree-navigation approach almost nothing to navigate, so vector RAG doing just as well there is expected, not evidence against the technique generally.",
        "Yes, because PageIndex's tree-navigation approach was specifically built for AI-regulation documents like the NIST AI RMF framework, and general technical or academic content falls outside what its tree-construction step can parse.",
        "No -- but the deciding factor here is the paper's total page count being too small, not the depth of its section structure; a longer paper with the same shallow ~7-section layout would have let PageIndex perform better."
      ],
      "correct": 1,
      "explanations": [
        "The page's scoping note says the opposite of what this implies: vector RAG's edge on a SHALLOW document doesn't mean vector RAG is generally stronger -- it means a shallow document doesn't have the structure PageIndex needs to show its advantage. The 5/5 vs. 3/5 result on a genuinely structured document is the fairer test of the technique's actual value, not the shallow-document result.",
        "Correct. This page states this directly: it deliberately avoided that exact paper as a demo document because its shallow structure wouldn't exercise PageIndex's real advantage, and used a genuinely 4-level-structured document instead to make the comparison meaningful.",
        "PageIndex isn't restricted to regulatory filings by design or by what it can parse -- the requirement is genuine hierarchical structure (headers, sections, a real table of contents), which happens to be common in filings, contracts, and technical manuals but isn't limited to any one content domain. An academic paper with deep structure would exercise the same advantage.",
        "This page's stated reasoning is about STRUCTURE (how many real hierarchical levels exist to navigate), not raw page or word count -- a much longer document with the same flat, ~7-section layout would give tree-navigation just as little to work with as the actual sample paper does."
      ]
    },
    {
      "scenario": "On the same document, a naive vector RAG system is asked 'What does MEASURE 2.7 evaluate?' and retrieves three chunks that are all about measurement, testing, and evaluation topics in general -- but none of them contain the actual MEASURE 2.7 entry, which is a specific numbered item elsewhere in the document.",
      "question": "What does this page identify as the specific, checked cause of this failure?",
      "options": [
        "A newer, higher-quality embedding model specifically trained on regulatory or legal text would have ranked the actual MEASURE 2.7 entry above the generically similar chunks in this case.",
        "Embedding similarity retrieved text that sounds semantically close to the query's own vocabulary (words like 'measure' and 'evaluate') rather than the document's actual structural answer -- a specific numbered subsection that happened not to score highest by that similarity measure.",
        "The chunk size was set too large, causing the boundaries of the actual MEASURE 2.7 entry to be split across multiple chunks so that no single chunk contained the complete answer.",
        "This failure mode is specific to documents with numbered technical entries like 'MEASURE 2.7,' and wouldn't occur in documents that reference concepts only by descriptive names rather than numeric identifiers."
      ],
      "correct": 1,
      "explanations": [
        "This page doesn't attribute the failure to embedding model quality or domain-training -- it attributes it to a structural property of similarity search itself: text sharing vocabulary with the query can outscore the document's actual, correctly-located answer regardless of how good or domain-specific the embedding model is.",
        "Correct. This is exactly what this page verifies directly: the retrieved chunks were generically similar-sounding text (a nearby table, generic process language) rather than the actual MEASURE 2.7 entry -- similarity to the query's wording, not the document's own organization, decided what got retrieved.",
        "Chunk size isn't identified as the cause here -- the retrieved chunks were topically related but drawn from the WRONG location in the document entirely (a nearby table, generic process language), not a MEASURE 2.7 entry split across chunk boundaries. This is a similarity-ranking issue, not a chunking-boundary issue.",
        "This failure mode is a general property of similarity-based retrieval whenever generically related prose competes against a document's own specific, correctly-located answer -- it isn't limited to numbered entries. A descriptively-named but specific section could just as easily lose to generically similar-sounding text elsewhere in the document."
      ]
    },
    {
      "scenario": "A team is choosing between vector RAG and PageIndex for a new system indexing thousands of short, loosely-organized customer support tickets with no consistent internal structure, where per-query latency needs to stay under a second.",
      "question": "Based on this page's 'when to use which' guidance, which approach fits better?",
      "options": [
        "PageIndex, since its structural-navigation approach gives it a real accuracy advantage on the documents this page tested, and that advantage should carry over to any document collection a team indexes next, including short and loosely-organized ones.",
        "Vector RAG -- support tickets have no real hierarchical structure for PageIndex to navigate, and this page measured PageIndex's per-query latency at 6-12 seconds (an actual reasoning loop) versus naive RAG's well under a tenth of a second, which conflicts directly with a sub-second latency requirement.",
        "PageIndex, because skipping the embedding step removes a processing stage compared to vector RAG, which should make each query resolve faster.",
        "Neither -- since this system will index many separate support tickets rather than one document, it needs Graph RAG's entity-and-relationship structure to handle the collection, the same way Graph RAG was used for multi-hop questions elsewhere on this site."
      ],
      "correct": 1,
      "explanations": [
        "This page explicitly ties PageIndex's advantage to genuine, exploitable document structure -- loosely-organized short documents like support tickets are exactly the case this page says favors vector RAG instead, not a case the measured advantage automatically carries over to.",
        "Correct. Two separate reasons from this page both point the same way here: PageIndex needs real structure to navigate (tickets don't have it), and this page's own measured latency (6-12s per PageIndex query vs. sub-0.1s for vector RAG) directly conflicts with a sub-second requirement.",
        "Removing the embedding step doesn't make PageIndex faster overall -- this page measured the opposite: PageIndex took 6-12 seconds per query (a multi-step agentic reasoning loop) against vector RAG's well under a tenth of a second. Skipping one stage (embedding) doesn't offset adding a much larger one (an LLM reasoning loop over the tree).",
        "Needing to handle many documents doesn't by itself imply a need for Graph RAG's relational structure -- Graph RAG is suited to questions whose answers depend on relationships BETWEEN entities, not simply to having a large collection of short, independent documents. This scenario's actual constraints (no internal structure per document, tight latency) point to a simpler vector RAG setup instead."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — the real, open-source library used in this recipe, local mode
- [PageIndex documentation](https://docs.pageindex.ai) — the tree-search reasoning pattern and API reference
- [Mafin2.5 / FinanceBench results](https://vectify.ai/blog/Mafin2.5) — PageIndex's own published benchmark on financial-filing QA, for a second data point beyond this page's own measurement
- [CAG](cag.md) — another technique measured on real API cost/latency numbers, contrasted with this page's own (partial) cost accounting
- [Modular RAG](modular-rag.md) and [Graph RAG](graph-rag.md) — routing across multiple documents and multi-source synthesis, both referenced above as the problems PageIndex's single-document tree doesn't solve on its own
