# Chunking Strategies

Every RAG system starts by cutting documents into pieces small enough to embed and retrieve individually. That decision — how you cut — turns out to matter more than most people expect: a [NAACL 2025 Vectara study](https://arxiv.org/abs/2410.13070) tested 25 chunking configurations against 48 embedding models and found that **chunking configuration influenced retrieval quality as much as, or more than, the embedding model choice**. Teams routinely obsess over which embedding model to use and treat chunking as an afterthought — the data says that's backwards.

The tension driving every strategy on this page is the same one: cut too small and a chunk loses the surrounding context it needs to make sense; cut too large and the embedding averages across multiple ideas, diluting the specific signal a query is searching for. Everything below is a different answer to that trade-off.

!!! example "Hands-on"
    Every strategy on this page is runnable and wraps a real, tested reference implementation — not a from-scratch reimplementation: [**Chunking Strategies →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/chunking-strategies) in the code repo. Same sample document, ten different ways of cutting it — run one, or run `--strategy all` and compare.

??? abstract "TL;DR — quick revision"
    - **Default:** recursive chunking, 256-512 tokens, ~10-20% overlap — cheap, structure-respecting, wins on end-to-end accuracy in 2026 benchmarks despite lower raw recall than semantic
    - **Chunking config rivals embedding-model choice** in impact on retrieval quality (Vectara, NAACL 2025) — don't skip tuning this
    - **Semantic chunking** gets higher recall (91.9% vs. recursive's 85-89%) but *lower* end-to-end accuracy (54% vs. 69%) — fragments can get topically pure but too small to actually answer from
    - **Small-to-big / parent-document** is the broadly useful upgrade once precise retrieval and sufficient generation context are in tension — which is most real systems
    - **Late chunking** flips the order entirely: embed the whole document first (full cross-attention), split the token embeddings after — chunks keep context from the rest of the document
    - **Proposition chunking** goes finer than a sentence: one atomic, self-contained fact per unit (Dense X Retrieval, Chen et al. 2023)
    - **Agentic chunking costs an LLM API call** per document; **proposition chunking** costs one local model inference call per chunk (free, no API) using the paper authors' released model — both still too slow for bulk indexing of a huge corpus
    - **Adaptive chunking** doesn't pick a technique for you — it runs several, scores them, and picks per document, which matters once your corpus is genuinely mixed-format

## Fixed-Size Chunking

The simplest thing that could possibly work: split every N tokens or characters, with a fixed overlap between consecutive chunks so a sentence sitting exactly on a boundary still appears intact in a neighboring chunk.

**Parameters:** `chunk_size` (typically 256-512 tokens to start), `overlap` (10-20% of chunk size).

It respects nothing about the document's actual structure — it will happily cut a sentence, a table row, or a function definition in half purely because the character count was hit. That's also its appeal: zero embedding or LLM calls to decide anything, completely predictable chunk sizes, trivial to implement and reason about.

**Use it for:** simple, uniform documents where structure barely matters, fast prototyping, or as a baseline before investing in anything smarter.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/chunking-strategies/chunking_strategies_docs.py:fixed_size"
```

## Recursive Chunking

Instead of blindly cutting at a fixed offset, try a hierarchy of separators — paragraph breaks first, then sentence breaks for any piece still too large, then word breaks as a last resort. LangChain's `RecursiveCharacterTextSplitter` is the reference implementation, and the code below uses it directly.

**Parameters:** `chunk_size`, `overlap`, and the separator hierarchy itself (defaults to `["\n\n", "\n", ". ", " ", ""]` — customize per document type, e.g. add `"```"` for code-heavy docs).

This is the **default almost everyone should start with**. It's still zero-cost (no embedding/LLM calls), respects natural boundaries far better than fixed-size, and — per the 2026 benchmarks cited above — actually wins on *end-to-end* accuracy (69%) despite semantic chunking's higher raw retrieval recall, because it doesn't over-fragment content the way topic-shift-triggered cuts can.

**Use it for:** the general-purpose default for markdown, HTML, and plain text — reach for something smarter only once you have eval evidence recursive isn't enough.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/chunking-strategies/chunking_strategies_docs.py:recursive"
```

## Document/Structure-Aware Chunking

Split along the document's *actual declared structure* — Markdown headers, HTML tags, PDF section breaks — instead of a generic separator that merely tends to correlate with structure. `MarkdownHeaderTextSplitter` in LangChain is the concrete analog.

**Parameters:** none in the traditional sense — the document's own header levels *are* the parameters. Optionally cap output chunk size and further split any single section that's still too large (compose with recursive chunking).

The upside is exact: chunk boundaries land exactly where the document's author already divided it, and you get rich metadata (section title, heading path) for free — genuinely useful for filtering and citations later. The downside shows up the moment the input lacks real structure: a scanned PDF or informally-written notes give this method nothing to split on, and it silently produces one giant chunk (see the demo below — this happens on the sample PDF, which is exactly why the code repo pairs this strategy with its own Markdown sample instead).

**Use it for:** structured documents with clear sections — Markdown files, well-tagged HTML, PDFs with genuine heading structure.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/chunking-strategies/chunking_strategies_docs.py:structure_aware"
```

## Semantic Chunking

Embed every sentence individually, walk through them in order, and cut a new chunk boundary wherever similarity to the next sentence drops below a threshold — a proxy for "the topic just shifted."

**Parameters:** `threshold` (a cosine-similarity cutoff, typically 0.5-0.75 — lower means fewer, larger chunks; needs tuning per embedding model and corpus, there's no universal default).

This produces the most topically coherent chunks of any non-LLM method — and it measurably wins on raw retrieval recall (91.9% vs. recursive's 85-89% in 2026 benchmarks). The catch, and it's an important one for interviews: **higher recall doesn't mean better answers**. The same benchmarks show semantic chunking's end-to-end accuracy landing at just 54% against recursive's 69%, because topic-pure fragments can end up too small or too narrow to actually contain a complete answer — a [2026 FloTorch benchmark](https://www.premai.io/blog/rag-chunking-strategies-the-2026-benchmark-guide/) found semantic fragments averaging just 43 tokens. It also costs real compute — embedding every sentence before you've even embedded the chunks you'll actually retrieve.

**Use it for:** narratives and technical documentation where topical coherence is genuinely the retrieval bottleneck — worth the extra indexing cost specifically when eval results show it, not by default.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/chunking-strategies/chunking_strategies_docs.py:semantic"
```

## Sentence-Window Retrieval

Index single sentences individually for maximally precise matching, but at retrieval time return a window of surrounding sentences instead of just the one that matched — decoupling "what gets matched" from "what the LLM actually sees."

**Parameters:** `window` (sentences on each side of the match — typically 1-3; larger windows give more context but reintroduce the dilution problem this technique exists to avoid).

A single matched sentence is often too little for the LLM to work with — a pronoun without its antecedent, a fact stated without the qualifier one sentence earlier — but indexing bigger chunks for the sake of context hurts matching precision. This pattern gets both: precise retrieval on the sentence, sufficient context from the window.

**Use it for:** precise fact-lookup where the exact matching unit matters (a specific clause, a specific number) but the model still needs a sentence or two of surrounding context to interpret it correctly.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/chunking-strategies/chunking_strategies_docs.py:sentence_window"
```

## Small-to-Big / Parent-Document Retrieval

Embed and index small child chunks for precise matching, but link each one back to a larger parent chunk — when a child matches at query time, fetch and pass the *parent* to generation, not the child itself.

**Parameters:** `child_size` (small, e.g. 150-300 tokens — the matching unit), `parent_size` (larger, e.g. 1000-2000 tokens — the generation-context unit), `overlap` on the child tier.

This is the general form of sentence-window (child = one sentence, parent = a fixed window) — generalized so the child can be any small unit and the parent any larger one, explicitly linked via a `parent_id` rather than a fixed N-sentences-around rule. It directly resolves the fixed-size/semantic dilemma above: small chunks retrieve precisely; big chunks carry context; this gets both by using different chunk sizes for different jobs.

**Use it for:** the default pattern to reach for whenever precise retrieval and sufficient generation context are in tension — which describes most production RAG systems once they've outgrown fixed-size or recursive alone.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/chunking-strategies/chunking_strategies_docs.py:small_to_big"
```

## Late Chunking

Every strategy above embeds each chunk **in isolation** — the embedding model sees only that chunk's text, nothing else. Late chunking inverts the order: run the *entire document* through a long-context embedding model first, so every token attends to every other token via full self-attention, and only *then* split the resulting token-level embeddings into chunks, mean-pooling each span. [Jina AI introduced the technique](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) specifically to fix a failure late chunking's every predecessor shares: a chunk like *"The company grew 3% that quarter"* is meaningless in isolation once split away from the sentence naming the company — late chunking keeps that reference grounded, because the token embeddings were computed with the whole document in view before any splitting happened.

**Parameters:** the boundary positions themselves (produced by any other chunking method — late chunking is about *how the embeddings are computed*, not where the cuts go), plus the underlying model's max context length (a hard ceiling on how much of the document can share attention — 8K tokens for many current long-context embedding models).

**Trade-off:** requires a long-context embedding model capable of producing token-level (not just pooled-sentence) output, and a full-document forward pass is more expensive than embedding small chunks independently. Anthropic's [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) solves a related problem — chunks losing their surrounding context — differently: it prepends an LLM-generated summary blurb (50-100 tokens) to each chunk before embedding, one LLM call per chunk (cacheable, ~90% cost savings via prompt caching on repeat runs against the same document), cutting retrieval failures 49% alone and 67% combined with reranking. Late chunking gets a similar effect without any LLM call, at the cost of needing a long-context embedding model instead.

**Use it for:** documents where chunks losing their surrounding reference (pronouns, "that quarter," "the aforementioned clause") is actually hurting retrieval — financial reports, legal documents, anything with heavy cross-referencing.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/chunking-strategies/chunking_strategies_docs.py:late_chunking"
```

## Proposition-Based Chunking

Go finer than a sentence: decompose a passage into **propositions** — atomic, self-contained factual statements, each rewritten so it makes sense with zero surrounding context (pronouns resolved, implicit subjects spelled out). This is the retrieval unit introduced by [Dense X Retrieval](https://arxiv.org/abs/2312.06648) (Chen et al., 2023, EMNLP 2024), which found proposition-level retrieval beats both passage- and sentence-level retrieval on open-domain QA benchmarks — a finer retrieval unit means a higher density of relevant information per retrieved item, and a better chance the exact right fact surfaces in the top-k instead of getting buried inside a longer, only-partially-relevant chunk. The paper's authors released a model fine-tuned specifically for this decomposition — `chentong00/propositionizer-wiki-flan-t5-large`, a 780M-parameter T5 model — which runs entirely locally with no API cost, rather than requiring a prompted call to a large general-purpose LLM.

**Parameters:** input format is fixed by how the model was trained — `Title: {title}. Section: {section}. Content: {content}` — so the lever is really just chunking your source text into reasonably-sized passages before feeding each one through.

**Trade-off:** one model inference call per source chunk (free/local with the released model, or an API call if you instead prompt a general LLM for this), and — as the notes on Dense X Retrieval point out — decomposing to this granularity can weaken broader contextual continuity, since closely related facts that used to sit in one chunk now get retrieved (or not) independently of each other.

**Use it for:** dense factual documents where a single query needs to land on one precise fact, not a paragraph containing it — knowledge bases, structured reference material, anywhere Recall@k on *exact facts* is the metric that matters.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/chunking-strategies/chunking_strategies_docs.py:proposition"
```

## Agentic Chunking

Skip the rule entirely — prompt an LLM to read the document and decide chunk boundaries directly, using genuine reading comprehension rather than a mechanical separator or similarity threshold.

**Parameters:** none numeric; the prompt is the entire mechanism (e.g., "keep a clause together with its stated exception, even though the exception discusses a different sub-topic").

**Trade-off:** the most expensive strategy on this page — an LLM call (or several, for a long document processed in windows) purely to decide *where to cut*, before any embedding for retrieval has even happened. It can incorporate domain/legal/logical coherence no mechanical rule can — recognizing that a clause and its exception belong together even though a similarity-threshold approach would see two different sub-topics and split them.

**Use it for:** high-value, complex documents where getting boundaries exactly right justifies real cost — contracts, medical records, regulatory filings. Not for bulk indexing of a large, lower-stakes corpus.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/chunking-strategies/chunking_strategies_docs.py:agentic"
```

## Adaptive Chunking

This one isn't a chunking method — it's a *strategy selector*. In a genuinely mixed document collection (clean Markdown docs, messy OCR'd contracts, choppy Slack exports), no single method above wins for every document type. Adaptive chunking runs several candidate strategies against a document, scores each for quality (a coherence score — average embedding similarity between sentences within each resulting chunk is one simple proxy), and picks whichever wins **for that specific document**, rather than a person hardcoding "use method X for type A, method Y for type B" by hand.

**Parameters:** the candidate strategy set to try (and each of *their* parameters), plus the scoring function itself — coherence is the simplest signal; production systems often also score boundary respect (did the cut land on a natural break?) and size validity.

**Trade-off:** real overhead — running and scoring multiple strategies per document at indexing time instead of picking one upfront. Worth it specifically because most real enterprise document collections *are* heterogeneous, making this more broadly applicable than it sounds at first.

**Use it for:** genuinely mixed-format corpora — which describes most real-world document repositories once you look closely.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/chunking-strategies/chunking_strategies_docs.py:adaptive"
```

## Comparison table

| Strategy | Mechanism | Cost | Best for |
|---|---|---|---|
| **Fixed-size** | Split every N tokens, fixed overlap | None | Simple/uniform docs, fast prototyping |
| **Recursive** | Paragraph → sentence → word fallback | None | General-purpose default — start here |
| **Structure-aware** | Split on the document's own headers/tags | Parsing only | Markdown, tagged HTML, structured PDFs |
| **Semantic** | Embed sentences, cut on similarity drop | Medium (embed every sentence) | Topical coherence is the actual bottleneck |
| **Sentence-window** | Index sentences, retrieve a window around each | Low-medium | Precise fact lookup needing some surrounding context |
| **Small-to-big** | Embed small children, generate from linked parent | Medium (2-tier storage) | Precision vs. context tension — most systems, eventually |
| **Late chunking** | Embed whole doc first, pool token spans after | Medium-high (long-context forward pass) | Cross-referencing documents, chunks losing meaning in isolation |
| **Proposition** | LLM extracts atomic self-contained facts | High (LLM call per chunk) | Dense factual QA needing exact-fact precision |
| **Agentic** | LLM decides boundaries directly | Highest (LLM call per document) | High-value complex documents — contracts, medical records |
| **Adaptive** | Run multiple strategies, score, pick winner | Medium-high (multiple strategies run) | Genuinely heterogeneous document collections |

## Don't hand-roll a splitter — these are solved problems

Every technique on this page is old and well-understood enough that a mature, widely-used library already implements it correctly — chunk boundaries, overlap semantics, and edge cases included. The code in this page's [companion repo](https://github.com/DhruvMakwana/rag-cookbook/tree/main/chunking-strategies) deliberately wraps real implementations rather than reimplementing the algorithms: `langchain-text-splitters` for fixed-size, recursive, and Markdown-header splitting; [**Chonkie**](https://pypi.org/project/chonkie) — a purpose-built, lightweight chunking library (`pip install chonkie`) — for semantic, late, and LLM-guided (`SlumberChunker`) chunking; LlamaIndex's `HierarchicalNodeParser` and `SentenceWindowNodeParser` for small-to-big and sentence-window; and, for proposition chunking, the actual fine-tuned model the Dense X Retrieval authors released, rather than an improvised prompt. Only the strategy-*selection* logic in adaptive chunking is inherently custom — no library owns "run several chunkers and pick the best," since that's an orchestration choice, not a chunking algorithm.

## Scenario Check

Ten chunking strategies is a lot to keep straight — these questions test whether you can match a symptom to its fix, the way a system-design interview actually probes this.

<div class="quiz-widget" data-title="Scenario Check: Chunking Strategies">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team benchmarks two chunking strategies on the same corpus. Strategy A gets 91% retrieval recall (the right chunk is almost always somewhere in the top-k). Strategy B gets 86% recall but a noticeably higher end-to-end answer-correctness score.",
      "question": "What's the most likely explanation, based on 2026 benchmark data?",
      "options": [
        "Strategy A is semantic chunking — high recall from topically pure chunks, but chunks small/narrow enough that individual ones often don't fully contain the answer.",
        "Strategy B is most likely using small-to-big retrieval, since fetching a larger parent chunk for generation is what typically closes this kind of recall-vs-accuracy gap.",
        "Strategy B's embedding model is simply better than Strategy A's.",
        "Strategy A is most likely using significantly less chunk overlap than Strategy B, and less overlap directly explains why its end-to-end accuracy comes in lower despite higher recall."
      ],
      "correct": 0,
      "explanations": [
        "Correct. This is exactly the semantic-vs-recursive pattern from the 2026 benchmarks: semantic chunking's topic-shift cuts produce highly coherent but sometimes very short fragments (averaging ~43 tokens in one benchmark) — great at surfacing something topically relevant, worse at containing a complete, generatable answer.",
        "Small-to-big retrieval is a real technique for balancing precise matching against generation context, but it's not what this specific numbers pattern indicates — a 91%-recall/54%-accuracy vs. 86%-recall/69%-accuracy split is the documented signature of semantic vs. recursive chunking specifically, not evidence of a parent/child architecture change.",
        "Both strategies could use the identical embedding model — chunking method, not embedding choice, is the variable described here, and per the Vectara study, chunking config rivals embedding choice in impact.",
        "Overlap affects how often the same content gets duplicated across chunk boundaries, not the specific recall-vs-accuracy pattern described here — and nothing in the scenario mentions overlap settings at all; the documented explanation for this exact gap is fragment size/coherence (semantic vs. recursive chunking), not overlap."
      ]
    },
    {
      "scenario": "A knowledge base is a folder of scanned, OCR'd legal contracts with no reliable heading tags (OCR mangled most of the formatting), but each contract is long, coherent legal prose.",
      "question": "Which strategy is the worst fit here, specifically?",
      "options": [
        "Recursive chunking, since its separator hierarchy leans on paragraph and sentence breaks — exactly the punctuation and layout cues OCR damage first, before it even gets to headers.",
        "Structure-aware chunking, since it has no reliable headers to split on and will produce one giant chunk per document, or split on garbage.",
        "Semantic chunking, since legal prose has no real topic shifts to detect.",
        "Fixed-size chunking, since a fixed token window will regularly cut legal clauses mid-sentence, making the retrieved text legally ambiguous."
      ],
      "correct": 1,
      "explanations": [
        "Recursive chunking's separator hierarchy falls back through multiple levels — paragraphs, then sentences, then words — so even damaged punctuation degrades it gracefully rather than breaking it; it still produces usable chunks throughout the document, unlike structure-aware chunking's complete failure when there's no structure to find at all.",
        "Correct. Structure-aware chunking's entire mechanism depends on the document actually declaring its structure. OCR'd text with mangled formatting gives it nothing reliable to split on — this is exactly the documented failure mode: one giant chunk, or splitting on noise.",
        "Legal prose often does have real topical structure (clauses, sections, exceptions) even without formatting tags — semantic chunking, working on the text itself rather than formatting, is actually a reasonable fit here, not the worst one.",
        "This is a real weakness of fixed-size chunking on legal text — mid-clause cuts are genuinely bad here — but it's not the worst fit in this specific scenario: fixed-size still produces complete, retrievable chunks across the entire document regardless of formatting, whereas structure-aware chunking can collapse the whole document into one unusable giant chunk when there's no header structure to find."
      ]
    },
    {
      "scenario": "A query asks about \"the company's Q3 revenue growth,\" and the retrieved chunk reads only: \"Revenue grew 3% that quarter, driven primarily by the segment discussed above.\" The chunk never names the company or the segment.",
      "question": "Which technique most directly targets this exact failure?",
      "options": [
        "Fixed-size chunking with more overlap between chunks.",
        "Small-to-big retrieval, so at least the parent chunk is available.",
        "Late chunking — since token embeddings were computed with the whole document in view before any splitting, the chunk's embedding itself carries context about which company/segment it discusses, unlike a chunk embedded in isolation.",
        "Sentence-window retrieval, since the answer is a single sentence anyway."
      ],
      "correct": 2,
      "explanations": [
        "More overlap only helps if the missing context happens to fall within the overlap window — it doesn't structurally fix a chunk referring to something named paragraphs earlier, which is the actual problem here.",
        "A reasonable partial fix (the parent chunk might include the company name) — but small-to-big doesn't change how the *embedding* itself was computed, so retrieval could still miss this chunk if its isolated embedding doesn't clearly signal what it's about. Late chunking fixes the retrieval signal itself, not just what gets shown after a match.",
        "Correct. This is precisely what late chunking is built for — the chunk's embedding was computed via full-document attention before splitting, so it already carries context from earlier in the document (the company name, the segment), making it both more findable and more interpretable, unlike every method that embeds the chunk in total isolation.",
        "Sentence-window would return the matched sentence plus a couple neighbors — helpful, but it doesn't guarantee the company name appears in that specific window if it was stated much earlier in the document."
      ]
    },
    {
      "scenario": "A team wants retrieval units precise enough that a single query about one specific fact (\"what's the maximum international remote work limit\") reliably surfaces exactly that fact, not a longer chunk that also discusses unrelated policies.",
      "question": "Which technique is purpose-built for this?",
      "options": [
        "Agentic chunking, since an LLM understands what's 'important' in the document.",
        "Proposition-based chunking, since each retrieval unit is a single atomic, self-contained fact rather than a multi-sentence passage.",
        "Fixed-size chunking with a very large chunk size, so the surrounding context around the fact naturally gets swept into the same chunk.",
        "Structure-aware chunking, since headers indicate which policy each section covers."
      ],
      "correct": 1,
      "explanations": [
        "Agentic chunking decides *boundaries* using LLM judgment, but the resulting units are still passage-sized — it doesn't push all the way down to one-fact-per-unit the way proposition chunking does.",
        "Correct. This is exactly what Dense X Retrieval's proposition-based chunking targets — decomposing text into atomic, self-contained facts as the retrieval unit, so a single-fact query has a good chance of matching a unit that IS that fact, not a passage that merely contains it among other things.",
        "A larger chunk size doesn't create precision — it does the opposite, pulling more unrelated policy content into the same embedding, which dilutes the vector's specific signal for a narrow single-fact query and makes it a worse, not better, match.",
        "Structure-aware chunking helps narrow down to the right *section*, but a section can still contain multiple facts bundled together — it doesn't get down to single-fact granularity the way proposition chunking does."
      ]
    },
    {
      "scenario": "An internal knowledge base has three very different document types: clean Markdown engineering docs, OCR'd legal contracts, and raw Slack thread exports. A team wants good chunking without hand-picking a strategy per document type and maintaining that mapping forever.",
      "question": "What should they reach for?",
      "options": [
        "Structure-aware chunking applied uniformly to everything, since it's the most sophisticated option.",
        "Adaptive chunking — run multiple candidate strategies per document, score each, and let the system pick the winner per document automatically.",
        "Agentic chunking for everything, since an LLM can adapt its judgment to any document type.",
        "Fixed-size chunking for everything, since a fixed character/token window is the simplest strategy to apply consistently across every document type without extra tooling."
      ],
      "correct": 1,
      "explanations": [
        "Structure-aware chunking is genuinely the wrong universal choice here — it fails outright on the OCR'd contracts and the Slack exports, which is precisely the kind of gap adaptive chunking exists to route around.",
        "Correct. This is exactly adaptive chunking's use case: genuinely heterogeneous document types where one fixed strategy can't serve all of them well, and where hand-maintaining a per-type mapping doesn't scale. It runs several strategies, scores them, and picks automatically per document.",
        "Agentic chunking would technically adapt via LLM judgment, but at the cost of an LLM call per document across the entire corpus — expensive at scale, and adaptive chunking gets a comparable per-document fit without paying for LLM reasoning on every single document.",
        "Consistency of application isn't the same as quality of output — fixed-size chunking ignores each document's actual structure (Markdown headers, contract clauses, Slack thread boundaries) equally in every case, which is exactly the gap adaptive chunking exists to close by tailoring the strategy per document instead of applying one blind rule everywhere."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [Dense X Retrieval: What Retrieval Granularity Should We Use?](https://arxiv.org/abs/2312.06648) — Chen et al., 2023/EMNLP 2024, the proposition-chunking paper
- [Late Chunking in Long-Context Embedding Models](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — Jina AI
- [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) — Anthropic engineering
- [Retrieval Augmented Generation (RAG) Done Right: Retrieval](https://vectara.com/retrieval-augmented-generation-rag-done-right-retrieval/) — Vectara's NAACL 2025 chunking/embedding study
- [RAG Chunking Strategies: The 2026 Benchmark Guide](https://www.premai.io/blog/rag-chunking-strategies-the-2026-benchmark-guide/) — semantic vs. recursive end-to-end accuracy numbers cited above
