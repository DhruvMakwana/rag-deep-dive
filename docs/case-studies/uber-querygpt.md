# Uber QueryGPT: Decomposing Text-to-SQL Into Narrow Sub-Agents

Uber runs roughly **1.2 million interactive SQL queries a month** across its internal data platform, with the Operations org alone responsible for 36% of that volume. Historically, each of those queries took an analyst about **10 minutes** to hand-write, which meant that democratizing access to Uber's data warehouse was bottlenecked less by data availability than by how much SQL and schema knowledge a given employee happened to have. QueryGPT, described on Uber's engineering blog, is Uber's answer: a multi-agent text-to-SQL system that turns a natural-language question into a working query without requiring the person asking it to know the schema at all. It's a useful case study specifically because it's a genuinely agentic pipeline — several narrow, independently-evaluable agents chained together — rather than one large prompt asked to do everything at once.

## The problem

Text-to-SQL over a data warehouse the size of Uber's runs into a scaling problem before it runs into a language-modeling problem: you cannot hand an LLM the entire schema and every plausible sample query and expect a clean, cheap, hallucination-free result. Context grows unmanageably, cost per query rises, and the surface area for the model to hallucinate a plausible-but-wrong table or column name grows right along with it. Uber's goal was to cut authoring time and let non-expert analysts self-serve, but doing that safely meant solving the schema-scoping problem first — deciding what subset of Uber's warehouse a given question was even about — before any SQL generation could happen.

## What they built, and why

QueryGPT is a **sequential four-stage pipeline**, not a single model call:

1. **Intent Agent** — classifies the natural-language question into a business "workspace": Mobility, Core Services, Platform Engineering, IT, Ads, or a custom workspace. This is the load-bearing decision in the whole architecture, because everything downstream operates only within that workspace's scope.
2. **Table Agent** — retrieves candidate tables using retrieval-augmented generation: a k-NN vector similarity search over a curated, per-workspace corpus of tier-1 tables and vetted sample queries. The proposed tables are surfaced to the user for confirmation before generation proceeds.
3. **Column Prune Agent** — strips irrelevant columns out of the candidate schema before it ever reaches the SQL-generation model, explicitly to control both token cost and the surface area available for the model to hallucinate against.
4. **Query Generation** — synthesizes the final SQL, with an accompanying explanation, using GPT-4 Turbo (the 1106 version, 128K context).

The key architectural decision underneath all four stages is that each **workspace is its own curated RAG corpus** — a scoped set of tier-1 tables and vetted SQL samples — rather than one global retrieval index spanning Uber's entire warehouse. That choice is what makes the Column Prune Agent's job tractable at all: pruning columns out of a scoped, already-relevant table set is a well-defined problem, whereas pruning columns out of a warehouse-wide candidate set would still leave enormous ambiguity and hallucination risk. Scoping the retrieval corpus per business domain is what turns "generate correct SQL over an entire company's warehouse" into a sequence of much narrower, checkable sub-problems.

That decomposition also pays off for evaluation. Uber tracks **intent-accuracy**, a **table-overlap score**, and a **successful-execution-rate** as separate metrics for separate stages of the pipeline, which means a failure can be localized to "the intent agent picked the wrong workspace" versus "the SQL is syntactically broken" instead of being one opaque end-to-end pass/fail.

## The numbers

Uber's own reported figure is that query-authoring time dropped from about **10 minutes to about 3 minutes**. In its limited-release phase, covering Operations and Support teams, QueryGPT saw roughly **300 daily active users**, and **78%** of users self-reported a reduction in query-authoring time. (Figures like a "70% reduction" or "$120M / 140,000 hours saved" circulate in secondary write-ups about QueryGPT, but those are not claims Uber itself made in its engineering blog post and shouldn't be treated as verified.)

!!! success "The lesson"
    The transferable idea here isn't "use RAG for text-to-SQL" — it's that scoping the retrieval corpus per business domain, rather than building one retrieval index over an entire warehouse, is what makes downstream steps like column pruning and hallucination control tractable in the first place. More generally, QueryGPT is a clean example of decomposing one agentic task into a sequence of narrow, independently-evaluable sub-agents — intent, then tables, then columns, then SQL — instead of asking a single large prompt to do all of it at once. Narrow agents with their own metrics are debuggable in a way monolithic prompts aren't: when something goes wrong, you know which stage to look at.

## Sources

- [Uber Engineering: QueryGPT — Natural Language to SQL Using Generative AI](https://www.uber.com/en-CA/blog/query-gpt/)
