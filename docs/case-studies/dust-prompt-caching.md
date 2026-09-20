# Dust: Tiered Prompt Caching for Long-Running Agents

Dust is an AI agent platform built for enterprises — what it calls "AI operators" — that lets non-engineers automate real business workflows across a company's internal systems, not just answer questions about them. That distinction matters for this case study: Dust's agents don't do a single retrieval-and-answer pass, they run extended, multi-step tool-use loops against live company data, sometimes for ten minutes or more. Working with Anthropic's Applied AI team, Dust re-architected both its prompt-caching strategy and its execution loop to make those longer, more capable runs cheaper rather than more expensive — a genuinely rare combination, since deeper agentic behavior usually costs more, not less.

## The problem

Dust's roadmap pointed toward agents that could do meaningfully more per run: its "Deep Research" agents needed to explore far more sources, chain far more tool calls, and stay useful over runs lasting many minutes. The obvious failure mode is that a more capable, longer-running agent burns proportionally more tokens on every call, since each turn of an agentic loop re-sends the accumulated conversation and tool history back to the model. Dust needed a way to let agents do substantially more work per run without letting the underlying model spend scale linearly with that extra work.

## What they built, and why

The core of the fix was recognizing that a single agent's prompt is not one undifferentiated block of text — it's made of layers that change at very different rates. Dust split its prompts into three tiers and gave each one a cache lifetime matched to how often that layer's content actually changes:

- **Global instructions** — the shared system-level behavior that's identical across nearly every run — were cached with a **1-hour TTL**, the longest window, because this layer is reused the most and changes the least.
- **Workspace-level context** — information specific to a given enterprise tenant — was cached on a shorter window, since it changes more often than global instructions but still far less often than per-request state.
- **Per-user context** was cached on the shortest window of the three, reflecting that it's the layer most likely to shift between requests.

The principle underneath this is straightforward but easy to get wrong in practice: cache lifetime should track content *volatility*, not be applied uniformly. A flat, single-TTL cache either expires the stable, highly-reused global instructions too aggressively (losing hit rate for no reason) or holds onto per-user context too long (serving stale state). Tiering the cache by how frequently each layer changes lets Dust maximize cache-hit rate on the parts of the prompt that barely change while keeping the parts that do change fresh.

Separately — and complementary to the caching work rather than in tension with it — Dust redesigned its execution loop to support far deeper tool use per run. Agents went from roughly 3 tool calls (capped at a maximum of 8) up to as many as **24 tool calls per run**, and Dust adopted the Model Context Protocol (MCP) as both a client and a server, giving its agents a standardized way to reach a much wider set of tools without bespoke integration work for each one.

## The numbers

The combined effect of the tiered caching and the redesigned loop showed up directly in Dust's spend and in what its agents were able to do:

- **~$10,000/day** in cost savings.
- **18-19%** reduction in overall model spend.
- **22%** reduction in input-token spend specifically.
- Cache-read share of input tokens rose from **30% to 65%** — more than double, meaning the majority of input tokens on a typical call were now served from cache rather than billed at full price.
- Autonomous tool-calling depth increased from **4-5 steps to 8+ steps** per run.
- Research sources consulted per task rose from **5 to 14**.

Stanislas Polu, Dust's co-founder and CTO, described the qualitative shift this way: "The new Claude models didn't just answer the question; it proactively explored adjacent information and synthesized across sources."

As a secondary, lower-confidence data point: Notion separately adopted the same Anthropic prompt-caching mechanism for Notion AI's Claude-powered features and reports a "90% cost reduction" and "up to 85% latency reduction." Those two figures are worth reading with real caution — they are identical to Anthropic's own generic marketing ceiling for prompt caching in general ("reduce costs by up to 90% and latency by up to 85%," from Anthropic's original prompt-caching launch announcement), and no independently-audited Notion-specific breakdown — actual dollar figures, before/after token counts — was found to back them up. That makes Notion's numbers directionally corroborating (another real production user seeing large gains from the same mechanism) but they should not be weighted the same as Dust's bespoke, mechanism-specific figures above, which come with a concrete tiering strategy and named before/after metrics attached.

!!! success "The lesson"
    Don't treat prompt caching as a single flat cache. Segment a prompt into layers by **volatility** — global instructions, tenant/workspace context, per-user context — and give each layer its own TTL matched to how often it actually changes. Done well, this is the single highest-leverage, lowest-risk cost lever available before touching model choice at all, and — as Dust's numbers show — it composes with capability increases like deeper tool-calling loops rather than trading off against them.

## Sources

- [Dust customer story on Claude.com](https://claude.com/customers/dust)
- [Notion customer story on Claude.com](https://claude.com/customers/notion)
