# Agentic RAG

Every technique before this one on the site still has a human — or at least a pre-written prompt — deciding the control flow. [Adaptive RAG](tutorials/advanced-rag-from-scratch.md)'s router picks a pre-retrieval technique via a classification prompt. [Corrective RAG](corrective-rag.md) and [Self-RAG](self-rag.md) check something and branch, but the branches themselves ("if Incorrect, web search") are rules someone wrote in advance. [Graph RAG](graph-rag.md) even ended with an explicit gap: choosing local vs. global search was a manual flag, not something the system decided. Agentic RAG is what closes that gap — instead of code deciding what happens next, an **LLM agent decides for itself**, at runtime, which tool to reach for.

!!! example "Hands-on"
    The full agent loop below is runnable, against real data: [**Agentic RAG →**](https://github.com/DhruvMakwana/rag-cookbook/tree/main/agentic-rag) in the code repo. This recipe's agent loop uses Anthropic's tool-calling API directly, so it needs an Anthropic key specifically (not the pluggable provider setup used elsewhere in this repo) — see below for why.

??? abstract "TL;DR — quick revision"
    - **The core mechanism is a ReAct loop** (Reason, Act, Observe, repeat) — but built here with the model's *actual* tool-calling API, not prompted text parsed with regex or tags the way every earlier recipe's routing worked
    - **Four tools, each a real technique already verified elsewhere on this site**: plain vector search, graph local search, graph global search, and a real web search fallback — the agent picks among them itself
    - **Verified correct on all 4 distinct question types, unprompted**: a simple factual question, a broad synthesis question, a specific relational question, and a question entirely outside the corpus — no routing logic was written for any of it; the tool *descriptions* alone were enough
    - **This is genuinely more expensive and less predictable than everything before it** — the number of tool calls varies per question, and a misinterpreted result can send the loop in an unproductive direction for several steps, which is why every real implementation needs a hard iteration cap and an honest "ran out of budget" exit
    - **Not free lunch, and not for everything** — high-volume, well-understood query patterns are cheaper and more predictable with a fixed or modular pipeline; agentic control flow earns its cost specifically when the right retrieval strategy genuinely can't be known in advance

## The ReAct loop, with real tool calls

Every prior "routing" technique on this site worked by asking an LLM to output a classification, then code parsed that text (a `<category>` tag, a comma-separated list) and branched on it by hand. That's a real, working pattern — but it's still code deciding what to do with the model's answer. **Tool calling flips that**: the model itself emits a structured `tool_use` request — a specific tool name and validated arguments — and the *only* thing left for code to do is actually run that tool and hand the result back. The model decides which tool, with what input, and whether it needs another one after seeing the result.

Here are the four tools this agent gets to choose from, each one a technique already built and measured on its own page elsewhere on this site:

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/agentic-rag/agentic_rag_docs.py:tools"
```

Notice what's actually doing the routing work here: not a classifier prompt, not a set of rules — just the `description` field on each tool. The model reads these the same way it would read documentation for any function it's deciding whether to call.

**What each tool actually does when called isn't repeated here** — every one of them is real, working code already shown and explained in full on its own page, and this page would just be duplicating it:

- `naive_vector_search` — plain bi-encoder top-k, the same retrieval step walked through on [Naive RAG](naive-rag.md#walking-through-it)
- `graph_local_search` / `graph_global_search` — entity-linking + traversal, and map-reduce over community summaries, both fully explained on [Graph RAG](graph-rag.md)
- `web_search` — a real Tavily call, the same fallback mechanism used on [Corrective RAG](corrective-rag.md#2b-incorrect-discard-everything-search-the-web-instead)

If you want to see the actual implementations side by side with this page's agent loop, [`agentic_rag.py`](https://github.com/DhruvMakwana/rag-cookbook/blob/main/agentic-rag/agentic_rag.py) in the repo has all four wired together in one file — worth reading directly rather than inferring from the tool descriptions alone, especially before adapting this pattern for your own tools.

## The loop itself

With those four tools declared, the agent loop is genuinely simple: ask the model, check whether it wants to call a tool, run the tool if so, feed the result back, and repeat — stopping either when the model has enough to answer directly, or when a hard cap is hit.

```python
--8<-- "https://raw.githubusercontent.com/DhruvMakwana/rag-cookbook/main/agentic-rag/agentic_rag_docs.py:agent_loop"
```

The iteration cap matters for a reason that has nothing to do with cost: a genuinely open-ended loop, given a misleading tool result, has no built-in reason to ever stop trying. Capping it and returning an honest "reached the iteration limit without a final answer" is the same discipline as [Self-RAG](self-rag.md)'s regenerate-once-then-fall-back-honestly pattern, applied to a loop whose length isn't fixed in advance.

## Verified: four different questions, four different (correct) tool choices

No routing logic was written anywhere in this recipe — the agent picked every one of these itself, from the tool descriptions alone.

!!! success "A simple factual question → naive_vector_search"
    Asked *"What optimizer did they use for training?"* — the agent called `naive_vector_search({'query': 'optimizer training'})` on the first step, got back the Adam-optimizer passage directly, and answered from it without needing anything else. The simplest tool for the simplest question — correctly recognized as such.

!!! success "A broad synthesis question → graph_global_search"
    Asked *"What are the main components of the Transformer architecture and how do they work together?"* — the same broad question [Graph RAG](graph-rag.md#global-search-for-questions-no-single-entity-or-chunk-can-fully-answer) measured naive top-5 retrieval failing on — the agent correctly reached for `graph_global_search`, and produced a genuinely comprehensive answer covering the encoder, decoder, multi-head attention mechanics, positional encoding, feed-forward networks, and residual connections. The tool's description ("best for broad questions that require synthesizing across many different parts") was enough to steer it here over the cheaper, narrower options.

!!! success "A specific relational question → graph_local_search"
    Asked *"What connects label smoothing to the BLEU score improvements?"* — the same question Graph RAG's local search demonstrated a real win on — the agent called `graph_local_search`, matched the entities `Label Smoothing` and `BLEU Score` directly, and answered from the traversed connection. It didn't reach for the more expensive global search or a plain vector lookup; the specific, two-entity nature of the question matched the tool built for exactly that.

!!! success "A question outside the corpus entirely → web_search"
    Asked *"What is the current stock price of Nvidia?"* — nothing in a 2017 machine learning paper could possibly answer this — the agent correctly skipped straight to `web_search`, made a real Tavily call, and returned genuinely current stock data (with the honest caveat that different sources in the results disagreed on the exact price, since stock prices move by the second). It didn't waste a call on `naive_vector_search` first and fail — it recognized the question was out of scope for the corpus-specific tools from the description alone.

## Why this needs more than everything before it

**Tool diversity beyond retrieval.** Nothing here restricts the agent to only choosing between retrieval strategies — the same mechanism extends to a calculator, a SQL database, a code execution tool, or another specialized agent, combined in whatever order a question actually demands. A fixed pipeline has to guess the tool sequence upfront; an agent works it out as it goes, including cases needing more than one tool in sequence.

**Real costs that don't show up on a fixed pipeline.** Latency and cost per query are no longer predictable the way a fixed 3-stage pipeline's are — a simple question might resolve in one tool call, a harder one in four or five. Error compounding is real: a misread tool result can send the loop down an unproductive path, which is exactly why the iteration cap above isn't optional. And evaluation gets harder — a fixed pipeline can be graded stage by stage (retrieval metrics, then generation metrics); an agent's *path itself* varies per question, so evaluating it well means reviewing whether each tool call was reasonable, not just whether the final answer was right.

## Framing within LangGraph / CrewAI / AutoGen

**LangGraph** defines an explicit graph of nodes (retrieve, grade, generate, call a tool) and edges — including conditional and looping edges — giving agentic flexibility while still constraining which paths are even possible, a middle ground between a free-form agent and a fully fixed pipeline. **CrewAI / AutoGen** lean toward multi-agent setups — several specialized agents (a retriever agent, a critic agent, a writer agent) collaborating or handing off tasks, useful specifically when one agent's context or role would get overloaded trying to do everything itself.

## Use it for / skip it for

**Use it for:** genuinely open-ended, multi-hop, multi-source questions where the retrieval strategy can't be enumerated in advance — research assistants, complex analyst tools, multi-system troubleshooting. **Skip it for:** high-volume, well-understood query patterns — simple FAQ lookups, single-document Q&A — where a fixed or modular pipeline is faster, cheaper, and just as accurate; agentic overhead buys nothing when the right strategy was already knowable in advance.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: Agentic RAG">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team builds a system where an LLM outputs a category label in <category> tags, and their own Python code parses that tag and calls the matching function. Another team gives an LLM a set of tool definitions with an API's native tool-calling parameter, and the LLM itself emits a structured tool-call request that the API returns as a distinct response type.",
      "question": "What's the key structural difference between these two approaches, based on how this page distinguishes them?",
      "options": [
        "Both approaches end up with the model producing a JSON payload describing which function to call embedded in its regular text output, which code then extracts with a parser before running it — the tool-calling API parameter is just a naming convention for the same underlying parse-then-execute mechanism.",
        "In the first approach, code decides what to do by parsing the model's free-text output after the fact; in the second, the model itself emits a structured, validated tool-use request as part of the API's actual response format — the routing decision is native to the model's output, not reconstructed by a parser.",
        "The second approach reaches the same routing decision as the first, just more cheaply — using the tool-calling parameter mainly saves the output tokens that would otherwise go into writing a <category> tag, without changing where the decision itself is made or validated.",
        "The first approach lets a developer define the categories in advance, while the second approach allows the model to invent and name new tools on its own without any tool definitions supplied beforehand."
      ],
      "correct": 1,
      "explanations": [
        "Native tool-calling doesn't embed the function call inside ordinary text for code to extract afterward — the API returns it as a distinct, schema-validated response type separate from any text the model generates. That's the actual structural difference the page draws, not a difference in naming.",
        "Correct. This is exactly the distinction the page draws: prompted routing has code parsing text after the fact to reconstruct a decision; native tool-calling has the model produce a structured, schema-validated request as a first-class part of the response — code's job shrinks to just executing what was already decided.",
        "The distinction the page draws isn't about token efficiency — it's about where the decision is represented and validated: prompted classification has code reconstruct a decision from free text after the fact, while native tool-calling has the model emit the decision as a structured, validated part of the response itself. Token savings, if any, are incidental to that.",
        "Native tool-calling still requires the developer to declare each tool's schema ahead of time — the model selects among predefined tools, it doesn't invent arbitrary ones at runtime. The difference the page describes is in how the chosen option gets represented in the response, not in whether tools must be predefined."
      ]
    },
    {
      "scenario": "An agentic RAG system is given no maximum number of loop iterations, on the reasoning that letting the agent decide for itself when to stop is the whole point of being agentic. In production, a misinterpreted tool result occasionally sends the agent down an unproductive path.",
      "question": "What does this page suggest is the correct way to handle this risk?",
      "options": [
        "Replace the fixed iteration cap with a wall-clock time budget instead — since the real risk is the loop running too long, capping by elapsed time rather than by a fixed number of tool calls better protects against a stuck agent without arbitrarily limiting how many steps a genuinely hard question can take.",
        "Set a hard iteration cap and have the loop return an honest 'ran out of budget without a final answer' response when the cap is hit — the same discipline as Self-RAG's regenerate-once-then-honestly-fall-back pattern, just applied to a loop of variable length.",
        "Add a step where the agent reviews its own previous tool result before choosing its next action, which removes the need for a hard iteration cap altogether, since self-checking catches misinterpretations before they can send the loop down an unproductive path.",
        "Limit the agent to using at most two distinct tools per conversation, so that even if one tool's result is misread, there are fewer remaining tools left for it to spiral through."
      ],
      "correct": 1,
      "explanations": [
        "This page's actual recommendation is a cap on the number of iterations, paired with an honest 'ran out of budget' exit — not a switch to a time-based budget. A wall-clock cap doesn't address the specific failure described (a misread result sending the loop down an unproductive path) any more precisely, and swaps out the exact mechanism the page describes for a different one it doesn't recommend.",
        "Correct. This is exactly what this page recommends: a hard cap plus an honest exit condition, mirroring the same 'don't loop forever, admit when you're out of budget' discipline already established for Self-RAG's retry logic — applied here to a loop whose length genuinely isn't known in advance.",
        "Self-review can help the agent catch some misreads, but it doesn't guarantee catching all of them, and it doesn't bound how long the loop can run if it doesn't. The page's fix is a hard cap plus an honest exit specifically because self-correction alone isn't a reliable enough safeguard against a loop that has no built-in reason to stop.",
        "The page's fix targets the loop's total LENGTH (an iteration cap), not which or how many distinct tools are available — restricting tool diversity would also block legitimate questions that genuinely need more than two tools, without actually bounding how many times the agent can loop on the tools it does have."
      ]
    },
    {
      "scenario": "A team is deciding whether to build an agentic RAG system (LLM-driven tool selection at runtime) or stick with their existing fixed 3-stage pipeline (retrieve → rerank → generate) for a high-volume FAQ chatbot that only ever answers simple, single-document lookup questions.",
      "question": "Based on this page's guidance, which approach is the better fit, and why?",
      "options": [
        "Agentic RAG, since giving the system access to more tools than it strictly needs can only help — a system that CAN call graph search or web search if useful will naturally perform at least as well as one that can't, even on questions simple enough not to need them.",
        "The fixed pipeline — the retrieval strategy for simple, well-understood FAQ lookups is already fully knowable in advance, so the fixed pipeline is faster, cheaper, and just as accurate; agentic overhead (variable latency/cost, harder evaluation) buys nothing when the right strategy doesn't actually need to be discovered at runtime.",
        "Agentic RAG, because having a wider variety of tools available is valuable in itself, independent of whether this particular FAQ chatbot's questions actually require any of that variety.",
        "Neither approach is ideal, since a genuinely simple FAQ chatbot should skip an LLM in the loop entirely and use a keyword or rules-based lookup instead of either RAG pattern."
      ],
      "correct": 1,
      "explanations": [
        "This ignores the real costs the page attaches to agentic control flow: variable latency/cost per query and harder evaluation, which are paid on every query regardless of whether the extra tools ever get used. For a query pattern that's already fully knowable in advance, that overhead buys no accuracy benefit — the fixed pipeline gets the same answer for less.",
        "Correct. This is exactly the 'skip it for' guidance stated directly on this page: when the retrieval strategy can already be fully specified in advance (as with simple, single-document FAQ lookups), a fixed pipeline gets the same accuracy without agentic RAG's unpredictable latency/cost and harder evaluation story.",
        "Having access to a wider tool variety only pays off when the query pattern actually needs it — for a narrow, already-knowable FAQ use case, that variety is unused capability paid for with agentic RAG's real costs (unpredictable latency/cost, harder evaluation), with no corresponding benefit.",
        "This page compares agentic RAG against a fixed 3-stage RAG pipeline specifically, and its guidance is about choosing between those two for a use case where the retrieval strategy is fully knowable in advance — it doesn't argue that RAG itself (in either form) is the wrong architecture for FAQ lookup, only that the FIXED version is the better-fitting one of the two being compared."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — the original ReAct paper this loop pattern is named after
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — the graph-of-nodes framing referenced above
- [Graph RAG](graph-rag.md) — the local/global search tools this agent chooses between
- [Corrective RAG](corrective-rag.md) — the web-search fallback pattern, here made a tool the agent picks itself instead of a fixed branch
