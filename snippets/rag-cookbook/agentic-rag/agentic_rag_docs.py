"""
Documentation-only companion to agentic_rag.py.

Not run top-to-bottom, not imported by anything. Every section below is
self-contained — its own imports, its own inlined helpers — so
copy-pasting any single section into a fresh .py works on its own. See
agentic_rag.py for the actual runnable CLI (graph construction, the
tool implementations, the full CLI).

If you change something in agentic_rag.py, mirror it here too — these
are NOT kept in sync automatically.
"""


# --8<-- [start:tools]
TOOLS = [
    {
        "name": "naive_vector_search",
        "description": (
            "Plain vector similarity search over the document. Best for a single, "
            "specific factual question already phrased using the document's own vocabulary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "graph_local_search",
        "description": (
            "Searches a knowledge graph by linking the query to specific entities and "
            "traversing their relationships. Best for questions about how two or more "
            "specific things are connected to each other."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "graph_global_search",
        "description": (
            "Searches using summaries of the whole knowledge graph's community structure. "
            "Best for broad questions that require synthesizing across many different parts "
            "of the document — e.g. 'what are the main components/themes' — where no single "
            "entity or chunk has the complete answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Searches the public web. Use this when the question is about something outside "
            "this technical paper entirely (current events, unrelated general knowledge), or "
            "when the other tools returned nothing useful for a question that clearly needs "
            "external information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
]
# --8<-- [end:tools]


# --8<-- [start:agent_loop]
import os

import anthropic

MAX_AGENT_ITERATIONS = 6


def agentic_rag(question: str, dispatch_tool, tools: list[dict] = TOOLS, max_iterations: int = MAX_AGENT_ITERATIONS) -> dict:
    """A ReAct-style loop, made of real API tool_use blocks: Claude
    picks a tool (Action), the tool runs and its output is fed back
    (Observation), and Claude decides — for itself, each iteration —
    whether it has enough to answer or needs to call another tool
    (Thought). Capped at max_iterations to avoid an unproductive loop
    running forever; if the cap is hit, an honest partial answer beats
    silently hanging.

    `dispatch_tool(name, tool_input) -> str` executes whichever tool
    Claude picked and returns its result as text — see
    agentic_rag.py's `_dispatch_tool` for the real implementation that
    routes to naive_vector_search / graph_local_search /
    graph_global_search / web_search."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    messages = [{"role": "user", "content": question}]
    trace = []

    for _ in range(max_iterations):
        response = client.messages.create(model=model, max_tokens=2048, tools=tools, messages=messages)

        if response.stop_reason != "tool_use":
            final_text = next((b.text for b in response.content if b.type == "text"), "")
            return {"trace": trace, "answer": final_text, "status": "ok"}

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = dispatch_tool(block.name, block.input)
                trace.append({"tool": block.name, "input": block.input, "result_preview": result[:200]})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "user", "content": tool_results})

    return {"trace": trace, "answer": "Reached the iteration cap without a final answer.", "status": "max_iterations"}
# --8<-- [end:agent_loop]
