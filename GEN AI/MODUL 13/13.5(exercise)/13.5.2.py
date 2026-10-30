from typing import TypedDict, Annotated
import operator, json, math

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

LIMIT_MESSAGE = "Reached step limit."


# ── Tools (sama seperti 13.3) ────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use Python math syntax."""
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error:{e}"


@tool
def token_cost_lookup(model: str) -> str:
    """Look up the cost per 1M tokens for a given LLM model."""
    costs = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    }
    if model not in costs:
        return f"Unknown model:{model}. Known:{list(costs.keys())}"
    c = costs[model]
    return json.dumps({"model": model, "input_per_1M": c["input"], "output_per_1M": c["output"]})


TOOLS = [calculator, token_cost_lookup]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

SYSTEM = SystemMessage(content="""You are a helpful AI assistant with access to two tools:
- calculator: for math calculations
- token_cost_lookup: for LLM pricing information
Always use tools when calculations or pricing lookups are needed. Think step by step.""")


class ReActState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_calls_made: int  # jumlah tool call yang sudah dieksekusi (tanpa reducer: ditimpa)
    final_answer: str


def build_agent(model_with_tools, max_tool_calls: int = 8):
    """model_with_tools: chat model yang sudah di-bind ke TOOLS."""

    def agent_node(state: ReActState) -> dict:
        response = model_with_tools.invoke([SYSTEM] + state["messages"])
        return {"messages": [response]}

    def tool_node(state: ReActState) -> dict:
        calls = state["messages"][-1].tool_calls
        results = []
        for call in calls:
            output = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        return {
            "messages": results,
            "tool_calls_made": state["tool_calls_made"] + len(calls),
        }

    def limit_node(state: ReActState) -> dict:
        return {
            "messages": [AIMessage(content=LIMIT_MESSAGE)],
            "final_answer": LIMIT_MESSAGE,
        }

    def route(state: ReActState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            # agent ingin tool call lagi, tapi jatah sudah habis -> berhenti
            if state["tool_calls_made"] >= max_tool_calls:
                return "limit"
            return "tools"
        return END

    graph = StateGraph(ReActState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("limit", limit_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "limit": "limit", END: END})
    graph.add_edge("tools", "agent")
    graph.add_edge("limit", END)
    return graph.compile()


def run(agent, question: str) -> dict:
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)], "tool_calls_made": 0, "final_answer": ""},
        config={"recursion_limit": 60},
    )
    answer = result["final_answer"] or result["messages"][-1].content
    return {"answer": answer, "tool_calls_made": result["tool_calls_made"]}


class LoopingLLM:
    """LLM tiruan yang tidak pernah memberi jawaban akhir: selalu minta tool call.
    Dipakai untuk membuktikan batas langkah bekerja, tanpa memakai kredit API."""

    def __init__(self):
        self.n = 0

    def invoke(self, messages):
        self.n += 1
        return AIMessage(content="", tool_calls=[
            {"name": "calculator", "args": {"expression": f"{self.n}+1"}, "id": f"call_{self.n}"}
        ])


if __name__ == "__main__":
    # Demo 1: agent yang tidak pernah selesai -> harus berhenti di batas 8
    looping_agent = build_agent(LoopingLLM(), max_tool_calls=8)
    out = run(looping_agent, "Keep calculating forever.")
    print("Demo loop  :", out)

    # Demo 2: agent asli (OpenRouter) pada soal 13.3 -> selesai normal sebelum batas
    from llm_utils import get_llm
    real_agent = build_agent(get_llm().bind_tools(TOOLS), max_tool_calls=8)
    out = run(real_agent, """I'm running a pipeline that processes 10,000 requests per day.
Each request uses 800 input tokens and 300 output tokens with gpt-4o.
What is my daily cost in USD?""")
    print("Agent asli :", out["tool_calls_made"], "tool call")
    print(out["answer"])