from typing import TypedDict, Annotated
import operator, json, math, re

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

from llm_utils import get_llm

llm = get_llm()
MAX_STEPS = 5
MAX_TOOL_ROUNDS = 3  # batas putaran tool call di dalam satu langkah


# ── Tools ────────────────────────────────────────────────────────────────────
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
llm_with_tools = llm.bind_tools(TOOLS)


# ── State ────────────────────────────────────────────────────────────────────
class PlanState(TypedDict):
    question: str
    plan: list[str]
    current_step: int  # indeks langkah yang akan dieksekusi berikutnya
    results: Annotated[list[str], operator.add]
    final_answer: str


# ── Nodes ────────────────────────────────────────────────────────────────────
def parse_plan(text: str) -> list[str]:
    steps = [m.group(1).strip() for m in re.finditer(r"^\s*\d+[.)]\s*(.+)$", text, re.MULTILINE)]
    return steps[:MAX_STEPS]


def planner_node(state: PlanState) -> dict:
    response = llm.invoke([
        SystemMessage(content=(
            "You are a planner. Break the user's task into 2 to 4 short, concrete steps. "
            "Available tools for execution: calculator, token_cost_lookup. "
            "Output ONLY a numbered list, one step per line, like:\n1. ...\n2. ..."
        )),
        HumanMessage(content=state["question"]),
    ])
    plan = parse_plan(response.content) or [state["question"]]  # fallback: satu langkah
    return {"plan": plan, "current_step": 0}


def executor_node(state: PlanState) -> dict:
    idx = state["current_step"]
    step = state["plan"][idx]
    previous = "\n".join(state["results"]) or "(none yet)"

    messages = [
        SystemMessage(content=(
            "You execute ONE step of a plan. Use tools when you need prices or math. "
            "Finish with a short factual result for this step only."
        )),
        HumanMessage(content=(
            f"Overall task: {state['question']}\n"
            f"Results so far:\n{previous}\n\n"
            f"Current step ({idx + 1}/{len(state['plan'])}): {step}"
        )),
    ]

    text = "No result."
    for _ in range(MAX_TOOL_ROUNDS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not getattr(response, "tool_calls", None):
            text = response.content
            break
        for call in response.tool_calls:
            output = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

    return {
        "results": [f"Step {idx + 1} ({step}): {text}"],
        "current_step": idx + 1,
    }


def after_executor(state: PlanState) -> str:
    return "executor" if state["current_step"] < len(state["plan"]) else "synthesize"


def synthesize_node(state: PlanState) -> dict:
    response = llm.invoke([
        SystemMessage(content="Write the final answer to the task using the step results. Be concise."),
        HumanMessage(content=f"Task: {state['question']}\n\nStep results:\n" + "\n".join(state["results"])),
    ])
    return {"final_answer": response.content}


# ── Graph ────────────────────────────────────────────────────────────────────
graph = StateGraph(PlanState)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("synthesize", synthesize_node)
graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", after_executor, {"executor": "executor", "synthesize": "synthesize"})
graph.add_edge("synthesize", END)
agent = graph.compile()


if __name__ == "__main__":
    question = (
        "A pipeline processes 5,000 requests per day with 1,000 input tokens and 400 output "
        "tokens per request. What is the monthly cost (30 days) with gpt-4o-mini, and how much "
        "would be saved compared to gpt-4o?"
    )
    result = agent.invoke(
        {"question": question, "plan": [], "current_step": 0, "results": [], "final_answer": ""},
        config={"recursion_limit": 30},
    )
    print("PLAN:")
    for i, s in enumerate(result["plan"], 1):
        print(f"  {i}. {s}")
    print("\nHASIL PER LANGKAH:")
    for r in result["results"]:
        print(" ", r)
    print(f"\nLangkah selesai: {result['current_step']}/{len(result['plan'])}")
    print("\nJAWABAN AKHIR:\n", result["final_answer"])