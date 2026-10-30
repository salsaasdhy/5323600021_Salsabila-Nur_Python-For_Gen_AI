from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from llm_utils import get_llm

llm = get_llm()

SYSTEM = SystemMessage(content=(
    "You are a helpful assistant. Use the conversation history to answer. "
    "If the user has not told you something, say you don't know it."
))


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # riwayat percakapan bertambah di tiap giliran


def chat_node(state: ChatState) -> dict:
    response = llm.invoke([SYSTEM] + state["messages"])
    return {"messages": [response]}


def build_app(checkpointer):
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=checkpointer)


def chat(app, thread_id: str, text: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    return result["messages"][-1].content


if __name__ == "__main__":
    checkpointer = MemorySaver()  # in-memory: hilang saat program berhenti (prod: SqliteSaver)
    THREAD = "user-1"             # thread_id tetap

    # ── Percakapan 1 ──────────────────────────────────────────────────────────
    app_1 = build_app(checkpointer)
    q1 = "Hi! My name is Alex and my favorite vector database is ChromaDB."
    print("[Sesi 1] User :", q1)
    print("[Sesi 1] Agent:", chat(app_1, THREAD, q1))

    # ── Percakapan 2: objek graf BARU + invoke terpisah, thread_id SAMA ──────
    app_2 = build_app(checkpointer)
    q2 = "What is my name and what is my favorite vector database?"
    print("\n[Sesi 2] User :", q2)
    print("[Sesi 2] Agent:", chat(app_2, THREAD, q2))

    # ── Kontrol: thread_id berbeda -> tidak ada ingatan ───────────────────────
    print("\n[Thread lain] User :", q2)
    print("[Thread lain] Agent:", chat(app_2, "user-2", q2))

    # ── Bukti: isi checkpoint per thread ──────────────────────────────────────
    for tid in (THREAD, "user-2"):
        msgs = app_2.get_state({"configurable": {"thread_id": tid}}).values["messages"]
        print(f"\nThread '{tid}' menyimpan {len(msgs)} pesan")