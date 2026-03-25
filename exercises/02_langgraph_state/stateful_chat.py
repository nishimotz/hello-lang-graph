"""Exercise 02: LangGraphグラフ + State設計

LangGraphの StateGraph を使い、明示的な状態管理と
思考プロセスの保存を行うチャットスクリプト。
"""

import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from hello_lang_graph.config import build_chat_llm, get_chat_config

# ========== State定義 ==========


class AgentState(TypedDict):
    """エージェントの状態。"""

    messages: Annotated[list, add_messages]
    thinking: str  # 最新の思考プロセス


# ========== LLM設定 ==========

CHAT_CONFIG = get_chat_config()
llm = build_chat_llm(temperature=0.8)

_THINKING_TAG = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)

SYSTEM_PROMPT = (
    "あなたは親切で簡潔に回答するアシスタントです。"
    "日本語で応答してください。"
    "回答の前に、推論や判断の要点を2〜4文程度で "
    "<thinking>...</thinking> タグで囲って出力し、その直後に本文を書いてください。"
)


def _thinking_and_display(message: AIMessage) -> tuple[str, str]:
    """APIの reasoning_content または <thinking> タグから思考と表示用本文を得る。"""
    content = message.content or ""
    thinking = ""
    ak = getattr(message, "additional_kwargs", None) or {}
    if isinstance(ak, dict):
        rc = ak.get("reasoning_content")
        if rc:
            thinking = str(rc).strip()

    if not thinking:
        m = _THINKING_TAG.search(content)
        if m:
            thinking = m.group(1).strip()

    display = _THINKING_TAG.sub("", content).strip() if _THINKING_TAG.search(content) else content.strip()
    return thinking, display


# ========== グラフノード ==========


def chat_node(state: AgentState) -> dict:
    """LLMを呼び出して応答を生成するノード。"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)

    thinking, _ = _thinking_and_display(response)

    return {
        "messages": [response],
        "thinking": thinking,
    }


# ========== グラフ構築 ==========


def build_graph() -> StateGraph:
    """チャットグラフを構築する。"""
    graph = StateGraph(AgentState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    return graph


def main() -> None:
    """メインのチャットループ。"""
    print("=== Stateful Chat (LangGraph) ===")
    print("LangGraph StateGraph + MemorySaver")
    print(f"Provider: {CHAT_CONFIG.app_name}")
    print(f"API Base URL: {CHAT_CONFIG.base_url}")
    print(f"Chat Model: {CHAT_CONFIG.model}")
    print("'exit' で終了\n")

    memory = MemorySaver()
    graph = build_graph()
    app = graph.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "session-1"}}

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("終了します。")
            break

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
        except Exception as exc:
            print(f"[エラー] {CHAT_CONFIG.app_name} に接続できませんでした。")
            print(f"  base_url={CHAT_CONFIG.base_url}")
            print("  API URL、APIキー、モデル名を確認してください。")
            print(f"  詳細: {exc}\n")
            continue

        # 思考プロセスがあれば表示（タグは本文から除いて表示）
        last_message = result["messages"][-1]
        if isinstance(last_message, AIMessage):
            t, display = _thinking_and_display(last_message)
            if t:
                snippet = t if len(t) <= 200 else f"{t[:200]}..."
                print(f"[思考] {snippet}")
            print(f"AI> {display}")
        print()


if __name__ == "__main__":
    main()
