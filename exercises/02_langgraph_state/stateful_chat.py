"""Exercise 02: LangGraphグラフ + State設計

LangGraphの StateGraph を使い、明示的な状態管理と
思考プロセスの保存を行うチャットスクリプト。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# ========== State定義 ==========


class AgentState(TypedDict):
    """エージェントの状態。"""

    messages: Annotated[list, add_messages]
    thinking: str  # 最新の思考プロセス


# ========== LLM設定 ==========

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="gpt-oss-20b",
    temperature=0.8,
)

SYSTEM_PROMPT = (
    "あなたは親切で簡潔に回答するアシスタントです。"
    "日本語で応答してください。"
)


# ========== グラフノード ==========


def chat_node(state: AgentState) -> dict:
    """LLMを呼び出して応答を生成するノード。"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)

    # 思考プロセスを抽出（reasoning_content がある場合）
    thinking = ""
    if hasattr(response, "additional_kwargs"):
        thinking = response.additional_kwargs.get("reasoning_content", "")

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

        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        # 思考プロセスがあれば表示
        thinking = result.get("thinking", "")
        if thinking:
            print(f"[思考] {thinking[:200]}...")

        # 最後のAIメッセージを表示
        last_message = result["messages"][-1]
        if isinstance(last_message, AIMessage):
            print(f"AI> {last_message.content}")
        print()


if __name__ == "__main__":
    main()
