"""Exercise 04: 要約メモリ + 構造化メモ

会話サマリと構造化メモをファイルに保存し、
再起動後も参照できるチャット。
"""

from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from hello_lang_graph.config import build_chat_llm, get_chat_config
from hello_lang_graph.memory import (
    format_memory_context,
    load_memory_snapshot,
    save_memory_snapshot,
    update_memory_snapshot,
)

MEMORY_DIR = Path("memory_store")
MAX_CONTEXT_CHARS = 8000  # 日本語では1文字≈1〜2トークン相当


# ========== State定義 ==========


class AgentState(TypedDict):
    """エージェントの状態。"""

    messages: Annotated[list, add_messages]
    thinking: str
    context: str


# ========== LLM設定 ==========

CHAT_CONFIG = get_chat_config()
llm = build_chat_llm(temperature=0.8)

SYSTEM_PROMPT = (
    "あなたは親切なアシスタントです。日本語で応答してください。\n"
    "過去の会話サマリやユーザー情報が与えられた場合は、それも参考にしてください。"
)

# ========== コンテキスト長管理 ==========


def count_chars(messages: list) -> int:
    """メッセージ列の合計文字数を返す。"""
    return sum(len(msg.content) if hasattr(msg, "content") else 0 for msg in messages)


def summarize_if_needed(state: AgentState) -> list:
    """文字数が多い場合、古い会話を要約する。"""
    messages = state["messages"]
    if count_chars(messages) <= MAX_CONTEXT_CHARS:
        return messages

    # 古い会話を要約
    old_messages = messages[:-4]  # 最新4メッセージは残す
    if not old_messages:
        return messages

    summary_prompt = (
        "以下の会話を3文以内で要約してください:\n\n"
        + "\n".join(
            f"{'ユーザー' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in old_messages
        )
    )
    summary = llm.invoke([HumanMessage(content=summary_prompt)])
    return [
        SystemMessage(content=f"[過去の会話の要約] {summary.content}")
    ] + messages[-4:]


# ========== ツール定義 ==========


@tool
def web_search(query: str) -> str:
    """DuckDuckGoでWeb検索を行い、結果を返す。"""
    from ddgs import DDGS

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as exc:
        return f"検索に失敗しました: {exc}"
    if not results:
        return "検索結果が見つかりませんでした。"
    return "\n\n".join(
        f"**{r['title']}**\n{r['body']}\n{r['href']}" for r in results
    )


all_tools = [web_search]
llm_with_tools = llm.bind_tools(all_tools)


# ========== グラフノード ==========


def load_memory_context(_: AgentState) -> dict:
    """保存済みのサマリとメモを読み込む。"""
    snapshot = load_memory_snapshot(MEMORY_DIR)
    return {"context": format_memory_context(snapshot)}


def chat_node(state: AgentState) -> dict:
    """LLMを呼び出すノード（コンテキスト付き）。"""
    messages = summarize_if_needed(state)

    system_content = SYSTEM_PROMPT
    context = state.get("context", "")
    if context:
        system_content += f"\n\n[保存済みメモ]\n{context}"

    full_messages = [SystemMessage(content=system_content)] + messages
    response = llm_with_tools.invoke(full_messages)

    thinking = ""
    if hasattr(response, "additional_kwargs"):
        thinking = response.additional_kwargs.get("reasoning_content", "")

    return {"messages": [response], "thinking": thinking}


def save_memory_node(state: AgentState) -> dict:
    """会話サマリと構造化メモを保存する。"""
    messages = state["messages"]
    human_msg = ""
    ai_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not ai_msg and not msg.tool_calls:
            ai_msg = msg.content
        elif isinstance(msg, HumanMessage) and not human_msg:
            human_msg = msg.content
        if human_msg and ai_msg:
            break
    if human_msg and ai_msg:
        snapshot = load_memory_snapshot(MEMORY_DIR)
        next_snapshot = update_memory_snapshot(llm, snapshot, human_msg, ai_msg)
        save_memory_snapshot(MEMORY_DIR, next_snapshot)
    return {}


# ========== グラフ構築 ==========


def build_graph() -> StateGraph:
    """長期記憶付きチャットグラフを構築する。"""
    graph = StateGraph(AgentState)

    graph.add_node("load_memory", load_memory_context)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", ToolNode(all_tools))
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "chat")

    def route_after_chat(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "save_memory"

    graph.add_conditional_edges("chat", route_after_chat)
    graph.add_edge("tools", "chat")
    graph.add_edge("save_memory", END)

    return graph


def main() -> None:
    """メインのチャットループ。"""
    print("=== Memory Chat (Summary + Profile) ===")
    print("会話サマリとユーザー情報を保存して参照します")
    print(f"Provider: {CHAT_CONFIG.app_name}")
    print(f"API Base URL: {CHAT_CONFIG.base_url}")
    print(f"Chat Model: {CHAT_CONFIG.model}")
    print(f"Memory File: {MEMORY_DIR / 'summary_memory.json'}")
    print("'exit' で終了\n")

    memory = MemorySaver()
    graph = build_graph()
    app = graph.compile(checkpointer=memory)

    config = {
        "configurable": {"thread_id": "session-1"},
        "recursion_limit": 24,
    }

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
            print("[エラー] 実行に失敗しました。")
            print("  チャット用 API URL、APIキー、モデル名を確認してください。")
            print(f"  詳細: {exc}\n")
            continue

        thinking = result.get("thinking", "")
        if thinking:
            print(f"[思考] {thinking[:200]}...")

        # 最後のAIメッセージを表示
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                print(f"AI> {msg.content}")
                break
        print()


if __name__ == "__main__":
    main()
