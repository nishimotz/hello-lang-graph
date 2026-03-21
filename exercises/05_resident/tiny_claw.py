"""Exercise 05: 常駐ループ完成版 — Tiny Claw

全機能を統合した常駐型AIエージェント。
asyncio で非同期に動作し、ツール呼び出し・human-in-the-loop・
要約メモリを備える。
"""

import asyncio
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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
    "あなたは Tiny Claw、ユーザーのローカルPCで常駐するAIアシスタントです。\n"
    "日本語で簡潔に応答してください。\n"
    "必要に応じてツールを使ってください。\n"
    "保存済みの会話サマリやユーザー情報があれば、それも参考にしてください。"
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

    old_messages = messages[:-4]
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
    from duckduckgo_search import DDGS

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


@tool
def write_file(filepath: str, content: str) -> str:
    """指定パスにテキストファイルを書き込む。既存ファイルには追記する。"""
    path = Path(filepath).expanduser()
    try:
        if path.exists():
            with path.open("a", encoding="utf-8") as f:
                f.write(content + "\n")
        else:
            path.write_text(content + "\n", encoding="utf-8")
        return f"ファイルに書き込みました: {path}"
    except Exception as exc:
        return f"ファイル書き込みに失敗しました: {path} ({exc})"


@tool
def read_file(filepath: str) -> str:
    """指定パスのテキストファイルを読み取る。"""
    path = Path(filepath).expanduser()
    if not path.exists():
        return f"ファイルが見つかりません: {path}"
    return path.read_text(encoding="utf-8")


# ========== ツール分類 ==========

safe_tools = [web_search, read_file]
dangerous_tools = [write_file]
all_tools = safe_tools + dangerous_tools
dangerous_tool_names = {t.name for t in dangerous_tools}

llm_with_tools = llm.bind_tools(all_tools)


# ========== グラフノード ==========


def retrieve_context(state: AgentState) -> dict:
    """保存済みの会話サマリと構造化メモを読み込む。"""
    snapshot = load_memory_snapshot(MEMORY_DIR)
    return {"context": format_memory_context(snapshot)}


def chat_node(state: AgentState) -> dict:
    """LLMを呼び出すノード。"""
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


def human_review_node(state: AgentState) -> dict:
    """危険なツール実行前のユーザー確認。"""
    last = state["messages"][-1]
    results = []

    for tc in last.tool_calls:
        if tc["name"] in dangerous_tool_names:
            print(f"\n⚠ ツール実行の確認: {tc['name']}")
            print(f"  引数: {tc['args']}")
            answer = input("  実行しますか？ [y/N] ").strip().lower()
            if answer != "y":
                results.append(
                    ToolMessage(
                        content="ユーザーが実行を拒否しました。",
                        tool_call_id=tc["id"],
                    )
                )
            else:
                # 承認されたツールを実行
                tool_fn = {t.name: t for t in all_tools}[tc["name"]]
                result = tool_fn.invoke(tc["args"])
                results.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )
        else:
            # 安全なツールはそのまま実行
            tool_fn = {t.name: t for t in all_tools}[tc["name"]]
            result = tool_fn.invoke(tc["args"])
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return {"messages": results}


def route_after_chat(state: AgentState) -> str:
    """ツール呼び出しの有無で分岐。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        # 危険なツールが含まれるか判定
        for tc in last.tool_calls:
            if tc["name"] in dangerous_tool_names:
                return "human_review"
        return "safe_tools"
    return "save_memory"


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
    """全機能統合チャットグラフを構築する。"""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_context)
    graph.add_node("chat", chat_node)
    graph.add_node("safe_tools", ToolNode(safe_tools))
    graph.add_node("human_review", human_review_node)
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "chat")
    graph.add_conditional_edges("chat", route_after_chat)
    graph.add_edge("safe_tools", "chat")
    graph.add_edge("human_review", "chat")
    graph.add_edge("save_memory", END)

    return graph


# ========== メインループ ==========


async def async_input(prompt: str) -> str:
    """非同期でユーザー入力を取得する。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input(prompt).strip())


async def main() -> None:
    """非同期メインループ。"""
    print("=" * 50)
    print("  Tiny Claw")
    print("  起動中... 待機します")
    print("=" * 50)
    print(f"Provider: {CHAT_CONFIG.app_name}")
    print(f"API Base URL: {CHAT_CONFIG.base_url}")
    print(f"Chat Model: {CHAT_CONFIG.model}")
    print(f"Memory File: {MEMORY_DIR / 'summary_memory.json'}")
    print("'exit' で終了\n")

    memory = MemorySaver()
    graph = build_graph()
    app = graph.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "session-1"}}

    while True:
        try:
            user_input = await async_input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nシャットダウンします。")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("シャットダウンします。")
            break

        # LangGraph を非同期で呼び出し
        try:
            result = await app.ainvoke(
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

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                print(f"\n{msg.content}")
                break
        print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n終了しました。")
        sys.exit(0)
