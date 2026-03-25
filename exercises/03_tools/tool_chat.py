"""Exercise 03: ツール呼び出し + human-in-the-loop

DuckDuckGo検索とファイル書き込みツールを使い、
危険な操作の前にユーザー確認を入れるエージェント。
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
from hello_lang_graph.tool_fallback import augment_ai_message_with_fallback

# ========== State定義 ==========


class AgentState(TypedDict):
    """エージェントの状態。"""

    messages: Annotated[list, add_messages]
    thinking: str


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


@tool
def write_file(filepath: str, content: str) -> str:
    """指定パスにテキストファイルを書き込む。既存ファイルには追記する。"""
    path = Path(filepath).expanduser()
    try:
        mode = "a" if path.exists() else "w"
        if mode == "w":
            path.write_text(content + "\n", encoding="utf-8")
        else:
            with path.open("a", encoding="utf-8") as f:
                f.write(content + "\n")
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

# 安全なツール（確認不要）
safe_tools = [web_search, read_file]
# 危険なツール（human-in-the-loop で確認）
dangerous_tools = [write_file]
all_tools = safe_tools + dangerous_tools
dangerous_tool_names = {t.name for t in dangerous_tools}
FALLBACK_TOOL_NAMES = frozenset(t.name for t in all_tools)

# ========== LLM設定 ==========

CHAT_CONFIG = get_chat_config()
llm = build_chat_llm(temperature=0.8).bind_tools(all_tools)

SYSTEM_PROMPT = (
    "あなたは親切なアシスタントです。日本語で応答してください。\n"
    "必要に応じてツールを使ってください。\n"
    "ファイル操作を頼まれたら write_file / read_file ツールを使います。\n"
    "デスクトップなど特定の場所へ書くときは filepath に ~/Desktop/ファイル名 のようにフルパスを使います。\n"
    "情報検索が必要なら web_search ツールを使います。\n"
    "すでにツール結果（ToolMessage）が会話に含まれている場合は、"
    "同じ目的で web_search を繰り返さず、その内容を要約して答えてください。"
)


# ========== グラフノード ==========


def chat_node(state: AgentState) -> dict:
    """LLMを呼び出すノード。"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    if isinstance(response, AIMessage):
        response = augment_ai_message_with_fallback(response, FALLBACK_TOOL_NAMES)
    thinking = ""
    if hasattr(response, "additional_kwargs"):
        thinking = response.additional_kwargs.get("reasoning_content", "")
    return {"messages": [response], "thinking": thinking}


def should_continue(state: AgentState) -> str:
    """ツール呼び出しの有無で分岐。危険なツールなら human 確認へ。"""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return END

    # 危険なツールが含まれるか判定
    for tc in last.tool_calls:
        if tc["name"] in dangerous_tool_names:
            return "human_review"
    return "safe_tools"


def human_review_node(state: AgentState) -> dict:
    """危険なツール実行前のユーザー確認。

    LangGraph の interrupt を使わず、シンプルに input() で確認する。
    """
    last = state["messages"][-1]
    for tc in last.tool_calls:
        if tc["name"] in dangerous_tool_names:
            print(f"\n⚠ ツール実行の確認: {tc['name']}")
            print(f"  引数: {tc['args']}")
            answer = input("  実行しますか？ [y/N] ").strip().lower()
            if answer != "y":
                from langchain_core.messages import ToolMessage

                return {
                    "messages": [
                        ToolMessage(
                            content="ユーザーが実行を拒否しました。",
                            tool_call_id=tc["id"],
                        )
                    ]
                }
    # 承認された場合はツールを実行
    tool_node = ToolNode(all_tools)
    return tool_node.invoke(state)


# ========== グラフ構築 ==========


def build_graph() -> StateGraph:
    """ツール付きチャットグラフを構築する。"""
    graph = StateGraph(AgentState)

    graph.add_node("chat", chat_node)
    graph.add_node("safe_tools", ToolNode(safe_tools))
    graph.add_node("human_review", human_review_node)

    graph.set_entry_point("chat")
    graph.add_conditional_edges("chat", should_continue)
    graph.add_edge("safe_tools", "chat")
    graph.add_edge("human_review", "chat")

    return graph


def main() -> None:
    """メインのチャットループ。"""
    print("=== Tool Chat (human-in-the-loop) ===")
    print("ツール: web_search, write_file, read_file")
    print(f"Provider: {CHAT_CONFIG.app_name}")
    print(f"API Base URL: {CHAT_CONFIG.base_url}")
    print(f"Chat Model: {CHAT_CONFIG.model}")
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
            print("  API URL、APIキー、モデル名、ネットワーク接続を確認してください。")
            print(f"  詳細: {exc}\n")
            continue

        thinking = result.get("thinking", "")
        if thinking:
            print(f"[思考] {thinking[:200]}...")

        last_message = result["messages"][-1]
        if isinstance(last_message, AIMessage):
            print(f"AI> {last_message.content}")
        print()


if __name__ == "__main__":
    main()
