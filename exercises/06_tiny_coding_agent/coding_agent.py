"""Exercise 06: 型ヒント必須Pythonを生成する Tiny Coding Agent

LangGraph の ReAct ループで「要求 → コード生成 → 静的解析 → 修正」を
自動で回すエージェント。型ヒントが必須の Python サブセットを対象とする。
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from hello_lang_graph.config import build_chat_llm, get_chat_config
from hello_lang_graph.tool_fallback import augment_ai_message_with_fallback

# ========== 型ヒント必須サブセットのプロンプト ==========

SUBSET_RULES = (
    "## 型ヒント必須サブセットのルール\n\n"
    "あなたは型ヒントが必須のPythonコードを生成するアシスタントです。\n"
    "以下のルールに従ってください:\n\n"
    "1. 全ての関数に引数と戻り値の型ヒントを付けること\n"
    "2. 全ての変数に型ヒントを付けること（=`value` の前に `name: Type`）\n"
    "3. `Any` は使用禁止\n"
    "4. 暗黙の `Optional` は禁止。`int | None` のように Union 構文を使うこと\n"
    "5. `list`, `dict` などの組み込み型は `list[int]` のようにパラメータ化すること\n"
    "6. `typing` モジュールより `collections.abc` を優先すること\n"
    "7. 戻り値が None の関数は `-> None` を明示すること\n"
    "8. `dataclasses.dataclass` を使用する場合は全フィールドに型ヒントを付けること\n"
    "9. `TypedDict` を使用する場合は全フィールドに型ヒントを付けること\n"
    "10. 空のコンテナは変数宣言と別行にしない（`items: list[int] = []`）\n\n"
    "コードブロックは ```python から始めてください。\n"
    "生成したコードは `run_lint` ツールで検証できるように、コードブロックの中身だけでなく、"
    "ファイルとして保存可能な形式で出力してください。"
)

LINT_SYSTEM_PROMPT = (
    "あなたは型ヒントの正確さをチェックするアシスタントです。\n"
    "与えられた lint エラーを分析し、修正後のコードを提案してください。\n"
    "エラーがない場合は「問題ありません」とだけ答えてください。\n"
    SUBSET_RULES
)

PROMPT_DIR = Path(__file__).parent / "prompts"


# ========== State定義 ==========


class CodingState(TypedDict):
    """コーディングエージェントの状態。"""
    messages: Annotated[list, add_messages]
    thinking: str
    iteration: int
    current_code: str
    lint_results: str


# ========== ツール定義 ==========


@tool
def generate_code(requirement: str) -> str:
    """与えられた要件に基づいて型ヒント必須のPythonコードを生成する。"""
    with open(PROMPT_DIR / "generate.txt", encoding="utf-8") as f:
        prompt_template = f.read()
    llm = build_chat_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=prompt_template),
        HumanMessage(content=requirement),
    ])
    return response.content


@tool
def run_lint(code: str) -> str:
    """型ヒントを含むPythonコードに対して静的解析を実行する。
    mypy, ruff の結果をまとめて返す。"""
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / "check_code.py"
        tmpfile.write_text(code, encoding="utf-8")

        # mypy
        try:
            mypy_result = subprocess.run(
                ["uv", "run", "mypy", "--python-version", "3.11", str(tmpfile)],
                capture_output=True, text=True, timeout=30,
            )
            mypy_out = mypy_result.stdout.strip() or mypy_result.stderr.strip()
            if mypy_result.returncode == 0:
                results.append("mypy: OK")
            else:
                lines = [l for l in mypy_out.split("\n") if "check_code.py" in l]
                results.append(f"mypy: {len(lines)}件")
                if lines:
                    results.append("  " + "\n  ".join(lines[:10]))
        except FileNotFoundError:
            results.append("mypy: 未インストール")
        except subprocess.TimeoutExpired:
            results.append("mypy: タイムアウト")

        # ruff
        try:
            ruff_result = subprocess.run(
                ["uv", "run", "ruff", "check", "--select", "ANN", str(tmpfile)],
                capture_output=True, text=True, timeout=30,
            )
            ruff_out = ruff_result.stdout.strip() or ruff_result.stderr.strip()
            if ruff_result.returncode == 0:
                results.append("ruff(ANN): OK")
            else:
                lines = [l for l in ruff_out.split("\n") if l.strip()]
                results.append(f"ruff(ANN): {len(lines)}件")
                if lines:
                    results.append("  " + "\n  ".join(lines[:10]))
        except FileNotFoundError:
            results.append("ruff: 未インストール")
        except subprocess.TimeoutExpired:
            results.append("ruff: タイムアウト")

    return "\n".join(results)


@tool
def fix_code(code: str, lint_results: str) -> str:
    """lint結果を元にコードを修正する。"""
    llm = build_chat_llm(temperature=0.2)
    prompt = (
        f"以下のPythonコードは型ヒントのチェックでエラーが出ています。\n"
        f"修正後のコード全体を ```python ブロックで返してください。\n\n"
        f"## lint結果\n{lint_results}\n\n"
        f"## 現在のコード\n```python\n{code}\n```"
    )
    response = llm.invoke([
        SystemMessage(content=LINT_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return response.content


# ========== 全てのツール ==========

all_tools = [generate_code, run_lint, fix_code]
FALLBACK_TOOL_NAMES = frozenset(t.name for t in all_tools)

CHAT_CONFIG = get_chat_config()
llm = build_chat_llm(temperature=0.3).bind_tools(all_tools)

SYSTEM_PROMPT = (
    "あなたは型ヒント必須Pythonコードを生成する Tiny Coding Agent です。\n"
    "日本語で応答してください。\n\n"
    SUBSET_RULES + "\n\n"
    "ワークフロー:\n"
    "1. 要件を聞かれたら `generate_code` でコードを生成する\n"
    "2. 生成したコードを `run_lint` で検証する\n"
    "3. エラーがあれば `fix_code` で修正する\n"
    "4. 修正後は再度 `run_lint` で検証する\n"
    "5. lint が通るまで繰り返す\n"
    "6. 最終的なコードを表示する"
)


# ========== グラフノード ==========


def chat_node(state: CodingState) -> dict:
    """LLMを呼び出すノード。"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    if isinstance(response, AIMessage):
        response = augment_ai_message_with_fallback(response, FALLBACK_TOOL_NAMES)
    thinking = ""
    if hasattr(response, "additional_kwargs"):
        thinking = response.additional_kwargs.get("reasoning_content", "")
    return {"messages": [response], "thinking": thinking}


def should_continue(state: CodingState) -> str:
    """ツール呼び出しの有無で分岐。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ========== グラフ構築 ==========


def build_graph() -> StateGraph:
    """コーディングエージェントのグラフを構築する。"""
    graph = StateGraph(CodingState)

    graph.add_node("chat", chat_node)
    graph.add_node("tools", lambda state: _run_tools(state))

    graph.set_entry_point("chat")
    graph.add_conditional_edges("chat", should_continue)
    graph.add_edge("tools", "chat")

    return graph


def _run_tools(state: CodingState) -> dict:
    """ツールを実行する。"""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}
    tool_map = {t.name: t for t in all_tools}
    results = []
    for tc in last.tool_calls:
        tool_fn = tool_map.get(tc["name"])
        if not tool_fn:
            available = ", ".join(sorted(tool_map))
            results.append(
                ToolMessage(
                    content=f"Unknown tool: {tc['name']}. Available: {available}",
                    tool_call_id=tc["id"],
                )
            )
            continue
        result = tool_fn.invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}


# ========== メイン ==========


def main() -> None:
    """メインのチャットループ。"""
    print("=" * 50)
    print("  Tiny Coding Agent")
    print("  型ヒント必須Pythonコード生成エージェント")
    print("=" * 50)
    print(f"Provider: {CHAT_CONFIG.app_name}")
    print(f"API Base URL: {CHAT_CONFIG.base_url}")
    print(f"Chat Model: {CHAT_CONFIG.model}")
    print("ツール: generate_code, run_lint, fix_code")
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
            print(f"[エラー] {exc}\n")
            continue

        thinking = result.get("thinking", "")
        if thinking:
            print(f"[思考] {thinking[:200]}...")

        last_msg = result["messages"][-1]
        if isinstance(last_msg, AIMessage):
            print(f"\n{last_msg.content}\n")


if __name__ == "__main__":
    main()
