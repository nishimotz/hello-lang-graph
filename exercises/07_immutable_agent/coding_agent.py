"""Exercise 07: イミュータブル制約付きPythonを生成する Tiny Coding Agent

Exercise 06 の型ヒント必須エージェントに「イミュータブル制約」を追加。
再代入・ミューテーションを禁止し、純粋関数スタイルを強制する。
"""

import ast
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from hello_lang_graph.config import build_chat_llm, get_chat_config
from hello_lang_graph.tool_fallback import augment_ai_message_with_fallback

# ========== サブセットのプロンプト ==========

SUBSET_RULES = (
    "## イミュータブル制約付きサブセットのルール\n\n"
    "あなたはイミュータブルな値のみを使う純粋関数スタイルのPythonコードを生成するアシスタントです。\n"
    "以下のルールに従ってください:\n\n"
    "1. 全ての関数に引数と戻り値の型ヒントを付けること（`run_lint` で自動検証）\n"
    "2. 変数にも可能な限り型ヒントを付けること（`name: Type = value` の形式。"
    "コード規約であり、静的解析では関数アノテーションを主に検証）\n"
    "3. `Any` は使用禁止\n"
    "4. 暗黙の `Optional` は禁止。`int | None` のように Union 構文を使うこと\n"
    "5. `list`, `dict` などの組み込み型は `list[int]` のようにパラメータ化すること\n"
    "6. `typing` モジュールより `collections.abc` を優先すること\n"
    "7. 戻り値が None の関数は `-> None` を明示すること\n"
    "8. `dataclasses.dataclass` を使用する場合は全フィールドに型ヒントを付けること\n"
    "9. `TypedDict` を使用する場合は全フィールドに型ヒントを付けること\n"
    "10. 空コンテナは型付きで書くこと（`items: list[int] = []`）。"
    "dataclass フィールドでは `field(default_factory=list)` を使うこと\n"
    "11. 関数のデフォルト引数に可変オブジェクト（`list`, `dict`, `set`）を使わないこと"
    "（`run_lint` で自動検証）\n"
    "12. `eval`, `exec`, `compile`, `__import__`, `getattr`, `setattr`, `delattr` の"
    "使用を禁止すること（`run_lint` で自動検証）\n"
    "13. トップレベルに実行コードを書かないこと。"
    "関数定義・クラス定義・import・定数定義のみ許可（`run_lint` で自動検証）\n"
    "14. 1関数あたりの循環複雑度は4以下にすること。"
    "複雑な処理はヘルパー関数に分割すること（`run_lint` で自動検証）\n"
    "15. 変数への再代入禁止。一度束縛した名前に別の値を代入してはいけない（`run_lint` で自動検証）\n"
    "16. `+=` `-=` などの累積代入禁止（`run_lint` で自動検証）\n"
    "17. オブジェクトの属性・添字への代入禁止（`obj.x = v`, `d[k] = v` など）（`run_lint` で自動検証）\n"
    "18. `del` 文禁止（`run_lint` で自動検証）\n"
    "19. `.append()` `.extend()` `.update()` `.add()` `.pop()` `.clear()` `.sort()` `.reverse()` など"
    "ミューテーションメソッドの呼び出し禁止（`run_lint` で自動検証）\n"
    "20. リスト・辞書の構築には内包表記や `tuple()` `frozenset()` を使うこと\n\n"
    "コードブロックは ```python から始めてください。\n"
    "生成したコードは `run_lint` ツールで検証できるように、"
    "コードブロックの中身だけでなく、ファイルとして保存可能な形式で出力してください。"
)

LINT_SYSTEM_PROMPT = (
    "あなたはイミュータブル制約と型ヒントの正確さをチェックするアシスタントです。\n"
    "与えられた lint エラーを分析し、修正後のコードを提案してください。\n"
    "エラーがない場合は「問題ありません」とだけ答えてください。\n"
    + SUBSET_RULES
)

PROMPT_DIR = Path(__file__).parent / "prompts"


# ========== State定義 ==========


class CodingState(TypedDict):
    """コーディングエージェントの状態。"""
    messages: Annotated[list, add_messages]
    thinking: str


# ========== ツール定義 ==========


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(?P<code>.*?)```", re.DOTALL)

_MUTATION_METHODS: frozenset[str] = frozenset({
    "append", "extend", "insert", "remove", "pop", "clear",
    "update", "add", "discard", "sort", "reverse",
    "setdefault", "__setitem__", "__delitem__",
})


def _extract_python_code(text: str) -> str:
    """MarkdownコードブロックからPythonコードを抽出する。
    コードブロックがない場合は全文をそのまま返す。"""
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group("code").strip()
    return text.strip()


def _check_toplevel(code: str) -> list[str]:
    """トップレベルに実行コードがないか AST で検査する。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"構文エラー: {e}"]
    violations: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (
            ast.Import, ast.ImportFrom,
            ast.FunctionDef, ast.AsyncFunctionDef,
            ast.ClassDef, ast.Assign, ast.AnnAssign,
        )):
            continue
        # モジュール docstring は許可
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            ):
                continue
        lineno = getattr(node, "lineno", "?")
        violations.append(
            f"  行{lineno}: トップレベルに実行コードがあります ({type(node).__name__})"
        )
    return violations


def _collect_target_names(target: ast.expr) -> list[ast.Name]:
    """代入ターゲットから Name ノードを再帰的に収集する（タプル展開に対応）。"""
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, ast.Tuple):
        return [name for elt in target.elts for name in _collect_target_names(elt)]
    return []


def _check_immutable(code: str) -> list[str]:
    """イミュータブル制約を AST で検査する。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"構文エラー: {e}"]
    violations: list[str] = []

    for node in ast.walk(tree):
        lineno: int | str = getattr(node, "lineno", "?")

        # 累積代入 (+=, -=, etc.)
        if isinstance(node, ast.AugAssign):
            op_name = type(node.op).__name__
            violations.append(f"  行{lineno}: 累積代入 ({op_name}) は禁止")

        # 属性・添字への代入（Assign / AnnAssign 両方）
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    violations.append(f"  行{lineno}: 属性への代入 (.{target.attr}) は禁止")
                elif isinstance(target, ast.Subscript):
                    violations.append(f"  行{lineno}: 添字への代入は禁止")

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Attribute):
                violations.append(f"  行{lineno}: 属性への代入 (.{node.target.attr}) は禁止")
            elif isinstance(node.target, ast.Subscript):
                violations.append(f"  行{lineno}: 添字への代入は禁止")

        # del 文
        elif isinstance(node, ast.Delete):
            violations.append(f"  行{lineno}: del 文は禁止")

        # ミューテーションメソッド呼び出し
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _MUTATION_METHODS
            ):
                violations.append(
                    f"  行{lineno}: ミューテーションメソッド .{node.func.attr}() は禁止"
                )

    # 関数スコープ内の変数再代入
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned: dict[str, int | str] = {}
        for child in ast.walk(func):
            targets: list[ast.expr] = []
            if isinstance(child, ast.Assign):
                targets = list(child.targets)
            elif isinstance(child, ast.AnnAssign) and child.value is not None:
                targets = [child.target]
            for target in targets:
                child_lineno: int | str = getattr(child, "lineno", "?")
                for name_node in _collect_target_names(target):
                    if name_node.id in assigned:
                        violations.append(
                            f"  行{child_lineno}: 変数 '{name_node.id}' への再代入は禁止"
                            f" (最初の代入: 行{assigned[name_node.id]})"
                        )
                    else:
                        assigned[name_node.id] = child_lineno

    return violations


def _invoke_llm(label: str, llm_instance: object, messages: list) -> object:
    """LLM を呼び出し、ラベルと所要時間を表示する。"""
    preview = str(getattr(messages[-1], "content", ""))[:60].replace("\n", " ")
    print(f"[LLM 推論中... ({label}) {preview}]")
    t0 = time.monotonic()
    response = llm_instance.invoke(messages)  # type: ignore[union-attr]
    elapsed = time.monotonic() - t0
    print(f"[LLM 完了 {elapsed:.1f}s]")
    return response


@tool
def generate_code(requirement: str) -> str:
    """与えられた要件に基づいてイミュータブル制約付きPythonコードを生成する。"""
    with open(PROMPT_DIR / "generate.txt", encoding="utf-8") as f:
        prompt_template = f.read()
    _llm = build_chat_llm(temperature=0.3)
    msgs = [SystemMessage(content=prompt_template), HumanMessage(content=requirement)]
    response = _invoke_llm("generate_code", _llm, msgs)
    return response.content  # type: ignore[union-attr]


@tool
def run_lint(code: str) -> str:
    """イミュータブル制約付きPythonコードに対して静的解析を実行する。
    mypy、ruff ANN/B006/C/S、トップレベル禁止、イミュータブル制約チェックの結果をまとめて返す。"""
    checked_code = _extract_python_code(code)
    results: list[str] = []

    # Any の簡易検査
    if re.search(r"\bAny\b", checked_code):
        results.append("Any検査: 失敗（Anyの使用は禁止されています）")

    # トップレベル実行コード禁止
    toplevel_violations = _check_toplevel(checked_code)
    if toplevel_violations:
        results.append(f"トップレベル検査: {len(toplevel_violations)}件")
        results.extend(toplevel_violations)
    else:
        results.append("トップレベル検査: OK")

    # イミュータブル制約
    immutable_violations = _check_immutable(checked_code)
    if immutable_violations:
        results.append(f"イミュータブル検査: {len(immutable_violations)}件")
        results.extend(immutable_violations[:10])
    else:
        results.append("イミュータブル検査: OK")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / "check_code.py"
        tmpfile.write_text(checked_code, encoding="utf-8")

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
                lines = [
                    line for line in mypy_out.split("\n") if "check_code.py" in line
                ]
                if lines:
                    results.append(f"mypy: {len(lines)}件")
                    results.append("  " + "\n  ".join(lines[:10]))
                else:
                    fallback = [line for line in mypy_out.split("\n") if line.strip()]
                    if fallback:
                        results.append("mypy: エラー")
                        results.append("  " + "\n  ".join(fallback[:10]))
                    else:
                        results.append("mypy: エラー（出力なし）")
        except FileNotFoundError:
            results.append("mypy: 未インストール")
        except subprocess.TimeoutExpired:
            results.append("mypy: タイムアウト")

        # ruff (ANN + B006 + C + S)
        try:
            ruff_result = subprocess.run(
                [
                    "uv", "run", "ruff", "check",
                    "--select", "ANN,B006,C,S,UP",
                    "--ignore", "ANN401,S101",
                    "--config", "lint.mccabe.max-complexity=4",
                    str(tmpfile),
                ],
                capture_output=True, text=True, timeout=30,
            )
            ruff_out = ruff_result.stdout.strip() or ruff_result.stderr.strip()
            if ruff_result.returncode == 0:
                results.append("ruff(ANN/B006/C/S/UP): OK")
            else:
                lines = [line for line in ruff_out.split("\n") if line.strip()]
                results.append(f"ruff(ANN/B006/C/S/UP): {len(lines)}件")
                if lines:
                    results.append("  " + "\n  ".join(lines[:10]))
        except FileNotFoundError:
            results.append("ruff: 未インストール")
        except subprocess.TimeoutExpired:
            results.append("ruff: タイムアウト")

    return "\n".join(results)


_OUTPUT_DIR = Path(__file__).parent / "output"


def _next_code_number(output_dir: Path) -> int:
    """output_dir 内の code_NNN.py の最大番号 + 1 を返す。"""
    nums = [
        int(m.group(1))
        for f in output_dir.glob("code_*.py")
        if (m := re.match(r"code_(\d+)\.py", f.name))
    ]
    return max(nums, default=0) + 1


@tool
def save_code(code: str) -> str:
    """lint が通ったコードをファイルに保存する。連番ファイル名で output/ に保存する。"""
    _OUTPUT_DIR.mkdir(exist_ok=True)
    next_num = _next_code_number(_OUTPUT_DIR)
    out_path = _OUTPUT_DIR / f"code_{next_num:03d}.py"
    out_path.write_text(_extract_python_code(code), encoding="utf-8")
    rel = out_path.relative_to(Path.cwd()) if out_path.is_relative_to(Path.cwd()) else out_path
    return (
        f"保存しました: {rel}\n\n"
        f"**実行コマンド:**\n"
        f"```bash\n"
        f"uv run python -i {rel}\n"
        f">>> \n"
        f"```"
    )


@tool
def fix_code(code: str, lint_results: str) -> str:
    """lint結果を元にコードを修正する。"""
    _llm = build_chat_llm(temperature=0.2)
    prompt = (
        f"以下のPythonコードは lint チェックでエラーが出ています。\n"
        f"修正後のコード全体を ```python ブロックで返してください。\n\n"
        f"## lint結果\n{lint_results}\n\n"
        f"## 現在のコード\n```python\n{code}\n```"
    )
    msgs = [SystemMessage(content=LINT_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = _invoke_llm("fix_code", _llm, msgs)
    return response.content  # type: ignore[union-attr]


# ========== 全てのツール ==========

all_tools = [generate_code, run_lint, fix_code, save_code]
FALLBACK_TOOL_NAMES = frozenset(t.name for t in all_tools)

CHAT_CONFIG = get_chat_config()
llm = build_chat_llm(temperature=0.3).bind_tools(all_tools)

SYSTEM_PROMPT = (
    "あなたはイミュータブル制約付きPythonコードを生成する Tiny Coding Agent です。\n"
    "日本語で応答してください。\n\n"
    + SUBSET_RULES + "\n\n"
    "ワークフロー:\n"
    "1. 要件を聞かれたら `generate_code` でコードを生成する\n"
    "2. 生成したコードを `run_lint` で検証する\n"
    "3. エラーがあれば `fix_code` で修正する\n"
    "4. 修正後は再度 `run_lint` で検証する\n"
    "5. lint が通るまで繰り返す\n"
    "6. lint が全て OK になったら `save_code` でファイルに保存する\n"
    "7. 保存パスと実行コマンドをユーザーに伝える"
)


# ========== グラフノード ==========


def chat_node(state: CodingState) -> dict:
    """LLMを呼び出すノード。"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = _invoke_llm(type(messages[-1]).__name__, llm, messages)
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


def _print_tool_result(tool_name: str, content: str) -> None:
    """ツール結果をツール種別に応じて整形して表示する。"""
    if tool_name == "run_lint":
        for line in content.strip().split("\n"):
            print(f"  {line}")
    elif tool_name in ("generate_code", "fix_code"):
        lines = content.strip().split("\n")
        for line in lines[:8]:
            print(f"  {line}")
        if len(lines) > 8:
            print(f"  ... ({len(lines) - 8}行省略)")
    else:
        print(f"  {content}")


def main() -> None:
    """メインのチャットループ。"""
    print("=" * 50)
    print("  Tiny Coding Agent — Immutable Edition")
    print("  イミュータブル制約付きPythonコード生成エージェント")
    print("=" * 50)
    print(f"Provider: {CHAT_CONFIG.app_name}")
    print(f"API Base URL: {CHAT_CONFIG.base_url}")
    print(f"Chat Model: {CHAT_CONFIG.model}")
    print("ツール: generate_code, run_lint, fix_code, save_code")
    print("複数行入力: 最後に '.' だけの行で送信")
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
            first_line = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not first_line:
            continue
        if first_line.lower() == "exit":
            print("終了します。")
            break

        if first_line == ".":
            continue

        lines: list[str] = [first_line]
        while True:
            try:
                line = input("... ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line == ".":
                break
            lines.append(line)
        user_input = "\n".join(lines)

        try:
            last_ai_msg: AIMessage | None = None
            last_thinking: str = ""
            pending_tool_calls: dict[str, str] = {}
            for chunk in app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_output in chunk.items():
                    if node_name == "chat":
                        thinking = node_output.get("thinking", "")
                        if thinking:
                            last_thinking = thinking
                        for msg in node_output.get("messages", []):
                            if isinstance(msg, AIMessage):
                                last_ai_msg = msg
                                if msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        print(f"[ツール: {tc['name']}]")
                                        pending_tool_calls[tc["id"]] = tc["name"]
                    elif node_name == "tools":
                        for msg in node_output.get("messages", []):
                            if isinstance(msg, ToolMessage):
                                tool_name = pending_tool_calls.get(msg.tool_call_id, "")
                                _print_tool_result(tool_name, msg.content)
        except Exception as exc:
            print(f"[エラー] {exc}\n")
            continue

        if last_thinking:
            print(f"[思考] {last_thinking[:200]}...")

        if last_ai_msg and not last_ai_msg.tool_calls:
            print(f"\nAI> {last_ai_msg.content}\n")


if __name__ == "__main__":
    main()
