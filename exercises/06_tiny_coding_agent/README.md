# Exercise 06: Tiny Coding Agent — 型ヒント必須Python生成

前回の Exercise 03-05 で作った「ツール呼び出し + ReAct ループ」の応用。
自然言語から型ヒントが必須の Python コードを生成・検証・修正するエージェント。

## 完成イメージ

~~~
% make run-06
==================================================
  Tiny Coding Agent
  型ヒント必須Pythonコード生成エージェント
==================================================
Provider: OpenRouter
API Base URL: https://openrouter.ai/api/v1
Chat Model: deepseek/deepseek-v4-flash
ツール: generate_code, run_lint, fix_code, save_code
'exit' で終了

You> 整数のリストを受け取って合計と最小値を返す関数    
[ツール: generate_code]
  ```python
  from collections.abc import Sequence
  
  
  def calculate_sum_and_min(numbers: Sequence[int]) -> tuple[int, int]:
      """整数のリストを受け取り、合計と最小値を返す。
  
      Args:
  ... (15行省略)
[ツール: run_lint]
  トップレベル検査: OK
  mypy: OK
  ruff(ANN/B006/S): OK
[ツール: save_code]
  保存しました: exercises/06_tiny_coding_agent/output/code_001.py

**実行コマンド:**
```bash
uv run python -i exercises/06_tiny_coding_agent/output/code_001.py
>>> 
```

AI> コードを生成・保存しました。

**保存先:** `exercises/06_tiny_coding_agent/output/code_001.py`

**実行コマンド:**
```bash
uv run python -i exercises/06_tiny_coding_agent/output/code_001.py
```

### 関数の説明

```python
def calculate_sum_and_min(numbers: Sequence[int]) -> tuple[int, int]:
```

- **引数:** `numbers: Sequence[int]` — 整数のシーケンス（`list[int]` など）
- **戻り値:** `tuple[int, int]` — 合計値と最小値のタプル

**使用例:**
```python
total, minimum = calculate_sum_and_min([3, 1, 4, 1, 5])
print(total, minimum)  # 14, 1
```

You> 
~~~

## ツール一覧

| ツール | 説明 |
|--------|------|
| `generate_code` | 要件に基づき型ヒント必須Pythonコードを生成 |
| `run_lint` | mypy + ruff(ANN/B006/C/S/UP) + トップレベル検査。結果をまとめて返す |
| `fix_code` | lint エラーを元にコードを修正 |

## 型ヒント必須サブセットのルール

1. 全ての関数に引数と戻り値の型ヒント（`run_lint` で自動検証）
2. 変数にも型ヒント（`name: Type = value` の形式。ローカル変数は努力目標、自動検証対象外）
3. `Any` 禁止
4. `int \| None` 構文（暗黙の Optional 禁止）
5. 組み込み型は `list[int]` のようにパラメータ化
6. `typing` より `collections.abc` を優先
7. 戻り値 None は `-> None` を明示
8. dataclass / TypedDict の全フィールドに型ヒント
9. 空コンテナは型付きで書く。dataclass フィールドは `field(default_factory=list)`
10. 関数のデフォルト引数に可変オブジェクト禁止（`run_lint` で自動検証）
11. `eval`, `exec`, `compile` など動的実行禁止（`run_lint` で自動検証）
12. トップレベルに実行コードを書かない（`run_lint` で自動検証）

## 実行

**OpenRouter（デフォルト設定例）:**
```bash
# .env に以下を設定
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-...
export OPENROUTER_MODEL=deepseek/deepseek-v4-flash
```

**Ollama Cloud:**
```bash
# .env に以下を設定
export LLM_PROVIDER=lmstudio
export LM_STUDIO_BASE_URL=https://ollama.com/v1
export OPENAI_API_KEY=your_ollama_api_key
export LM_STUDIO_CHAT_MODEL=deepseek-v4-flash:cloud  # ollama.com/api/tags で確認
```

**LM Studio（ローカル）:**
```bash
# .env に以下を設定
export LLM_PROVIDER=lmstudio
export LM_STUDIO_BASE_URL=http://localhost:1234/v1
export LM_STUDIO_CHAT_MODEL=your-model-name
```

```bash
make run-06
# または: uv run python exercises/06_tiny_coding_agent/coding_agent.py
```

## 試す入力例

### 基本: シンプルな関数
```
整数のリストを受け取って合計と平均を返す関数
文字列を受け取って、数字のみなら int、小数点を含むなら float、それ以外は str で返す関数
```

### 中級: データ構造
```
社員クラス（名前・年齢・部署）と、上司を辿る関数
```

### 応用: ファイル操作
```
CSVファイルを読んで、各列のデータ型を推測する関数
```

### 実践: 型チェックを通す
```
このコードを型ヒント付きに直して: [エラーが出るコードを貼る]
```

## ReAct パターン

このエージェントは **ReAct（Reasoning + Acting）** パターンで動いている。

```
[思考] 何をすべきか判断
  ↓
[行動] ツールを呼ぶ
  ↓
[観察] ツール結果を受け取る
  ↓
[思考] 結果を踏まえて次を判断
  ↓
  ... 繰り返し ...
  ↓
[回答] ループを終了して応答
```

| ReAct | このコードでの対応 |
|---|---|
| 思考 | `chat_node` で LLM が次のツールを決める |
| 行動 | `tool_calls` にツール名と引数が入る |
| 観察 | `_run_tools` が実行して `ToolMessage` を返す |
| 思考 | 再び `chat_node` に戻って結果を評価 |
| 回答 | ツール呼び出しなしで返答 → `should_continue` が `END` |

`save_code` の後にもう一度 LLM が動くのは、保存結果（観察）を受け取った LLM が「ループを終了して回答する」フェーズに入るため。

## ポイント

- LLM は型ヒントの正確さに弱い。`run_lint` で検証してフィードバックループを回す
- `run_lint` は mypy と ruff の両方を実行。後で pyright などを追加してもツールは同じ
- 自分でプロンプト (`prompts/generate.txt`) を編集すると生成品質が変わる

## 循環複雑度（McCabe Complexity）

循環複雑度はコードの分岐の多さを数値化したもの。`if`・`elif`・`for`・`while`・`except` などが1つ増えるたびに+1される。

| 複雑度 | 目安 |
|---|---|
| 1 | 分岐なし（直線的なコード） |
| 2〜4 | シンプル・テストしやすい |
| 5〜7 | やや複雑 |
| 8以上 | 複雑・分割を検討 |

このエージェントは **複雑度4以下** を強制している。`if/elif/else` が3つ重なるだけで超えるので、LLM はヘルパー関数に分割せざるを得ない。

```python
# 複雑度5（アウト）
def parse(s: str) -> int | float | str:
    if s.isdigit():        # +1
        return int(s)
    elif "." in s:         # +1
        try:               # +1
            return float(s)
        except ValueError: # +1
            return s
    else:                  # +1
        return s

# 複雑度2+2（OK）— ヘルパーに分割
def _is_float(s: str) -> bool:  # 複雑度2
    try:
        float(s)
        return True
    except ValueError:
        return False

def parse(s: str) -> int | float | str:  # 複雑度2
    if s.isdigit():
        return int(s)
    return float(s) if _is_float(s) else s
```

## 制限事項

- **Any 禁止は簡易検査**: `\bAny\b` の文字列一致で検出。型エイリアスや動的な None チェックまではカバーしない
- **ルール2（変数アノテーション）は自動検証対象外**: mypy は関数の引数・戻り値はチェックするが、ローカル変数のアノテーション漏れは一部しか検出しない。コード規約として提示している
- **S（bandit）ルールは false positive あり**: 標準的な subprocess 呼び出しなど一部の合法コードも警告対象になる場合がある
- **実行は信頼できるコードに限定**: このエージェントはコード生成のみ。生成されたコードを実行する機能はない
