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
| `run_lint` | mypy + ruff(ANN/B006/S) + トップレベル検査。結果をまとめて返す |
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

```bash
make run-06
# または: uv run python exercises/06_tiny_coding_agent/coding_agent.py
```

## 試す入力例

### 基本: シンプルな関数
```
整数のリストを受け取って合計と平均を返す関数
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

## ポイント

- LLM は型ヒントの正確さに弱い。`run_lint` で検証してフィードバックループを回す
- `run_lint` は mypy と ruff の両方を実行。後で pyright などを追加してもツールは同じ
- 自分でプロンプト (`prompts/generate.txt`) を編集すると生成品質が変わる

## 制限事項

- **Any 禁止は簡易検査**: `\bAny\b` の文字列一致で検出。型エイリアスや動的な None チェックまではカバーしない
- **ルール2（変数アノテーション）は自動検証対象外**: mypy は関数の引数・戻り値はチェックするが、ローカル変数のアノテーション漏れは一部しか検出しない。コード規約として提示している
- **S（bandit）ルールは false positive あり**: 標準的な subprocess 呼び出しなど一部の合法コードも警告対象になる場合がある
- **実行は信頼できるコードに限定**: このエージェントはコード生成のみ。生成されたコードを実行する機能はない
