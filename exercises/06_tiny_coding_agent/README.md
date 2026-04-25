# Exercise 06: Tiny Coding Agent — 型ヒント必須Python生成

前回の Exercise 03-05 で作った「ツール呼び出し + ReAct ループ」の応用。
自然言語から型ヒントが必須の Python コードを生成・検証・修正するエージェント。

## 完成イメージ

```
You> CSVを読んで各列の合計を計算する関数
[思考] 要件を分析中...
[ツール] generate_code → コード生成
[ツール] run_lint → mypy: 2件, ruff(ANN): 1件
[ツール] fix_code → 修正
[ツール] run_lint → mypy: OK, ruff(ANN): OK

AI> 型ヒント付きのコードが完成しました:
```python
def read_and_sum_csv(path: str) -> dict[str, int]:
    ...
```...
```

## ツール一覧

| ツール | 説明 |
|--------|------|
| `generate_code` | 要件に基づき型ヒント必須Pythonコードを生成 |
| `run_lint` | mypy + ruff(ANN) で静的解析。結果をまとめて返す |
| `fix_code` | lint エラーを元にコードを修正 |

## 型ヒント必須サブセットのルール

1. 全ての関数に引数と戻り値の型ヒント
2. 全ての変数に型ヒント
3. `Any` 禁止
4. `int \| None` 構文（暗黙の Optional 禁止）
5. 組み込み型は `list[int]` のようにパラメータ化
6. `typing` より `collections.abc` を優先
7. 戻り値 None は `-> None` を明示
8. dataclass / TypedDict の全フィールドに型ヒント

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
