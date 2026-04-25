# Exercise 07: Tiny Coding Agent — Immutable Edition

Exercise 06 の型ヒント必須エージェントに「イミュータブル制約」を追加。
再代入・ミューテーションを禁止し、純粋関数スタイルを強制する。

## 実行

**OpenRouter:**
```bash
# .env に以下を設定
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-...
export OPENROUTER_MODEL=deepseek/deepseek-v4-flash
```

**Ollama Cloud:**
```bash
export LLM_PROVIDER=lmstudio
export LM_STUDIO_BASE_URL=https://ollama.com/v1
export OPENAI_API_KEY=your_ollama_api_key
export LM_STUDIO_CHAT_MODEL=deepseek-v4-flash:cloud
```

```bash
make run-07
# または: uv run python exercises/07_immutable_agent/coding_agent.py
```

## ツール一覧

| ツール | 説明 |
|--------|------|
| `generate_code` | 要件に基づきイミュータブル制約付きPythonコードを生成 |
| `run_lint` | mypy + ruff(ANN/B006/C/S) + トップレベル検査 + イミュータブル検査。結果をまとめて返す |
| `fix_code` | lint エラーを元にコードを修正 |
| `save_code` | lint が通ったコードを `output/` に連番で保存 |

## イミュータブル制約とは

一度束縛した変数に別の値を代入しない、オブジェクトの状態を変更しない、というスタイル。

```python
# NG: 再代入
total: int = 0
for x in numbers:
    total += x  # AugAssign 禁止

# OK: 内包表記・組み込み関数
total: int = sum(numbers)
```

```python
# NG: ミューテーション
result: list[int] = []
result.append(x * 2)  # .append() 禁止

# OK: リスト内包表記
result: list[int] = [x * 2 for x in numbers]
```

## run_lint が検査するもの

| 検査 | 内容 |
|---|---|
| Any検査 | `Any` の使用禁止（文字列一致） |
| トップレベル検査 | 実行コードがトップレベルにないか（AST） |
| イミュータブル検査 | 再代入・累積代入・属性代入・ミューテーションメソッド（AST） |
| mypy | 型チェック |
| ruff ANN/B006/C/S | 関数アノテーション・可変デフォルト引数・複雑度4以下・動的実行禁止 |

## 試す入力例

### イミュータブル制約が効くもの

```
整数のリストを受け取って合計と平均を返す関数（ループで書こうとすると引っかかる）
```

```
文字列のリストを受け取って、各文字列の長さをキー、文字列のリストを値とする辞書を返す関数
```

```
整数のリストを受け取って昇順ソートした新しいリストを返す関数（.sort() 禁止）
```

```
二分探索で値の位置を返す関数（whileループで書くこと）
```

> **ポイント**: while ループ版はイミュータブル制約と衝突する（`left`/`right` の再代入が必須）。
> LLM は制約を説明しながら再帰版で解決する。

### 複数行入力でコードを渡す

```
このコードをイミュータブルに書き直して:
def process(items):
    result = []
    for x in items:
        result.append(x * 2)
    return result
.
```

## 制限事項

- **変数再代入検査は関数スコープのみ**: モジュールレベルの再代入は検出しない
- **ミューテーションメソッドは名前ベース**: カスタムクラスの同名メソッドも禁止になる
- **for ループのイテレータ変数は再代入扱いしない**: `for x in xs` の `x` は毎回新たな束縛として扱う
- **イミュータブル検査は false positive あり**: 正当なコードでも引っかかる場合がある
