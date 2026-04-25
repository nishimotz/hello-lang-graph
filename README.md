# 自分専用常駐AIエージェント

**「軽くて賢くて、24時間自分のPCで動いてくれるエージェントが欲しい」**

それをLangGraphと LM Studio または OpenRouter を使って、2時間で作ってみよう！というハンズオンです。

[hello-litestar-htmx](https://github.com/nishimotz/hello-litestar-htmx)、
[hello-pandas3](https://github.com/nishimotz/hello-pandas3) に続くシリーズ第3弾です。

## 完成イメージ（2時間後）

```text
$ make run-05

Tiny Claw
起動中... 待機します
> 今日の大阪の天気 + デスクトップに「買い物リスト.txt」追記して

[思考プロセス表示] → Web検索 → 天気取得
→ ファイル書き込み提案 → 「実行しますか？ [y/N]」
y → 「晴れ 21℃です。リストに追記完了！」
```

## 対象者

- LangGraphを触ったことはあるけど「結局何作れば…？」状態の人
- ローカルで常駐型AIエージェントを作ってみたい人
- LM Studio や OpenRouter でエージェントを試したい人
- 「DockerとかFastAPIとか全部入れるのもう疲れた…」と思ってる人

## 必要なもの（2026年3月時点）

- **ソフトウェア**
  - Python 3.13（3.14では依存ライブラリが警告を出すため）
  - [uv](https://docs.astral.sh/uv/)（パッケージマネージャ）
  - LM Studio（ローカル実行する場合）
  - OpenRouter アカウント（クラウド実行する場合）

LM Studio を使う場合の推奨ハード:

- 16GB RAM以上（RTX 3060〜4070 or M2/M3/M4 Macが快適）
- `gpt-oss-20b` GGUF（Q5_K_M または Q6_K 推奨）

## セットアップ

```bash
git clone https://github.com/nishimotz/hello-lang-graph.git
cd hello-lang-graph
make sync
# または: uv sync
```

## プロバイダの選び方

- `LM Studio`:
  ローカル完結。回線に依存しにくいが、モデル準備とマシンスペックが必要
- `OpenRouter`:
  すぐ始めやすい。LM Studio が使えない参加者向けの代替ルート

この教材では `Exercise 01-05` を LM Studio / OpenRouter のどちらでも進められます。
後半は embeddings を使わず、要約メモリと構造化メモで長期記憶を作ります。

## セットアップ例

### 1. LM Studio を使う場合

1. LM Studio でチャットモデル `gpt-oss-20b` をロードする
2. LM Studio の `Local Server` を有効にする
3. そのまま実行する

```bash
make run-01
# または: uv run python exercises/01_minimal_chat/chat.py
```

### 2. OpenRouter を使う場合

1. APIキーを設定する
2. `LLM_PROVIDER=openrouter` を設定する
3. モデル名を設定する（未設定時のデフォルトは `meta-llama/llama-3.1-8b-instruct`）

```bash
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=your_api_key
export OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
make run-01
# または: make run-openrouter-01
# または: uv run python exercises/01_minimal_chat/chat.py
```

#### OpenRouter で `404` / 「No endpoints available matching your guardrail…」になるとき

アカウントの [Privacy / Guardrails](https://openrouter.ai/settings/privacy) と、選択したモデルのデータポリシーが一致していないと、利用可能なエンドポイントが 0 件になりこのエラーになります。Privacy で無料モデル（学習利用あり等）を許可する、別モデルに `OPENROUTER_MODEL` を変える、組織や API キー単位の Guardrail を緩める、のいずれかで解消することが多いです。LM Studio と名前を揃えたい場合は `openai/gpt-oss-20b:free` なども試せますが、環境によっては上記制約で使えません。

## 実演前チェック

2時間のハンズオンをスムーズに進めるため、開始前に以下だけ確認してください。

1. 使うプロバイダを `LM Studio` か `OpenRouter` で決める
2. `Exercise 01` を先に起動してチャット疎通を確認する
3. 別ターミナルで最小動作を確認する
   ```bash
   make run-01
   ```

接続先の準備が未完了でも、各スクリプトはヒントを表示して停止するようにしてあります。

## 録画・自学習向けメモ

- **ツールが本文にしか出ないモデル**: LM Studio の一部モデルは、API の `tool_calls` が空のまま `<web_search>{"query":"..."}</web_search>` や単体 JSON だけを返すことがあります。Exercise **03〜05** では `hello_lang_graph/tool_fallback.py` がこれを解析し、既存のツール実行フローに載せ替えます。
- **危険ツールの承認**: `y` を押した事実は `HumanMessage` としては残りません。拒否時は拒否の旨が `ToolMessage` に入り、承認して実行できた場合は **ツール結果の `ToolMessage`** が間接的な記録になります。
- **ファイルの保存先**: 相対パス（例: `memo.txt`）は **`make run-03` を実行したカレントディレクトリ**に作られます。デスクトップへは **`~/Desktop/ファイル名.txt`** と依頼してください。
- **Web 検索**: 依存は **`ddgs`** パッケージです（DuckDuckGo 系の検索ラッパ）。

## 2時間タイムテーブル

| 時間       | 内容                                              | Exercise |
|------------|---------------------------------------------------|----------|
| 0:00–0:10  | 使用プロバイダ設定（LM Studio または OpenRouter） | -        |
| 0:10–0:25  | LangGraph最小グラフ（chat + stream）              | 01       |
| 0:25–0:45  | State設計（記憶＋思考プロセス保存）               | 02       |
| 0:45–1:10  | ツール呼び出し + human-in-the-loop                | 03       |
| 1:10–1:35  | 簡易長期記憶（要約メモリ + 構造化メモ）           | 04       |
| 1:35–1:50  | 常駐ループ（asyncio + CLI入力）                   | 05       |
| 1:50–2:00  | デモ・デバッグ・Q&A                               | -        |

## 第4回（2026年4月29日）タイムテーブル

| 時間       | 内容                                              | Exercise |
|------------|---------------------------------------------------|----------|
| 0:00–0:10  | 前回のおさらい                                    | 01-05    |
| 0:10–0:35  | 型ヒント必須サブセット + コード生成               | 06       |
| 0:35–0:55  | run_lint（mypy + ruff）の仕組み                   | 06       |
| 0:55–1:20  | フィードバックループ（生成→検証→修正→再検証）     | 06       |
| 1:20–1:40  | 自由に試す・ツール追加                            | 06       |
| 1:40–2:00  | デモ・共有・Q&A                                   | -        |

## Exercises

各ステップの詳細な解説は `exercises/` ディレクトリに格納されています。

| # | ディレクトリ | 内容 | スクリプト |
|---|---|---|---|
| 01 | `exercises/01_minimal_chat/` | OpenAI互換API接続 + 最小チャット | `chat.py` |
| 02 | `exercises/02_langgraph_state/` | LangGraphグラフ + State設計 | `stateful_chat.py` |
| 03 | `exercises/03_tools/` | ツール呼び出し + human-in-the-loop | `tool_chat.py` |
| 04 | `exercises/04_memory/` | 要約メモリ + 構造化メモ | `memory_chat.py` |
| 05 | `exercises/05_resident/` | 常駐ループ完成版 | `tiny_claw.py` |
| 06 | `exercises/06_tiny_coding_agent/` | 型ヒント必須Python生成エージェント | `coding_agent.py` |
| 07 | `exercises/07_immutable_agent/` | イミュータブル制約付きPython生成エージェント | `coding_agent.py` |

各 Exercise の進め方：

1. Exercise の `README.md` を読む
2. スクリプトを実行して動作を確認する
   ```bash
   make run-01
   # または: uv run python exercises/01_minimal_chat/chat.py
   ```
3. コードを読んで、編集して、再実行してみる

## 実演の進め方

実演では、毎章で次の順に進めると詰まりにくいです。

1. その章の `README.md` を30秒で説明する
2. スクリプトを起動して成功ケースを1つ見せる
3. コードの差分ポイントを2〜3箇所だけ読む
4. 受講者に「ここを変えると何が起きるか」を1つ試してもらう

各章のおすすめデモ入力:

- 01: `LangGraphって何？`
- 02: `私の好きな言語はPythonです。覚えて`
- 03: `大阪の天気を教えて`
- 03: `~/Desktop/demo_memo.txt に 今日の買い物 を書いて`
- 04: `さっき覚えてって言った好きな言語は？`
- 05: `大阪の天気を調べて ~/Desktop/demo_memo.txt に追記して`

## デモ時の注意

- `MemorySaver` はプロセス内メモリです。Exercise 02/03 の会話は再起動後には残りません。
- Exercise 04/05 の長期記憶は `summary_memory.json` に保存されるため、再起動後も参照できます。
- Exercise 05 は `asyncio` ベースですが、危険ツールの承認確認は同期入力です。
- Web 検索（`ddgs`）が使えない環境でも、エラーメッセージを返して進行できるようにしています。
- OpenRouter でも `Exercise 01-05` を通しやすくなっています。

## 推奨LM Studio設定（gpt-oss-20b）

- nGPU Layers: -1（全部GPU）
- Reasoning Effort: low（常駐） / medium（本気タスク）
- Temperature: 0.75–0.85
- Min P: 0.05
- Repeat Penalty: 1.1–1.15
- Context Length: 8192〜16384（メモリ次第）

## 発展ネタ

- Reasoning Effortを動的にlow/medium/high切り替え
- LangGraph Studioでグラフ可視化
- Langfuseでログ・トレース
- Telegram / Discord bot化
- 複数ツール（カレンダー、メール、ブラウザ操作など）
- mypy / ruff 以外の linter 追加（pyright, basedpyright）
- Human-in-the-loop 強化（承認プロセスのUI化）

## 主な環境変数

- `LLM_PROVIDER`: `lmstudio` または `openrouter`
- `LLM_CHAT_MODEL`: 共通のチャットモデル名上書き
- `LM_STUDIO_BASE_URL`: LM Studio の API URL
- `LM_STUDIO_CHAT_MODEL`: LM Studio 利用時のモデル名
- `OPENROUTER_API_KEY`: OpenRouter の APIキー
- `OPENROUTER_MODEL`: OpenRouter 利用時のモデル名（デフォルト `meta-llama/llama-3.1-8b-instruct`）
- `OPENROUTER_SITE_URL` / `OPENROUTER_APP_NAME`: OpenRouter 推奨の識別用ヘッダ（任意）
