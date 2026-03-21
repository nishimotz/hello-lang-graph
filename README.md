# 自分専用常駐AIエージェント

**「軽くて賢くて、24時間自分のPCで動いてくれるエージェントが欲しい」**

それをLangGraphとLM Studio + gpt-oss-20bだけで、2時間で作ってみよう！というハンズオンです。

[hello-litestar-htmx](https://github.com/nishimotz/hello-litestar-htmx)、
[hello-pandas3](https://github.com/nishimotz/hello-pandas3) に続くシリーズ第3弾です。

## 完成イメージ（2時間後）

```text
$ uv run python exercises/05_resident/tiny_claw.py

Tiny Claw (gpt-oss-20b / low effort) 起動中... 待機します
> 今日の大阪の天気 + デスクトップに「買い物リスト.txt」追記して

[思考プロセス表示] → DuckDuckGo検索 → 天気取得
→ ファイル書き込み提案 → 「実行しますか？ [y/N]」
y → 「晴れ 21℃です。リストに追記完了！」
```

## 対象者

- LangGraphを触ったことはあるけど「結局何作れば…？」状態の人
- ローカルで常駐型AIエージェントを作ってみたい人
- LM Studio + gpt-oss-20b を本気で活用したい人
- 「DockerとかFastAPIとか全部入れるのもう疲れた…」と思ってる人

## 必要なもの（2026年3月時点）

- **ハード**：16GB RAM以上（RTX 3060〜4070 or M2/M3/M4 Macが快適）
- **ソフトウェア**
  - Python 3.11+
  - [uv](https://docs.astral.sh/uv/)（パッケージマネージャ）
  - LM Studio（最新版）
  - gpt-oss-20b GGUF（Q5_K_M または Q6_K 推奨）

## セットアップ

```bash
git clone https://github.com/nishimotz/hello-lang-graph.git
cd hello-lang-graph
uv sync
```

## 2時間タイムテーブル

| 時間       | 内容                                              | Exercise |
|------------|---------------------------------------------------|----------|
| 0:00–0:10  | LM Studioでgpt-oss-20bを起動 & Local Server設定   | -        |
| 0:10–0:25  | LangGraph最小グラフ（chat + stream）              | 01       |
| 0:25–0:45  | State設計（記憶＋思考プロセス保存）               | 02       |
| 0:45–1:10  | ツール呼び出し + human-in-the-loop                | 03       |
| 1:10–1:35  | 簡易長期記憶（MemorySaver + 直近会話ベクトル）    | 04       |
| 1:35–1:50  | 常駐ループ（asyncio + CLI入力）                   | 05       |
| 1:50–2:00  | デモ・デバッグ・Q&A                               | -        |

## Exercises

各ステップの詳細な解説は `exercises/` ディレクトリに格納されています。

| # | ディレクトリ | 内容 | スクリプト |
|---|---|---|---|
| 01 | `exercises/01_minimal_chat/` | LM Studio接続 + 最小チャット | `chat.py` |
| 02 | `exercises/02_langgraph_state/` | LangGraphグラフ + State設計 | `stateful_chat.py` |
| 03 | `exercises/03_tools/` | ツール呼び出し + human-in-the-loop | `tool_chat.py` |
| 04 | `exercises/04_memory/` | 簡易長期記憶 | `memory_chat.py` |
| 05 | `exercises/05_resident/` | 常駐ループ完成版 | `tiny_claw.py` |

各 Exercise の進め方：

1. Exercise の `README.md` を読む
2. スクリプトを実行して動作を確認する
   ```bash
   uv run python exercises/01_minimal_chat/chat.py
   ```
3. コードを読んで、編集して、再実行してみる

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
