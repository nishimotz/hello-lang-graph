# Exercise 01: OpenAI互換API接続 + 最小チャット

## 目標

- OpenAI互換APIに LangChain から接続する
- 最小限のチャットループを作る
- ストリーミング出力を体験する

## 前提

以下のどちらかを準備してください。

- LM Studio でモデルをロードし、Local Server を起動する
- OpenRouter の API キーを用意する

## 実行

```bash
make run-01
# OpenRouter 利用時は環境変数を付けたうえで上と同じ、または:
# make run-openrouter-01
# または: uv run python exercises/01_minimal_chat/chat.py
```

OpenRouter で `404` と「guardrail / data policy」の文言が出る場合は、リポジトリ直下 `README.md` の「OpenRouter で 404…」の項を参照してください。

## ポイント

- `ChatOpenAI` は OpenAI互換API なら LM Studio / OpenRouter のどちらにも接続できる
- 接続先は `LLM_PROVIDER` などの環境変数で切り替えられる
- `model` には利用中のプロバイダのモデル名を指定する
- `streaming=True` でトークン単位のリアルタイム出力が得られる

## やってみよう

1. スクリプトを実行して会話してみる
2. `temperature` や `max_tokens` を変えて応答の違いを観察する
3. `exit` で終了
