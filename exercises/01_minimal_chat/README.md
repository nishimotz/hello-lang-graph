# Exercise 01: LM Studio接続 + 最小チャット

## 目標

- LM Studioの Local Server に LangChain から接続する
- 最小限のチャットループを作る
- ストリーミング出力を体験する

## 前提

LM Studio で gpt-oss-20b をロードし、Local Server を起動済みであること。
デフォルトでは `http://localhost:1234/v1` でOpenAI互換APIが利用できます。

## 実行

```bash
uv run python exercises/01_minimal_chat/chat.py
```

## ポイント

- `ChatOpenAI` の `base_url` を LM Studio のローカルサーバーに向ける
- `api_key` は LM Studio では不要だが、ライブラリ的に空文字列を渡す必要がある
- `model` にはLM Studioでロード中のモデル名を指定（何でもOK、LM Studio側で無視される）
- `streaming=True` でトークン単位のリアルタイム出力が得られる

## やってみよう

1. スクリプトを実行して会話してみる
2. `temperature` や `max_tokens` を変えて応答の違いを観察する
3. `exit` で終了
