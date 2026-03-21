# Exercise 02: LangGraphグラフ + State設計

## 目標

- LangGraphの `StateGraph` を使ってチャットをグラフとして構成する
- `TypedDict` で状態（State）を明示的に定義する
- 思考プロセス（reasoning）を状態に保存して表示する

## Exercise 01 との違い

01 ではただのループでしたが、ここでは LangGraph の **グラフ** として処理フローを定義します。
これにより、後のステップでツール呼び出しや分岐を自然に追加できます。
この段階では `chat -> END` の最小グラフだけを扱います。
図で追うより、まずは `StateGraph` と `AgentState` の対応を見る方が理解しやすい章です。

## 実行

```bash
make run-02
# または: uv run python exercises/02_langgraph_state/stateful_chat.py
```

## ポイント

- `AgentState` に `messages` と `thinking` を持たせる
- `add_messages` アノテーションで会話履歴を自動管理
- グラフのノードとして `chat_node` を定義し、エッジで繋ぐ
- `MemorySaver` でチェックポイントを有効にし、`thread_id` で会話を区別

## やってみよう

1. 実行して会話し、思考プロセスが表示されることを確認
2. `AgentState` にフィールドを追加して、好きな情報を保存してみる
3. 同じプロセス中では `thread_id` ごとに会話が保持されることを確認

## 注意

このサンプルの `MemorySaver` はプロセス内メモリです。
スクリプトを再起動すると会話は消えます。
