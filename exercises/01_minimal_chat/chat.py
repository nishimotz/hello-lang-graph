"""Exercise 01: OpenAI互換API接続 + 最小チャット

OpenAI互換APIにLangChainから接続し、
ストリーミングで対話するミニマルなチャットスクリプト。
"""

from langchain_core.messages import HumanMessage, SystemMessage
from hello_lang_graph.config import build_chat_llm, get_chat_config

CHAT_CONFIG = get_chat_config()
llm = build_chat_llm(temperature=0.8, streaming=True)

SYSTEM_PROMPT = (
    "あなたは親切で簡潔に回答するアシスタントです。"
    "日本語で応答してください。"
)


def main() -> None:
    """メインのチャットループ。"""
    print("=== Minimal Chat ===")
    print(f"Provider: {CHAT_CONFIG.app_name}")
    print(f"API Base URL: {CHAT_CONFIG.base_url}")
    print(f"Chat Model: {CHAT_CONFIG.model}")
    print("'exit' で終了\n")

    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("終了します。")
            break

        messages.append(HumanMessage(content=user_input))

        # ストリーミングで応答を表示
        print("AI> ", end="", flush=True)
        full_response = ""
        try:
            for chunk in llm.stream(messages):
                content = chunk.content
                if content:
                    print(content, end="", flush=True)
                    full_response += content
            print()  # 改行
        except Exception as exc:
            print(f"\n[エラー] {CHAT_CONFIG.app_name} に接続できませんでした。")
            print(f"  base_url={CHAT_CONFIG.base_url}")
            print("  設定した API URL、APIキー、モデル名を確認してください。")
            print(f"  詳細: {exc}\n")
            messages.pop()
            continue

        # アシスタントの応答を履歴に追加
        from langchain_core.messages import AIMessage

        messages.append(AIMessage(content=full_response))


if __name__ == "__main__":
    main()
