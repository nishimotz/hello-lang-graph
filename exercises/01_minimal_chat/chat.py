"""Exercise 01: LM Studio接続 + 最小チャット

LM StudioのLocal ServerにLangChainから接続し、
ストリーミングで対話するミニマルなチャットスクリプト。
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# LM Studio Local Server に接続
# api_key は LM Studio では検証されないが、ライブラリが要求するため設定
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="gpt-oss-20b",
    temperature=0.8,
    streaming=True,
)

SYSTEM_PROMPT = (
    "あなたは親切で簡潔に回答するアシスタントです。"
    "日本語で応答してください。"
)


def main() -> None:
    """メインのチャットループ。"""
    print("=== Minimal Chat (LM Studio) ===")
    print("LM Studio Local Server に接続中...")
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
        for chunk in llm.stream(messages):
            content = chunk.content
            if content:
                print(content, end="", flush=True)
                full_response += content
        print()  # 改行

        # アシスタントの応答を履歴に追加
        from langchain_core.messages import AIMessage

        messages.append(AIMessage(content=full_response))


if __name__ == "__main__":
    main()
