"""Exercise 04: 簡易長期記憶

FAISSベクトルストアで過去の会話を検索・参照する長期記憶付きチャット。
会話が長くなったら要約してコンテキスト長を管理する。
"""

from pathlib import Path
from typing import Annotated, TypedDict

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

MEMORY_DIR = Path("memory_store")
MAX_CONTEXT_TOKENS = 4096


# ========== State定義 ==========


class AgentState(TypedDict):
    """エージェントの状態。"""

    messages: Annotated[list, add_messages]
    thinking: str
    context: str  # ベクトル検索で取得した過去の会話


# ========== LLM・Embeddings設定 ==========

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="gpt-oss-20b",
    temperature=0.8,
)

embeddings = OpenAIEmbeddings(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="text-embedding-nomic-embed-text-v1.5",
)

SYSTEM_PROMPT = (
    "あなたは親切なアシスタントです。日本語で応答してください。\n"
    "過去の会話コンテキストが与えられた場合は、それも参考にしてください。"
)

encoding = tiktoken.get_encoding("cl100k_base")


# ========== ベクトルストア管理 ==========


def get_vector_store():
    """FAISSベクトルストアを取得（既存があればロード）。"""
    from langchain_community.vectorstores import FAISS

    index_path = MEMORY_DIR / "index.faiss"
    if index_path.exists():
        return FAISS.load_local(
            str(MEMORY_DIR), embeddings, allow_dangerous_deserialization=True
        )
    return None


def save_to_memory(user_msg: str, ai_msg: str) -> None:
    """会話ペアをベクトルストアに保存する。"""
    from langchain_community.vectorstores import FAISS

    text = f"ユーザー: {user_msg}\nアシスタント: {ai_msg}"
    store = get_vector_store()
    if store is None:
        MEMORY_DIR.mkdir(exist_ok=True)
        store = FAISS.from_texts([text], embeddings)
    else:
        store.add_texts([text])
    store.save_local(str(MEMORY_DIR))


def search_memory(query: str, k: int = 3) -> str:
    """クエリに関連する過去の会話を検索する。"""
    store = get_vector_store()
    if store is None:
        return ""
    docs = store.similarity_search(query, k=k)
    if not docs:
        return ""
    return "\n---\n".join(doc.page_content for doc in docs)


# ========== トークン管理 ==========


def count_tokens(messages: list) -> int:
    """メッセージ列のトークン数を概算する。"""
    total = 0
    for msg in messages:
        total += len(encoding.encode(msg.content if hasattr(msg, "content") else ""))
    return total


def summarize_if_needed(state: AgentState) -> list:
    """トークン数が多い場合、古い会話を要約する。"""
    messages = state["messages"]
    if count_tokens(messages) <= MAX_CONTEXT_TOKENS:
        return messages

    # 古い会話を要約
    old_messages = messages[:-4]  # 最新4メッセージは残す
    if not old_messages:
        return messages

    summary_prompt = (
        "以下の会話を3文以内で要約してください:\n\n"
        + "\n".join(
            f"{'ユーザー' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in old_messages
        )
    )
    summary = llm.invoke([HumanMessage(content=summary_prompt)])
    return [
        SystemMessage(content=f"[過去の会話の要約] {summary.content}")
    ] + messages[-4:]


# ========== ツール定義 ==========


@tool
def web_search(query: str) -> str:
    """DuckDuckGoでWeb検索を行い、結果を返す。"""
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "検索結果が見つかりませんでした。"
    return "\n\n".join(
        f"**{r['title']}**\n{r['body']}\n{r['href']}" for r in results
    )


all_tools = [web_search]
llm_with_tools = llm.bind_tools(all_tools)


# ========== グラフノード ==========


def retrieve_context(state: AgentState) -> dict:
    """最新のユーザーメッセージで過去の会話を検索する。"""
    messages = state["messages"]
    last_human = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break
    context = search_memory(last_human) if last_human else ""
    return {"context": context}


def chat_node(state: AgentState) -> dict:
    """LLMを呼び出すノード（コンテキスト付き）。"""
    messages = summarize_if_needed(state)

    system_content = SYSTEM_PROMPT
    context = state.get("context", "")
    if context:
        system_content += f"\n\n[過去の関連する会話]\n{context}"

    full_messages = [SystemMessage(content=system_content)] + messages
    response = llm_with_tools.invoke(full_messages)

    thinking = ""
    if hasattr(response, "additional_kwargs"):
        thinking = response.additional_kwargs.get("reasoning_content", "")

    return {"messages": [response], "thinking": thinking}


def save_memory_node(state: AgentState) -> dict:
    """会話をベクトルストアに保存する。"""
    messages = state["messages"]
    # 最新の Human-AI ペアを保存
    human_msg = ""
    ai_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not ai_msg and not msg.tool_calls:
            ai_msg = msg.content
        elif isinstance(msg, HumanMessage) and not human_msg:
            human_msg = msg.content
        if human_msg and ai_msg:
            break
    if human_msg and ai_msg:
        save_to_memory(human_msg, ai_msg)
    return {}


# ========== グラフ構築 ==========


def build_graph() -> StateGraph:
    """長期記憶付きチャットグラフを構築する。"""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_context)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", ToolNode(all_tools))
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "chat")

    def route_after_chat(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "save_memory"

    graph.add_conditional_edges("chat", route_after_chat)
    graph.add_edge("tools", "chat")
    graph.add_edge("save_memory", END)

    return graph


def main() -> None:
    """メインのチャットループ。"""
    print("=== Memory Chat (FAISS長期記憶) ===")
    print("過去の会話をベクトル検索で参照します")
    print("'exit' で終了\n")

    memory = MemorySaver()
    graph = build_graph()
    app = graph.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "session-1"}}

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

        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        thinking = result.get("thinking", "")
        if thinking:
            print(f"[思考] {thinking[:200]}...")

        # 最後のAIメッセージを表示
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                print(f"AI> {msg.content}")
                break
        print()


if __name__ == "__main__":
    main()
