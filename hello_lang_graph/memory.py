"""Helpers for summary-based persistent memory.

This module keeps the workshop memory design simple:
- a rolling summary for long conversations
- a small structured profile memory
- a few recent topics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


def load_memory_snapshot(memory_dir: Path) -> dict[str, Any]:
    """Load persistent memory from disk."""
    path = memory_dir / "summary_memory.json"
    if not path.exists():
        return {"summary": "", "profile": {}, "recent_topics": []}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"summary": "", "profile": {}, "recent_topics": []}

    return {
        "summary": str(data.get("summary", "")),
        "profile": dict(data.get("profile", {})),
        "recent_topics": list(data.get("recent_topics", [])),
    }


def save_memory_snapshot(memory_dir: Path, snapshot: dict[str, Any]) -> None:
    """Persist memory to disk."""
    memory_dir.mkdir(exist_ok=True)
    path = memory_dir / "summary_memory.json"
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_memory_context(snapshot: dict[str, Any]) -> str:
    """Render persistent memory into a prompt-friendly string."""
    sections: list[str] = []

    summary = snapshot.get("summary", "").strip()
    if summary:
        sections.append(f"[会話サマリ]\n{summary}")

    profile = snapshot.get("profile", {})
    if profile:
        profile_lines = [f"- {key}: {value}" for key, value in profile.items()]
        sections.append("[覚えていること]\n" + "\n".join(profile_lines))

    recent_topics = snapshot.get("recent_topics", [])
    if recent_topics:
        topic_lines = "\n".join(f"- {topic}" for topic in recent_topics)
        sections.append("[最近の話題]\n" + topic_lines)

    return "\n\n".join(sections)


def update_summary(llm: Any, current_summary: str, user_msg: str, ai_msg: str) -> str:
    """Refresh the rolling summary with the latest turn."""
    prompt = (
        "あなたは会話メモ更新係です。\n"
        "既存の要約と最新の会話を見て、今後の対話に必要な情報だけを3〜5文で日本語要約してください。\n"
        "雑談の言い回しは削り、継続的に役立つ事実・依頼・結論を残してください。\n\n"
        f"[既存要約]\n{current_summary or 'なし'}\n\n"
        f"[最新の会話]\nユーザー: {user_msg}\nアシスタント: {ai_msg}\n"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return str(response.content).strip()


def extract_profile_updates(llm: Any, user_msg: str, ai_msg: str) -> dict[str, str]:
    """Extract durable user facts as a small JSON object."""
    prompt = (
        "以下の会話から、今後の対話で役立つユーザー情報だけを "
        "JSON object で抽出してください。\n"
        "条件:\n"
        '- キーは短い日本語。例: "好きな言語", "住んでいる場所", "進行中タスク"\n'
        "- 値は短い文字列\n"
        "- 新しい記憶がない場合は {} を返す\n"
        "- JSON 以外を出力しない\n\n"
        f"ユーザー: {user_msg}\nアシスタント: {ai_msg}\n"
    )
    response = llm.invoke(
        [
            SystemMessage(content="あなたはJSONだけを返す情報抽出器です。"),
            HumanMessage(content=prompt),
        ]
    )
    content = str(response.content).strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key).strip(): str(value).strip()
        for key, value in data.items()
        if str(key).strip() and str(value).strip()
    }


def extract_recent_topics(llm: Any, user_msg: str, ai_msg: str) -> list[str]:
    """Extract up to three short topic labels from the latest turn."""
    prompt = (
        "以下の会話のトピックを最大3個、短い日本語フレーズの "
        "JSON array で返してください。\n"
        '例: ["大阪の天気", "買い物メモ"]\n'
        "- JSON 以外を出力しない\n\n"
        f"ユーザー: {user_msg}\nアシスタント: {ai_msg}\n"
    )
    response = llm.invoke(
        [
            SystemMessage(content="あなたはJSONだけを返すトピック抽出器です。"),
            HumanMessage(content=prompt),
        ]
    )
    content = str(response.content).strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    topics: list[str] = []
    for item in data:
        topic = str(item).strip()
        if topic and topic not in topics:
            topics.append(topic)
    return topics[:3]


def update_memory_snapshot(
    llm: Any,
    snapshot: dict[str, Any],
    user_msg: str,
    ai_msg: str,
) -> dict[str, Any]:
    """Build the next persistent memory snapshot."""
    summary = update_summary(llm, str(snapshot.get("summary", "")), user_msg, ai_msg)
    profile = dict(snapshot.get("profile", {}))
    profile.update(extract_profile_updates(llm, user_msg, ai_msg))

    recent_topics = list(snapshot.get("recent_topics", []))
    for topic in extract_recent_topics(llm, user_msg, ai_msg):
        if topic in recent_topics:
            recent_topics.remove(topic)
        recent_topics.insert(0, topic)

    return {
        "summary": summary,
        "profile": profile,
        "recent_topics": recent_topics[:5],
    }
