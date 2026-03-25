"""ツール呼び出しが本文にしか現れないモデル向けのフォールバック。

OpenAI 互換の `tool_calls` が空でも、`<tool>{...}</tool>` 形式や
単体 JSON (`{"tool":"web_search","args":{...}}`) を解析して
`AIMessage.tool_calls` に載せ替える。
"""

from __future__ import annotations

import json
import re
from uuid import uuid4

from langchain_core.messages import AIMessage

_COMMENTARY_RE = re.compile(r"(?s)<\|channel\|>commentary.*$")


def _tag_pattern(allowed_names: frozenset[str]) -> re.Pattern[str]:
    if not allowed_names:
        return re.compile("a^")
    alt = "|".join(re.escape(n) for n in sorted(allowed_names))
    return re.compile(
        rf"<(?P<name>{alt})>\s*(?P<body>\{{.*?\}})\s*</(?P=name)>",
        re.DOTALL,
    )


def parse_fallback_tool_calls(content: str, allowed_names: frozenset[str]) -> list[dict]:
    """本文から擬似ツール呼び出しを抽出する。`allowed_names` 以外は無視。"""
    text = _COMMENTARY_RE.sub("", content).strip()
    calls: list[dict] = []
    tag_re = _tag_pattern(allowed_names)

    for m in tag_re.finditer(text):
        name = m.group("name")
        if name not in allowed_names:
            continue
        try:
            args = json.loads(m.group("body"))
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict):
            calls.append(
                {
                    "id": f"fallback_{uuid4().hex[:8]}",
                    "type": "tool_call",
                    "name": name,
                    "args": args,
                }
            )

    if not calls:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            name = obj.get("tool") or obj.get("name")
            args = obj.get("args") or obj.get("arguments")
            if isinstance(name, str) and name in allowed_names and isinstance(args, dict):
                calls.append(
                    {
                        "id": f"fallback_{uuid4().hex[:8]}",
                        "type": "tool_call",
                        "name": name,
                        "args": args,
                    }
                )
    return calls


def augment_ai_message_with_fallback(
    response: AIMessage, allowed_names: frozenset[str]
) -> AIMessage:
    """`tool_calls` が空のときだけフォールバック解析を適用する。"""
    if response.tool_calls:
        return response
    fallback = parse_fallback_tool_calls(response.content or "", allowed_names)
    if not fallback:
        return response
    ak = getattr(response, "additional_kwargs", None) or {}
    return AIMessage(
        content="",
        tool_calls=fallback,
        additional_kwargs=dict(ak) if isinstance(ak, dict) else {},
    )
