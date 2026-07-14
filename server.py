"""
AnySearch Agent 服务 — 同一端口自动适配 nc 交互和 HTTP 调用。

启动:
  python server.py                  # 默认 0.0.0.0:9090
  python server.py --port 9090

同事使用:
  # 交互模式
  nc 127.0.0.1 9090

  # curl 一键搜索+模型总结
  curl -s http://127.0.0.1:9090/chat \
    -H "Content-Type: application/json" \
    -d '{"query":"比特币价格"}'

  # curl 指定模型
  curl -s http://127.0.0.1:9090/chat \
    -H "Content-Type: application/json" \
    -d '{"query":"今天新闻","model":"deepseek-v4-flash","max_results":5}'

  # curl 只看搜索结果（不走模型）
  curl -s http://127.0.0.1:9090/search \
    -H "Content-Type: application/json" \
    -d '{"query":"比特币价格"}'
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import parse_qs, urlparse

from search_client import SearchResult
from llm_client import chat, chat_stream
from agent import (
    format_search_results, format_sources,
    SYSTEM_PROMPT, SYSTEM_PROMPT_DIRECT,
    SEARCH_DECISION_MODEL,
    judge_query, rewrite_query,
    is_standalone_smalltalk,
    is_frontend_meta_task,
    should_rewrite_query,
    search_web_only,
)
from app_capabilities import (
    AVAILABLE_MODELS,
    is_model_capability_query,
    model_capability_answer,
)

DEFAULT_MODEL = os.getenv("LLM_HUB_MODEL", "deepseek-v4-pro")
DEFAULT_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "10"))
DEFAULT_FRESHNESS = os.getenv("SEARCH_FRESHNESS", "month")
GEMMA_STREAM_TIMEOUT = float(os.getenv("GEMMA_STREAM_TIMEOUT", "240"))
GEMMA_MAX_TOKENS = int(os.getenv("GEMMA_MAX_TOKENS", "2048"))
KIMI_CODING_MAX_TOKENS = int(os.getenv("KIMI_CODING_MAX_TOKENS", "16384"))
KIMI_CODING_STREAM_TIMEOUT = float(os.getenv("KIMI_CODING_STREAM_TIMEOUT", "240"))
THINKING_MODEL_MAX_TOKENS = int(os.getenv("THINKING_MODEL_MAX_TOKENS", "8192"))
THINKING_MODEL_STREAM_TIMEOUT = float(os.getenv("THINKING_MODEL_STREAM_TIMEOUT", "180"))
OPENCODE_MAX_TOKENS = int(os.getenv("OPENCODE_MAX_TOKENS", "16384"))
OPENCODE_STREAM_TIMEOUT = float(os.getenv("OPENCODE_STREAM_TIMEOUT", "240"))
FIRST_TOKEN_HEARTBEAT = float(os.getenv("FIRST_TOKEN_HEARTBEAT", "8"))
FIRST_TOKEN_ERROR_TIMEOUT = float(os.getenv("FIRST_TOKEN_ERROR_TIMEOUT", "30"))
KIMI_K2_FIRST_TOKEN_ERROR_TIMEOUT = float(os.getenv("KIMI_K2_FIRST_TOKEN_ERROR_TIMEOUT", "120"))
GEMMA_PROMPT_MAX_TOKENS = int(os.getenv("GEMMA_PROMPT_MAX_TOKENS", "8000"))
GEMMA_HISTORY_MAX_TURNS = int(os.getenv("GEMMA_HISTORY_MAX_TURNS", "10"))
GEMMA_HISTORY_MESSAGE_MAX_CHARS = int(os.getenv("GEMMA_HISTORY_MESSAGE_MAX_CHARS", "600"))
GEMMA_OLD_CONTEXT_SUMMARY_MAX_CHARS = int(os.getenv("GEMMA_OLD_CONTEXT_SUMMARY_MAX_CHARS", "1600"))
DEFAULT_VISION_MODELS = (
    "opencode/qwen3.6-plus",
    "gemini-3-flash",
    "gemini-3.5-flash-low",
    "gemini-3.5-flash-agent",
    "moonshot/kimi-for-coding",
    "opencode/glm-5.2",
    "opencode/glm-5.1",
    "opencode/glm-5",
    "opencode/kimi-k2.6",
    "opencode/qwen3.5-plus",
    "opencode/minimax-m2.7",
    "opencode/minimax-m2.5",
)
DEFAULT_VISION_MODEL = os.getenv("VISION_MODEL", "opencode/qwen3.6-plus")
VISION_PREPROCESS_MAX_TOKENS = int(os.getenv("VISION_PREPROCESS_MAX_TOKENS", "900"))
VISION_PREPROCESS_TIMEOUT = float(os.getenv("VISION_PREPROCESS_TIMEOUT", "60"))
IMAGE_CONTEXT_MARKER = "[图片识别结果]"
_IMAGE_REFERENCE_TERMS = (
    "图片", "图里", "图中", "图上", "这张图", "那张图", "上图", "截图",
    "照片", "画面", "图像", "识别", "看图", "刚才那张", "刚刚那张",
    "上传的图", "上传的图片", "这是什么图", "图里的", "图上的",
)


def _default_max_tokens_for_model(model: str, need_search: bool = True) -> int:
    """Reasoning-heavy models need enough completion budget for thought + answer."""
    model_name = model or DEFAULT_MODEL
    if model_name == "gemma4-crack":
        return GEMMA_MAX_TOKENS
    if model_name == "moonshot/kimi-for-coding":
        return KIMI_CODING_MAX_TOKENS
    if model_name in {"local/qwen3.5-27b-distill", "local/minimax-m2.7"}:
        return THINKING_MODEL_MAX_TOKENS
    if model_name.startswith("opencode/"):
        return OPENCODE_MAX_TOKENS
    return 4096 if need_search else 2048


def _max_tokens_for_request(requested_max_tokens, model: str, need_search: bool = True) -> int:
    model_default = _default_max_tokens_for_model(model, need_search=need_search)
    if requested_max_tokens is None:
        return model_default

    requested = int(requested_max_tokens)
    # Open WebUI often sends a generic 4k default. For Kimi Coding that can be
    # exhausted by reasoning before the final answer, so keep a model-specific floor.
    if model == "moonshot/kimi-for-coding":
        return max(requested, model_default)
    return requested


def _stream_timeout_for_model(model: str) -> float:
    if model == "gemma4-crack":
        return GEMMA_STREAM_TIMEOUT
    if model == "moonshot/kimi-for-coding":
        return KIMI_CODING_STREAM_TIMEOUT
    if model in {"local/qwen3.5-27b-distill", "local/minimax-m2.7"}:
        return THINKING_MODEL_STREAM_TIMEOUT
    if model.startswith("opencode/"):
        return OPENCODE_STREAM_TIMEOUT
    return 60.0


def _first_token_error_timeout_for_model(model: str) -> float:
    if model == "opencode/kimi-k2.6":
        return KIMI_K2_FIRST_TOKEN_ERROR_TIMEOUT
    return FIRST_TOKEN_ERROR_TIMEOUT

# ── 回答缓存 ──────────────────────────────────────────────────────
_answer_cache: dict[str, tuple[float, str]] = {}  # key → (expire_at, answer_json)
CACHE_TTL = int(os.getenv("ANSWER_CACHE_TTL", "300"))  # 默认 5 分钟
CACHE_SCHEMA_VERSION = "v4"
_image_summary_cache: dict[str, tuple[float, str]] = {}
IMAGE_SUMMARY_CACHE_TTL = int(os.getenv("IMAGE_SUMMARY_CACHE_TTL", "86400"))
_last_image_summary_by_user: dict[str, tuple[float, str]] = {}


def _message_text(content) -> str:
    if isinstance(content, list):
        return "".join(
            c.get("text", "")
            for c in content
            if isinstance(c, dict) and c.get("type", "text") == "text"
        )
    return str(content) if content else ""


def _clean_message_content(content):
    """Drop provider-specific thinking/reasoning blocks from returned history."""
    if isinstance(content, str):
        return _strip_ui_status_lines(content)
    if not isinstance(content, list):
        return content

    cleaned = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "text")
        if item_type in {"thinking", "reasoning", "reasoning_content"}:
            continue
        if "thinking" in item or "reasoning_content" in item:
            continue
        if item_type == "text":
            text = _strip_ui_status_lines(str(item.get("text", "")))
            if text:
                cleaned.append({**item, "text": text})
            continue
        cleaned.append(item)
    return cleaned


_UI_STATUS_PREFIXES = ("🔍", "🧠", "⚡", "⚠️", "🤖 正在生成回答", "💭")


def _strip_ui_status_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and any(stripped.startswith(p) for p in _UI_STATUS_PREFIXES):
            continue
        if stripped.startswith("找到 ") and " 条结果" in stripped:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _history_digest(messages: list[dict]) -> str:
    # 只取最近几轮且截断内容，避免缓存 key 过长，同时区分多轮上下文。
    relevant = []
    for m in messages[-8:]:
        role = m.get("role", "")
        content = _strip_ui_status_lines(_message_text(m.get("content", "")))[:500]
        if content:
            relevant.append({"role": role, "content": content})
    raw = json.dumps(relevant, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _messages_chars(messages: list[dict]) -> int:
    return sum(len(_message_text(m.get("content", ""))) for m in messages)


def _estimate_tokens(text: str) -> int:
    # Conservative mixed Chinese/English estimate for prompt budgeting.
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = max(len(text) - cjk, 0)
    return int(cjk * 1.2 + other / 4) + 1


def _messages_tokens(messages: list[dict]) -> int:
    return sum(_estimate_tokens(_message_text(m.get("content", ""))) for m in messages)


def _truncate_for_prompt(text: str, limit: int) -> str:
    text = text.strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _summarize_old_context(messages: list[dict]) -> str:
    lines: list[str] = []
    used = 0
    for m in messages:
        role = "用户" if m.get("role") == "user" else "助手"
        content = _strip_ui_status_lines(_message_text(m.get("content", "")))
        if not content:
            continue
        piece = f"{role}: {_truncate_for_prompt(content, 220)}"
        if used + len(piece) > GEMMA_OLD_CONTEXT_SUMMARY_MAX_CHARS:
            break
        lines.append(piece)
        used += len(piece)
    if not lines:
        return ""
    return "早期对话摘要（自动压缩，仅供理解上下文）：\n" + "\n".join(lines)


def _trim_gemma_history_for_prompt(history: list[dict]) -> list[dict]:
    max_messages = max(GEMMA_HISTORY_MAX_TURNS * 2, 0)
    old_messages = history[:-max_messages] if max_messages and len(history) > max_messages else []
    recent_messages = history[-max_messages:] if max_messages else []
    trimmed: list[dict] = []
    old_summary = _summarize_old_context(old_messages)
    if old_summary:
        trimmed.append({"role": "system", "content": old_summary})
    for m in recent_messages:
        role = m.get("role", "")
        if role not in {"user", "assistant"}:
            continue
        content = _strip_ui_status_lines(_message_text(m.get("content", "")))
        if not content:
            continue
        trimmed.append({
            "role": role,
            "content": _truncate_for_prompt(content, GEMMA_HISTORY_MESSAGE_MAX_CHARS),
        })
    return trimmed


def _cap_gemma_prompt_history(
    final_messages: list[dict],
    history_messages: list[dict],
    query_message: dict,
    *,
    date_str: str,
    search_mode: bool,
) -> list[dict]:
    before_tokens = _messages_tokens(final_messages)
    before_chars = _messages_chars(final_messages)
    if before_tokens <= GEMMA_PROMPT_MAX_TOKENS:
        print(f"  ⏱️ prompt_size chars={before_chars} tokens~{before_tokens} gemma_trimmed=False", flush=True)
        return final_messages

    system_prompt = SYSTEM_PROMPT if search_mode else SYSTEM_PROMPT_DIRECT
    history = _trim_gemma_history_for_prompt(history_messages)
    capped = [
        {"role": "system", "content": system_prompt.format(date=date_str)},
        *history,
        query_message,
    ]
    while _messages_tokens(capped) > GEMMA_PROMPT_MAX_TOKENS and history:
        history.pop(0)
        capped = [
            {"role": "system", "content": system_prompt.format(date=date_str)},
            *history,
            query_message,
        ]
    after_tokens = _messages_tokens(capped)
    after_chars = _messages_chars(capped)
    print(
        f"  ⏱️ prompt_size chars={before_chars}->{after_chars} "
        f"tokens~{before_tokens}->{after_tokens} gemma_trimmed=True",
        flush=True,
    )
    return capped


def _effective_freshness(query: str, freshness: str) -> str:
    """Tighten freshness for explicitly recent questions."""
    if freshness and freshness != DEFAULT_FRESHNESS:
        return freshness
    q = query or ""
    if any(k in q for k in ("24小时", "24 小时", "今天", "今日", "昨天", "昨日")):
        return "day"
    if any(k in q for k in ("本周", "这周", "一周", "7天", "7 天")):
        return "week"
    return freshness


def _bool_param(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "开启", "打开"}
    return default


def _cache_key(
    search_query: str,
    model: str,
    *,
    freshness: str,
    max_results: int,
    messages: list[dict],
) -> str:
    payload = {
        "version": CACHE_SCHEMA_VERSION,
        "search_query": search_query,
        "model": model,
        "freshness": freshness,
        "max_results": max_results,
        "history": _history_digest(messages),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(
    search_query: str,
    model: str,
    *,
    freshness: str,
    max_results: int,
    messages: list[dict],
) -> str | None:
    key = _cache_key(
        search_query,
        model,
        freshness=freshness,
        max_results=max_results,
        messages=messages,
    )
    entry = _answer_cache.get(key)
    if entry is None:
        return None
    expire_at, payload = entry
    if time.time() > expire_at:
        del _answer_cache[key]
        return None
    return payload


def _cache_set(
    search_query: str,
    model: str,
    payload: str,
    *,
    freshness: str,
    max_results: int,
    messages: list[dict],
) -> None:
    key = _cache_key(
        search_query,
        model,
        freshness=freshness,
        max_results=max_results,
        messages=messages,
    )
    _answer_cache[key] = (time.time() + CACHE_TTL, payload)
    # 防止内存泄漏：超过 1000 条清最老的
    if len(_answer_cache) > 1000:
        oldest = min(_answer_cache, key=lambda k: _answer_cache[k][0])
        del _answer_cache[oldest]


VISION_MODELS = {
    m.strip()
    for m in os.getenv("VISION_MODELS", ",".join(DEFAULT_VISION_MODELS)).split(",")
    if m.strip()
}


def _available_vision_models() -> list[str]:
    models = [m for m in AVAILABLE_MODELS if m in VISION_MODELS]
    if DEFAULT_VISION_MODEL in models:
        models.remove(DEFAULT_VISION_MODEL)
        return [DEFAULT_VISION_MODEL, *models]
    return models


def _vision_model_hint(model: str, image_count: int) -> str:
    image_text = "一张图片" if image_count == 1 else f"{image_count} 张图片"
    vision_models = _available_vision_models()
    if vision_models:
        return (
            f"用户上传过{image_text}，但当前模型 {model} 不是视觉模型，无法识别图片内容。"
            f"当前可用的视觉模型：{', '.join(vision_models)}。"
            "如需解析图片，请在左上角切换到视觉模型后重试。"
        )
    return (
        f"用户上传过{image_text}，但当前模型 {model} 不是视觉模型，无法识别图片内容。"
        "当前没有配置可用的视觉模型。"
    )


def _image_items(content) -> list[dict]:
    if not isinstance(content, list):
        return []
    return [
        item for item in content
        if isinstance(item, dict) and item.get("type") == "image_url"
    ]


def _content_text(content) -> str:
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type", "text") == "text"
        ).strip()
    return str(content or "").strip()


def _references_image(query: str) -> bool:
    q = (query or "").strip()
    return bool(q) and any(term in q for term in _IMAGE_REFERENCE_TERMS)


def _inject_last_image_summary(messages: list[dict], query: str, user_id: str = "default") -> tuple[list[dict], bool]:
    summary = _last_image_summary_get(user_id)
    if not summary:
        return messages, False
    injected = [
        *messages[:-1],
        {
            "role": "system",
            "content": (
                f"{IMAGE_CONTEXT_MARKER}\n"
                "以下是最近一次图片识别摘要。用户当前问题明确指向图片时，可基于它回答；"
                "不要把它用于与图片无关的问题。\n\n"
                f"{summary}"
            ),
        },
        messages[-1],
    ]
    return injected, True


def _image_cache_key(item: dict) -> str:
    image_url = item.get("image_url", {})
    if isinstance(image_url, dict):
        url = str(image_url.get("url", ""))
    else:
        url = str(image_url)
    return hashlib.sha256(url.encode()).hexdigest()


def _image_cache_get(item: dict) -> str | None:
    key = _image_cache_key(item)
    cached = _image_summary_cache.get(key)
    if not cached:
        return None
    expire_at, summary = cached
    if expire_at < time.time():
        del _image_summary_cache[key]
        return None
    return summary


def _image_cache_set(item: dict, summary: str) -> None:
    key = _image_cache_key(item)
    _image_summary_cache[key] = (time.time() + IMAGE_SUMMARY_CACHE_TTL, summary)
    if len(_image_summary_cache) > 500:
        oldest = min(_image_summary_cache, key=lambda k: _image_summary_cache[k][0])
        del _image_summary_cache[oldest]


def _last_image_summary_get(user_id: str = "default") -> str | None:
    user_key = _safe_path_part(user_id)
    entry = _last_image_summary_by_user.get(user_key)
    if entry is None:
        return None
    expire_at, summary = entry
    if expire_at < time.time():
        _last_image_summary_by_user.pop(user_key, None)
        return None
    return summary


def _last_image_summary_set(summary: str, user_id: str = "default") -> None:
    if summary.strip():
        user_key = _safe_path_part(user_id)
        _last_image_summary_by_user[user_key] = (time.time() + IMAGE_SUMMARY_CACHE_TTL, summary.strip())
    if len(_last_image_summary_by_user) > 500:
        oldest = min(_last_image_summary_by_user, key=lambda k: _last_image_summary_by_user[k][0])
        del _last_image_summary_by_user[oldest]


def _fallback_image_context(model: str, image_count: int) -> str:
    return IMAGE_CONTEXT_MARKER + "\n" + _vision_model_hint(model, image_count)


async def _describe_images_with_vision_model(text_parts: list[str], image_items: list[dict]) -> str:
    prompt = (
        "请识别用户上传的图片，输出可供纯文本模型继续回答的中文摘要。\n"
        "要求：\n"
        "1. 默认输出 5-8 条要点，总字数尽量控制在 500 字内。\n"
        "2. 如果图片中有关键文字、表格、代码或错误信息，优先 OCR 并保留必要原文。\n"
        "3. 描述主要对象、场景、截图界面和关键细节。\n"
        "4. 区分确定信息和不确定推测。\n"
        "5. 不要回答用户问题，只输出图片内容摘要。\n\n"
        f"用户随图文字：{' '.join(p for p in text_parts if p).strip() or '无'}"
    )
    content = [{"type": "text", "text": prompt}, *image_items]
    summary = await chat(
        [{"role": "user", "content": content}],
        model=DEFAULT_VISION_MODEL,
        stream=False,
        max_tokens=VISION_PREPROCESS_MAX_TOKENS,
        temperature=0.0,
        timeout=VISION_PREPROCESS_TIMEOUT,
    )
    return summary.strip()


async def _preprocess_images_for_model(
    messages: list[dict],
    model: str,
    use_image_context: bool,
    user_id: str = "default",
    status_cb=None,
) -> tuple[list[dict], int, int, int]:
    """Replace images with vision summaries before sending to text-only models."""
    if model in VISION_MODELS:
        return messages, 0, 0, 0

    cleaned_messages: list[dict] = []
    processed_images = 0
    failed_images = 0
    cached_images = 0
    latest_image_message_idx = -1
    for idx, m in enumerate(messages):
        if m.get("role") == "user" and _image_items(m.get("content", "")):
            latest_image_message_idx = idx

    for idx, m in enumerate(messages):
        content = m.get("content", "")
        if m.get("role") != "user" or not isinstance(content, list):
            cleaned_messages.append(m)
            continue

        new_content: list[dict] = []
        image_items: list[dict] = []
        text_parts: list[str] = []
        had_image_in_message = False
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url":
                had_image_in_message = True
                if use_image_context and idx == latest_image_message_idx:
                    image_items.append(item)
                continue
            if item.get("type", "text") == "text":
                text = str(item.get("text", ""))
                if text:
                    text_parts.append(text)
            new_content.append(item)

        if image_items:
            try:
                cached_summaries: list[str] = []
                missing_items: list[dict] = []
                for item in image_items:
                    cached = _image_cache_get(item)
                    if cached:
                        cached_summaries.append(cached)
                        cached_images += 1
                    else:
                        missing_items.append(item)

                new_summary = ""
                if missing_items:
                    await _emit_status(
                        status_cb,
                        "vision_start",
                        images=len(missing_items),
                        model=DEFAULT_VISION_MODEL,
                    )
                    vision_start = time.perf_counter()
                    new_summary = await _describe_images_with_vision_model(text_parts, missing_items)
                    vision_elapsed_ms = int((time.perf_counter() - vision_start) * 1000)
                    if len(missing_items) == 1:
                        _image_cache_set(missing_items[0], new_summary)
                    _last_image_summary_set(new_summary, user_id)
                    await _emit_status(
                        status_cb,
                        "vision_done",
                        images=len(missing_items),
                        model=DEFAULT_VISION_MODEL,
                        elapsed_ms=vision_elapsed_ms,
                    )

                summary_parts = [*cached_summaries]
                if new_summary:
                    summary_parts.append(new_summary)
                summary = "\n\n".join(summary_parts).strip()
                _last_image_summary_set(summary, user_id)
            except Exception as e:
                failed_images += len(image_items)
                print(
                    f"  ⚠️ vision_preprocess_error model={DEFAULT_VISION_MODEL}: {e}",
                    flush=True,
                )
                summary = _fallback_image_context(model, len(image_items))
            else:
                processed_images += len(image_items)
                summary = (
                    f"{IMAGE_CONTEXT_MARKER}\n"
                    f"视觉模型：{DEFAULT_VISION_MODEL}\n"
                    f"图片数量：{len(image_items)}\n\n"
                    f"{summary}"
                )
            new_content.append({"type": "text", "text": summary})
            cleaned_messages.append({**m, "content": new_content})
        elif had_image_in_message:
            if new_content:
                cleaned_messages.append({**m, "content": new_content})
        else:
            cleaned_messages.append(m)

    if cached_images:
        await _emit_status(
            status_cb,
            "vision_cache_hit",
            images=cached_images,
            model=DEFAULT_VISION_MODEL,
        )

    return cleaned_messages, processed_images, failed_images, cached_images

BANNER = r"""
╔══════════════════════════════════════════════════╗
║       🌐 AnySearch Agent 联网搜索助手             ║
║       模型: {model:32s} ║
║       搜索: anysearch.com                        ║
║                                                  ║
║  可用模型: /model <名> 切换                       ║
║    opencode/deepseek-v4-pro (默认)              ║
║    opencode/deepseek-v4-flash gemini-3-flash    ║
║    gemini-3.5-flash-low    gemini-3.5-flash-agent║
║    deepseek-v4-pro          deepseek-v4-flash    ║
║    moonshot/kimi-for-coding                      ║
║  OpenCode Go:                                    ║
║    opencode/glm-5.2         opencode/glm-5.1     ║
║    opencode/glm-5           opencode/kimi-k2.6   ║
║    opencode/mimo-v2.5-pro                       ║
║    opencode/qwen3.7-max     opencode/qwen3.6-plus║
║    opencode/qwen3.5-plus    opencode/minimax-m2.7║
║    opencode/minimax-m2.5                        ║
║  local:                                          ║
║    local/qwen3.5-27b-distill  local/minimax-m2.7 ║
║                                                  ║
║   输入 /help 更多    /exit 退出                   ║
╚══════════════════════════════════════════════════╝
"""

HELP = """
命令:
  /exit          退出
  /help          帮助
  /model <名>    切换模型 (当前: {model})
  /fresh <值>    搜索时效 day/week/month/year (当前: {fresh})
  /results <n>   搜索结果数 1-100 (当前: {max_results})

可用模型:
  Gemini:   gemini-3-flash  gemini-3.5-flash-low  gemini-3.5-flash-agent
  DeepSeek: opencode/deepseek-v4-pro  opencode/deepseek-v4-flash
            deepseek-v4-pro  deepseek-v4-flash
  Kimi:     moonshot/kimi-for-coding
  OpenCode Go:
    opencode/glm-5.2  opencode/glm-5.1  opencode/glm-5
    opencode/kimi-k2.6  opencode/mimo-v2.5-pro
    opencode/qwen3.7-max  opencode/qwen3.6-plus
    opencode/qwen3.5-plus  opencode/minimax-m2.7  opencode/minimax-m2.5
  本地:
    gemma4-crack  local/qwen3.5-27b-distill  local/minimax-m2.7

curl 一键调用:
  curl -s http://HOST:PORT/chat -H "Content-Type: application/json" \\
    -d '{{"query":"问题","model":"deepseek-v4-flash"}}'
"""


# ── 工具函数 ────────────────────────────────────────────────────

async def write_line(writer: asyncio.StreamWriter, text: str):
    writer.write((text + "\r\n").encode())
    await writer.drain()


async def write_raw(writer: asyncio.StreamWriter, data: bytes | str):
    if isinstance(data, str):
        data = data.encode()
    try:
        writer.write(data)
        await writer.drain()
    except (BrokenPipeError, ConnectionResetError):
        writer.close()
        pass


async def _nc_stream_answer(writer: asyncio.StreamWriter, messages: list[dict],
                            model: str, max_tokens: int = 4096):
    """nc 模式流式输出：思考用 '思考中...' 提示，最终输出直接展示"""
    think_parts: list[str] = []
    content_parts: list[str] = []
    showed_thinking = False
    async for kind, token in chat_stream(
        messages,
        model=model,
        max_tokens=max_tokens,
        fallback_on_503=True,
        timeout=_stream_timeout_for_model(model),
    ):
        if kind == "think":
            think_parts.append(token)
            if not showed_thinking:
                await write_line(writer, "💭 思考中...")
                showed_thinking = True
        else:
            if showed_thinking and not content_parts:
                await write_line(writer, "📝 回答:")
            await write_raw(writer, token)
            content_parts.append(token)
    # 只有思考没有输出时（如 gemma4 低 token），用思考作 fallback
    if not content_parts and think_parts:
        await write_line(writer, "📝 回答:")
        await write_raw(writer, "".join(think_parts))


@dataclass
class AnswerPipelineResult:
    query: str
    search_query: str
    need_search: bool
    results: list[SearchResult]
    messages: list[dict]
    direct_answer: str = ""
    timings: dict[str, int] = field(default_factory=dict)


async def _emit_status(status_cb, stage: str, **data):
    if status_cb is not None:
        await status_cb(stage, **data)


async def prepare_answer_pipeline(
    query: str,
    *,
    messages: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    search_query_override: str | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    freshness: str = DEFAULT_FRESHNESS,
    allow_rewrite: bool = False,
    status_cb=None,
) -> AnswerPipelineResult:
    """共享问答准备流程：改写追问、判断搜索、搜索、构造最终 LLM messages。"""
    base_messages = messages or [{"role": "user", "content": query}]
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    timings: dict[str, int] = {}
    has_image_context = any(
        IMAGE_CONTEXT_MARKER in _message_text(m.get("content", ""))
        for m in base_messages
    )
    has_raw_image_input = any(
        _image_items(m.get("content", ""))
        for m in base_messages
        if m.get("role") == "user"
    )
    image_only_query = (
        (has_image_context or has_raw_image_input)
        and query == "请根据图片识别结果回答用户问题。"
    )

    search_query = search_query_override or query
    if is_frontend_meta_task(query) or image_only_query:
        await _emit_status(status_cb, "direct_answer")
        return AnswerPipelineResult(
            query=query,
            search_query=query,
            need_search=False,
            results=[],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DIRECT.format(date=date_str)},
                *base_messages,
            ],
            timings=timings,
        )

    if is_model_capability_query(search_query):
        await _emit_status(status_cb, "direct_answer")
        return AnswerPipelineResult(
            query=query,
            search_query=search_query,
            need_search=False,
            results=[],
            messages=[],
            direct_answer=model_capability_answer(search_query),
            timings=timings,
        )

    if (
        search_query_override is None
        and allow_rewrite
        and query
        and len(base_messages) > 1
        and should_rewrite_query(query)
    ):
        user_count = sum(1 for m in base_messages if m.get("role") == "user")
        if user_count > 1:
            rewrite_start = time.perf_counter()
            await _emit_status(status_cb, "rewrite_start")
            try:
                search_query = await rewrite_query(query, base_messages)
                rewrite_ms = int((time.perf_counter() - rewrite_start) * 1000)
                timings["rewrite_ms"] = rewrite_ms
                if search_query != query:
                    print(f"  🔄 改写: {query!r} → {search_query!r}", flush=True)
                print(f"  ⏱️ rewrite {rewrite_ms}ms changed={search_query != query}", flush=True)
                await _emit_status(
                    status_cb,
                    "rewrite_done",
                    elapsed_ms=rewrite_ms,
                    changed=search_query != query,
                    search_query=search_query,
                )
            except Exception as e:
                rewrite_ms = int((time.perf_counter() - rewrite_start) * 1000)
                timings["rewrite_ms"] = rewrite_ms
                print(f"  ⚠️ 改写失败: {e} ({rewrite_ms}ms)", flush=True)
                await _emit_status(status_cb, "rewrite_error", elapsed_ms=rewrite_ms, error=str(e))

    need_search = False
    if search_query:
        judge_start = time.perf_counter()
        await _emit_status(status_cb, "judge_start", query=search_query)
        try:
            need_search = await judge_query(search_query)
            judge_ms = int((time.perf_counter() - judge_start) * 1000)
            timings["judge_ms"] = judge_ms
            print(f"  ⏱️ judge {judge_ms}ms need_search={need_search}", flush=True)
            await _emit_status(
                status_cb,
                "judge_done",
                elapsed_ms=judge_ms,
                need_search=need_search,
            )
        except Exception as e:
            judge_ms = int((time.perf_counter() - judge_start) * 1000)
            timings["judge_ms"] = judge_ms
            need_search = True
            print(f"  ⚠️ judge_error {judge_ms}ms: {e}", flush=True)
            await _emit_status(
                status_cb,
                "judge_error",
                elapsed_ms=judge_ms,
                fallback="search",
                error=str(e),
            )

    results: list[SearchResult] = []
    if not need_search or not search_query:
        await _emit_status(status_cb, "direct_answer")
        direct_messages = (
            [{"role": "user", "content": query}]
            if is_standalone_smalltalk(query)
            else base_messages
        )
        final_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_DIRECT.format(date=date_str)},
            *direct_messages,
        ]
        if model == "gemma4-crack" and not is_standalone_smalltalk(query):
            final_messages = _cap_gemma_prompt_history(
                final_messages,
                base_messages,
                {"role": "user", "content": query},
                date_str=date_str,
                search_mode=False,
            )
    else:
        await _emit_status(status_cb, "search_start", query=search_query)
        search_start = time.perf_counter()
        results = await search_web_only(
            search_query,
            max_results=max_results,
            freshness=freshness,
        )
        search_ms = int((time.perf_counter() - search_start) * 1000)
        timings["search_ms"] = search_ms
        print(f"  ⏱️ search {search_ms}ms results={len(results)}", flush=True)
        await _emit_status(
            status_cb,
            "search_done",
            elapsed_ms=search_ms,
            results=len(results),
        )

        search_text = format_search_results(results)
        search_context = (
            f"## 搜索结果摘要\n{search_text}\n\n"
            "请根据以上搜索信息和对话历史综合回答。引用来源时用 [序号] 标注。"
        )
        final_messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(date=date_str)},
            *base_messages[:-1],
            {"role": "user", "content": f"{query}\n\n{search_context}"},
        ]
        if model == "gemma4-crack":
            final_messages = _cap_gemma_prompt_history(
                final_messages,
                base_messages[:-1],
                {"role": "user", "content": f"{query}\n\n{search_context}"},
                date_str=date_str,
                search_mode=True,
            )

    return AnswerPipelineResult(
        query=query,
        search_query=search_query,
        need_search=need_search,
        results=results,
        messages=final_messages,
        timings=timings,
    )


# ── HTTP 响应 ────────────────────────────────────────────────────

def http_ok(writer: asyncio.StreamWriter, body: str, ct: str = "application/json"):
    data = body.encode()
    writer.write(b"HTTP/1.1 200 OK\r\n")
    writer.write(f"Content-Type: {ct}\r\nContent-Length: {len(data)}\r\n".encode())
    writer.write(b"Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n")
    writer.write(data)


def http_sse_start(writer: asyncio.StreamWriter):
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n")
    writer.write(b"Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n")
    writer.write(b"Connection: close\r\n\r\n")


async def http_sse(writer: asyncio.StreamWriter, event: str, data: str):
    await write_raw(writer, f"event: {event}\ndata: {data}\n\n")


async def http_openai_content(
    writer: asyncio.StreamWriter,
    chat_id: str,
    created: int,
    model: str,
    content: str,
):
    chunk = json.dumps({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }, ensure_ascii=False)
    await http_sse(writer, "", chunk)


def _upstream_error_summary(exc: object, model: str = "") -> str:
    raw = str(exc or "")
    lowered = raw.lower()
    model_text = f"模型 `{model}` " if model else "上游模型"

    if (
        "insufficient balance" in lowered
        or "余额不足" in raw
        or "余额不够" in raw
        or "额度不足" in raw
        or "quota" in lowered
        or "billing" in lowered
    ):
        return f"{model_text}余额不足"

    if (
        "authenticationerror" in lowered
        or "unauthorized" in lowered
        or "invalid api key" in lowered
        or "401" in lowered
        or "认证失败" in raw
    ):
        return f"{model_text}认证失败"

    if (
        "all accounts failed" in lowered
        or "unhealthy" in lowered
        or "serviceunavailable" in lowered
        or "503" in lowered
        or "token error" in lowered
        or "首 token 超时" in raw
        or "上游账号不健康" in raw
        or "服务不可用" in raw
    ):
        return f"{model_text}上游账号不健康或服务不可用"

    return f"{model_text}调用失败"


def _friendly_llm_error(exc: Exception, model: str) -> str:
    raw = str(exc)
    summary = _upstream_error_summary(exc, model)
    if "余额不足" in summary:
        return (
            f"当前{summary}，暂时无法生成回答。\n\n"
            "请切换到其他模型后重试，或联系管理员检查对应上游账号余额。"
        )
    if "认证失败" in summary:
        return (
            f"当前{summary}，暂时无法生成回答。\n\n"
            "请切换到其他模型后重试，或联系管理员检查对应上游账号/API Key。"
        )
    unhealthy_markers = (
        "All accounts failed",
        "unhealthy",
        "ServiceUnavailable",
        "503",
        "Token error",
        "首 token 超时",
        "上游账号不健康",
        "服务不可用",
    )
    if any(marker in raw for marker in unhealthy_markers):
        return (
            f"当前{summary}，暂时无法生成回答。\n\n"
            "请在左上角切换到其他模型后重试，例如 `deepseek-v4-flash`、"
            "`deepseek-v4-pro`、`gemini-3-flash` 或 `opencode/qwen3.6-plus`。"
        )
    return f"模型调用失败：{raw}"


async def http_openai_stop(writer: asyncio.StreamWriter, chat_id: str, created: int, model: str):
    stop_chunk = json.dumps({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }, ensure_ascii=False)
    await http_sse(writer, "", stop_chunk)
    await write_raw(writer, "data: [DONE]\n\n")


def _safe_path_part(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value.strip())
    safe = safe.strip(".-")
    return safe[:80] or "default"


def _request_user_id(body: dict) -> str:
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    for source in (metadata, body):
        for key in ("anysearch_user_id", "user_id", "user", "username", "email"):
            value = source.get(key)
            if value:
                return _safe_path_part(str(value))
    return "default"


def _messages_from_history_object(value) -> list[dict]:
    if isinstance(value, list):
        return [msg for msg in value if isinstance(msg, dict)]
    if not isinstance(value, dict):
        return []

    history = value.get("history") if isinstance(value.get("history"), dict) else value
    raw_messages = history.get("messages") if isinstance(history, dict) else None
    if isinstance(raw_messages, list):
        return [msg for msg in raw_messages if isinstance(msg, dict)]
    if not isinstance(raw_messages, dict):
        return []

    current_id = history.get("currentId") or history.get("current_id") or value.get("currentId")
    if current_id and current_id in raw_messages:
        ordered = []
        seen = set()
        cursor = current_id
        while cursor and cursor in raw_messages and cursor not in seen:
            seen.add(cursor)
            msg = raw_messages[cursor]
            if isinstance(msg, dict):
                ordered.append(msg)
                cursor = msg.get("parentId") or msg.get("parent_id")
            else:
                break
        ordered.reverse()
    else:
        ordered = [msg for msg in raw_messages.values() if isinstance(msg, dict)]

    messages = []
    for msg in ordered:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant", "system", "tool"} and _content_text(content):
            messages.append({"role": role, "content": content})
    return messages


def _coerce_messages_for_model(body: dict, model: str) -> list[dict]:
    messages = _messages_from_history_object(body.get("messages", []))
    if messages:
        return messages

    messages = _messages_from_history_object(body)
    if messages:
        return messages

    for container_key in ("chat", "payload", "data", "params"):
        container = body.get(container_key)
        if isinstance(container, dict):
            nested = _messages_from_history_object(container.get("messages", []))
            if nested:
                return nested
            nested = _messages_from_history_object(container)
            if nested:
                return nested

    return []


async def http_sse_comment(writer: asyncio.StreamWriter, text: str):
    """发送 SSE comment；前端会收到保活/状态，但不会作为回答正文保存。"""
    safe_text = text.replace("\r", " ").replace("\n", " ")
    await write_raw(writer, f": {safe_text}\n\n")


def http_err(writer: asyncio.StreamWriter, status: int, msg: str):
    body = json.dumps({"error": msg}).encode()
    writer.write(f"HTTP/1.1 {status} Error\r\nContent-Type: application/json\r\n".encode())
    writer.write(f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode())
    writer.write(body)


# ── HTTP 路由处理 ────────────────────────────────────────────────

async def handle_http(writer: asyncio.StreamWriter, req: dict):
    parsed_url = urlparse(req["path"])
    path = parsed_url.path
    params = parse_qs(parsed_url.query)
    print(f"[http] {req['method']} {path} body={req.get('body', b'')[:200]}", flush=True)
    try:
        body = json.loads(req["body"].decode()) if req["body"] else {}
    except json.JSONDecodeError:
        http_err(writer, 400, "Invalid JSON")
        return

    if req["method"] == "OPTIONS":
        # CORS 预检
        writer.write(b"HTTP/1.1 204 No Content\r\n")
        writer.write(b"Access-Control-Allow-Origin: *\r\n")
        writer.write(b"Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n")
        writer.write(b"Access-Control-Allow-Headers: Content-Type, Authorization\r\n")
        writer.write(b"Connection: close\r\n\r\n")
        return
    if path == "/chat" and req["method"] == "POST":
        await handle_chat_http(writer, body)
    elif path == "/search" and req["method"] == "POST":
        await handle_search_http(writer, body)
    elif path in ("/v1/models", "/v1/models/"):
        await handle_models(writer)
    elif path in ("/v1/chat/completions", "/v1/chat/completions/"):
        await handle_chat_completions(writer, body)
    else:
        msg = json.dumps({
            "endpoints": {
                "POST /chat":   "搜索+模型总结（SSE 流式），参数: query, model?, max_results?, freshness?",
                "POST /search": "仅搜索（JSON），参数: query, max_results?",
            },
            "example": 'curl -s http://HOST:PORT/chat -H "Content-Type: application/json" -d \'{"query":"比特币"}\'',
        }, ensure_ascii=False, indent=2)
        http_ok(writer, msg)


async def handle_search_http(writer: asyncio.StreamWriter, body: dict):
    query = body.get("query", "")
    max_results = body.get("max_results", DEFAULT_MAX_RESULTS)
    freshness = body.get("freshness", DEFAULT_FRESHNESS)
    if not query:
        http_err(writer, 400, "Missing query")
        return

    results = await search_web_only(query=query, max_results=max_results, freshness=freshness)
    resp = json.dumps({
        "query": query,
        "results": [{"title": r.title, "url": r.url, "snippet": r.snippet[:300]} for r in results],
    }, ensure_ascii=False, indent=2)
    http_ok(writer, resp)


async def handle_chat_http(writer: asyncio.StreamWriter, body: dict):
    query = body.get("query", "")
    model = body.get("model", DEFAULT_MODEL)
    max_results = body.get("max_results", DEFAULT_MAX_RESULTS)
    freshness = body.get("freshness", DEFAULT_FRESHNESS)

    if not query:
        http_err(writer, 400, "Missing query")
        return

    # SSE 流式输出
    http_sse_start(writer)
    await writer.drain()

    try:
        async def chat_status(stage: str, **data):
            if stage == "judge_done":
                await http_sse(writer, "judge", json.dumps({
                    "need_search": data["need_search"],
                    "elapsed_ms": data.get("elapsed_ms"),
                }, ensure_ascii=False))
            elif stage == "judge_error":
                error_summary = _upstream_error_summary(
                    data.get("error", ""),
                    SEARCH_DECISION_MODEL,
                )
                await http_sse(writer, "judge", json.dumps({
                    "need_search": True,
                    "error": f"{error_summary}: {data.get('error', '')}",
                    "elapsed_ms": data.get("elapsed_ms"),
                }, ensure_ascii=False))
            elif stage == "search_start":
                await http_sse(writer, "status", json.dumps({
                    "msg": "搜索中",
                    "query": data.get("query", query),
                }, ensure_ascii=False))
            elif stage == "search_done":
                await http_sse(writer, "status", json.dumps({
                    "msg": "搜索完成",
                    "results": data.get("results", 0),
                    "elapsed_ms": data.get("elapsed_ms"),
                }, ensure_ascii=False))

        pipeline = await prepare_answer_pipeline(
            query,
            messages=[{"role": "user", "content": query}],
            model=model,
            max_results=max_results,
            freshness=freshness,
            status_cb=chat_status,
        )
        if pipeline.need_search:
            await http_sse(writer, "sources", json.dumps([
                {"title": r.title, "url": r.url, "snippet": r.snippet[:200]} for r in pipeline.results
            ], ensure_ascii=False))

        if pipeline.direct_answer:
            await http_sse(writer, "token", json.dumps(pipeline.direct_answer, ensure_ascii=False))
            await http_sse(writer, "done", json.dumps({
                "answer": pipeline.direct_answer,
                "model": model,
                "need_search": False,
                "sources": [],
            }, ensure_ascii=False))
            return

        answer_parts = []
        think_parts = []
        max_tokens = _default_max_tokens_for_model(model, need_search=pipeline.need_search)
        async for kind, token in chat_stream(
            pipeline.messages,
            model=model,
            max_tokens=max_tokens,
            fallback_on_503=True,
            timeout=_stream_timeout_for_model(model),
        ):
            if kind == "think":
                think_parts.append(token)
                await http_sse(writer, "think", json.dumps(token, ensure_ascii=False))
            else:
                answer_parts.append(token)
                await http_sse(writer, "token", json.dumps(token, ensure_ascii=False))

        await http_sse(writer, "done", json.dumps({
            "answer": "".join(answer_parts) or "".join(think_parts),
            "model": model,
            "need_search": pipeline.need_search,
            "sources": [{"title": r.title, "url": r.url} for r in pipeline.results],
        }, ensure_ascii=False))
    except Exception as e:
        await http_sse(writer, "error", json.dumps({"error": str(e)}, ensure_ascii=False))


# ── OpenAI 兼容端点（供 LobeChat / NextChat 等前端接入）───────────

async def handle_models(writer: asyncio.StreamWriter):
    """GET /v1/models — 返回可用模型列表"""
    models = [{"id": m, "object": "model", "created": 0, "owned_by": "anysearch"} for m in AVAILABLE_MODELS]
    resp = json.dumps({"object": "list", "data": models}, ensure_ascii=False)
    http_ok(writer, resp)


async def handle_chat_completions(writer: asyncio.StreamWriter, body: dict):
    """POST /v1/chat/completions — OpenAI 兼容接口，内置搜索增强 + 缓存"""
    req_start = time.perf_counter()
    model = body.get("model", DEFAULT_MODEL)
    messages = _coerce_messages_for_model(body, model)
    max_results = body.get("max_results", DEFAULT_MAX_RESULTS)
    freshness = body.get("freshness", DEFAULT_FRESHNESS)
    stream = body.get("stream", True)
    requested_max_tokens = body.get("max_tokens")
    user_id = _request_user_id(body)

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    if not messages:
        http_err(writer, 400, "Missing messages")
        return

    # 清理上轮回流的 UI 状态和 thinking/reasoning 块，避免污染下一轮请求。
    cleaned = []
    for m in messages:
        content = m.get("content", "")
        if m.get("role") == "assistant":
            cleaned_content = _clean_message_content(content)
            if not cleaned_content:
                continue
            m = {**m, "content": cleaned_content}
        cleaned.append(m)
    messages = cleaned

    if stream:
        http_sse_start(writer)
        await writer.drain()
        role_chunk = json.dumps({
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }, ensure_ascii=False)
        await http_sse(writer, "", role_chunk)

    async def preprocess_status(stage: str, **data):
        if not stream:
            return
        if stage == "vision_start":
            await http_openai_content(
                writer,
                chat_id,
                created,
                model,
                f"🖼️ 正在识别图片（{data.get('model')}）...\n\n",
            )
            await http_sse_comment(
                writer,
                f"status=vision_start model={data.get('model')} images={data.get('images')}",
            )
        elif stage == "vision_cache_hit":
            await http_openai_content(
                writer,
                chat_id,
                created,
                model,
                "🖼️ 已复用图片识别结果\n\n",
            )
            await http_sse_comment(
                writer,
                f"status=vision_cache_hit model={data.get('model')} images={data.get('images')}",
            )
        elif stage == "vision_done":
            elapsed = data.get("elapsed_ms")
            elapsed_text = f"（{elapsed}ms）" if elapsed is not None else ""
            await http_openai_content(
                writer,
                chat_id,
                created,
                model,
                f"🖼️ 图片识别完成{elapsed_text}\n\n",
            )
            await http_sse_comment(
                writer,
                f"status=vision_done model={data.get('model')} images={data.get('images')} elapsed_ms={elapsed}",
            )

    had_image_input = any(_image_items(m.get("content", "")) for m in messages if m.get("role") == "user")
    latest_user_content = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    latest_has_image = isinstance(latest_user_content, list) and bool(_image_items(latest_user_content))
    latest_text = _content_text(latest_user_content)
    latest_image_only = False
    if latest_has_image:
        latest_image_only = not latest_text
    use_image_context = latest_has_image or _references_image(latest_text)
    injected_last_image_summary = False
    if use_image_context and not latest_has_image:
        messages, injected_last_image_summary = _inject_last_image_summary(messages, latest_text, user_id)

    direct_vision_image_items = _image_items(latest_user_content) if latest_image_only else []
    if had_image_input and latest_image_only and model not in VISION_MODELS:
        if stream:
            await http_openai_content(
                writer,
                chat_id,
                created,
                model,
                f"🖼️ 纯图片问题，切换视觉模型回答（{DEFAULT_VISION_MODEL}）\n\n",
            )
        model = DEFAULT_VISION_MODEL

    messages, processed_images, failed_images, cached_images = await _preprocess_images_for_model(
        messages,
        model,
        use_image_context=use_image_context,
        user_id=user_id,
        status_cb=preprocess_status,
    )
    if injected_last_image_summary:
        cached_images += 1
        await preprocess_status(
            "vision_cache_hit",
            images=1,
            model=DEFAULT_VISION_MODEL,
        )
    if processed_images or failed_images:
        print(
            f"  🖼️ vision_preprocess processed={processed_images} cached={cached_images} failed={failed_images} "
            f"text_model={model} vision_model={DEFAULT_VISION_MODEL}",
            flush=True,
        )

    # 提取最后一条 user 消息作为搜索 query（兼容 string 和 array 两种 content 格式）
    query = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                text_items = [
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type", "text") == "text"
                ]
                user_text_items = [
                    t for t in text_items
                    if t and not str(t).startswith(IMAGE_CONTEXT_MARKER)
                ]
                query = "".join(user_text_items).strip()
                if not query and had_image_input:
                    query = "请根据图片识别结果回答用户问题。"
            else:
                query = str(content) if content else ""
            break

    if stream:
        await http_openai_content(writer, chat_id, created, model, "🔍 分析问题中...\n\n")
        print(f"  ⏱️ sse_open {int((time.perf_counter() - req_start) * 1000)}ms query={query[:80]!r}", flush=True)

    # ── 多轮对话：追问改写为独立搜索 query ──────────────────────
    search_query = query
    if query and len(messages) > 1 and should_rewrite_query(query):
        user_count = sum(1 for m in messages if m.get("role") == "user")
        if user_count > 1:
            rewrite_start = time.perf_counter()
            if stream:
                await http_sse_comment(writer, "status=rewrite_start")
            try:
                search_query = await rewrite_query(query, messages)
                rewrite_ms = int((time.perf_counter() - rewrite_start) * 1000)
                if search_query != query:
                    print(f"  🔄 改写: {query!r} → {search_query!r}", flush=True)
                print(f"  ⏱️ rewrite {rewrite_ms}ms changed={search_query != query}", flush=True)
                if stream:
                    await http_sse_comment(
                        writer,
                        f"status=rewrite_done elapsed_ms={rewrite_ms} changed={search_query != query}",
                    )
                    if search_query != query:
                        await http_openai_content(
                            writer,
                            chat_id,
                            created,
                            model,
                            f"🔍 已结合上下文改写搜索词（{rewrite_ms}ms）\n\n",
                        )
            except Exception as e:
                rewrite_ms = int((time.perf_counter() - rewrite_start) * 1000)
                print(f"  ⚠️ 改写失败: {e} ({rewrite_ms}ms)", flush=True)
                if stream:
                    await http_sse_comment(writer, f"status=rewrite_error elapsed_ms={rewrite_ms}")

    freshness = _effective_freshness(search_query, freshness)

    # ── 0. 缓存检查 ──────────────────────────────────────────────
    if search_query:
        cached = _cache_get(
            search_query,
            model,
            freshness=freshness,
            max_results=max_results,
            messages=messages,
        )
        if cached is not None:
            cached_data = json.loads(cached)
            if stream:
                await http_sse_comment(writer, "status=cache_hit")
                cache_hit_ms = int((time.perf_counter() - req_start) * 1000)
                await http_openai_content(writer, chat_id, created, model, f"⚡ 缓存命中（{cache_hit_ms}ms）\n\n")
                print(f"  ⏱️ cache_hit {cache_hit_ms}ms", flush=True)
                # 重放答案
                chunks_out = 0
                cached_answer = cached_data.get("answer", "")
                for i in range(0, len(cached_answer), 80):
                    piece = cached_answer[i:i+80]
                    chunks_out += 1
                    chunk = json.dumps({
                        "id": chat_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }, ensure_ascii=False)
                    await http_sse(writer, "", chunk)
                # 来源
                sources_list = cached_data.get("sources", [])
                if sources_list:
                    sources_text = "\n\n---\n📎 **参考来源**\n" + "\n".join(
                        f"- [{i}] [{r['title'] or '无标题'}]({r['url']})"
                        for i, r in enumerate(sources_list, 1)
                    )
                    sources_chunk = json.dumps({
                        "id": chat_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": sources_text}, "finish_reason": None}],
                    }, ensure_ascii=False)
                    await http_sse(writer, "", sources_chunk)
                stop_chunk = json.dumps({
                    "id": chat_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }, ensure_ascii=False)
                await http_sse(writer, "", stop_chunk)
                await write_raw(writer, "data: [DONE]\n\n")
            else:
                resp = cached_data.get("full_response", "")
                if resp:
                    http_ok(writer, resp)
                else:
                    # 兼容旧缓存：构造 response
                    resp = json.dumps({
                        "id": chat_id, "object": "chat.completion",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "message": {"role": "assistant",
                            "content": cached_data.get("answer", "")}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    }, ensure_ascii=False)
                    http_ok(writer, resp)
            return

    # ── 1. 判断 + 搜索 + 输出 ──────────────────────────────────
    try:
        async def openai_status(stage: str, **data):
            if stream:
                if stage == "judge_start":
                    await http_sse_comment(writer, "status=judge_start")
                elif stage == "judge_done":
                    need_search_text = "需要搜索" if data.get("need_search") else "无需搜索，直接回答"
                    await http_openai_content(
                        writer,
                        chat_id,
                        created,
                        model,
                        f"🧠 判断：{need_search_text}（{data.get('elapsed_ms')}ms）\n\n",
                    )
                    await http_sse_comment(
                        writer,
                        f"status=judge_done elapsed_ms={data.get('elapsed_ms')} need_search={data.get('need_search')}",
                    )
                elif stage == "judge_error":
                    error_summary = _upstream_error_summary(
                        data.get("error", ""),
                        SEARCH_DECISION_MODEL,
                    )
                    await http_openai_content(
                        writer,
                        chat_id,
                        created,
                        model,
                        f"⚠️ {error_summary}，已自动改为搜索（{data.get('elapsed_ms')}ms）\n\n",
                    )
                    await http_sse_comment(
                        writer,
                        f"status=judge_error elapsed_ms={data.get('elapsed_ms')} fallback=search",
                    )
                elif stage == "direct_answer":
                    await http_sse_comment(writer, "status=direct_answer")
                elif stage == "search_start":
                    await http_openai_content(
                        writer,
                        chat_id,
                        created,
                        model,
                        f"🔍 搜索中：{str(data.get('query', ''))[:80]}\n",
                    )
                    await http_sse_comment(writer, f"status=search_start query={data.get('query', '')[:80]!r}")
                elif stage == "search_done":
                    await http_openai_content(
                        writer,
                        chat_id,
                        created,
                        model,
                        f"找到 {data.get('results')} 条结果（{data.get('elapsed_ms')}ms）\n\n",
                    )
                    await http_sse_comment(
                        writer,
                        f"status=search_done elapsed_ms={data.get('elapsed_ms')} results={data.get('results')}",
                    )

        pipeline = await prepare_answer_pipeline(
            query,
            messages=messages,
            model=model,
            search_query_override=search_query,
            max_results=max_results,
            freshness=freshness,
            status_cb=openai_status,
        )
        need_search = pipeline.need_search
        results = pipeline.results
        final_messages = pipeline.messages
        print(f"  ⏱️ pipeline_ready {int((time.perf_counter() - req_start) * 1000)}ms", flush=True)

        if pipeline.direct_answer:
            if stream:
                await http_openai_content(writer, chat_id, created, model, pipeline.direct_answer)
                stop_chunk = json.dumps({
                    "id": chat_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }, ensure_ascii=False)
                await http_sse(writer, "", stop_chunk)
                await write_raw(writer, "data: [DONE]\n\n")
            else:
                resp = json.dumps({
                    "id": chat_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": pipeline.direct_answer},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }, ensure_ascii=False)
                http_ok(writer, resp)
            return

        # ── 2. 流式 / 非流式输出 ────────────────────────────────
        if stream:
            full_answer: list[str] = []
            think_answer: list[str] = []
            llm_start = time.perf_counter()
            first_token_sent = False
            local_max_tokens = _max_tokens_for_request(requested_max_tokens, model, need_search=need_search)
            local_timeout = _stream_timeout_for_model(model)
            first_token_error_timeout = _first_token_error_timeout_for_model(model)
            await http_sse_comment(writer, "status=llm_start")
            await http_openai_content(writer, chat_id, created, model, "🤖 正在生成回答...\n\n")
            try:
                token_queue: asyncio.Queue = asyncio.Queue()

                async def produce_tokens():
                    try:
                        async for item in chat_stream(
                            final_messages,
                            model=model,
                            max_tokens=local_max_tokens,
                            fallback_on_503=True,
                            timeout=local_timeout,
                        ):
                            await token_queue.put(item)
                        await token_queue.put(("done", ""))
                    except Exception as exc:
                        await token_queue.put(("error", exc))

                producer = asyncio.create_task(produce_tokens())
                heartbeat_count = 0
                while True:
                    try:
                        kind, token = await asyncio.wait_for(
                            token_queue.get(),
                            timeout=FIRST_TOKEN_HEARTBEAT if not first_token_sent else None,
                        )
                    except asyncio.TimeoutError:
                        heartbeat_count += 1
                        elapsed = int((time.perf_counter() - llm_start) * 1000)
                        print(f"  ⏱️ llm_waiting_first_token {elapsed}ms model={model}", flush=True)
                        await http_sse_comment(
                            writer,
                            f"status=llm_waiting_first_token elapsed_ms={elapsed} model={model}",
                        )
                        if elapsed >= int(first_token_error_timeout * 1000):
                            producer.cancel()
                            raise TimeoutError(
                                f"模型 {model} 首 token 超时，可能是上游账号不健康或服务不可用"
                            )
                        if writer.is_closing():
                            producer.cancel()
                            raise ConnectionResetError("client disconnected")
                        continue

                    if kind == "done":
                        break
                    if kind == "error":
                        raise token
                    if writer.is_closing():
                        producer.cancel()
                        raise ConnectionResetError("client disconnected")

                    if not first_token_sent:
                        first_token_ms = int((time.perf_counter() - llm_start) * 1000)
                        total_ms = int((time.perf_counter() - req_start) * 1000)
                        first_token_sent = True
                        print(f"  ⏱️ llm_first_token {first_token_ms}ms total={total_ms}ms kind={kind}", flush=True)
                        await http_sse_comment(
                            writer,
                            f"status=llm_first_token elapsed_ms={first_token_ms} total_ms={total_ms} kind={kind}",
                        )
                    if kind == "think":
                        think_answer.append(token)
                        think_chunk = json.dumps({
                            "id": chat_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"reasoning_content": token}, "finish_reason": None}],
                        }, ensure_ascii=False)
                        await http_sse(writer, "", think_chunk)
                    else:
                        full_answer.append(token)
                        chunk = json.dumps({
                            "id": chat_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                        }, ensure_ascii=False)
                        await http_sse(writer, "", chunk)
                producer.cancel()
            except Exception:
                if not full_answer and think_answer:
                    await http_sse_comment(writer, "status=llm_think_fallback_after_error")
                else:
                    raise

            if not full_answer and think_answer:
                fallback_answer = "".join(think_answer)
                full_answer.append(fallback_answer)
                await http_sse_comment(writer, "status=llm_think_fallback")
                fallback_chunk = json.dumps({
                    "id": chat_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"content": fallback_answer}, "finish_reason": None}],
                }, ensure_ascii=False)
                await http_sse(writer, "", fallback_chunk)

            if direct_vision_image_items and len(direct_vision_image_items) == 1:
                direct_answer = "".join(full_answer).strip()
                if direct_answer:
                    direct_summary = f"该图片已由视觉模型 {model} 直接分析并回答：\n{direct_answer}"
                    _image_cache_set(direct_vision_image_items[0], direct_summary)
                    _last_image_summary_set(direct_summary, user_id)

            # 追加来源
            if need_search and results:
                sources_text = "\n\n---\n📎 **参考来源**\n" + "\n".join(
                    f"- [{i}] [{r.title or '无标题'}]({r.url})" for i, r in enumerate(results, 1)
                )
                sources_chunk = json.dumps({
                    "id": chat_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"content": sources_text}, "finish_reason": None}],
                }, ensure_ascii=False)
                await http_sse(writer, "", sources_chunk)

            stop_chunk = json.dumps({
                "id": chat_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }, ensure_ascii=False)
            await http_sse(writer, "", stop_chunk)
            await write_raw(writer, "data: [DONE]\n\n")
            print(f"  ⏱️ total {int((time.perf_counter() - req_start) * 1000)}ms", flush=True)

            # 缓存答案
            if search_query:
                cache_payload = json.dumps({
                    "answer": "".join(full_answer),
                    "need_search": need_search,
                    "sources": [{"title": r.title, "url": r.url} for r in results],
                }, ensure_ascii=False)
                _cache_set(
                    search_query,
                    model,
                    cache_payload,
                    freshness=freshness,
                    max_results=max_results,
                    messages=messages,
                )
        else:
            llm_start = time.perf_counter()
            final_max_tokens = _max_tokens_for_request(requested_max_tokens, model, need_search=need_search)
            final_answer = await chat(final_messages, model=model, stream=False, max_tokens=final_max_tokens, fallback_on_503=True)
            print(f"  ⏱️ llm_complete {int((time.perf_counter() - llm_start) * 1000)}ms", flush=True)
            if direct_vision_image_items and len(direct_vision_image_items) == 1 and final_answer.strip():
                direct_summary = f"该图片已由视觉模型 {model} 直接分析并回答：\n{final_answer.strip()}"
                _image_cache_set(direct_vision_image_items[0], direct_summary)
                _last_image_summary_set(direct_summary, user_id)
            if need_search and results:
                sources_text = "\n\n---\n📎 **参考来源**\n" + "\n".join(
                    f"- [{i}] [{r.title or '无标题'}]({r.url})" for i, r in enumerate(results, 1)
                )
                final_answer += sources_text
            resp = json.dumps({
                "id": chat_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": final_answer},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }, ensure_ascii=False)
            http_ok(writer, resp)

            # 缓存答案
            if search_query:
                cache_payload = json.dumps({
                    "answer": final_answer,
                    "need_search": need_search,
                    "sources": [{"title": r.title, "url": r.url} for r in results],
                    "full_response": resp,
                }, ensure_ascii=False)
                _cache_set(
                    search_query,
                    model,
                    cache_payload,
                    freshness=freshness,
                    max_results=max_results,
                    messages=messages,
                )
            print(f"  ⏱️ total {int((time.perf_counter() - req_start) * 1000)}ms", flush=True)
    except Exception as e:
        import traceback
        print(f"❌ handle_chat_completions error: {e}", flush=True)
        traceback.print_exc()
        if stream:
            try:
                error_text = _friendly_llm_error(e, model)
                await http_openai_content(writer, chat_id, created, model, error_text)
                await http_openai_stop(writer, chat_id, created, model)
            except Exception:
                pass
        else:
            error_text = _friendly_llm_error(e, model)
            resp = json.dumps({
                "id": chat_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": error_text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }, ensure_ascii=False)
            http_ok(writer, resp)


async def nc_qa(writer: asyncio.StreamWriter, query: str, model: str,
                max_results: int, freshness: str):
    async def nc_status(stage: str, **data):
        if stage == "judge_done":
            await write_line(
                writer,
                f"🧠 判断: {'需要搜索' if data.get('need_search') else '无需搜索，直接回答'}"
                f" ({data.get('elapsed_ms')}ms)",
            )
        elif stage == "judge_error":
            error_summary = _upstream_error_summary(
                data.get("error", ""),
                SEARCH_DECISION_MODEL,
            )
            await write_line(
                writer,
                f"⚠️ {error_summary}，将自动搜索"
                f" ({data.get('elapsed_ms')}ms)",
            )
        elif stage == "search_start":
            await write_line(writer, f"🔍 搜索: {data.get('query', query)}")
        elif stage == "search_done":
            await write_line(writer, f"   共 {data.get('results', 0)} 条结果 ({data.get('elapsed_ms')}ms)")

    pipeline = await prepare_answer_pipeline(
        query,
        messages=[{"role": "user", "content": query}],
        model=model,
        max_results=max_results,
        freshness=freshness,
        status_cb=nc_status,
    )
    await write_line(writer, f"\n🤖 {model}:")
    if pipeline.direct_answer:
        await write_line(writer, pipeline.direct_answer)
        await write_line(writer, "")
        return
    try:
        max_tokens = _default_max_tokens_for_model(model, need_search=pipeline.need_search)
        await _nc_stream_answer(writer, pipeline.messages, model, max_tokens=max_tokens)
    except Exception as e:
        await write_line(writer, f"\n❌ LLM 错误: {e}")
    await write_line(writer, "")
    if pipeline.need_search:
        await write_line(writer, format_sources(pipeline.results))
    await write_line(writer, "")


# ── 协议自动分发 ─────────────────────────────────────────────────

async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """检测协议：快速 peek 判断 HTTP / nc"""
    try:
        peek = await asyncio.wait_for(reader.read(4), timeout=0.3)
    except asyncio.TimeoutError:
        # 0.3s 内无数据 → nc 模式
        await handle_tcp(reader, writer)
        return

    if not peek:
        writer.close()
        return

    if peek.startswith(b"GET ") or peek.startswith(b"POST") or peek.startswith(b"HEAD"):
        # HTTP 模式：手动读取完整请求
        rest = await reader.readuntil(b"\r\n\r\n")
        raw = (peek + rest).decode(errors="replace")
        lines = raw.split("\r\n")
        method, path, _ = (lines[0].split() + ["", "", ""])[:3]

        headers = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        clen = int(headers.get("content-length", 0))
        body = b""
        if clen > 0:
            body = await reader.readexactly(clen)

        req = {"method": method, "path": path, "headers": headers, "body": body}
        print(f"[http] {method} {path}")
        await handle_http(writer, req)
    else:
        # 非 HTTP 数据 → nc 模式，把 peek 的字节作为第一行输入
        await handle_tcp_with_first(reader, writer, peek)

    try:
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    writer.close()


async def handle_tcp_with_first(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                                first_bytes: bytes):
    """nc 模式，但已有 peek 到的初始字节，补读完整第一行"""
    peer = writer.get_extra_info("peername", ("?", 0))
    print(f"[nc] {peer[0]}:{peer[1]} 已连接")

    model = DEFAULT_MODEL
    max_results = DEFAULT_MAX_RESULTS
    freshness = DEFAULT_FRESHNESS

    await write_line(writer, BANNER.format(model=model))

    # peek 到的字节只是一行的开头，补读该行剩余部分
    rest = await reader.readline()
    first_line = (first_bytes + rest).decode(errors="replace").strip()

    if first_line:
        result = await _process_input(writer, first_line, model, max_results, freshness)
        if result is False:
            return
        if isinstance(result, str):
            model = result
        elif isinstance(result, tuple):
            model, freshness, max_results = result

    await _nc_loop(reader, writer, model, max_results, freshness, peer)


async def handle_tcp(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """nc 模式，标准入口"""
    peer = writer.get_extra_info("peername", ("?", 0))
    print(f"[nc] {peer[0]}:{peer[1]} 已连接")

    model = DEFAULT_MODEL
    max_results = DEFAULT_MAX_RESULTS
    freshness = DEFAULT_FRESHNESS

    await write_line(writer, BANNER.format(model=model))
    await _nc_loop(reader, writer, model, max_results, freshness, peer)


async def _nc_loop(reader, writer, model, max_results, freshness, peer):
    """nc 交互主循环"""
    try:
        while True:
            writer.write(b"\xf0\x9f\x92\xac ")
            await writer.drain()
            line = await reader.readline()
            if not line:
                break

            query = line.decode(errors="replace").strip()
            if not query:
                continue

            keep_going = await _process_input(writer, query, model, max_results, freshness)
            if keep_going is False:
                break
            if isinstance(keep_going, str):
                model = keep_going
            elif isinstance(keep_going, tuple):
                model, freshness, max_results = keep_going

    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        pass
    finally:
        print(f"[nc] {peer[0]}:{peer[1]} 断开")
        writer.close()


async def _process_input(writer, query, model, max_results, freshness):
    """处理单行输入，返回: False=断开, str=新model, tuple=(model,fresh,max_results), True=继续"""
    if query == "/exit":
        await write_line(writer, "👋 再见")
        return False
    if query == "/help":
        await write_line(writer, HELP.format(model=model, fresh=freshness, max_results=max_results))
        return True
    if query.startswith("/model "):
        new_model = query.split(" ", 1)[1].strip()
        await write_line(writer, f"✅ 模型 → {new_model}")
        return new_model
    if query.startswith("/fresh "):
        new_fresh = query.split(" ", 1)[1].strip()
        await write_line(writer, f"✅ 时效 → {new_fresh}")
        return (model, new_fresh, max_results)
    if query.startswith("/results "):
        try:
            new_max = int(query.split(" ", 1)[1])
            await write_line(writer, f"✅ 结果数 → {new_max}")
            return (model, freshness, new_max)
        except ValueError:
            await write_line(writer, "❌ 请输入数字")
            return True

    await write_line(writer, "")
    try:
        await nc_qa(writer, query, model, max_results, freshness)
    except Exception as e:
        await write_line(writer, f"\n❌ 出错: {e}")
    return True


# ── 入口 ─────────────────────────────────────────────────────────

async def serve(host: str = "0.0.0.0", port: int = 9090):
    server = await asyncio.start_server(handle_connection, host, port)
    print(f"\n{'='*60}")
    print(f"🌐 AnySearch Agent 服务已启动  (自动适配 nc / HTTP)")
    print(f"   监听: {host}:{port}")
    print(f"   模型: {DEFAULT_MODEL}")
    print(f"")
    print(f"   # 交互模式")
    print(f"   nc {host} {port}")
    print(f"")
    print(f"   # curl 搜索+模型总结")
    print(f'   curl -s http://{host}:{port}/chat -H "Content-Type: application/json" \\')
    print(f"     -d '{{\"query\":\"比特币价格\"}}'")
    print(f"")
    print(f'   # curl 仅搜索')
    print(f'   curl -s http://{host}:{port}/search -H "Content-Type: application/json" \\')
    print(f"     -d '{{\"query\":\"比特币价格\"}}'")
    print(f"{'='*60}\n")
    async with server:
        await server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="AnySearch Agent 服务（nc + HTTP）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=9090)
    args = parser.parse_args()
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
