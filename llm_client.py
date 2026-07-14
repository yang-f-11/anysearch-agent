"""
LLM Hub 客户端。
通过 OpenAI 兼容协议调用 llm_hub 网关。
"""
from __future__ import annotations

import os
from typing import AsyncIterator

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIStatusError, APITimeoutError, APIConnectionError

load_dotenv()

BASE_URL = os.getenv("LLM_HUB_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("LLM_HUB_API_KEY", "not-needed")
DEFAULT_MODEL = os.getenv("LLM_HUB_MODEL", "deepseek-v4-pro")
GEMMA_MODEL = "gemma4-crack"
LOCAL_FALLBACK_MODEL = os.getenv("LLM_HUB_LOCAL_FALLBACK_MODEL", "local/minimax-m2.7")

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            http_client=httpx.AsyncClient(timeout=60.0, trust_env=False),
        )
    return _client


def is_encoding_error(exc: APIStatusError) -> bool:
    """检测 latin-1 编码错误（部分模型不支持中文）"""
    if exc.status_code != 500 or not isinstance(getattr(exc, 'body', None), dict):
        return False
    body = exc.body
    # body 可能是 {"message": "..."} 或 {"error": {"message": "..."}}
    msg = body.get("message", "") or body.get("error", {}).get("message", "")
    return "latin-1" in str(msg)


def _chat_extra_body(model_name: str) -> dict | None:
    """Model-specific OpenAI-compatible extensions for llm_hub."""
    return None


def _fallback_model_for(model_name: str) -> str:
    if model_name != DEFAULT_MODEL:
        return DEFAULT_MODEL
    return LOCAL_FALLBACK_MODEL


async def chat(
    messages: list[dict],
    model: str = "",
    stream: bool = True,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    timeout: float = 60.0,
    fallback_on_503: bool = False,
) -> str:
    """非流式对话（内部用 stream=True，兼容所有模型）。timeout 秒。"""
    # 短任务（judge/rewrite）用独立短超时 client，避免卡死
    _client = get_client() if timeout >= 60.0 else AsyncOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        http_client=httpx.AsyncClient(timeout=timeout, trust_env=False),
    )
    client = _client
    model_name = model or DEFAULT_MODEL
    try:
        stream_resp = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=_chat_extra_body(model_name),
        )
        parts: list[str] = []
        parts_think: list[str] = []
        async for chunk in stream_resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                parts_think.append(rc)
            if delta.content:
                parts.append(delta.content)
        # 优先 content，部分思考链模型（gemma4/qwen）只有 reasoning_content
        return "".join(parts) or "".join(parts_think)
    except APIStatusError as e:
        if is_encoding_error(e) and model_name != DEFAULT_MODEL:
            import sys
            print(f"⚠️ {model_name} 编码错误，fallback → {DEFAULT_MODEL}", file=sys.stderr, flush=True)
            return await chat(messages, model=DEFAULT_MODEL,
                              max_tokens=max_tokens, temperature=temperature, timeout=timeout)
        if fallback_on_503 and e.status_code == 503 and model_name == DEFAULT_MODEL:
            fallback = _fallback_model_for(model_name)
            if fallback == model_name:
                raise
            import sys
            print(f"⚠️ {model_name} 503，fallback → {fallback}", file=sys.stderr, flush=True)
            return await chat(messages, model=fallback,
                              max_tokens=max_tokens, temperature=temperature, timeout=timeout,
                              fallback_on_503=fallback != LOCAL_FALLBACK_MODEL)
        raise
    except (APITimeoutError, APIConnectionError) as e:
        if fallback_on_503 and model_name == DEFAULT_MODEL:
            fallback = _fallback_model_for(model_name)
            if fallback == model_name:
                raise
            import sys
            print(f"⚠️ {model_name} 超时/连接错误，fallback → {fallback}", file=sys.stderr, flush=True)
            return await chat(messages, model=fallback,
                              max_tokens=max_tokens, temperature=temperature, timeout=timeout,
                              fallback_on_503=fallback != LOCAL_FALLBACK_MODEL)
        raise


async def chat_stream(
    messages: list[dict],
    model: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    fallback_on_503: bool = False,
    timeout: float = 60.0,
) -> AsyncIterator[tuple[str, str]]:
    """流式对话，返回 (kind, token) 迭代器。kind: 'think' | 'content'"""
    client = get_client() if timeout >= 60.0 and timeout == 60.0 else AsyncOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        http_client=httpx.AsyncClient(timeout=timeout, trust_env=False),
    )
    model_name = model or DEFAULT_MODEL
    try:
        stream_resp = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=_chat_extra_body(model_name),
        )
        has_content = False
        async for chunk in stream_resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                yield ("think", rc)
            if delta.content:
                has_content = True
                yield ("content", delta.content)
        # 如果全程没有 content（如 gemma4 低 token 预算），
        # 思考内容已作为 'think' 产出，调用方可 fallback
    except APIStatusError as e:
        if is_encoding_error(e) and model_name != DEFAULT_MODEL:
            import sys
            print(f"⚠️ {model_name} 编码错误，fallback → {DEFAULT_MODEL}", file=sys.stderr, flush=True)
            async for kind, token in chat_stream(messages, model=DEFAULT_MODEL,
                                                  max_tokens=max_tokens, temperature=temperature,
                                                  timeout=timeout):
                yield (kind, token)
            return
        if fallback_on_503 and e.status_code == 503 and model_name == DEFAULT_MODEL:
            fallback = _fallback_model_for(model_name)
            if fallback == model_name:
                raise
            import sys
            print(f"⚠️ {model_name} 503，fallback → {fallback}", file=sys.stderr, flush=True)
            async for kind, token in chat_stream(messages, model=fallback,
                                                  max_tokens=max_tokens, temperature=temperature,
                                                  fallback_on_503=fallback != LOCAL_FALLBACK_MODEL,
                                                  timeout=timeout):
                yield (kind, token)
            return
        raise
    except (APITimeoutError, APIConnectionError) as e:
        if fallback_on_503 and model_name == DEFAULT_MODEL:
            fallback = _fallback_model_for(model_name)
            if fallback == model_name:
                raise
            import sys
            print(f"⚠️ {model_name} 超时/连接错误，fallback → {fallback}", file=sys.stderr, flush=True)
            async for kind, token in chat_stream(messages, model=fallback,
                                                  max_tokens=max_tokens, temperature=temperature,
                                                  fallback_on_503=fallback != LOCAL_FALLBACK_MODEL,
                                                  timeout=timeout):
                yield (kind, token)
            return
        raise
