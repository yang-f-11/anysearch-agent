"""Deterministic answers for AnySearch Agent's own runtime capabilities."""
from __future__ import annotations

import re


AVAILABLE_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "opencode/deepseek-v4-flash",
    "opencode/deepseek-v4-pro",
    "gemini-3-flash",
    "gemini-3.5-flash-low",
    "gemini-3.5-flash-agent",
    "moonshot/kimi-for-coding",
    "opencode/glm-5.2",
    "opencode/glm-5.1",
    "opencode/glm-5",
    "opencode/kimi-k2.6",
    "opencode/mimo-v2.5-pro",
    "opencode/qwen3.7-max",
    "opencode/qwen3.6-plus",
    "opencode/qwen3.5-plus",
    "opencode/minimax-m2.7",
    "opencode/minimax-m2.5",
    "gemma4-crack",
    "local/qwen3.5-27b-distill",
    "local/minimax-m2.7",
]

def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def is_model_capability_query(query: str) -> bool:
    q = _compact(query)
    if not q:
        return False

    if "v1/models" in q or "v1/chat/completions" in q or "chat/completions" in q:
        return True

    mentioned_model = any(model.lower() in q for model in AVAILABLE_MODELS)
    if mentioned_model and any(term in q for term in ("怎么调用", "如何调用", "调用", "接口", "api")):
        return True

    if any(term in q for term in ("当前可用模型", "可用模型列表", "可用的模型", "模型列表是什么")):
        return True

    service_scope = any(
        term in q
        for term in (
            "我们现在接入",
            "我们接入",
            "当前接入",
            "现在接入",
            "本服务",
            "anysearchagent",
            "anysearch",
            "openwebui",
            "openwebui",
            "这个系统",
            "这个服务",
        )
    )
    model_scope = any(term in q for term in ("ai模型", "大模型", "模型列表", "可用模型", "接入的模型"))
    call_scope = any(term in q for term in ("怎么调用", "如何调用", "调用方式", "接口", "api", "有哪些", "列表"))
    return service_scope and model_scope and call_scope


def _chat_example(model: str = "deepseek-v4-flash") -> str:
    return (
        "```bash\n"
        "curl -N http://127.0.0.1:9090/v1/chat/completions \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        f"  -d '{{\"model\":\"{model}\",\"messages\":[{{\"role\":\"user\",\"content\":\"你好\"}}],\"stream\":true}}'\n"
        "```\n"
    )


def model_capability_answer(query: str = "") -> str:
    q = _compact(query)
    if "v1/models" in q:
        return (
            "`/v1/models` 用来获取当前 AnySearch Agent 暴露给 Open WebUI 的模型列表。\n\n"
            "调用示例：\n\n"
            "```bash\n"
            "curl -s http://127.0.0.1:9090/v1/models\n"
            "```\n\n"
            "返回是 OpenAI 兼容格式：`{\"object\":\"list\",\"data\":[...]}`，其中每个 `data[].id` "
            "就是聊天接口里可传的 `model`。"
        )

    if "v1/chat/completions" in q or "chat/completions" in q:
        return (
            "`/v1/chat/completions` 是 OpenAI 兼容聊天接口，Open WebUI 主要通过它调用模型。\n\n"
            "最小调用示例：\n\n"
            f"{_chat_example()}\n"
            "常用字段：`model` 选择模型，`messages` 传对话，`stream` 控制是否流式输出。"
        )

    mentioned = next((model for model in AVAILABLE_MODELS if model.lower() in q), "")
    if mentioned:
        return (
            f"`{mentioned}` 可以通过 OpenAI 兼容聊天接口调用。\n\n"
            "示例：\n\n"
            f"{_chat_example(mentioned)}\n"
            "如果在 Open WebUI 中使用，直接在模型下拉框选择该模型后发送消息即可。"
        )

    models = "\n".join(f"- `{model}`" for model in AVAILABLE_MODELS)
    if any(term in q for term in ("列表", "可用", "当前可用")) and not any(term in q for term in ("调用", "怎么")):
        return f"当前可用模型列表：\n\n{models}\n\n可通过 `GET /v1/models` 获取同样的列表。"

    return (
        "当前 AnySearch Agent 接入的 AI 模型如下：\n\n"
        f"{models}\n\n"
        "调用方式：\n\n"
        "1. 查看模型列表：`GET /v1/models`\n"
        "2. OpenAI 兼容聊天接口：`POST /v1/chat/completions`\n"
        "3. 便捷接口：`POST /chat`，参数为 `query`，可选 `model`、`max_results`、`freshness`\n\n"
        "示例：\n\n"
        f"{_chat_example()}"
    )
