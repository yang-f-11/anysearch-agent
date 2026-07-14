"""
AnySearch Agent 调用
服务地址: http://127.0.0.1:9090

用法:
  python3 ask.py              # 搜索 + LLM 总结（流式）
  python3 ask.py --search     # 仅搜索，不总结
"""
import sys
import json
import requests

from openai import OpenAI

BASE = os.getenv("ANYSEARCH_AGENT_BASE_URL", "http://127.0.0.1:9090")
client = OpenAI(base_url=f"{BASE}/v1", api_key="not-needed")

model = "gemini-3-flash"
# model = "gemini-3-pro"
# model = "gemini-3.1-pro"
# model = "gemini-3.5-flash-low"
# model = "gemini-3.5-flash-agent"
# model = "claude-opus-4-6-thinking"
# model = "claude-sonnet-4-6"
# model = "deepseek-v4-pro"
# model = "deepseek-v4-flash"
# model = "opencode/deepseek-v4-pro"
# model = "opencode/deepseek-v4-flash"
# model = "gemma4-crack"
# model = "moonshot/kimi-for-coding"
# model = "local/qwen3.5-27b-distill"
# model = "local/minimax-m2.7"
# model = "opencode/glm-5.2"
# model = "opencode/glm-5.1"
# model = "opencode/glm-5"
# model = "opencode/kimi-k2.6"
# model = "opencode/mimo-v2.5-pro"
# model = "opencode/qwen3.7-max"
# model = "opencode/qwen3.6-plus"
# model = "opencode/qwen3.5-plus"
# model = "opencode/minimax-m2.7"
# model = "opencode/minimax-m2.5"

QUERY = "SiliconFlow 当前有哪些免费模型和付费模型？"

# ── 仅搜索（不总结） ──
def search_only(query: str):
    resp = requests.post(f"{BASE}/search", json={"query": query})
    data = resp.json()
    for i, r in enumerate(data.get("results", []), 1):
        print(f"[{i}] {r['title']}")
        print(f"    {r['url']}")
        print(f"    {r.get('snippet', '')}")
        print()

# ── 搜索 + LLM 总结（流式） ──
def ask(query: str):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        max_tokens=4096,
        temperature=0.0,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
    print()


if __name__ == "__main__":
    if "--search" in sys.argv:
        search_only(QUERY)
    else:
        ask(QUERY)
