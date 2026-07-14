"""
联网搜索 Agent — 核心编排逻辑。

流程：判断是否需要搜索 → 不需要则直接回答 → 需要则搜索 → LLM 总结
"""
from __future__ import annotations

import os
from datetime import datetime

from dotenv import load_dotenv

from search_client import AnySearchClient, SearchResult
from llm_client import chat, chat_stream
from app_capabilities import is_model_capability_query, model_capability_answer

load_dotenv()

DEFAULT_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "10"))
DEFAULT_FRESHNESS = os.getenv("SEARCH_FRESHNESS", "month")
DEFAULT_MODEL = os.getenv("LLM_HUB_MODEL", "deepseek-v4-pro")
SEARCH_DECISION_MODEL = os.getenv("SEARCH_DECISION_MODEL", "deepseek-v4-flash")
SEARCH_DECISION_MODE = os.getenv("SEARCH_DECISION_MODE", "auto").lower()
JUDGE_TIMEOUT = float(os.getenv("JUDGE_TIMEOUT", "2.5"))
REWRITE_TIMEOUT = float(os.getenv("REWRITE_TIMEOUT", "2.5"))

# 关键词预判：命中则直接搜索（跳过 LLM judge）
_SEARCH_KEYWORDS = [
    "最新", "最近", "今天", "本周", "今年", "当前", "现在",
    "多少钱", "价格", "股价", "汇率", "天气", "新闻",
    "版本", "财报", "发布", "上市", "上线", "推出",
    "政策", "规定", "入境", "签证", "免签",
    "报错", "错误码", "报错码",
    "2025", "2026", "2027",
]

_CONTEXTUAL_REWRITE_HINTS = [
    "它", "他", "她", "这个", "那个", "这家", "那家", "该", "其",
    "上述", "上面", "前面", "刚才", "上一条", "上一个",
    "呢", "那", "还有", "继续", "价格", "股价", "收盘", "怎么样",
]

_STANDALONE_SMALLTALK = {
    "你好", "您好", "hello", "hi", "嗨", "在吗", "谢谢", "多谢",
    "早上好", "下午好", "晚上好",
}

_FRONTEND_META_TASK_PREFIXES = (
    "### task:\nsuggest 3-5 relevant follow-up questions",
    "### task:\ngenerate a concise",
    "### task:\ngenerate 1-3 broad tags",
)


def is_frontend_meta_task(query: str) -> bool:
    compact = query.strip().lower()
    return any(compact.startswith(prefix) for prefix in _FRONTEND_META_TASK_PREFIXES)


def is_standalone_smalltalk(query: str) -> bool:
    compact = query.strip().lower().replace(" ", "")
    return compact in _STANDALONE_SMALLTALK


def _quick_judge(query: str) -> bool | None:
    """关键词命中返回 True，否则返回 None 表示需要 LLM 判断"""
    if is_standalone_smalltalk(query) or is_frontend_meta_task(query):
        return False
    for kw in _SEARCH_KEYWORDS:
        if kw in query:
            return True
    return None


def should_rewrite_query(query: str) -> bool:
    """判断当前用户输入是否像需要结合历史的追问。"""
    q = query.strip()
    if not q:
        return False
    compact = q.lower().replace(" ", "")
    if is_standalone_smalltalk(q):
        return False
    if len(q) <= 8 and not any(h in q for h in _CONTEXTUAL_REWRITE_HINTS):
        return False
    return any(h in q for h in _CONTEXTUAL_REWRITE_HINTS)

SYSTEM_PROMPT = """\
你是一个专业的搜索增强助手。你会根据提供的搜索结果来回答用户的问题。

回答规则：
1. 优先使用搜索结果中的信息，如果搜索结果不足以回答，请如实告知。
2. 引用来源时使用 [序号] 标注，例如 "根据 [1]，..."。
3. 保持回答简洁、准确、客观。
4. 如果涉及实时信息（股价、天气、新闻等），务必注明信息的时间。
5. 用中文回答，除非用户要求其他语言。

当前日期：{date}
"""

SYSTEM_PROMPT_DIRECT = """\
你是一个专业的 AI 助手。直接回答用户问题，简洁准确，用中文。

当前日期：{date}
"""

JUDGE_PROMPT = """\
判断用户问题是否需要联网搜索最新信息。只回复 YES 或 NO。

需要搜索 (YES)：
- 包含时效词：最新、最近、今天、本周、今年、当前、现在、2026
- 实时数据：新闻、股价、汇率、天气、赛事、价格
- 具体事实：产品版本、公司财报、某人动态、政策规定、入境要求
- 排障/报错：特定错误码的解决方法（解决方案可能随版本更新）
- 工具/软件/库的最新用法、版本特性、变更

不需要搜索 (NO)：
- 纯理论：数学证明、物理定律、历史事实（已固定的知识）
- 基础概念：什么是X、X的定义、X的工作原理
- 通用编程：语法怎么写、算法原理、数据结构（不涉及特定版本/错误码）
- 写作/翻译/总结/观点建议/头脑风暴

用户问题：{query}

只回复一个词：YES 或 NO。"""

REWRITE_PROMPT = """\
根据对话历史，将用户追问改写为可独立搜索的完整查询。

对话历史：
{history}

用户追问：{query}

规则：
- 补充指代词（它、他、她、这个、那个、其、该）指向的具体实体
- 补充缩写、简称的完整名称
- 用中文输出，只输出改写后的查询，不要任何解释

改写后的搜索查询："""


async def rewrite_query(query: str, messages: list[dict], model: str = "") -> str:
    """用对话历史将追问改写为可独立搜索的完整 query。

    仅在 handle_chat_completions (Open WebUI) 多轮对话路径调用。
    model 默认使用 DEFAULT_MODEL 保证低延迟。
    """
    # 构建简化历史（最近 10 条，每条截断 200 字）
    history_parts = []
    for m in messages[-10:]:
        role = "用户" if m["role"] == "user" else "助手"
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        if content:
            history_parts.append(f"{role}: {content[:200]}")
    history_text = "\n".join(history_parts)

    prompt = REWRITE_PROMPT.format(history=history_text, query=query)
    rewrite_msg = [{"role": "user", "content": prompt}]
    rewritten = await chat(
        rewrite_msg,
        model=DEFAULT_MODEL,
        stream=False,
        max_tokens=100,
        temperature=0.0,
        timeout=REWRITE_TIMEOUT,
    )
    result = rewritten.strip()
    # 防止 LLM 输出格式异常，取最后一行
    if "\n" in result:
        result = result.split("\n")[-1].strip()
    # 防呆：检测 LLM 是否输出指令/解释/prompt 尾巴而非真实查询
    garbage_prefix = (
        "*", "To ", "I ", "Here", "The ", "Rules", "Note", "Please",
        "You ", "This", "Based", "Let", "改写",
    )
    garbage_contain = ("搜索查询", "改写后", "以下是根据")
    if any(result.startswith(p) for p in garbage_prefix):
        return query
    if any(p in result for p in garbage_contain):
        return query
    # LLM 可能输出英文解释，搜索结果需要是中文 query
    if len(result) < 3 and not result.isdigit():
        return query
    return result or query


def format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "（未找到相关搜索结果）"
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.title}")
        if r.url:
            parts.append(f"    URL: {r.url}")
        if r.snippet:
            parts.append(f"    {r.snippet}")
        if r.published_date:
            parts.append(f"    日期: {r.published_date}")
        parts.append("")
    return "\n".join(parts)


def format_sources(results: list[SearchResult]) -> str:
    lines = ["\n📎 参考来源:"]
    for i, r in enumerate(results, 1):
        title = r.title or "无标题"
        url = r.url or ""
        lines.append(f"  [{i}] {title}")
        if url:
            lines.append(f"      {url}")
    return "\n".join(lines)


async def search_web_only(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    freshness: str = DEFAULT_FRESHNESS,
) -> list[SearchResult]:
    """Search AnySearch only, bypassing the local internal KB."""
    client = AnySearchClient()
    return (await client.search(query, max_results=max_results, freshness=freshness))[:max_results]


async def judge_query(query: str, model: str = "") -> bool:
    """判断是否需要搜索。关键词命中直接搜，否则 LLM 判断。
    LLM 不可用时抛出异常，由调用方兜底搜索。"""
    if SEARCH_DECISION_MODE in {"always_search", "search"}:
        return True
    quick = _quick_judge(query)
    if quick is not None:
        return quick
    if SEARCH_DECISION_MODE in {"heuristic", "quick"}:
        return False
    judge_msg = [{"role": "user", "content": JUDGE_PROMPT.format(query=query)}]
    resp = await chat(
        judge_msg,
        model=model or SEARCH_DECISION_MODEL,
        stream=False,
        max_tokens=8,
        temperature=0.0,
        timeout=JUDGE_TIMEOUT,
    )
    return "YES" in resp.upper()


async def _stream_answer(messages: list[dict], model_name: str) -> str:
    """流式输出，思考内容显示为 '思考中...'，最终输出直接展示"""
    think_parts: list[str] = []
    content_parts: list[str] = []
    showed_thinking = False
    async for kind, token in chat_stream(messages, model=model_name):
        if kind == "think":
            think_parts.append(token)
            if not showed_thinking:
                print("💭 思考中...", end="", flush=True)
                showed_thinking = True
        else:
            if showed_thinking and not content_parts:
                print("\n📝 回答: ", end="", flush=True)
            print(token, end="", flush=True)
            content_parts.append(token)
    print()
    if content_parts:
        return "".join(content_parts)
    # 只有思考没有输出时（如 gemma4 低 token），用思考作 fallback
    if think_parts:
        print("📝 (fallback 思考内容):")
        think_text = "".join(think_parts)
        print(think_text)
        return think_text
    return ""


async def search_and_answer(
    query: str,
    model: str = "",
    max_results: int = DEFAULT_MAX_RESULTS,
    freshness: str = DEFAULT_FRESHNESS,
    stream: bool = True,
) -> tuple[str, list[SearchResult]]:
    """核心流程：判断 → 直接回答 或 搜索 → LLM 总结"""
    model_name = model or DEFAULT_MODEL
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 1. 服务自身能力查询直接回答，不进入联网搜索。
    if is_model_capability_query(query):
        answer = model_capability_answer(query)
        print("🧠 判断: 服务能力查询，无需搜索")
        print(answer)
        return answer, []

    need_search = await judge_query(query)
    reason = "需要搜索" if need_search else "无需搜索"
    print(f"🧠 判断: {reason}  |  模型: {model_name}")

    if not need_search:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_DIRECT.format(date=date_str)},
            {"role": "user", "content": query},
        ]
        if stream:
            answer = await _stream_answer(messages, model_name)
        else:
            print("⏳ 正在生成回答...")
            answer = await chat(messages, model=model_name, stream=False)
            print(answer)
        return answer, []

    # 2. 搜索
    print(f"🔍 搜索: {query}")
    results = await search_web_only(query, max_results=max_results, freshness=freshness)
    print(f"   共 {len(results)} 条结果")

    # 3. 构建搜索增强消息
    system = SYSTEM_PROMPT.format(date=date_str)
    search_text = format_search_results(results)
    user_msg = (f"## 用户问题\n{query}\n\n"
                f"## 搜索结果摘要\n{search_text}\n\n"
                f"请根据以上信息综合回答。需要引用来源时用 [序号] 标注。")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    # 4. LLM 生成
    if stream:
        answer = await _stream_answer(messages, model_name)
    else:
        print("⏳ 正在生成回答...")
        answer = await chat(messages, model=model_name, stream=False)
        print(answer)

    print(format_sources(results))
    return answer, results


async def ask(
    query: str,
    model: str = "",
    max_results: int = DEFAULT_MAX_RESULTS,
    freshness: str = DEFAULT_FRESHNESS,
) -> tuple[str, list[SearchResult]]:
    """快捷接口"""
    return await search_and_answer(
        query=query,
        model=model,
        max_results=max_results,
        freshness=freshness,
        stream=True,
    )
