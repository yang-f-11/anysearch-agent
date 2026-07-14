"""
AnySearch 搜索客户端（纯 HTTP/JSON-RPC，无 MCP SDK 依赖）。

curl 等效调用:
  curl -s -X POST "https://api.anysearch.com/mcp" \\
    -H "Content-Type: application/json" \\
    -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"search","arguments":{"query":"关键词","max_results":5}},"id":1}'
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

ANYSEARCH_URL = "https://api.anysearch.com/mcp"
ANYSEARCH_API_KEY = os.getenv("ANYSEARCH_API_KEY", "")
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "180"))

_async_client: httpx.AsyncClient | None = None
_search_cache: dict[str, tuple[float, list["SearchResult"]]] = {}


def _get_async_client(timeout: float) -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _async_client


def _search_cache_key(args: dict) -> str:
    return json.dumps(args, ensure_ascii=False, sort_keys=True)


@dataclass
class SearchResult:
    """统一的搜索结果结构"""
    title: str = ""
    url: str = ""
    snippet: str = ""
    domain: str = ""
    published_date: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_text(self, index: int = 0) -> str:
        lines = [f"[{index}] {self.title}"]
        if self.url:
            lines.append(f"    URL: {self.url}")
        if self.snippet:
            lines.append(f"    {self.snippet}")
        if self.published_date:
            lines.append(f"    日期: {self.published_date}")
        return "\n".join(lines)


class AnySearchClient:
    """AnySearch HTTP 客户端（JSON-RPC 2.0 over HTTP）"""

    def __init__(self, api_key: str = "", timeout: float = 30.0):
        self.api_key = api_key or ANYSEARCH_API_KEY
        self.timeout = timeout
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"

    async def _rpc(self, tool_name: str, arguments: dict, req_id: int = 1) -> str:
        """
        JSON-RPC 调用 anysearch MCP 端点，返回 text 内容。
        等效 curl:
          curl -s -X POST "https://api.anysearch.com/mcp" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $ANYSEARCH_API_KEY" \
            -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"TOOL","arguments":{...}},"id":1}'
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": req_id,
        }
        client = _get_async_client(self.timeout)
        resp = await client.post(ANYSEARCH_URL, json=payload, headers=self._headers)
        resp.raise_for_status()
        data = resp.json()

        # 提取 result.content[0].text
        result = data.get("result", {})
        for item in result.get("content", []):
            if item.get("type") == "text":
                return item.get("text", "")
        return ""

    # ── Markdown 解析 ──────────────────────────────────────────────

    @staticmethod
    def _parse_markdown_results(text: str) -> list[SearchResult]:
        """解析 anysearch 返回的 Markdown 格式搜索结果"""
        items: list[SearchResult] = []
        if not text:
            return items

        # 结果块: ### N. Title\n- **URL**: ...\n- snippet...
        blocks = re.split(r"\n### \d+\.\s*", text)
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n")
            title = lines[0].strip() if lines else ""
            # 跳过 header
            if not title or title.startswith("Search Results") or title.startswith("##"):
                continue

            url = ""
            snippet_parts = []
            i = 1
            while i < len(lines):
                line = lines[i].strip()
                m = re.match(r"-\s*\*\*URL\*\*:\s*(.+)", line)
                if m:
                    url = m.group(1).strip()
                    i += 1
                    continue
                if line.startswith("- **"):
                    i += 1
                    continue
                if not line or line.startswith("---"):
                    i += 1
                    continue
                snippet_parts.append(line)
                i += 1

            items.append(SearchResult(
                title=title,
                url=url,
                snippet=" ".join(snippet_parts),
            ))

        return items

    # ── 公开方法 ───────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        max_results: int = 10,
        freshness: str = "month",
        domain: str = "",
        sub_domain: str = "",
    ) -> list[SearchResult]:
        """通用 / 垂直领域搜索

        等效 curl:
          curl -s -X POST "$ANYSEARCH_URL" -H "Content-Type: application/json" \\
            -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"search","arguments":{"query":"关键词","max_results":10}},"id":1}'
        """
        args: dict = {"query": query, "max_results": max_results}
        if freshness:
            args["freshness"] = freshness
        if domain:
            args["domain"] = domain
        if sub_domain:
            args["sub_domain"] = sub_domain

        cache_key = _search_cache_key(args)
        cached = _search_cache.get(cache_key)
        if cached is not None:
            expire_at, results = cached
            if time.time() <= expire_at:
                return list(results)
            del _search_cache[cache_key]

        text = await self._rpc("search", args)
        results = self._parse_markdown_results(text)
        if SEARCH_CACHE_TTL > 0:
            _search_cache[cache_key] = (time.time() + SEARCH_CACHE_TTL, results)
            if len(_search_cache) > 1000:
                oldest = min(_search_cache, key=lambda k: _search_cache[k][0])
                del _search_cache[oldest]
        return results

    async def batch_search(self, queries: list[dict]) -> list[list[SearchResult]]:
        """批量搜索（已废弃，当前项目不使用。保留接口供参考。）

        等效 curl:
          curl ... -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"batch_search","arguments":{"queries":[{...},{...}]}},"id":1}'
        """
        text = await self._rpc("batch_search", {"queries": queries})
        items = self._parse_markdown_results(text)
        # 注意：anysearch batch 返回的 Markdown 不按 query 分组，
        # 此处无法拆分，统一返回单列表。
        return [items] if items else []

    async def extract(self, url: str) -> str:
        """提取网页内容为 Markdown（上限约 50k 字符）

        等效 curl:
          curl ... -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"extract","arguments":{"url":"https://..."}},"id":1}'
        """
        return await self._rpc("extract", {"url": url})

    async def list_domains(self, domain: str = "") -> str:
        """列出可用的垂直领域

        等效 curl:
          curl ... -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"list_domains","arguments":{}},"id":1}'
        """
        args: dict = {}
        if domain:
            args["domain"] = domain
        return await self._rpc("list_domains", args)


class AnySearchClientSync:
    """AnySearch HTTP 客户端（同步版，基于 requests）"""

    def __init__(self, api_key: str = "", timeout: float = 30.0):
        import requests as _requests
        self._requests = _requests
        self.api_key = api_key or ANYSEARCH_API_KEY
        self.timeout = timeout
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"

    def _rpc(self, tool_name: str, arguments: dict, req_id: int = 1) -> str:
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": req_id,
        }
        resp = self._requests.post(
            ANYSEARCH_URL, json=payload, headers=self._headers, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", {})
        for item in result.get("content", []):
            if item.get("type") == "text":
                return item.get("text", "")
        return ""

    def search(self, query: str, max_results: int = 10, freshness: str = "month",
               domain: str = "", sub_domain: str = "") -> list[SearchResult]:
        args: dict = {"query": query, "max_results": max_results}
        if freshness:
            args["freshness"] = freshness
        if domain:
            args["domain"] = domain
        if sub_domain:
            args["sub_domain"] = sub_domain
        text = self._rpc("search", args)
        return AnySearchClient._parse_markdown_results(text)

    def extract(self, url: str) -> str:
        return self._rpc("extract", {"url": url})

    def list_domains(self, domain: str = "") -> str:
        args: dict = {}
        if domain:
            args["domain"] = domain
        return self._rpc("list_domains", args)


# 便捷函数
async def search_web(query: str, max_results: int = 10) -> list[SearchResult]:
    """异步快速搜索"""
    client = AnySearchClient()
    return await client.search(query, max_results=max_results)


def search_sync(query: str, max_results: int = 10) -> list[SearchResult]:
    """同步快速搜索（无需 asyncio，同事可直接调用）"""
    client = AnySearchClientSync()
    return client.search(query, max_results=max_results)
