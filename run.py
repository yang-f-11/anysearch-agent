#!/usr/bin/env python3
"""
AnySearch Agent — 联网搜索智能助手

用法:
    python run.py "今天比特币价格"
    python run.py --model gemini-3-pro "量子计算最新进展"
  python run.py --interactive          # 交互模式
  python run.py --no-stream "问题"     # 非流式输出
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from agent import search_and_answer, DEFAULT_MAX_RESULTS, DEFAULT_FRESHNESS, DEFAULT_MODEL


async def run_once(query: str, model: str, max_results: int, freshness: str, stream: bool):
    """单次问答"""
    await search_and_answer(
        query=query,
        model=model,
        max_results=max_results,
        freshness=freshness,
        stream=stream,
    )


async def run_interactive(model: str, max_results: int, freshness: str):
    """交互模式"""
    print("=" * 60)
    print("🌐 AnySearch Agent — 联网搜索助手")
    print(f"   模型: {model or DEFAULT_MODEL}")
    print(f"   搜索: anysearch.com")
    print(f"   LLM:  llm_hub ({DEFAULT_MODEL})")
    print("=" * 60)
    print("输入 /help 查看帮助，/exit 退出\n")

    while True:
        try:
            query = input("💬 提问: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见")
            break

        if not query:
            continue

        if query == "/exit":
            print("👋 再见")
            break
        if query == "/help":
            print(f"""
  命令:
    /exit        退出
    /help        显示帮助
    /model <名>  切换模型
    /fresh <值>  设置搜索时效 (day/week/month/year)
    /results <n> 设置搜索结果数 (1-100)
""")
            continue
        if query.startswith("/model "):
            model = query.split(" ", 1)[1].strip()
            print(f"✅ 模型已切换为: {model}")
            continue
        if query.startswith("/fresh "):
            freshness = query.split(" ", 1)[1].strip()
            print(f"✅ 搜索时效已设为: {freshness}")
            continue
        if query.startswith("/results "):
            try:
                max_results = int(query.split(" ", 1)[1])
                print(f"✅ 搜索结果数已设为: {max_results}")
            except ValueError:
                print("❌ 请输入有效数字")
            continue

        print()
        try:
            await search_and_answer(
                query=query,
                model=model,
                max_results=max_results,
                freshness=freshness,
                stream=True,
            )
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="AnySearch Agent — 联网搜索智能助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py "今天北京天气"
  python run.py -i                   # 交互模式（本地）
  python run.py -s                   # 启动 TCP 服务（同事 nc 直连）
  python run.py --model gemini-3-pro "量子计算进展"
        """,
    )
    parser.add_argument("query", nargs="?", help="搜索问题")
    parser.add_argument("--model", "-m", default="", help=f"LLM 模型名 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--max-results", "-n", type=int, default=DEFAULT_MAX_RESULTS, help=f"最大搜索结果数 (默认: {DEFAULT_MAX_RESULTS})")
    parser.add_argument("--freshness", "-f", default=DEFAULT_FRESHNESS, help=f"搜索时效: day/week/month/year (默认: {DEFAULT_FRESHNESS})")
    parser.add_argument("--no-stream", action="store_true", help="禁用流式输出")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--serve", "-s", action="store_true", help="启动 TCP 服务（同事用 nc 直连进入交互模式）")
    parser.add_argument("--port", "-p", type=int, default=9090, help="TCP 服务端口 (配合 --serve，默认: 9090)")

    args = parser.parse_args()

    if args.serve:
        from server import serve as run_server
        asyncio.run(run_server(port=args.port))
    elif args.interactive:
        asyncio.run(run_interactive(args.model, args.max_results, args.freshness))
    elif args.query:
        asyncio.run(run_once(args.query, args.model, args.max_results, args.freshness, not args.no_stream))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
