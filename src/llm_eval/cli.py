"""Command-line entry point.

Examples:
    # offline demo with the built-in mock model
    llm-eval --tasks tasks/tasks.json --mock

    # against a real OpenAI-compatible endpoint
    export LLM_API_KEY=sk-...
    llm-eval --tasks tasks/tasks.json --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .client import MockClient, OpenAICompatibleClient
from .harness import Harness, load_tasks
from .report import to_json, to_markdown

_DEMO_RESPONSES = {
    "two integers": "```python\ndef add(a, b):\n    return a + b\n```",
    "palindrome": (
        "```python\ndef is_palindrome(s):\n"
        "    s = ''.join(c.lower() for c in s if c.isalnum())\n"
        "    return s == s[::-1]\n```"
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate an LLM on Python coding tasks.")
    parser.add_argument("--tasks", required=True, help="Path to a tasks JSON file")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--mock", action="store_true", help="Use the offline mock client")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-task sandbox timeout (s)")
    parser.add_argument("--out", default="results.json", help="JSON results path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = (
        MockClient(responses=_DEMO_RESPONSES)
        if args.mock
        else OpenAICompatibleClient(model=args.model, base_url=args.base_url)
    )
    harness = Harness(client, max_concurrency=args.concurrency, timeout_s=args.timeout)

    tasks = load_tasks(args.tasks)
    results = asyncio.run(harness.run(tasks))

    to_json(results, args.out)
    print(to_markdown(results))
    print(f"\nFull results written to {args.out}", file=sys.stderr)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
