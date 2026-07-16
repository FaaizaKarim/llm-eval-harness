"""Turns a list of EvalResults into JSON and human-readable reports."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .models import EvalResult


def summarize(results: list[EvalResult]) -> dict:
    """Aggregate metrics for a run."""
    total = len(results)
    passed = sum(r.passed for r in results)
    modes = Counter(r.failure_mode.value for r in results if not r.passed)
    return {
        "model": results[0].model if results else "n/a",
        "tasks": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "avg_duration_s": round(sum(r.duration_s for r in results) / total, 3)
        if total else 0.0,
        "failure_modes": dict(modes.most_common()),
    }


def to_json(results: list[EvalResult], path: str | Path) -> None:
    """Write full, reproducible results (including error traces) to JSON."""
    payload = {
        "summary": summarize(results),
        "results": [r.to_dict() for r in results],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def to_markdown(results: list[EvalResult]) -> str:
    """Render a compact markdown report for humans."""
    s = summarize(results)
    lines = [
        f"# Evaluation report — {s['model']}",
        "",
        f"**Pass rate:** {s['passed']}/{s['tasks']} ({s['pass_rate']:.0%})  ",
        f"**Avg duration:** {s['avg_duration_s']}s",
        "",
        "| Task | Result | Failure mode | Tests |",
        "|---|---|---|---|",
    ]
    for r in results:
        status = "✅ pass" if r.passed else "❌ fail"
        lines.append(
            f"| {r.task_id} | {status} | {r.failure_mode.value} "
            f"| {r.tests_passed}/{r.tests_total} |"
        )
    failures = [r for r in results if not r.passed and r.error_trace]
    if failures:
        lines += ["", "## Failure details", ""]
        for r in failures:
            lines += [f"### {r.task_id} — {r.failure_mode.value}", "",
                      "```", r.error_trace.strip(), "```", ""]
    return "\n".join(lines)
