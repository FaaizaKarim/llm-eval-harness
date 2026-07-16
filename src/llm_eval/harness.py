"""Orchestrates an evaluation run: prompt -> generate -> sandbox -> score.

Tasks are evaluated concurrently with a bounded semaphore so a large
benchmark neither serializes on network latency nor overwhelms the
machine with subprocesses.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from .client import LLMClient, extract_code
from .models import EvalResult, FailureMode, Task

logger = logging.getLogger(__name__)


def load_tasks(path: str | Path) -> list[Task]:
    """Load a task suite from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Task.from_dict(item) for item in raw]


class Harness:
    """Runs a task suite against one model and collects EvalResults."""

    def __init__(
        self,
        client: LLMClient,
        max_concurrency: int = 4,
        timeout_s: float = 10.0,
        memory_limit_mb: int = 256,
    ) -> None:
        self._client = client
        self._sem = asyncio.Semaphore(max_concurrency)
        self._timeout_s = timeout_s
        self._memory_limit_mb = memory_limit_mb

    async def evaluate_task(self, task: Task) -> EvalResult:
        """Evaluate a single task end to end."""
        from .scorer import score  # local import avoids cycle at module load

        async with self._sem:
            try:
                reply = await self._client.generate(task.prompt)
            except Exception as exc:  # network/provider errors are data too
                logger.warning("generation failed for %s: %s", task.task_id, exc)
                return EvalResult(
                    task_id=task.task_id, model=self._client.name, passed=False,
                    failure_mode=FailureMode.SANDBOX_ERROR, tests_passed=0,
                    tests_total=len(task.test_cases), duration_s=0.0,
                    error_trace=f"generation error: {exc!r}",
                )
            code = extract_code(reply)
            result = await score(
                code, task, self._client.name,
                timeout_s=self._timeout_s,
                memory_limit_mb=self._memory_limit_mb,
            )
            logger.info(
                "%s | %s | %s (%d/%d)",
                task.task_id, self._client.name,
                result.failure_mode.value, result.tests_passed, result.tests_total,
            )
            return result

    async def run(self, tasks: list[Task]) -> list[EvalResult]:
        """Evaluate all tasks concurrently and return results in task order."""
        return list(await asyncio.gather(*(self.evaluate_task(t) for t in tasks)))
