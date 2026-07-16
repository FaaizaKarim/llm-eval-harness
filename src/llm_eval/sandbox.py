"""Sandboxed execution of untrusted, model-generated Python code.

Code runs in a separate Python subprocess with:
  * a hard wall-clock timeout,
  * an address-space (memory) limit via ``resource`` on POSIX,
  * no inherited environment variables,
  * an isolated temporary working directory.

This is process-level isolation, suitable for local benchmarking. For
hostile inputs, run the harness itself inside the provided Docker image
(defense in depth).
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

from .models import ExecutionResult

_PREAMBLE = """\
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
except (ImportError, ValueError, OSError):
    pass  # resource module is POSIX-only (absent on Windows); timeout still applies
"""


async def run_python(
    code: str,
    timeout_s: float = 10.0,
    memory_limit_mb: int = 256,
) -> ExecutionResult:
    """Execute Python source in an isolated subprocess.

    Returns an ExecutionResult; never raises for failures of the
    *executed* code — those are captured as data.
    """
    mem_bytes = memory_limit_mb * 1024 * 1024
    wrapped = _PREAMBLE.format(mem=mem_bytes) + "\n" + code

    with tempfile.TemporaryDirectory(prefix="llm_eval_") as tmp:
        script = Path(tmp) / "solution.py"
        script.write_text(wrapped, encoding="utf-8")

        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", str(script),  # -I: isolated mode
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tmp,
            env={},  # do not leak API keys or host env to untrusted code
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecutionResult(
                timed_out=True,
                exit_code=-9,
                duration_s=time.monotonic() - start,
                stderr=f"Timed out after {timeout_s}s",
            )

        return ExecutionResult(
            stdout=stdout_b.decode(errors="replace"),
            stderr=stderr_b.decode(errors="replace"),
            exit_code=proc.returncode or 0,
            duration_s=time.monotonic() - start,
        )
