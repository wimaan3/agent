"""Child process that executes one generated program.

Run as ``python -m agent.codeact.worker <program> <result>``. Deliberately a
separate process: a program that hangs, allocates without bound or segfaults
takes only this process with it, and the parent can kill the whole group.
"""

from __future__ import annotations

import contextlib
import io
import json
import resource
import sys
import traceback
from pathlib import Path
from typing import Any

#: Address space and CPU ceilings. Generous enough for pandas on a spreadsheet,
#: tight enough that a runaway loop dies rather than swapping the machine.
MEMORY_BYTES = 1_024 * 1024 * 1024
CPU_SECONDS = 60


def _apply_limits() -> None:
    for limit, value in ((resource.RLIMIT_AS, MEMORY_BYTES), (resource.RLIMIT_CPU, CPU_SECONDS)):
        # Platform dependent; a missing ceiling is worth less than refusing to run.
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(limit, (value, value))


def execute(code: str) -> dict[str, Any]:
    """Run ``code`` with the tool namespace, capturing everything it emits."""
    from agent.codeact.api import NAMESPACE, FinalAnswer

    namespace: dict[str, Any] = {"__name__": "__main__", **NAMESPACE}
    buffer = io.StringIO()
    answer: str | None = None
    error: str | None = None

    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(code, namespace)  # noqa: S102 - the entire point of this module
    except FinalAnswer as done:
        answer = str(done.value)
    except BaseException:  # noqa: BLE001 - report every failure, including exits
        error = traceback.format_exc(limit=3).strip()

    return {"stdout": buffer.getvalue(), "answer": answer, "error": error}


def main(argv: list[str]) -> int:
    program_path, result_path = Path(argv[1]), Path(argv[2])
    _apply_limits()
    result = execute(program_path.read_text(encoding="utf-8"))
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main(sys.argv))
