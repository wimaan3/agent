"""Parent side of program execution: spawn, bound, kill, collect."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agent.codeact.parse import UnsafeProgramError, vet
from agent.obs.logging import get_logger

log = get_logger("codeact.sandbox")


@dataclass(frozen=True, slots=True)
class Execution:
    """What running one program produced."""

    stdout: str = ""
    answer: str | None = None
    error: str | None = None

    @property
    def observation(self) -> str:
        """What the model is shown next step."""
        parts = [self.stdout.strip()]
        if self.error:
            # Only label it a traceback when it is one; a refusal or a timeout
            # is a fact about the run, and mislabelling it invites the model to
            # "fix" code that was never the problem.
            label = "Traceback" if self.error.startswith("Traceback") else "Result"
            parts.append(f"{label}:\n{self.error}")
        joined = "\n".join(p for p in parts if p)
        return joined or "(the program printed nothing)"


#: This file is src/agent/codeact/sandbox.py, so parents[2] is the source root.
_SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _child_env() -> dict[str, str]:
    """Environment pinning the child to the parent's own source tree.

    Without this the child imports whatever ``agent`` happens to be installed,
    which on a machine with more than one checkout is the wrong one.
    """
    existing = os.environ.get("PYTHONPATH", "")
    path = os.pathsep.join(p for p in (str(_SOURCE_ROOT), existing) if p)
    return {**os.environ, "PYTHONPATH": path}


def run(code: str, timeout_s: float, observation_chars: int) -> Execution:
    """Execute a generated program in a child process and collect its output.

    The child is put in its own process group so a timeout can kill everything
    it spawned, not just the interpreter we can see.
    """
    try:
        vet(code)
    except UnsafeProgramError as exc:
        log.warning("refused program: %s", exc)
        return Execution(error=f"Refused: {exc}")

    with tempfile.TemporaryDirectory(prefix="codeact-") as workdir:
        program = Path(workdir) / "program.py"
        result = Path(workdir) / "result.json"
        program.write_text(code, encoding="utf-8")

        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "agent.codeact.worker", str(program), str(result)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
            env=_child_env(),
        )
        try:
            _, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=5)
            log.warning("program exceeded %.0fs and was killed", timeout_s)
            return Execution(error=f"Timed out after {timeout_s:.0f}s.")

        if not result.exists():
            detail = (stderr or "").strip()[-400:] or f"exit code {process.returncode}"
            log.error("worker produced no result: %s", detail)
            return Execution(error=f"The program could not be run: {detail}")

        payload = json.loads(result.read_text(encoding="utf-8"))

    return Execution(
        stdout=_elide(str(payload.get("stdout", "")), observation_chars),
        answer=payload.get("answer"),
        error=payload.get("error"),
    )


def _elide(text: str, limit: int) -> str:
    """Trim the middle, not the tail.

    The end of a program's output is usually the part that matters - a total, a
    final row - so truncating tail-first throws away the answer.
    """
    if len(text) <= limit:
        return text
    head, tail = limit // 2, limit - limit // 2
    dropped = len(text) - limit
    return f"{text[:head]}\n... [{dropped} characters elided] ...\n{text[-tail:]}"
