"""CodeAct: the agent acts by writing Python programs rather than tool calls.

Lazy on purpose. The sandbox worker imports only ``codeact.api`` for the tool
namespace, and an eager re-export of the loop made it pay for langchain and the
model client on every program - measured at 7 seconds per step.
"""

from __future__ import annotations

from typing import Any

__all__ = ["NoAnswerError", "solve"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from agent.codeact import loop

        return getattr(loop, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
