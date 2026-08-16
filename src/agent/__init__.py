"""3MagicLabs agent.

Public API::

    from agent import solve                 # CodeAct: write a program, run it
    from agent import answer_question       # supervisor: route to specialists

Attributes resolve lazily. Importing this package used to pull in langgraph and
the whole orchestrator, so a module needing nothing but ``Settings`` paid for
the entire stack - and every sandboxed program paid it again per step, several
seconds each. Nothing here imports a model client or a graph until asked for.
"""

from __future__ import annotations

from typing import Any

from agent.config import Settings, get_settings, load_settings, set_settings

__version__ = "0.1.0"

#: Attribute -> module that defines it, imported on first access.
_LAZY = {
    "solve": "agent.codeact",
    "NoAnswerError": "agent.codeact",
    "Orchestrator": "agent.core.graph",
    "answer_question": "agent.core.graph",
    "get_orchestrator": "agent.core.graph",
    "reset_orchestrator": "agent.core.graph",
}

__all__ = [
    "NoAnswerError",
    "Orchestrator",
    "Settings",
    "__version__",
    "answer_question",
    "get_orchestrator",
    "get_settings",
    "load_settings",
    "reset_orchestrator",
    "set_settings",
    "solve",
]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
