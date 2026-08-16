"""Extract and vet the program a model produced.

Two jobs: find the code among the prose, and refuse code that has no business
running. The gate is a denylist over the AST rather than a regex, because the
interesting attacks are all spelled differently than they read.
"""

from __future__ import annotations

import ast
import re

#: Fenced blocks, optionally tagged ```python.
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

#: Modules a benchmark program never legitimately needs, and that give away the
#: machine if a prompt injection reaches the generator.
BANNED_MODULES = frozenset(
    {"subprocess", "shutil", "socket", "ctypes", "multiprocessing", "importlib", "pty", "signal"}
)

#: Builtins that defeat the module gate by resolving names at runtime.
BANNED_CALLS = frozenset({"eval", "exec", "compile", "__import__", "breakpoint", "input"})

#: os is allowed for path work but not for process or filesystem destruction.
BANNED_ATTRIBUTES = frozenset(
    {
        "system",
        "popen",
        "spawn",
        "spawnl",
        "spawnv",
        "fork",
        "execv",
        "execl",
        "rmtree",
        "remove",
        "unlink",
        "rmdir",
    }
)


class UnsafeProgramError(Exception):
    """The program asked for something outside what a benchmark task needs."""


def extract(text: str) -> str:
    """The code from a model reply: the fenced block, or the whole reply.

    Models emit prose around code more often than not, and an unfenced reply is
    usually bare code rather than a refusal - so falling back to the whole text
    recovers more than it breaks.
    """
    blocks = _FENCE.findall(text)
    if blocks:
        return str(max(blocks, key=len)).strip()
    return text.strip()


def _module_roots(node: ast.AST) -> set[str]:
    roots: set[str] = set()
    if isinstance(node, ast.Import):
        roots |= {alias.name.split(".")[0] for alias in node.names}
    elif isinstance(node, ast.ImportFrom) and node.module:
        roots.add(node.module.split(".")[0])
    return roots


def vet(code: str) -> ast.Module:
    """Parse and reject anything on the denylist. Raises UnsafeProgram."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeProgramError(f"not valid Python: {exc}") from exc

    for node in ast.walk(tree):
        banned = _module_roots(node) & BANNED_MODULES
        if banned:
            raise UnsafeProgramError(f"import of {', '.join(sorted(banned))} is not allowed")

        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id in BANNED_CALLS:
                raise UnsafeProgramError(f"{target.id}() is not allowed")
            if isinstance(target, ast.Attribute) and target.attr in BANNED_ATTRIBUTES:
                raise UnsafeProgramError(f".{target.attr}() is not allowed")

    return tree


def calls_final_answer(tree: ast.Module) -> bool:
    """Whether the program can terminate the task."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "final_answer"
        for node in ast.walk(tree)
    )
