"""The namespace a generated program runs inside.

Every name here is callable from model-written code. Tools are plain functions
rather than schema-bound tools: the model expresses a call as Python source, so
there is no JSON argument to escape and no function-calling syntax to malform -
which is what 84 of the supervisor's HTTP 400s were.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.config import get_settings


class FinalAnswer(Exception):  # noqa: N818 - control flow, not an error
    """Raised by ``final_answer`` to stop the program with a result."""

    def __init__(self, value: Any) -> None:
        self.value = value
        super().__init__(str(value))


def final_answer(value: Any) -> None:
    """Submit the answer and stop. The only way a program can finish."""
    raise FinalAnswer(value)


def _unwrap(tool: Any) -> Any:
    """The plain function behind a LangChain tool object."""
    return getattr(tool, "func", tool)


def search(query: str) -> str:
    """Web search. Returns titles, urls and snippets as text."""
    from agent.tools.web import web_search

    return str(_unwrap(web_search)(query))


def fetch(url: str) -> str:
    """Fetch a page and return its readable text."""
    from agent.tools.web import scrape_webpage

    return str(_unwrap(scrape_webpage)(url))


def wiki(title: str) -> str:
    """Plain-text extract of a Wikipedia article."""
    from agent.tools.web import wikipedia_lookup

    return str(_unwrap(wikipedia_lookup)(title))


def download(task_id: str) -> str:
    """Download this task's attachment. Memoised: safe to call repeatedly."""
    from agent.tools.files import download_task_file

    return str(_unwrap(download_task_file)(task_id))


def read(path: str) -> str:
    """Read a downloaded file as text. Spreadsheets come back as a table."""
    from agent.tools.files import read_file

    return str(_unwrap(read_file)(path))


def attachment(task_id: str) -> str:
    """Local path of this task's attachment, downloading it if needed.

    Returns a path usable with pandas, open(), etc. - the common case is two
    calls (download then read) collapsed into one.
    """
    download(task_id)
    matches = sorted(p for p in get_settings().download_dir.glob(f"{task_id}*") if p.is_file())
    return str(matches[0]) if matches else ""


def files() -> list[str]:
    """Paths of everything downloaded so far this run."""
    directory: Path = get_settings().download_dir
    if not directory.exists():
        return []
    return sorted(str(p) for p in directory.iterdir() if p.is_file())


#: Injected into every program's globals. Ordering here is the order documented
#: to the model, so the cheapest and most specific tools come first.
NAMESPACE = {
    "final_answer": final_answer,
    "attachment": attachment,
    "read": read,
    "download": download,
    "wiki": wiki,
    "search": search,
    "fetch": fetch,
    "files": files,
}


def signatures() -> str:
    """Tool documentation, generated from the functions themselves.

    Generated rather than written out, so a rename can never leave the prompt
    describing a function that no longer exists.
    """
    lines = []
    for name, fn in NAMESPACE.items():
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        annotations = getattr(fn, "__annotations__", {})
        params = ", ".join(
            f"{p}: {getattr(t, '__name__', t)}" for p, t in annotations.items() if p != "return"
        )
        lines.append(f"{name}({params}) - {doc}")
    return "\n".join(lines)
