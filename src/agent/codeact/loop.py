"""The CodeAct loop: write a program, run it, look at the output, repeat.

Chosen over the supervisor topology on measured grounds. In the supervisor,
84 provider errors were malformed function-calling syntax - code escaped inside
a JSON string field - which a fenced code block cannot produce. 96.8% of tokens
were transcript replay, which a bounded step window removes. And 71% of routing
decisions went to one specialist anyway, so the router was mostly cost.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.codeact import prompt as prompt_module
from agent.codeact.parse import extract
from agent.codeact.sandbox import run
from agent.config import Settings, get_settings
from agent.core.llm import get_llm
from agent.obs.logging import get_logger
from agent.obs.tracing import trace_config

log = get_logger("codeact.loop")


class NoAnswerError(RuntimeError):
    """The loop ended without the program ever calling final_answer."""


def solve(
    question: str,
    task_id: str = "local",
    callbacks: list[Any] | None = None,
    settings: Settings | None = None,
) -> str:
    """Answer one task. Raises rather than inventing an answer.

    Raising is deliberate: the supervisor's habit of returning its best guess
    when nothing had worked is what let a 0/20 run report zero errors.
    """
    resolved = settings or get_settings()
    llm = get_llm().bind(max_tokens=resolved.max_program_tokens)
    config = cast(RunnableConfig, trace_config(task_id, callbacks))
    steps: list[tuple[str, str]] = []

    for step in range(1, resolved.max_program_steps + 1):
        system, user = prompt_module.build(question, steps, resolved.program_window)
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        reply = llm.invoke(messages, config=config)
        code = extract(str(reply.content))

        if not code:
            log.warning("step %d/%d produced no code", step, resolved.max_program_steps)
            steps.append(("", "You replied with no code block. Reply with one ```python block."))
            continue

        execution = run(code, resolved.program_timeout_s, resolved.max_observation_chars)
        log.info(
            "step %d/%d: %d chars of code -> %s",
            step,
            resolved.max_program_steps,
            len(code),
            "ANSWER"
            if execution.answer is not None
            else execution.observation[:80].replace("\n", " "),
        )

        if execution.answer is not None:
            return execution.answer.strip()

        steps.append((code, execution.observation))

    raise NoAnswerError(f"No final_answer after {resolved.max_program_steps} programs.")
