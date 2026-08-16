"""Prompt construction for the CodeAct loop."""

from __future__ import annotations

from agent.codeact.api import signatures

SYSTEM = """You solve tasks by writing Python. Each turn you write one program;
it runs; you see what it printed; you write the next one.

Available in every program (already imported, do not import them):
{tools}

The standard library is available, plus pandas and numpy.

Rules:
- Reply with exactly ONE ```python code block and nothing else.
- print() anything you need to see. Printed output is all you get back;
  variables do not survive between programs. Files on disk do.
- Call final_answer(value) the moment you know the answer. Nothing after it runs.
- Never invent a fact. If you need information, fetch it in code.
- Character-level work (reversing, decoding, counting) must be done in code,
  never in your head - you read tokens, not characters, and you will be wrong.
- If a program errors, read the traceback and write different code. Do not
  repeat a program that just failed.

final_answer takes the graded answer and nothing else. It is compared by exact
string match, so: a number is digits only, no separators, no units, no currency
symbol; a string is as few words as possible with no leading article; a list is
comma-separated with one space after each comma."""

STEP = """Task:
{question}

{history}
Write the next program."""


def build(question: str, steps: list[tuple[str, str]], keep: int) -> tuple[str, str]:
    """Return (system, user) for the next turn.

    Only the most recent ``keep`` steps are replayed. Older programs are
    summarised to one line: an agent that already read a page rarely needs its
    text again, and replaying it is what took 96.8% of the supervisor's tokens.
    """
    system = SYSTEM.format(tools=signatures())

    if not steps:
        return system, STEP.format(question=question, history="")

    older, recent = steps[:-keep], steps[-keep:]
    lines: list[str] = []
    if older:
        lines.append(f"({len(older)} earlier step(s) omitted.)\n")
    for index, (code, observation) in enumerate(recent, start=len(older) + 1):
        lines.append(f"--- program {index} ---\n{code}\n--- output {index} ---\n{observation}\n")

    return system, STEP.format(question=question, history="\n".join(lines) + "\n")
