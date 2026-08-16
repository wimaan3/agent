"""The CodeAct kernel: parsing, the gate, execution bounds, and the loop."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from agent.codeact import loop as loop_module
from agent.codeact.parse import UnsafeProgramError, calls_final_answer, extract, vet
from agent.codeact.prompt import build
from agent.codeact.sandbox import Execution, _elide, run
from agent.config import Settings

pytestmark = pytest.mark.unit


class TestExtract:
    def test_takes_the_fenced_block(self):
        assert extract("Sure!\n```python\nx = 1\n```\nDone.") == "x = 1"

    def test_accepts_an_untagged_fence(self):
        assert extract("```\ny = 2\n```") == "y = 2"

    def test_takes_the_longest_block_when_several(self):
        reply = "```python\nx=1\n```\nand\n```python\nprint('much longer program')\n```"
        assert "much longer" in extract(reply)

    def test_falls_back_to_the_whole_reply(self):
        """An unfenced reply is usually bare code, not prose."""
        assert extract("final_answer(3)") == "final_answer(3)"


class TestGate:
    @pytest.mark.parametrize(
        "code",
        [
            "import subprocess",
            "from shutil import rmtree",
            "import socket",
            "eval('1+1')",
            "exec('x=1')",
            "__import__('os')",
            "import os; os.system('ls')",
            "import os; os.remove('/tmp/x')",
        ],
    )
    def test_rejects_what_a_task_never_needs(self, code):
        with pytest.raises(UnsafeProgramError):
            vet(code)

    @pytest.mark.parametrize(
        "code",
        [
            "import pandas as pd",
            "import os; print(os.path.join('a', 'b'))",
            "import re, json, math",
            "print(read(attachment('abc')))",
        ],
    )
    def test_allows_ordinary_work(self, code):
        assert vet(code) is not None

    def test_syntax_errors_are_refusals_not_crashes(self):
        with pytest.raises(UnsafeProgramError, match="not valid Python"):
            vet("def broken(")

    def test_detects_the_termination_call(self):
        assert calls_final_answer(vet("final_answer(1)"))
        assert not calls_final_answer(vet("print(1)"))


class TestSandbox:
    def test_final_answer_is_captured(self):
        assert run("final_answer(6*7)", 30, 2000).answer == "42"

    def test_printed_output_becomes_the_observation(self):
        assert run("print('seen')", 30, 2000).observation == "seen"

    def test_a_traceback_is_returned_not_raised(self):
        result = run("1/0", 30, 2000)

        assert result.answer is None
        assert "ZeroDivisionError" in result.observation

    def test_a_refused_program_is_not_labelled_a_traceback(self):
        """Mislabelling invites the model to fix code that was never the problem."""
        result = run("import socket", 30, 2000)

        assert "Refused" in result.observation
        assert not result.observation.startswith("Traceback")

    def test_an_endless_program_is_killed(self):
        assert "Timed out" in run("while True: pass", 3, 2000).error

    def test_silence_is_reported_explicitly(self):
        assert "printed nothing" in run("x = 1", 30, 2000).observation


class TestElide:
    def test_short_output_is_untouched(self):
        assert _elide("abc", 100) == "abc"

    def test_the_middle_goes_not_the_tail(self):
        """The end of a program's output is usually where the total is."""
        elided = _elide("START" + "x" * 500 + "END", 100)

        assert elided.startswith("START")
        assert elided.endswith("END")
        assert "elided" in elided


class TestPrompt:
    def test_the_first_turn_has_no_history(self):
        _, user = build("how many?", [], keep=3)

        assert "how many?" in user
        assert "program 1" not in user

    def test_tools_are_documented_from_the_functions(self):
        system, _ = build("q", [], keep=3)

        assert "final_answer" in system
        assert "attachment" in system

    def test_older_steps_collapse_to_a_count(self):
        steps = [(f"code{i}", f"out{i}") for i in range(6)]

        _, user = build("q", steps, keep=2)

        assert "4 earlier step(s) omitted" in user
        assert "out5" in user
        assert "out0" not in user


class ScriptedLLM:
    """Replays canned replies, one per step."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[Any] = []

    def bind(self, **_kwargs: Any) -> ScriptedLLM:
        return self

    def invoke(self, messages: Any, **_kwargs: Any) -> AIMessage:
        self.prompts.append(messages)
        return AIMessage(content=self.replies.pop(0) if self.replies else "```python\npass\n```")


class TestLoop:
    def _install(self, monkeypatch, replies):
        llm = ScriptedLLM(replies)
        monkeypatch.setattr(loop_module, "get_llm", lambda: llm)
        return llm

    def test_returns_the_answer(self, settings, monkeypatch):
        self._install(monkeypatch, ["```python\nfinal_answer('right')\n```"])

        assert loop_module.solve("q", settings=settings) == "right"

    def test_an_observation_is_fed_back(self, settings, monkeypatch):
        llm = self._install(
            monkeypatch,
            ["```python\nprint('intermediate fact')\n```", "```python\nfinal_answer('done')\n```"],
        )

        assert loop_module.solve("q", settings=settings) == "done"
        assert "intermediate fact" in str(llm.prompts[1][-1].content)

    def test_never_answering_raises(self, settings, monkeypatch):
        """Returning a guess is what let a 0/20 run report zero errors."""
        capped = Settings(
            log_dir=settings.log_dir, download_dir=settings.download_dir, max_program_steps=2
        )
        self._install(monkeypatch, ["```python\nprint(1)\n```"] * 2)

        with pytest.raises(loop_module.NoAnswerError, match="No final_answer after 2"):
            loop_module.solve("q", settings=capped)

    def test_a_reply_without_code_is_corrected(self, settings, monkeypatch):
        llm = self._install(monkeypatch, ["", "```python\nfinal_answer('recovered')\n```"])

        assert loop_module.solve("q", settings=settings) == "recovered"
        assert "```python" in str(llm.prompts[1][-1].content)


class TestExecutionShape:
    def test_observation_prefers_stdout(self):
        assert Execution(stdout="out").observation == "out"

    def test_empty_execution_is_explicit(self):
        assert "printed nothing" in Execution().observation
