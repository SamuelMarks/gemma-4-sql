"""Module docstring."""

import sys
from unittest import mock

import gemma_4_sql.backends.maxtext.few_shot as fs


def test_build_few_shot_prompt_maxtext() -> object:  # type: ignore[return]
    """Initialize function test_build_few_shot_prompt_maxtext."""
    res = fs.build_few_shot_prompt("foo", "prompt", [{"input": "in", "output": "out"}])
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError


def test_build_few_shot_prompt_maxtext_missing() -> object:  # type: ignore[return]
    """Initialize function test_build_few_shot_prompt_maxtext_missing."""
    with mock.patch.dict(sys.modules, {"maxtext.models.gemma4": None, "maxtext.models": None, "maxtext": None}):
        importlib = __import__("importlib")
        importlib.reload(fs)
        res = fs.build_few_shot_prompt("foo", "prompt", [])
        if not res["status"] == "mocked_missing_maxtext":
            raise AssertionError


import pytest


def test_few_shot_success(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.maxtext.few_shot as m_few

    monkeypatch.setattr(m_few, "gemma4", object())
    examples = [{"input": "i", "output": "o"}]
    res = m_few.build_few_shot_prompt("model", "prompt", examples)
    assert res["status"] == "success_maxtext_few_shot"
    assert "i\nOutput: o" in res["few_shot_prompt"]
