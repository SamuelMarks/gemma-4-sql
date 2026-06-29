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
