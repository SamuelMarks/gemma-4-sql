"""Module docstring."""

import sys
from unittest import mock

import gemma_4_sql.backends.pytorch.few_shot as fs


def test_build_few_shot_prompt_pytorch() -> object:  # type: ignore[return]
    """Initialize function test_build_few_shot_prompt_pytorch."""
    res = fs.build_few_shot_prompt("foo", "prompt", [{"input": "in", "output": "out"}])
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError
    if "Input: in" not in res["few_shot_prompt"]:  # type: ignore[operator]
        raise AssertionError


def test_build_few_shot_prompt_pytorch_missing() -> object:  # type: ignore[return]
    """Initialize function test_build_few_shot_prompt_pytorch_missing."""
    with mock.patch.dict(sys.modules, {"torch": None}):
        importlib = __import__("importlib")
        importlib.reload(fs)
        res = fs.build_few_shot_prompt("foo", "prompt", [])
        if not res["status"] == "mocked_missing_pytorch":
            raise AssertionError
