"""Module docstring."""

import sys
from unittest import mock

import gemma_4_sql.backends.keras.few_shot as fs


def test_build_few_shot_prompt_keras() -> object:  # type: ignore[return]
    """Initialize function test_build_few_shot_prompt_keras."""
    res = fs.build_few_shot_prompt("foo", "prompt", [{"input": "in", "output": "out"}])
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError
    if "Input: in" not in res["few_shot_prompt"]:  # type: ignore[operator]
        raise AssertionError


def test_build_few_shot_prompt_keras_missing() -> object:  # type: ignore[return]
    """Initialize function test_build_few_shot_prompt_keras_missing."""
    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        importlib = __import__("importlib")
        importlib.reload(fs)
        res = fs.build_few_shot_prompt("foo", "prompt", [])
        if not res["status"] == "mocked_missing_keras":
            raise AssertionError
