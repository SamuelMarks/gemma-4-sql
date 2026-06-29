"""Module docstring."""

import sys
from unittest import mock

import gemma_4_sql.backends.jax.few_shot as fs


def test_build_few_shot_prompt_jax() -> object:  # type: ignore[return]
    """Initialize function test_build_few_shot_prompt_jax."""
    res = fs.build_few_shot_prompt("foo", "prompt", [{"input": "in", "output": "out"}])
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError
    if "Input: in" not in res["few_shot_prompt"]:  # type: ignore[operator]
        raise AssertionError


def test_build_few_shot_prompt_jax_missing() -> object:  # type: ignore[return]
    """Initialize function test_build_few_shot_prompt_jax_missing."""
    with mock.patch.dict(sys.modules, {"jax": None}):
        importlib = __import__("importlib")
        importlib.reload(fs)
        res = fs.build_few_shot_prompt("foo", "prompt", [])
        if not res["status"] == "mocked_missing_jax":
            raise AssertionError
