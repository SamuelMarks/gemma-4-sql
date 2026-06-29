"""Module docstring."""

import pytest
from gemma_4_sql.sdk.few_shot import build_few_shot_prompt


def test_few_shot_routing() -> object:  # type: ignore[return]
    """Initialize function test_few_shot_routing."""
    for backend in ["jax", "keras", "maxtext", "pytorch"]:
        res = build_few_shot_prompt("foo", "prompt", [], backend=backend)
        if not res["backend"] == backend:
            raise AssertionError
        if not res["model"] == "foo":
            raise AssertionError


def test_few_shot_routing_error() -> object:  # type: ignore[return]
    """Initialize function test_few_shot_routing_error."""
    with pytest.raises(ValueError, match=r".*"):
        build_few_shot_prompt("foo", "prompt", [], backend="unknown")
