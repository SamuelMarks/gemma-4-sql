"""Provide module docstring."""

from gemma_4_sql.sdk.few_shot import build_few_shot_prompt


def test_few_shot_routing() -> None:
    """Initialize function test_few_shot_routing."""
    for backend in ["jax", "keras", "maxtext", "pytorch", "unknown"]:
        res = build_few_shot_prompt("foo", "prompt", [{"input": "a", "output": "b"}], backend=backend)
        if res["backend"] != backend:
            raise AssertionError
        if res["model"] != "foo":
            raise AssertionError
        if res["status"] != f"success_{backend}_few_shot":
            raise AssertionError
