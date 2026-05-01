import pytest

from gemma_4_sql.sdk.few_shot import build_few_shot_prompt


def test_few_shot_routing():
    for backend in ["jax", "keras", "maxtext", "pytorch"]:
        res = build_few_shot_prompt("foo", "prompt", [], backend=backend)
        assert res["backend"] == backend
        assert res["model"] == "foo"


def test_few_shot_routing_error():
    with pytest.raises(ValueError):
        build_few_shot_prompt("foo", "prompt", [], backend="unknown")
