import pytest

from gemma_4_sql.sdk.serve import serve_model


def test_serve_model_routing():
    for backend in ["jax", "keras", "maxtext", "pytorch"]:
        res = serve_model("foo", backend=backend)
        assert res["backend"] == backend
        assert res["model"] == "foo"

def test_serve_model_routing_error():
    with pytest.raises(ValueError):
        serve_model("foo", backend="unknown")
