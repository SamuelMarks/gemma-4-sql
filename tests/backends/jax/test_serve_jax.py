import sys
from unittest import mock

import gemma_4_sql.backends.jax.serve as srv


def test_serve_model_jax():
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    assert res["backend"] == "jax"
    assert res["model"] == "foo"
    assert res["port"] == 8000
    assert res["max_batch_size"] == 16
    assert res["mode"] == "continuous_batching"

def test_serve_model_jax_missing():
    with mock.patch.dict(sys.modules, {"jax": None}):
        import importlib
        importlib.reload(srv)
        res = srv.serve_model("foo")
        assert res["status"] == "mocked_missing_jax"
