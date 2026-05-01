import sys
from unittest import mock

import gemma_4_sql.backends.pytorch.serve as srv


def test_serve_model_pytorch():
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    assert res["backend"] == "pytorch"
    assert res["model"] == "foo"
    assert res["port"] == 8000
    assert res["max_batch_size"] == 16
    assert res["mode"] == "continuous_batching"


def test_serve_model_pytorch_missing():
    with mock.patch.dict(sys.modules, {"vllm": None}):
        import importlib

        importlib.reload(srv)
        res = srv.serve_model("foo")
        assert res["status"] == "mocked_missing_pytorch"
