import sys
from unittest.mock import MagicMock


def test_pytorch_serve_vllm_import(monkeypatch):
    vllm_mock = MagicMock()
    vllm_mock.AsyncEngineArgs = MagicMock()
    vllm_mock.AsyncLLMEngine = MagicMock()
    vllm_utils_mock = MagicMock()
    vllm_utils_mock.random_uuid = MagicMock()
    vllm_mock.utils = vllm_utils_mock

    monkeypatch.setitem(sys.modules, "vllm", vllm_mock)

    import importlib

    import gemma_4_sql.backends.pytorch.serve as pserve

    importlib.reload(pserve)
    assert pserve.AsyncLLMEngine is not None
