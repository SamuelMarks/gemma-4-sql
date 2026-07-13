"""Tests for Models SDK module."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.models import pretrain_model
from gemma_4_sql.type_hints import TrainingConfig


def test_pretrain_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test pretraining a model."""
    res = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="jax"))
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["action"] == "pretrain":
        raise AssertionError
    if not res["model"] == "my-model":
        raise AssertionError

    # Instead of just pt_train.torch = None, mock train_model to raise the error
    # since other tests might have patched the module
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.train_model", lambda *args, **kwargs: (_ for _ in ()).throw(DependencyMissingError("PyTorch dependencies are missing.")))
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="pytorch"))

    res = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="keras"))
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["action"] == "pretrain":
        raise AssertionError

    import gemma_4_sql.backends.maxtext.train as mx_train

    monkeypatch.setattr(mx_train, "jax", None)
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="maxtext"))

    with pytest.raises(ValueError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="mlx"))


def test_pretrain_model_error() -> None:
    """Test pretraining a model with an unknown backend."""
    with pytest.raises(ValueError, match=r".*"):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="unknown"))


import pytest


def test_chat_no_sql(monkeypatch):
    import builtins

    from gemma_4_sql.sdk import chat

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "gemma_4_sql.sdk.registry":

            class MockReg:
                @staticmethod
                def get_backend(b):
                    return type("Backend", (), {"generate_sql": lambda *a, **k: {}})()

            return MockReg
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(RuntimeError) as e:
        chat.chat_turn("m", [], "p")
    assert "did not return SQL" in str(e.value)


def test_etl_defaults():
    # We mock _route_backend to just return config

    from gemma_4_sql.sdk import etl

    orig_route = etl._route_backend
    etl._route_backend = lambda c, b: (c, b)

    try:
        c, _b = etl.etl_pretrain()
        assert c.dataset_name == "my-custom-dataset"

        c, _b = etl.etl_sft()
        assert c.dataset_name == etl.DEFAULT_SFT_DATASET

        c, _b = etl.etl_posttrain()
        assert c.dataset_name == etl.DEFAULT_POSTTRAIN_DATASET
    finally:
        etl._route_backend = orig_route


def test_evaluation_max_batches(monkeypatch):
    import gemma_4_sql.sdk.evaluation as ev

    class MockBackend:
        def build_dataloader(self, c):
            return {"loader": [{"inputs": [1], "targets": [2]}] * (ev.MAX_BATCHES + 2)}

        def generate_sql(self, *a, **k):
            return {"sql": "A", "confidence_score": 0.5}

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "gemma_4_sql.sdk.registry":

            class MockReg:
                @staticmethod
                def get_backend(b):
                    return MockBackend()

            return MockReg
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Also need to mock _process_batch_inputs if it is used
    monkeypatch.setattr(ev, "_process_batch_inputs", lambda b: ([1], [2]))
    monkeypatch.setattr(ev, "SQLTokenizer", type("Tok", (), {"__init__": lambda self, **kw: None, "decode": lambda self, x: "T"}))

    preds, _truths, _scores = ev._run_evaluation_inference("m", "d", MockBackend())
    assert len(preds) == ev.MAX_BATCHES


def test_models_defaults(monkeypatch):
    import gemma_4_sql.sdk.models as mod

    def mock_route(c):
        return c

    monkeypatch.setattr(mod, "_route_training", mock_route)

    class FakeTC:
        def __init__(self, **kwargs):
            self.backend = kwargs.get("backend", "jax")
            self.action = None

    monkeypatch.setattr(mod, "TrainingConfig", FakeTC)

    c = mod.sft_model()
    assert c.action == "sft"

    c2 = mod.posttrain_model()
    assert c2.action == "posttrain"
    assert c2.backend == "keras"


import pytest

from gemma_4_sql.sdk.models import train_from_scratch


def test_train_from_scratch_pytorch(monkeypatch: object) -> object:
    """Initialize function test_train_from_scratch_pytorch.

    Args:
    ----
    monkeypatch: Description of monkeypatch.


    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    pt_train = get_backend("pytorch")
    monkeypatch.setattr(pt_train, "train_model", lambda _config, **_kw: {"status": "mock"})
    res = train_from_scratch(TrainingConfig(action="pretrain", model_name="mock", dataset="mock", backend="pytorch"))
    if not res == {"status": "mock"}:
        raise AssertionError


def test_train_from_scratch_unknown() -> object:
    """Initialize function test_train_from_scratch_unknown."""
    with pytest.raises(ValueError, match=r".*"):
        train_from_scratch(TrainingConfig(action="pretrain", model_name="mock", dataset="mock", backend="unknown"))
