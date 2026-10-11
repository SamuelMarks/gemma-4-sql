"""Tests for Models SDK module."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.models import pretrain_model, train_from_scratch
from gemma_4_sql.type_hints import TrainingConfig


def test_pretrain_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test pretraining a model."""
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    jax_agent = get_backend("jax")
    monkeypatch.setattr(jax_agent, "train_model", lambda cfg, **k: {"backend": cfg.backend, "action": cfg.action, "model": cfg.model_name, "dataset": cfg.dataset, "epochs": cfg.epochs})
    res = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="jax"))
    assert res["backend"] == "jax"
    assert res["action"] == "pretrain"
    assert res["model"] == "my-model"

    # Instead of just pt_train.torch = None, mock train_model to raise the error
    # since other tests might have patched the module
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.train_model", lambda *args, **kwargs: (_ for _ in ()).throw(DependencyMissingError("PyTorch dependencies are missing.")))
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="pytorch"))

    import gemma_4_sql.backends.keras as kb
    import gemma_4_sql.backends.keras.train as ktrain

    monkeypatch.setattr(ktrain, "tf", None)
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="keras"))

    def mock_train(*args, **kwargs):
        """Docstring for mock_train."""
        return {"backend": "keras", "action": "pretrain", "model": "my-model"}

    monkeypatch.setattr(ktrain, "keras", object())
    monkeypatch.setattr(ktrain, "tf", object())
    monkeypatch.setattr(ktrain, "train_model", mock_train)
    monkeypatch.setattr(kb, "train_model", mock_train)

    res_k = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="keras"))
    assert res_k["backend"] == "keras"
    assert res_k["action"] == "pretrain"

    import gemma_4_sql.backends.maxtext.train as mx_train

    monkeypatch.setattr(mx_train, "jax", None)
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="maxtext"))

    # cover mlx pretraining by mocking registry and mlx modules
    import sys
    from unittest.mock import Mock

    monkeypatch.setitem(sys.modules, "mlx.core", Mock())
    monkeypatch.setitem(sys.modules, "mlx.nn", Mock())
    monkeypatch.setitem(sys.modules, "mlx.optimizers", Mock())
    monkeypatch.setitem(sys.modules, "mlx_lm", Mock())

    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    mlx_agent = get_backend("mlx")
    monkeypatch.setattr(mlx_agent, "train_model", lambda cfg, **k: {"backend": cfg.backend, "action": cfg.action, "model": cfg.model_name, "dataset": cfg.dataset, "epochs": cfg.epochs})

    res_mlx = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="mlx"))
    assert res_mlx["backend"] == "mlx"


def test_pretrain_model_error() -> None:
    """Test pretraining a model with an unknown backend."""
    with pytest.raises(ValueError, match=r".*"):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="unknown"))


def test_chat_no_sql(monkeypatch):
    """Test chat no sql functionality."""
    from gemma_4_sql.sdk import chat

    monkeypatch.setattr("gemma_4_sql.sdk.registry.get_backend", lambda b: type("Backend", (), {"generate_sql": lambda *a, **k: {}})())

    with pytest.raises(RuntimeError) as e:
        chat.chat_turn("m", [], "p")
    assert "did not return SQL" in str(e.value)


def test_etl_defaults():
    # We mock _route_backend to just return config

    """Test etl defaults functionality."""
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
    """Test evaluation max batches functionality."""
    import gemma_4_sql.sdk.evaluation as ev

    class MockBackend:
        """Test class for MockBackend."""

        def build_dataloader(self, c):
            """Execute build dataloader helper."""
            return {"loader": [{"inputs": [1], "targets": [2]}] * (ev.MAX_BATCHES + 2)}

        def generate_sql(self, *a, **k):
            """Execute generate sql helper."""
            return {"sql": "A", "confidence_score": 0.5}

    # Also need to mock _process_batch_inputs if it is used
    monkeypatch.setattr(ev, "_process_batch_inputs", lambda b: ([1], [2]))
    monkeypatch.setattr(ev, "SQLTokenizer", type("Tok", (), {"__init__": lambda self, **kw: None, "decode": lambda self, x: "T"}))

    preds, _truths, _scores = ev._run_evaluation_inference("m", "d", MockBackend())
    assert len(preds) == ev.MAX_BATCHES


def test_models_defaults(monkeypatch):
    """Test models defaults functionality."""
    import gemma_4_sql.sdk.models as mod

    def mock_route(c):
        """Execute mock route helper."""
        return c

    monkeypatch.setattr(mod, "_route_training", mock_route)

    class FakeTC:
        """Test class for FakeTC."""

        def __init__(self, **kwargs):
            """Initialize __init__."""
            self.backend = kwargs.get("backend", "jax")
            self.action = None

    monkeypatch.setattr(mod, "TrainingConfig", FakeTC)

    c = mod.sft_model()
    assert c.action == "sft"

    c2 = mod.posttrain_model()
    assert c2.action == "posttrain"
    assert c2.backend == "keras"


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
    assert res == {"status": "mock"}


def test_train_from_scratch_unknown() -> object:
    """Initialize function test_train_from_scratch_unknown."""
    with pytest.raises(ValueError, match=r".*"):
        train_from_scratch(TrainingConfig(action="pretrain", model_name="mock", dataset="mock", backend="unknown"))


def test_training_config_immutability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that pretrain_model, sft_model, train_from_scratch, posttrain_model do not mutate input config."""
    from gemma_4_sql.sdk.models import posttrain_model, sft_model

    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    pt_train = get_backend("pytorch")
    monkeypatch.setattr(pt_train, "train_model", lambda _config, **_kw: {"status": "ok"})

    orig_cfg = TrainingConfig(action="custom_action", model_name="mock", dataset="mock", backend="pytorch")
    pretrain_model(orig_cfg)
    assert orig_cfg.action == "custom_action"

    sft_model(orig_cfg)
    assert orig_cfg.action == "custom_action"

    train_from_scratch(orig_cfg)
    assert orig_cfg.action == "custom_action"

    posttrain_model(orig_cfg)
    assert orig_cfg.action == "custom_action"
