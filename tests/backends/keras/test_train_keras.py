"""Tests for Keras training pipeline."""

import pytest

import gemma_4_sql.backends.keras.train as tr
from gemma_4_sql.backends.keras.train import train_model


class MockTfTensor:
    def __init__(self, shape):
        self.shape = shape


class MockTf:
    int32 = 1

    @staticmethod
    def zeros(shape, dtype=None):
        return MockTfTensor(shape)


class MockHistory:
    def __init__(self):
        self.history = {"loss": [0.5, 0.1]}


class MockModel:
    def compile(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        return MockHistory()


class MockLayers:
    @staticmethod
    def Embedding(*args, **kwargs):
        return lambda x: x

    @staticmethod
    def Dense(*args, **kwargs):
        return lambda x: x


class MockOptimizers:
    @staticmethod
    def AdamW(*args, **kwargs):
        return "adamw"


class MockLosses:
    @staticmethod
    def SparseCategoricalCrossentropy(*args, **kwargs):
        return "loss"


class MockKeras:
    Input = staticmethod(lambda *args, **kwargs: "input")
    Model = staticmethod(lambda *args, **kwargs: MockModel())
    layers = MockLayers()
    optimizers = MockOptimizers()
    losses = MockLosses()


@pytest.fixture
def mock_keras_env(monkeypatch):
    monkeypatch.setattr(tr, "keras", MockKeras())
    monkeypatch.setattr(tr, "tf", MockTf())

    def mock_build_dataloader(*args, **kwargs):
        return {
            "loader": [{"inputs": MockTfTensor((1,)), "targets": MockTfTensor((1,))}]
        }

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)


def test_train_model_keras_real(mock_keras_env):
    res = train_model("sft", "mod", "dat", 2, 0.1)
    assert res["status"] == "completed"
    assert res["final_loss"] == 0.1
    assert res["backend"] == "keras"


def test_train_model_keras_missing():
    orig_keras = tr.keras
    orig_tf = tr.tf
    tr.keras = None
    tr.tf = None
    res = train_model("sft", "mod", "dat", 2, 0.1)
    assert res["status"] == "mocked_missing_keras"
    # also test model fallback
    model = tr.KerasSQLModel()
    assert model(None) is None
    tr.keras = orig_keras
    tr.tf = orig_tf


def test_train_model_keras_error(mock_keras_env, monkeypatch):
    def raise_error(*args, **kwargs):
        raise ValueError("err")

    monkeypatch.setattr(tr, "build_dataloader", raise_error)
    res = train_model("sft", "mod", "dat", 2, 0.1)
    assert "failed: err" in res["status"]


def test_train_model_keras_no_loader_fallback(mock_keras_env, monkeypatch):
    def mock_build_dataloader(*args, **kwargs):
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    res = train_model("sft", "mod", "dat", 2, 0.1)
    assert res["status"] == "completed"
    assert res["final_loss"] == 0.1
