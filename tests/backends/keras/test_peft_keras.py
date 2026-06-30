"""Tests for Keras PEFT."""

import pytest

import gemma_4_sql.backends.keras.peft as pt


class MockLayers:
    @staticmethod
    def Embedding(*args: object, **kwargs: object) -> object:
        return lambda x: "x"

    @staticmethod
    def Dense(*args: object, **kwargs: object) -> object:
        return lambda x: "x"


class MockModel:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.backbone = self

    def enable_lora(self, rank: int) -> None:
        self.lora_enabled = True


class MockKeras:
    @staticmethod
    def Input(*args: object, **kwargs: object) -> object:
        return "inputs"

    Model = MockModel
    layers = MockLayers()


def test_apply_lora_keras_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "keras", None)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "mocked_missing_keras":
        raise AssertionError
    if not res["backend"] == "keras":
        raise AssertionError


def test_apply_lora_keras_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "keras", MockKeras())
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "completed":
        raise AssertionError


def test_apply_lora_keras_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "keras", MockKeras())

    def raise_err(*args: object, **kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockKeras, "Input", raise_err)

    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if "failed" not in str(res["status"]):
        raise AssertionError
