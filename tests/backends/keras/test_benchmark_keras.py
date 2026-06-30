"""Tests for Keras Benchmark."""

import pytest

import gemma_4_sql.backends.keras.benchmark as bm


class MockTfTensor:
    def numpy(self) -> float:
        return 0.0


class MockTf:
    int32 = "int32"

    def zeros(self, *args: object, **kwargs: object) -> object:
        return MockTfTensor()

    def function(self, fn: object) -> object:
        return fn

    class config:
        class experimental:
            @staticmethod
            def get_memory_info(device: str) -> dict:
                return {"current": 1024 * 1024 * 100}


def test_benchmark_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "keras", None)
    res = bm.benchmark_model("model", "gpu", 1)
    if not res["status"] == "mocked_missing_keras":
        raise AssertionError


def test_benchmark_keras_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "tf", MockTf())
    monkeypatch.setattr(bm, "keras", object())

    res = bm.benchmark_model("model", "gpu", 1, test_mode=True, num_runs=2)
    if not res["status"] == "success":
        raise AssertionError
    if not res["tokens_per_sec"] > 0:
        raise AssertionError


def test_benchmark_keras_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "tf", MockTf())
    monkeypatch.setattr(bm, "keras", object())

    def raise_err(*args: object, **kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockTf, "zeros", raise_err)

    res = bm.benchmark_model("model", "gpu", 1, test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError
