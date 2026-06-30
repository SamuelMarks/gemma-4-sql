"""Tests for PyTorch Benchmark."""

from __future__ import annotations

import typing

import gemma_4_sql.backends.pytorch.benchmark as pt_bm
from gemma_4_sql.backends.pytorch.benchmark import benchmark_model

if typing.TYPE_CHECKING:
    import pytest


class MockTorch:
    long = "long"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    def zeros(self, *args: object, **kwargs: object) -> object:
        return [0]


class MockAutoModelForCausalLM:
    pass


def test_benchmark_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt_bm, "torch", None)
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", None)

    res = benchmark_model("model", "gpu", 1)
    if not res["status"] == "mocked_missing_torch":
        raise AssertionError


def test_benchmark_pytorch_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt_bm, "torch", MockTorch())
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", MockAutoModelForCausalLM)

    res = benchmark_model("model", "gpu", 1, test_mode=True, num_runs=2)
    if not res["status"] == "success":
        raise AssertionError
    if not res["tokens_per_sec"] > 0:
        raise AssertionError


def test_benchmark_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt_bm, "torch", MockTorch())
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", MockAutoModelForCausalLM)

    def raise_err(*args: object, **kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockTorch, "zeros", raise_err)

    res = benchmark_model("model", "gpu", 1, test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError
