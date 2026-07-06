"""Tests for PyTorch Benchmark."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.pytorch.benchmark as pt_bm
from gemma_4_sql.backends.pytorch.benchmark import benchmark_model


class MockTorch:
    """Provide class docstring."""

    long = "long"

    class MockCuda:
        """Provide class docstring."""

        @staticmethod
        def is_available() -> bool:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return False

    cuda = MockCuda

    def zeros(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [0]


class MockAutoModelForCausalLM:
    """Mock Model."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Mock method.

        Returns:
            object: Description of return.

        """
        return cls()

    """Provide class docstring."""


def test_benchmark_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(pt_bm, "torch", None)
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", None)
    with pytest.raises(DependencyMissingError, match="PyTorch dependencies are missing."):
        benchmark_model("model", "gpu", 1)


def test_benchmark_pytorch_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt_bm, "torch", MockTorch())
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", MockAutoModelForCausalLM)
    res = benchmark_model("model", "gpu", 1, test_mode=True, num_runs=2)
    if not res["status"] == "success":
        raise AssertionError
    if not res["tokens_per_sec"] > 0:
        raise AssertionError


def test_benchmark_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt_bm, "torch", MockTorch())
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", MockAutoModelForCausalLM)

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockTorch, "zeros", raise_err)
    res = benchmark_model("model", "gpu", 1, test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError


class MockModel:
    """Provide class docstring."""

    def to(self, device: object) -> None:
        """Execute function."""

    def eval(self) -> None:
        """Execute function."""

    def __call__(self, x: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return x


def test_benchmark_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_benchmark = __import__("gemma_4_sql.backends.pytorch.benchmark", fromlist=[""])
    monkeypatch.setattr(m_benchmark, "torch", MockTorch)
    monkeypatch.setattr(m_benchmark, "AutoModelForCausalLM", object())
    res = m_benchmark.benchmark_model("m", "cuda", 1, test_mode=True)
    if res["status"] != "success":
        raise AssertionError
    "Execute function."
    pt_bm = __import__("gemma_4_sql.backends.pytorch.benchmark", fromlist=[""])
    monkeypatch.setattr(pt_bm, "torch", MockTorch())
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", MockAutoModelForCausalLM())
    res = pt_bm.benchmark_model("model", "gpu", 1, test_mode=False, num_runs=2)
    if res["status"] != "success":
        raise AssertionError
    res = pt_bm.benchmark_model("model", "cpu", 1, test_mode=False, num_runs=2)
    if res["status"] != "success":
        raise AssertionError
