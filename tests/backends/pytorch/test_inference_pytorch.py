"""Tests for PyTorch inference."""

from __future__ import annotations

import typing

import gemma_4_sql.backends.pytorch.inference as pt_inf
from gemma_4_sql.backends.pytorch.inference import generate_sql

if typing.TYPE_CHECKING:
    import pytest


class MockTorch:
    """Provide class docstring."""


class MockAutoModelForCausalLM:
    """Provide class docstring."""


class MockAutoTokenizer:
    """Provide class docstring."""


def test_inference_pytorch_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(pt_inf, "torch", MockTorch())
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockAutoTokenizer)
    res = generate_sql("mock", "hi", beam_width=1, max_length=2, test_mode=True)
    if not res["status"] == "success":
        raise AssertionError
    if not res["model"] == "mock":
        raise AssertionError


def test_inference_pytorch_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(pt_inf, "torch", None)
    res = generate_sql("mock", "hi")
    if not res["status"] == "mocked_missing_torch":
        raise AssertionError


def test_inference_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(pt_inf, "torch", MockTorch())
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockAutoTokenizer)

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function."""
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(pt_inf, "_run_generation", raise_err)
    res = generate_sql("mock", "hi", test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError


class MockTokenizerObj:
    """Provide class docstring."""

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return type("M", (), {"to": lambda *_a, **_k: {"input_ids": [1]}})()

    def decode(self, *_args: object, **_kwargs: object) -> str:
        """Execute function."""
        return "prompt SELECT * FROM x"


class MockTokenizer:
    """Provide class docstring."""

    @classmethod
    def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return MockTokenizerObj()


class MockModelObj:
    """Provide class docstring."""

    device = "cpu"

    def generate(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""

        class MockOutputs:
            """Provide class docstring."""

            sequences: typing.ClassVar = [[1, 2]]
            sequences_scores: typing.ClassVar = [type("T", (), {"item": lambda _self: 0.99})()]

        return MockOutputs()


class MockModel:
    """Provide class docstring."""

    @classmethod
    def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return MockModelObj()


def test_inference_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    m_inf = __import__("gemma_4_sql.backends.pytorch.inference")
    monkeypatch.setattr(m_inf, "torch", object())
    monkeypatch.setattr(m_inf, "AutoTokenizer", MockTokenizer)
    monkeypatch.setattr(m_inf, "AutoModelForCausalLM", MockModel)
    res = m_inf.generate_sql("m", "prompt", test_mode=False)
    if res["status"] != "success":
        raise AssertionError
    if res["sql"] != "SELECT * FROM x":
        raise AssertionError
