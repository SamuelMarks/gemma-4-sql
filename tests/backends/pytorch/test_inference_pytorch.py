"""Tests for PyTorch inference."""

from __future__ import annotations

import typing

import gemma_4_sql.backends.pytorch.inference as pt_inf
from gemma_4_sql.backends.pytorch.inference import generate_sql

if typing.TYPE_CHECKING:
    import pytest


class MockTorch:
    pass


class MockAutoModelForCausalLM:
    pass


class MockAutoTokenizer:
    pass


def test_inference_pytorch_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt_inf, "torch", MockTorch())
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockAutoTokenizer)

    res = generate_sql("mock", "hi", beam_width=1, max_length=2, test_mode=True)
    if not res["status"] == "success":
        raise AssertionError
    if not res["model"] == "mock":
        raise AssertionError


def test_inference_pytorch_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt_inf, "torch", None)
    res = generate_sql("mock", "hi")
    if not res["status"] == "mocked_missing_torch":
        raise AssertionError


def test_inference_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt_inf, "torch", MockTorch())
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockAutoTokenizer)

    def raise_err(*args: object, **kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(pt_inf, "SQLTokenizer", raise_err)

    res = generate_sql("mock", "hi", test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError
