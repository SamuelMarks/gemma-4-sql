"""Tests for MaxText PEFT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gemma_4_sql.backends.maxtext.peft as pt

if TYPE_CHECKING:
    import pytest


class MockJnp:
    int32 = 1

    @staticmethod
    def zeros(shape: object, **kwargs: object) -> object:
        return [0]


class MockJaxRandom:
    @staticmethod
    def PRNGKey(seed: object) -> object:
        return seed


class MockJax:
    random = MockJaxRandom()


class MockGemma4Model:
    def __init__(self, name: object) -> None:
        pass

    def init(self, rng: object, inputs: object) -> object:
        return "params"


def test_apply_lora_maxtext_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "jax", None)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "mocked_missing_jax":
        raise AssertionError
    if not res["backend"] == "maxtext":
        raise AssertionError


def test_apply_lora_maxtext_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "jax", MockJax())
    monkeypatch.setattr(pt, "jnp", MockJnp())
    monkeypatch.setattr(pt, "Gemma4Model", MockGemma4Model)

    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "completed":
        raise AssertionError


def test_apply_lora_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "jax", MockJax())
    monkeypatch.setattr(pt, "jnp", MockJnp())
    monkeypatch.setattr(pt, "Gemma4Model", MockGemma4Model)

    def raise_err(*args: object, **kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockJnp, "zeros", raise_err)

    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if "failed" not in str(res["status"]):
        raise AssertionError
