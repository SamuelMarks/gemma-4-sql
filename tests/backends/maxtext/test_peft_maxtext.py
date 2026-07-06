"""Tests for MaxText PEFT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gemma_4_sql.backends.maxtext.peft as pt

if TYPE_CHECKING:
    import pytest


class MockJnp:
    """Provide class docstring."""

    int32 = 1

    @staticmethod
    def zeros(_shape: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [0]


class MockJaxRandom:
    """Provide class docstring."""

    @staticmethod
    def mock_prngkey(seed: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return seed

    PRNGKey = mock_prngkey


class MockJax:
    """Provide class docstring."""

    random = MockJaxRandom()


class MockGemma4Model:
    """Provide class docstring."""

    def __init__(self, name: object) -> None:
        """Execute function."""

    def init(self, _rng: object, _inputs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "params"


def test_apply_lora_maxtext_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    import pytest

    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(pt, "jax", None)
    with pytest.raises(DependencyMissingError):
        pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)


def test_apply_lora_maxtext_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt, "jax", MockJax())
    monkeypatch.setattr(pt, "jnp", MockJnp())
    monkeypatch.setattr(pt, "Gemma4Model", MockGemma4Model)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "completed":
        raise AssertionError


def test_apply_lora_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt, "jax", MockJax())
    monkeypatch.setattr(pt, "jnp", MockJnp())
    monkeypatch.setattr(pt, "Gemma4Model", MockGemma4Model)

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockJnp, "zeros", raise_err)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if "failed" not in str(res["status"]):
        raise AssertionError


def test_peft_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    m_peft = __import__("gemma_4_sql.backends.maxtext.peft", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_peft)
    monkeypatch.undo()
    importlib.reload(m_peft)
