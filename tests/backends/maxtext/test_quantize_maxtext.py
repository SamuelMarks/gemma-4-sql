"""Tests for MaxText quantization logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gemma_4_sql.backends.maxtext.quantize as maxtext_quantize
from gemma_4_sql.backends.maxtext.quantize import quantize_model

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


def test_quantize_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MaxText quantize when missing.

    Raises:
        AssertionError: Description.

    """
    import pytest

    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(maxtext_quantize, "jnp", None)
    with pytest.raises(DependencyMissingError):
        quantize_model("model", "int8")


def test_quantize_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MaxText quantize.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(maxtext_quantize, "jax", MockJax())
    monkeypatch.setattr(maxtext_quantize, "jnp", MockJnp())
    monkeypatch.setattr(maxtext_quantize, "Gemma4Model", MockGemma4Model)
    res = quantize_model("model", "int8")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["status"] == "quantized_int8":
        raise AssertionError
    res = quantize_model("model", "int4")
    if not res["status"] == "quantized_int4":
        raise AssertionError
    res = quantize_model("model", "awq")
    if not res["method"] == "awq":
        raise AssertionError
    if res["status"] not in {"quantized_awq", "mocked_missing_maxtext"}:
        raise AssertionError


def test_quantize_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(maxtext_quantize, "jax", MockJax())
    monkeypatch.setattr(maxtext_quantize, "jnp", MockJnp())
    monkeypatch.setattr(maxtext_quantize, "Gemma4Model", MockGemma4Model)

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockJnp, "zeros", raise_err)
    res = quantize_model("model", "int8")
    if "failed" not in str(res["status"]):
        raise AssertionError


def test_quantize_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    m_quantize = __import__("gemma_4_sql.backends.maxtext.quantize", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_quantize)
    monkeypatch.undo()
    importlib.reload(m_quantize)
