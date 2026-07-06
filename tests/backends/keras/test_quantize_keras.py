"""Tests for Keras quantization logic."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.keras.quantize as kr_quantize
from gemma_4_sql.backends.keras.quantize import quantize_model
from gemma_4_sql.exceptions import DependencyMissingError


def test_quantize_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras quantize when missing.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(kr_quantize, "keras", None)
    with pytest.raises(DependencyMissingError):
        quantize_model("model", "int8")


def test_quantize_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras quantize.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(kr_quantize, "keras", object())
    res = quantize_model("model", "int8")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["status"] == "quantized_int8":
        raise AssertionError
    res = quantize_model("model", "int4")
    if not res["status"] == "quantized_int4":
        raise AssertionError
    res = quantize_model("model", "awq")
    if not res["status"] == "quantized_awq":
        raise AssertionError
    res = quantize_model("model", "unknown")
    if "unsupported" not in str(res["status"]):
        raise AssertionError


def test_quantize_keras_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    mdl = __import__("gemma_4_sql.backends.keras.quantize", fromlist=[""])
    monkeypatch.setitem(sys.modules, "keras", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)


def test_quantize_keras_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(kr_quantize, "keras", type("MockKeras", (), {}))
    res = kr_quantize.quantize_model("model", "awq")
    if res["status"] != "quantized_awq":
        raise AssertionError


def test_quantize_keras_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(kr_quantize, "keras", type("MockKeras", (), {}))

    def raise_err(*_args: object, **_kwargs: object) -> None:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(kr_quantize.logger, "warning", raise_err)
    res = kr_quantize.quantize_model("model", "unsupported")
    if "failed" not in res["status"]:
        raise AssertionError
