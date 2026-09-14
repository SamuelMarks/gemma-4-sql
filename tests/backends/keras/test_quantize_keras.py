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


def test_quantize_keras_with_policies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras quantize setting dtype_policies and config policies."""
    called_policies: list[str] = []

    class MockPolicies:
        """Mock dtype policies."""

        def set_dtype_policy(self, policy: str) -> None:
            """Set policy."""
            called_policies.append(policy)

    class MockConfig:
        """Mock keras config."""

        def set_dtype_policy(self, policy: str) -> None:
            """Set policy."""
            called_policies.append(policy)

    mock_keras_1 = type("MockKeras1", (), {"dtype_policies": MockPolicies()})
    monkeypatch.setattr(kr_quantize, "keras", mock_keras_1)
    res1 = kr_quantize.quantize_model("model", "int8")
    assert res1["status"] == "quantized_int8"
    assert "int8_from_float32" in called_policies

    mock_keras_2 = type("MockKeras2", (), {"config": MockConfig()})
    monkeypatch.setattr(kr_quantize, "keras", mock_keras_2)
    res2 = kr_quantize.quantize_model("model", "int4")
    assert res2["status"] == "quantized_int4"
    assert "int4_from_float32" in called_policies

    class MockFailingPolicies:
        """Mock policies raising ValueError."""

        def set_dtype_policy(self, _policy: str) -> None:
            """Raise ValueError."""
            raise ValueError("policy error")

    mock_keras_fail = type("MockKerasFail", (), {"dtype_policies": MockFailingPolicies()})
    monkeypatch.setattr(kr_quantize, "keras", mock_keras_fail)
    res_fail = kr_quantize.quantize_model("model", "int8")
    assert res_fail["status"] == "quantized_int8"

    mock_keras_fail_cfg = type("MockKerasFailCfg", (), {"config": MockFailingPolicies()})
    monkeypatch.setattr(kr_quantize, "keras", mock_keras_fail_cfg)
    res_fail_cfg = kr_quantize.quantize_model("model", "int4")
    assert res_fail_cfg["status"] == "quantized_int4"
