"""
Tests for SDK Export module.
"""

import pytest

from gemma_4_sql.sdk.export import export_model


def test_export_jax() -> None:
    """Test export with jax."""
    res = export_model("model1", "path1", "jax")
    assert res["backend"] == "jax"
    assert res["model"] == "model1"
    assert res["export_path"] == "path1"
    assert res["status"] in ["exported", "mock_exported"]
    assert res["format"] == "orbax/saved_model"


def test_export_keras() -> None:
    """Test export with keras."""
    res = export_model("model1", "path1", "keras")
    assert res["backend"] == "keras"
    assert res["model"] == "model1"
    assert res["export_path"] == "path1"
    assert res["status"] in ["exported", "mock_exported"]
    assert res["format"] == "keras_v3/keras_tensor"


def test_export_maxtext() -> None:
    """Test export with maxtext."""
    res = export_model("model1", "path1", "maxtext")
    assert res["backend"] == "maxtext"
    assert res["model"] == "model1"
    assert res["export_path"] == "path1"
    assert res["status"] in ["exported", "mock_exported"]
    assert res["format"] == "maxtext/checkpoint"


def test_export_pytorch() -> None:
    """Test export with pytorch."""
    res = export_model("model1", "path1", "pytorch")
    assert res["backend"] == "pytorch"
    assert res["model"] == "model1"
    assert res["export_path"] == "path1"
    assert res["status"] in ["exported", "mock_exported"]
    assert res["format"] == "safetensors"


def test_export_invalid() -> None:
    """Test export with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        export_model("model1", "path1", "invalid")
