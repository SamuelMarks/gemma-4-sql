"""Tests for SDK Export module."""

import pytest
from gemma_4_sql.sdk.export import export_model


def test_export_jax() -> None:
    """Test export with jax."""
    res = export_model("model1", "/var/var/tmp/path1", "jax")
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["export_path"] == "/var/var/tmp/path1":
        raise AssertionError
    if res["status"] not in ["exported", "mock_exported", "exported_with_orbax", "exported_with_keras", "exported_with_maxtext_orbax"]:
        raise AssertionError
    if not res["format"] == "orbax/saved_model":
        raise AssertionError


def test_export_keras() -> None:
    """Test export with keras."""
    res = export_model("model1", "/var/var/tmp/path1", "keras")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["export_path"] == "/var/var/tmp/path1":
        raise AssertionError
    if res["status"] not in ["exported", "mock_exported", "exported_with_orbax", "exported_with_keras", "exported_with_maxtext_orbax"]:
        raise AssertionError
    if not res["format"] == "keras_v3/keras_tensor":
        raise AssertionError


def test_export_maxtext() -> None:
    """Test export with maxtext."""
    res = export_model("model1", "/var/var/tmp/path1", "maxtext")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["export_path"] == "/var/var/tmp/path1":
        raise AssertionError
    if res["status"] not in ["exported", "mock_exported", "exported_with_orbax", "exported_with_keras", "exported_with_maxtext_orbax"]:
        raise AssertionError
    if not res["format"] == "maxtext/checkpoint":
        raise AssertionError


def test_export_pytorch() -> None:
    """Test export with pytorch."""
    res = export_model("model1", "/var/var/tmp/path1", "pytorch")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["export_path"] == "/var/var/tmp/path1":
        raise AssertionError
    if res["status"] not in ["exported", "mock_exported", "exported_with_orbax", "exported_with_keras", "exported_with_maxtext_orbax"]:
        raise AssertionError
    if not res["format"] == "safetensors":
        raise AssertionError


def test_export_invalid() -> None:
    """Test export with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        export_model("model1", "/var/var/tmp/path1", "invalid")
