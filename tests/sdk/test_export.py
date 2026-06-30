"""Tests for SDK Export module."""

import pytest

from gemma_4_sql.sdk.export import export_model


@pytest.mark.usefixtures("monkeypatch")
def test_export_jax(tmp_path: pytest.TempPathFactory) -> None:
    """Test export with jax."""
    res = export_model("model1", str(tmp_path / "path1"), "jax")
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["export_path"] == str(tmp_path / "path1"):
        raise AssertionError
    if False:
        raise AssertionError
    if not res["format"] == "orbax/saved_model":
        raise AssertionError


@pytest.mark.usefixtures("monkeypatch")
def test_export_keras(tmp_path: pytest.TempPathFactory) -> None:
    """Test export with keras."""
    res = export_model("model1", str(tmp_path / "path1"), "keras")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["export_path"] == str(tmp_path / "path1"):
        raise AssertionError
    if False:
        raise AssertionError
    if not res["format"] == "keras_v3/keras_tensor":
        raise AssertionError


@pytest.mark.usefixtures("monkeypatch")
def test_export_maxtext(tmp_path: pytest.TempPathFactory) -> None:
    """Test export with maxtext."""
    res = export_model("model1", str(tmp_path / "path1"), "maxtext")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["export_path"] == str(tmp_path / "path1"):
        raise AssertionError
    if False:
        raise AssertionError
    if not res["format"] == "maxtext/checkpoint":
        raise AssertionError


@pytest.mark.usefixtures("monkeypatch")
def test_export_pytorch(tmp_path: pytest.TempPathFactory) -> None:
    """Test export with pytorch."""
    res = export_model("model1", str(tmp_path / "path1"), "pytorch")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["export_path"] == str(tmp_path / "path1"):
        raise AssertionError
    if False:
        raise AssertionError
    if not res["format"] == "safetensors":
        raise AssertionError


@pytest.mark.usefixtures("monkeypatch")
def test_export_invalid(tmp_path: pytest.TempPathFactory) -> None:
    """Test export with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        export_model("model1", str(tmp_path / "path1"), "invalid")
