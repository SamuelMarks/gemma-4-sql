"""Tests for SDK Export module."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.export import export_model


@pytest.mark.usefixtures("monkeypatch")
def test_export_jax(tmp_path: pytest.TempPathFactory) -> None:
    """Test export with jax.

    Raises:
        AssertionError: Description.

    """
    res = export_model("model1", str(tmp_path / "path1"), "jax")
    assert res["backend"] == "jax"

    assert res["model"] == "model1"

    assert res["export_path"] == str(tmp_path / "path1")

    assert res["format"] == "orbax/saved_model"


@pytest.mark.usefixtures("monkeypatch")
def test_export_keras(tmp_path: pytest.TempPathFactory) -> None:
    """Test export with keras."""
    with pytest.raises(ValueError):
        export_model("model1", str(tmp_path / "path1"), "keras")


def test_export_maxtext(tmp_path: pytest.TempPathFactory) -> None:
    """Test export with maxtext."""
    with pytest.raises(DependencyMissingError):
        export_model("model1", str(tmp_path / "path1"), "maxtext")


@pytest.mark.usefixtures("monkeypatch")
def test_export_pytorch(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    """Test export with pytorch.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    pytorch_agent = get_backend("pytorch")
    monkeypatch.setattr(pytorch_agent, "export_model", lambda m, p: {"backend": "pytorch", "model": m, "export_path": p, "format": "safetensors", "status": "success"})
    res = export_model("model1", str(tmp_path / "path1"), "pytorch")
    assert res["backend"] == "pytorch"

    assert res["model"] == "model1"

    assert res["export_path"] == str(tmp_path / "path1")

    assert res["format"] == "safetensors"


@pytest.mark.usefixtures("monkeypatch")
def test_export_invalid(tmp_path: pytest.TempPathFactory) -> None:
    """Test export with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        export_model("model1", str(tmp_path / "path1"), "invalid")
