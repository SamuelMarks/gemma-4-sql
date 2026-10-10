"""Unified tests for model export across supported backends."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, ExportError
from gemma_4_sql.sdk.export import export_model
from gemma_4_sql.sdk.registry import get_backend


def test_export_model_jax_unified(monkeypatch, tmp_path: Path) -> None:
    """Test unified model export using JAX backend."""
    from gemma_4_sql.sdk.registry import get_backend

    backend_agent = get_backend("jax")
    monkeypatch.setattr(backend_agent, "export_model", lambda *a, **k: {"backend": "jax", "format": "orbax/saved_model", "file_path": str(tmp_path / "f.txt")})
    (tmp_path / "f.txt").touch()

    export_dir = tmp_path / "jax_unified_export"
    res = export_model("test_model", str(export_dir), backend="jax")
    assert res["backend"] == "jax"
    assert res["format"] == "orbax/saved_model"
    assert Path(res["file_path"]).exists()


def test_export_model_pytorch_unified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test unified model export using PyTorch backend."""

    # We need to mock torch and safetensors inside the module properly
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.pytorch.export as pt_export

    mock_torch = MagicMock()
    mock_save_file = MagicMock()

    monkeypatch.setattr(pt_export, "torch", mock_torch)
    monkeypatch.setattr(pt_export, "save_file", mock_save_file)

    export_dir = tmp_path / "pytorch_unified_export"
    res = export_model(
        "test_model",
        str(export_dir),
        backend="pytorch",
        backend_alias="pytorch_native",
    )
    assert res["backend"] == "pytorch_native"
    assert res["status"] in ("exported_with_safetensors", "skipped_non_rank_zero")


def test_export_model_maxtext_unified(tmp_path: Path) -> None:
    """Test unified model export using MaxText backend."""
    export_dir = tmp_path / "maxtext_unified_export"
    with pytest.raises((DependencyMissingError, ExportError)):
        export_model("test_model", str(export_dir), backend="maxtext")


def test_export_model_invalid_backend(tmp_path: Path) -> None:
    """Test unified model export with unrecognized backend name."""
    with pytest.raises(ValueError, match="Unknown backend: nonexistent_backend"):
        export_model("test_model", str(tmp_path), backend="nonexistent_backend")


def test_export_model_custom_backend_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test unified export dispatch with mocked backend agent."""
    backend_agent = get_backend("jax")
    mock_export = MagicMock(return_value={"backend": "jax", "model": "mock", "status": "ok"})
    monkeypatch.setattr(backend_agent, "export_model", mock_export)

    res = export_model("mock", str(tmp_path), backend="jax")
    assert res["status"] == "ok"
    mock_export.assert_called_once_with("mock", str(tmp_path))
