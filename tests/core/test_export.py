"""Unified behavioral tests for model export across supported backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from gemma_4_sql.sdk.export import export_model


def test_export_model_jax_unified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify export_model routes to the JAX backend and forwards model_name, export_path, and kwargs."""
    import gemma_4_sql.sdk.export as sdk_exp

    captured: list[tuple[str, str, dict[str, Any]]] = []

    class FakeJaxBackend:
        """Backend stub writing an export artifact to disk."""

        def export_model(self, model_name: str, export_path: str, **kwargs: Any) -> dict[str, Any]:
            """Write artifact file and return ExportResultDict."""
            captured.append((model_name, export_path, dict(kwargs)))
            out_dir = Path(export_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            artifact = out_dir / "checkpoint"
            artifact.write_text("orbax_ckpt", encoding="utf-8")
            return {
                "backend": "jax",
                "model": model_name,
                "export_path": export_path,
                "file_path": str(artifact),
                "format": "orbax/saved_model",
                "status": "exported_with_orbax",
            }

    monkeypatch.setattr(sdk_exp, "get_backend", lambda name: FakeJaxBackend() if name == "jax" else None)

    export_dir = tmp_path / "jax_unified_export"
    res = export_model("test_model", str(export_dir), backend="jax", step=5)
    assert len(captured) == 1
    assert captured[0] == ("test_model", str(export_dir), {"step": 5})
    assert res["backend"] == "jax"
    assert res["format"] == "orbax/saved_model"
    assert Path(res["file_path"]).read_text(encoding="utf-8") == "orbax_ckpt"


def test_export_model_pytorch_unified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test unified model export using PyTorch native backend."""
    import gemma_4_sql.backends.pytorch.export as pt_export

    saved_tensors: dict[str, Any] = {}

    def _capture_save(tensors: dict[str, Any], filename: str) -> None:
        """Capture saved state_dict and write placeholder file."""
        saved_tensors.update(tensors)
        Path(filename).write_bytes(b"safetensors")

    monkeypatch.setattr(pt_export, "save_file", _capture_save)

    export_dir = tmp_path / "pytorch_unified_export"
    res = export_model(
        "test_model",
        str(export_dir),
        backend="pytorch",
        backend_alias="pytorch_native",
    )
    assert res["backend"] == "pytorch_native"
    assert res["status"] == "exported_with_safetensors"
    assert len(saved_tensors) > 0
    assert Path(res["file_path"]).exists()


def test_export_model_maxtext_unified(tmp_path: Path) -> None:
    """Test unified model export using MaxText backend."""
    from gemma_4_sql.exceptions import DependencyMissingError, ExportError

    export_dir = tmp_path / "maxtext_unified_export"
    with pytest.raises((DependencyMissingError, ExportError)):
        export_model("test_model", str(export_dir), backend="maxtext")


def test_export_model_invalid_backend(tmp_path: Path) -> None:
    """Test unified model export with unrecognized backend name."""
    with pytest.raises(ValueError, match="Unknown backend: nonexistent_backend"):
        export_model("test_model", str(tmp_path), backend="nonexistent_backend")


def test_export_model_custom_backend_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test unified export dispatch forwarding extra kwargs to backend adapter."""
    import gemma_4_sql.sdk.export as sdk_exp

    mock_be = MagicMock()
    mock_export = MagicMock(return_value={"backend": "jax", "model": "mock", "status": "ok"})
    mock_be.export_model = mock_export
    monkeypatch.setattr(sdk_exp, "get_backend", lambda _: mock_be)

    res = export_model("mock", str(tmp_path), backend="jax", format="gguf")
    assert res["status"] == "ok"
    mock_export.assert_called_once_with("mock", str(tmp_path), format="gguf")
