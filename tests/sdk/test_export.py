"""Behavioral tests for the SDK Export module across backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.export import export_model
from gemma_4_sql.sdk.registry import get_backend


def test_export_jax(tmp_path: Path) -> None:
    """Verify export_model exports a JAX checkpoint artifact to disk."""
    dest = str(tmp_path / "jax_export")
    res = export_model("model1", dest, "jax")
    assert res["backend"] == "jax"
    assert res["model"] == "model1"
    assert res["export_path"] == dest
    assert res["format"] == "orbax/saved_model"
    assert Path(str(res["file_path"])).exists()


def test_export_keras(tmp_path: Path) -> None:
    """Verify export_model raises ValueError for unresolvable Keras preset names."""
    with pytest.raises(ValueError):
        export_model("model1", str(tmp_path / "keras_export"), "keras")


def test_export_maxtext(tmp_path: Path) -> None:
    """Verify export_model raises DependencyMissingError when MaxText is not installed."""
    with pytest.raises(DependencyMissingError):
        export_model("model1", str(tmp_path / "maxtext_export"), "maxtext")


def test_export_pytorch_routing_and_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify export_model forwards arguments to the PyTorch backend and returns export metadata."""
    dest = str(tmp_path / "pytorch_export")
    pytorch_agent = get_backend("pytorch")
    captured: dict[str, Any] = {}

    def spy_export(model_name: str, export_path: str, **kwargs: Any) -> dict[str, str]:
        """Record forwarded export arguments and write a marker file."""
        captured["model_name"] = model_name
        captured["export_path"] = export_path
        captured["kwargs"] = kwargs
        out_dir = Path(export_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / "model.safetensors"
        marker.write_bytes(b"safetensors")
        return {
            "backend": "pytorch",
            "model": model_name,
            "export_path": export_path,
            "file_path": str(marker),
            "format": "safetensors",
            "status": "exported_with_safetensors",
        }

    monkeypatch.setattr(pytorch_agent, "export_model", spy_export)
    res = export_model("model1", dest, "pytorch")
    assert captured["model_name"] == "model1"
    assert captured["export_path"] == dest
    assert res["backend"] == "pytorch"
    assert res["model"] == "model1"
    assert res["export_path"] == dest
    assert res["format"] == "safetensors"
    assert Path(str(res["file_path"])).exists()


def test_export_invalid(tmp_path: Path) -> None:
    """Verify export_model raises ValueError on an invalid backend name."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        export_model("model1", str(tmp_path / "path1"), "invalid")
