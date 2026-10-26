"""Behavioral tests for the SDK Export module across backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.export import export_model


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


def test_export_pytorch_routing_and_execution(tmp_path: Path) -> None:
    """Verify export_model serializes a real tiny PyTorch Gemma4ForCausalLM to model.safetensors."""
    from safetensors.torch import load_file

    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config

    dest = str(tmp_path / "pytorch_export")
    tiny_cfg = Gemma4Config.gemma4_tiny()
    res = export_model("gemma-4-tiny", dest, "pytorch_native")
    assert res["backend"] == "pytorch_native"
    assert res["model"] == "gemma-4-tiny"
    assert res["export_path"] == dest
    assert res["format"] == "safetensors"
    file_path = Path(str(res["file_path"]))
    assert file_path.exists()
    tensors = load_file(str(file_path))
    assert "embed_tokens.weight" in tensors
    assert tensors["embed_tokens.weight"].shape == (tiny_cfg.vocab_size, tiny_cfg.hidden_size)


def test_export_invalid(tmp_path: Path) -> None:
    """Verify export_model raises ValueError on an invalid backend name."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        export_model("model1", str(tmp_path / "path1"), "invalid")
