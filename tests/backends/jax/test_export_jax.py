"""Unit and behavioral tests for JAX model export with real Orbax checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.conftest import requires_jax

pytestmark = requires_jax

import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
from flax import nnx

import gemma_4_sql.backends.jax.export as jax_export
from gemma_4_sql.backends.jax.gemma4 import Gemma4Config
from gemma_4_sql.exceptions import DependencyMissingError, ExportError


def test_export_model_with_real_orbax_checkpoint(tmp_path: Path) -> None:
    """Verify export_model saves a real Orbax checkpoint that can be restored."""
    export_dir = tmp_path / "orbax_ckpt"
    weights = {"w": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)}

    res = jax_export.export_model("test_model", str(export_dir), weights=weights)
    assert res["status"] == "exported_with_orbax"
    assert res["backend"] == "jax"
    assert res["format"] == "orbax/saved_model"
    assert Path(res["file_path"]).exists()

    restored = ocp.PyTreeCheckpointer().restore(Path(res["file_path"]))
    assert jnp.allclose(restored["w"], weights["w"])


def test_export_model_extracts_real_gemma4_weights(tmp_path: Path) -> None:
    """Verify export_model instantiates a tiny Gemma4ForCausalLM model and exports its NNX state."""
    cfg = Gemma4Config.gemma4_tiny()
    export_dir = tmp_path / "model_export"

    res = jax_export.export_model("gemma-4-tiny", str(export_dir), config=cfg)
    assert res["status"] == "exported_with_orbax"
    assert Path(res["file_path"]).exists()

    # Also test preset resolution when config is omitted
    export_dir2 = tmp_path / "preset_export"
    res2 = jax_export.export_model("test_tiny_preset", str(export_dir2))
    assert res2["status"] == "exported_with_orbax"
    assert Path(res2["file_path"]).exists()


def test_export_model_error_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify export_model raises DependencyMissingError and ExportError on failures."""
    with patch.dict(sys.modules, {"flax": None}), pytest.raises(DependencyMissingError, match="Flax NNX dependency missing"):
        jax_export.export_model("test_model", str(tmp_path / "no_flax"))

    with patch.object(nnx, "state", side_effect=RuntimeError("State Extraction Failed")), pytest.raises(ExportError, match="Failed to extract state for model 'test_model'"):
        jax_export.export_model("test_model", str(tmp_path / "state_err"))

    mock_ocp = MagicMock()
    mock_ocp.PyTreeCheckpointer.return_value.save.side_effect = OSError("Disk full")
    monkeypatch.setattr(jax_export, "ocp", mock_ocp)
    with pytest.raises(ExportError, match="Failed to save Orbax checkpoint"):
        jax_export.export_model("test_model", str(tmp_path / "save_err"), weights={"a": 1})

    monkeypatch.setattr(jax_export, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX export dependencies are missing\."):
        jax_export.export_model("model", str(tmp_path / "no_jax"))
