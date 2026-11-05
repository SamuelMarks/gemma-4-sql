"""Unit and behavioral tests for export maxtext with real Orbax checkpointing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest

from gemma_4_sql.backends.maxtext.export import export_model
from gemma_4_sql.exceptions import DependencyMissingError, ExportError


class _TinyStatelessMaxTextModel:
    """Deterministic stateless MaxText model stub producing real JAX arrays."""

    def __init__(self, model_name: str) -> None:
        """Initialize model with name."""
        self.model_name = model_name

    def init(self, rng: jax.Array, dummy_input: jax.Array) -> dict[str, jax.Array]:
        """Initialize real JAX array parameters."""
        _ = (rng, dummy_input)
        return {"w": jnp.array([2.0, 4.0], dtype=jnp.float32)}

    def apply(self, params: dict[str, jax.Array], inputs: jax.Array) -> jax.Array:
        """Apply model parameters to inputs."""
        _ = params
        return jnp.ones((inputs.shape[0], inputs.shape[1], 8), dtype=jnp.float32)


def test_export_model_success_with_kwargs(tmp_path: Path) -> None:
    """Verify export_model writes a real Orbax checkpoint to disk when params are supplied."""
    export_path = tmp_path / "export_dir"
    weights = {"w": jnp.array([1.5, 3.5], dtype=jnp.float32)}

    result = export_model("test_model", str(export_path), params=weights)

    assert result["backend"] == "maxtext"
    assert result["model"] == "test_model"
    assert result["export_path"] == str(export_path)
    ckpt_dir = export_path / "maxtext_orbax_ckpt"
    assert result["file_path"] == str(ckpt_dir)
    assert result["status"] == "exported_with_maxtext_orbax"
    assert result["format"] == "maxtext/checkpoint"
    assert (ckpt_dir / "0").exists()

    with ocp.CheckpointManager(ckpt_dir, ocp.PyTreeCheckpointer()) as mngr:
        restored = mngr.restore(0)
    assert jnp.allclose(restored["w"], weights["w"])


def test_export_model_success_without_kwargs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify export_model initializes Gemma4Model and writes a real Orbax checkpoint."""
    export_path = tmp_path / "export_dir_no_kwargs"
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.export.Gemma4Model", _TinyStatelessMaxTextModel)

    result = export_model("test_model", str(export_path))
    assert result["status"] == "exported_with_maxtext_orbax"
    ckpt_dir = export_path / "maxtext_orbax_ckpt"
    assert (ckpt_dir / "0").exists()

    with ocp.CheckpointManager(ckpt_dir, ocp.PyTreeCheckpointer()) as mngr:
        restored = mngr.restore(0)
    assert jnp.allclose(restored["w"], jnp.array([2.0, 4.0], dtype=jnp.float32))
    out = _TinyStatelessMaxTextModel("test_model").apply(restored, jnp.zeros((1, 2), dtype=jnp.int32))
    assert out.shape == (1, 2, 8)


def test_export_model_missing_jax(tmp_path: Path) -> None:
    """Verify export model missing jax behavior."""
    with patch("gemma_4_sql.backends.maxtext.export.jax", None), pytest.raises(DependencyMissingError, match=r"MaxText export dependencies \(jax, orbax\.checkpoint\) are missing\."):
        export_model("test_model", str(tmp_path))


def test_export_model_missing_gemma4model(tmp_path: Path) -> None:
    """Verify export model missing gemma4model behavior."""
    with (
        patch("gemma_4_sql.backends.maxtext.export.Gemma4Model", None),
        pytest.raises(DependencyMissingError, match=r"MaxText dependency \(maxtext\.models\.gemma4\.Gemma4Model\) is missing\."),
    ):
        export_model("test_model", str(tmp_path))


def test_export_model_init_error(tmp_path: Path) -> None:
    """Verify export model init error behavior."""
    mock_gemma4 = MagicMock(side_effect=RuntimeError("Init failed"))
    with (
        patch("gemma_4_sql.backends.maxtext.export.Gemma4Model", mock_gemma4),
        pytest.raises(ExportError, match="Failed to initialize MaxText model 'test_model': Init failed"),
    ):
        export_model("test_model", str(tmp_path))


def test_export_model_save_error(tmp_path: Path) -> None:
    """Verify export model save error behavior."""
    export_path = tmp_path / "export_dir"

    mock_mngr_class = MagicMock(side_effect=RuntimeError("Save failed"))
    mock_ocp = MagicMock()
    mock_ocp.CheckpointManager = mock_mngr_class

    with (
        patch("gemma_4_sql.backends.maxtext.export.ocp", mock_ocp),
        pytest.raises(ExportError, match="Failed to save MaxText Orbax checkpoint"),
    ):
        export_model("test_model", str(export_path), params={"w": jnp.array([1.0])})
