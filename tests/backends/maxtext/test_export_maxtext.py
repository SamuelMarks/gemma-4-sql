"""Unit and behavioral tests for MaxText export with real Orbax checkpointing without MagicMock."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import jax
import orbax.checkpoint as ocp
import pytest

from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.maxtext import inference as maxtext_inference
from gemma_4_sql.backends.maxtext.export import export_model
from gemma_4_sql.exceptions import DependencyMissingError, ExportError


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Clear MaxText caches before each export test."""
    maxtext_inference._MODEL_CACHE.clear()
    maxtext_inference._PARAMS_ONLY_CACHE.clear()


class _TinyStatelessMaxTextModel:
    """Deterministic stateless MaxText model stub producing real JAX arrays."""

    def __init__(self, model_name: str) -> None:
        """Initialize model with name."""
        self.model_name = model_name

    def init(self, rng: jax.Array, dummy_input: jax.Array) -> dict[str, jax.Array]:
        """Initialize real JAX array parameters.

        Returns:
            A dictionary containing initialized float32 parameter `'w'`.

        """
        _ = (rng, dummy_input)
        return {"w": jnp.array([2.0, 4.0], dtype=jnp.float32)}

    def apply(self, params: dict[str, jax.Array], inputs: jax.Array) -> jax.Array:
        """Apply model parameters to inputs.

        Returns:
            A float32 ones logits array of shape `(batch, seq, 8)`.

        """
        _ = params
        return jnp.ones((inputs.shape[0], inputs.shape[1], 8), dtype=jnp.float32)


def test_export_model_success_with_kwargs_and_cached_params(tmp_path: Path) -> None:
    """Verify `export_model` writes explicit `params` or cached MaxText weights to Orbax."""
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

    from gemma_4_sql.backends.maxtext import export as export_mod
    from gemma_4_sql.type_hints import JaxArrayProtocol

    ocp_typed = export_mod.ocp
    assert ocp_typed is not None
    with ocp_typed.CheckpointManager(ckpt_dir, ocp_typed.PyTreeCheckpointer()) as mngr:
        restored = mngr.restore(0)
    restored_w = restored["w"]
    assert isinstance(restored_w, JaxArrayProtocol)
    assert jnp.allclose(jnp.asarray(restored_w), weights["w"])

    # Verify cached parameters are exported even when `Gemma4Model` is `None`!
    cached_weights = {"w": jnp.array([7.0, 9.0], dtype=jnp.float32)}
    maxtext_inference.cache_maxtext_params("cached_model", cached_weights)
    cached_export_dir = tmp_path / "cached_export"
    with patch("gemma_4_sql.backends.maxtext.export.Gemma4Model", None):
        cached_res = export_model("cached_model", str(cached_export_dir))
    assert cached_res["status"] == "exported_with_maxtext_orbax"
    with ocp_typed.CheckpointManager(
        cached_export_dir / "maxtext_orbax_ckpt",
        ocp_typed.PyTreeCheckpointer(),
    ) as mngr:
        restored_cached = mngr.restore(0)
    restored_cached_w = restored_cached["w"]
    assert isinstance(restored_cached_w, JaxArrayProtocol)
    assert jnp.allclose(jnp.asarray(restored_cached_w), cached_weights["w"])


def test_export_model_success_without_kwargs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `export_model` initializes `Gemma4Model`, caches weights, and writes a real Orbax checkpoint."""
    from gemma_4_sql.backends.maxtext import export as export_mod
    from gemma_4_sql.type_hints import JaxArrayProtocol

    export_path = tmp_path / "export_dir_no_kwargs"
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.export.Gemma4Model", _TinyStatelessMaxTextModel)

    result = export_model("test_model", str(export_path))
    assert result["status"] == "exported_with_maxtext_orbax"
    ckpt_dir = export_path / "maxtext_orbax_ckpt"
    assert (ckpt_dir / "0").exists()

    ocp_typed = export_mod.ocp
    assert ocp_typed is not None
    with ocp_typed.CheckpointManager(ckpt_dir, ocp_typed.PyTreeCheckpointer()) as mngr:
        restored = mngr.restore(0)
    restored_w = restored["w"]
    assert isinstance(restored_w, JaxArrayProtocol)
    assert jnp.allclose(jnp.asarray(restored_w), jnp.array([2.0, 4.0], dtype=jnp.float32))
    restored_arrs = {k: jnp.asarray(v) for k, v in restored.items() if isinstance(v, JaxArrayProtocol)}
    out = _TinyStatelessMaxTextModel("test_model").apply(restored_arrs, jnp.zeros((1, 2), dtype=jnp.int32))
    assert out.shape == (1, 2, 8)


def test_export_model_missing_jax(tmp_path: Path) -> None:
    """Verify `export_model` raises `DependencyMissingError` when `jax` is `None`."""
    with (
        patch("gemma_4_sql.backends.maxtext.export.jax", None),
        pytest.raises(
            DependencyMissingError, match=r"MaxText export dependencies \(jax, orbax\.checkpoint\) are missing\."
        ),
    ):
        export_model("test_model", str(tmp_path))


def test_export_model_missing_gemma4model(tmp_path: Path) -> None:
    """Verify `export_model` raises `DependencyMissingError` when `Gemma4Model` is `None`."""
    with (
        patch("gemma_4_sql.backends.maxtext.export.Gemma4Model", None),
        pytest.raises(
            DependencyMissingError, match=r"MaxText dependency \(maxtext\.models\.gemma4\.Gemma4Model\) is missing\."
        ),
    ):
        export_model("test_model", str(tmp_path))


def test_export_model_init_error(tmp_path: Path) -> None:
    """Verify `export_model` wraps `Gemma4Model` initialization failures in `ExportError`."""

    def _failing_gemma4(_name: str) -> _TinyStatelessMaxTextModel:
        """Raise `RuntimeError` during model construction.

        Raises:
            RuntimeError: Always raised to test `ExportError` wrapping.

        """
        raise RuntimeError("Init failed")

    with (
        patch("gemma_4_sql.backends.maxtext.export.Gemma4Model", _failing_gemma4),
        pytest.raises(ExportError, match="Failed to initialize MaxText model 'test_model': Init failed"),
    ):
        export_model("test_model", str(tmp_path))


def test_export_model_save_error(tmp_path: Path) -> None:
    """Verify `export_model` wraps Orbax checkpoint save failures in `ExportError`."""
    export_path = tmp_path / "export_dir"

    with (
        patch.object(ocp, "CheckpointManager", side_effect=RuntimeError("Save failed")),
        pytest.raises(ExportError, match="Failed to save MaxText Orbax checkpoint"),
    ):
        export_model("test_model", str(export_path), params={"w": jnp.array([1.0])})
