"""Unit and behavioral tests for JAX model export with real Orbax checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Never
from unittest.mock import patch

from tests.conftest import requires_jax

pytestmark = requires_jax

import orbax.checkpoint as ocp
import pytest
from flax import nnx

import gemma_4_sql.backends.jax.export as jax_export
from gemma_4_sql.backends.jax.gemma4 import Gemma4Config
from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.exceptions import DependencyMissingError, ExportError


class _FailingCheckpointer:
    """Typed Orbax checkpointer stub raising `OSError` on `save`."""

    def save(self, _path: Path, _weights: jax_export.ExportWeightsTree) -> Never:
        """Raise `OSError` to simulate disk failure.

        Args:
            _path: Destination checkpoint path.
            _weights: Export weights dictionary.

        Raises:
            OSError: Always raised for error-path verification.

        """
        raise OSError("Disk full")

    def close(self) -> None:
        """Close the stub checkpointer."""


class _FailingOcpModule:
    """Typed `orbax.checkpoint` module stub exposing `_FailingCheckpointer`."""

    PyTreeCheckpointer = _FailingCheckpointer


def test_export_model_with_real_orbax_checkpoint(tmp_path: Path) -> None:
    """Verify export_model saves a real Orbax checkpoint (including nested dicts and NNX variables)."""
    export_dir = tmp_path / "orbax_ckpt"
    weights: dict[str, jax_export.ExportWeightLeaf | jax_export.ExportWeightsTree] = {
        "w": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        "nested": {"scale": 1.5},
    }
    var_tree: dict[str | int, jax_export.ExportWeightLeaf | str | nnx.Variable[jax_export.Array]] = {
        "param": nnx.Param(jnp.array([4.0], dtype=jnp.float32)),
        "ignored": "skip_str",
    }
    normalized = jax_export._normalize_export_tree(var_tree)
    assert normalized is not None
    assert "param" in normalized and "ignored" not in normalized

    res = jax_export.export_model("test_model", str(export_dir), weights=weights)
    assert res["status"] == "exported_with_orbax"
    assert res["backend"] == "jax"
    assert res["format"] == "orbax/saved_model"
    assert Path(res["file_path"]).exists()

    from collections.abc import Callable, Mapping
    from typing import Protocol

    from gemma_4_sql.type_hints import OrbaxCheckpointerProtocol

    class _ClosableCheckpointerProtocol(OrbaxCheckpointerProtocol[Mapping[str, jax_export.Array]], Protocol):
        """Protocol for an Orbax `PyTreeCheckpointer` exposing `.close()`."""

        def close(self) -> None:
            """Release background Orbax checkpointer resources."""
            ...

    ckpt_factory: Callable[[], _ClosableCheckpointerProtocol] = vars(ocp)["PyTreeCheckpointer"]
    ckptr = ckpt_factory()
    try:
        restored = ckptr.restore(Path(res["file_path"]))
    finally:
        ckptr.close()
    assert jnp.allclose(restored["w"], jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))


def test_export_model_extracts_real_gemma4_weights(tmp_path: Path) -> None:
    """Verify export_model instantiates a tiny Gemma4ForCausalLM model and exports its NNX state."""
    cfg = Gemma4Config.gemma4_tiny()
    export_dir = tmp_path / "model_export"

    res = jax_export.export_model("gemma-4-tiny", str(export_dir), config=cfg)
    assert res["status"] == "exported_with_orbax"
    assert Path(res["file_path"]).exists()

    export_dir2 = tmp_path / "preset_export"
    res2 = jax_export.export_model("gemma-4-tiny", str(export_dir2))
    assert res2["status"] == "exported_with_orbax"
    assert Path(res2["file_path"]).exists()


def test_export_model_error_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify export_model raises DependencyMissingError and ExportError on failures."""
    with (
        patch.dict(sys.modules, {"flax": None}),
        pytest.raises(DependencyMissingError, match="Flax NNX dependency missing"),
    ):
        jax_export.export_model("gemma-4-tiny", str(tmp_path / "no_flax"))

    with (
        patch.object(nnx, "state", side_effect=RuntimeError("State Extraction Failed")),
        pytest.raises(ExportError, match="Failed to extract state for model 'gemma-4-tiny'"),
    ):
        jax_export.export_model("gemma-4-tiny", str(tmp_path / "state_err"))

    monkeypatch.setattr(jax_export, "ocp", _FailingOcpModule)
    with pytest.raises(ExportError, match="Failed to save Orbax checkpoint"):
        jax_export.export_model("test_model", str(tmp_path / "save_err"), weights={"a": 1})

    monkeypatch.setattr(jax_export, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX export dependencies are missing\."):
        jax_export.export_model("model", str(tmp_path / "no_jax"))
