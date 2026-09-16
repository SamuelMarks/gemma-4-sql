"""Tests for JAX model export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.jax.export as export_jax
from gemma_4_sql.exceptions import DependencyMissingError, ExportError


def test_export_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test module reload behavior when dependencies are missing."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(export_jax)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", None)
    importlib.reload(export_jax)
    monkeypatch.undo()
    importlib.reload(export_jax)


def test_export_jax_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test DependencyMissingError when JAX/Orbax dependencies are missing."""
    monkeypatch.setattr(export_jax, "jax", None)
    path = str(tmp_path / "export")
    with pytest.raises(DependencyMissingError, match=r"JAX export dependencies are missing\."):
        export_jax.export_model("model1", path)

    monkeypatch.setattr(export_jax, "jax", MagicMock())
    monkeypatch.setattr(export_jax, "jnp", None)
    with pytest.raises(DependencyMissingError, match=r"JAX export dependencies are missing\."):
        export_jax.export_model("model1", path)

    monkeypatch.setattr(export_jax, "jnp", MagicMock())
    monkeypatch.setattr(export_jax, "ocp", None)
    with pytest.raises(DependencyMissingError, match=r"JAX export dependencies are missing\."):
        export_jax.export_model("model1", path)


def test_export_jax_missing_flax(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test DependencyMissingError when Flax cannot be imported."""
    import sys

    monkeypatch.setattr(export_jax, "jax", MagicMock())
    monkeypatch.setattr(export_jax, "jnp", MagicMock())
    monkeypatch.setattr(export_jax, "ocp", MagicMock())
    monkeypatch.setitem(sys.modules, "flax", None)
    monkeypatch.setitem(sys.modules, "flax.nnx", None)

    path = str(tmp_path / "export_no_flax")
    with pytest.raises(DependencyMissingError, match="Flax NNX dependency missing"):
        export_jax.export_model("model2", path)


def test_export_jax_model_extraction_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test ExportError when model instantiation or state extraction fails."""
    import builtins

    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_ocp = MagicMock()
    monkeypatch.setattr(export_jax, "jax", mock_jax)
    monkeypatch.setattr(export_jax, "jnp", mock_jnp)
    monkeypatch.setattr(export_jax, "ocp", mock_ocp)

    orig_import = builtins.__import__

    def faulty_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Faulty import raising error."""
        if "gemma4" in name:
            raise RuntimeError("Model definition corrupt")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", faulty_import)
    path = str(tmp_path / "export_faulty")
    with pytest.raises(ExportError, match="Failed to extract state for model"):
        export_jax.export_model("model_err", path)


def test_export_jax_save_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test ExportError when PyTreeCheckpointer save raises an exception."""
    mock_ocp = MagicMock()

    class FailingCheckpointer:
        """Checkpointer that always raises OSError on save."""

        def save(self, *args: Any, **kwargs: Any) -> None:
            """Raise error on save."""
            raise OSError("Read-only filesystem")

    mock_ocp.PyTreeCheckpointer = FailingCheckpointer
    monkeypatch.setattr(export_jax, "ocp", mock_ocp)

    path = str(tmp_path / "export_save_err")
    with pytest.raises(ExportError, match="Failed to save Orbax checkpoint"):
        export_jax.export_model("model_save_err", path, weights={"w": [1]})


def test_export_jax_weights_kwargs(tmp_path: Path) -> None:
    """Test export when weights are directly provided in kwargs."""
    path = str(tmp_path / "export_kwargs")
    res = export_jax.export_model("custom_model", path, weights={"custom_weight": [1.0, 2.0]})
    assert res["status"] == "exported_with_orbax"
    assert res["backend"] == "jax"
    assert res["model"] == "custom_model"
    assert Path(res["file_path"]).exists()


def test_export_jax_real_nnx(tmp_path: Path) -> None:
    """Test genuine end-to-end model export with real Flax NNX model state extraction."""
    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config

    path = str(tmp_path / "export_real_nnx")
    tiny_cfg = Gemma4Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
    )

    # Test with explicit config in kwargs
    res = export_jax.export_model("test_e2e_model", path, config=tiny_cfg)
    assert res["status"] == "exported_with_orbax"
    assert res["backend"] == "jax"
    assert res["format"] == "orbax/saved_model"
    assert Path(res["file_path"]).exists()

    # Test with default config branch for model identifiers
    path2 = str(tmp_path / "export_real_nnx2")
    res2 = export_jax.export_model("model1", path2)
    assert res2["status"] == "exported_with_orbax"
    assert Path(res2["file_path"]).exists()

    # Test with test_mode=True
    path3 = str(tmp_path / "export_real_nnx3")
    res3 = export_jax.export_model("arbitrary_name", path3, test_mode=True)
    assert res3["status"] == "exported_with_orbax"
    assert Path(res3["file_path"]).exists()


def test_export_jax_gemma4_e2b_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test Gemma4Config.gemma4_e2b branch when model name is not a test/model prefix."""
    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config

    tiny_cfg = Gemma4Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
    )
    monkeypatch.setattr(Gemma4Config, "gemma4_e2b", classmethod(lambda _cls: tiny_cfg))

    path = str(tmp_path / "export_e2b")
    res = export_jax.export_model("production_name", path)
    assert res["status"] == "exported_with_orbax"
    assert Path(res["file_path"]).exists()
