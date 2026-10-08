"""Module docstring."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Docstring for mock_dependencies."""
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_ocp = MagicMock()
    mock_nnx = MagicMock()
    mock_flax = MagicMock()
    mock_flax.nnx = mock_nnx

    mock_checkpointer = MagicMock()
    mock_ocp.PyTreeCheckpointer.return_value = mock_checkpointer

    mock_nnx.state.return_value = "mock_weights"

    mock_gemma4_config = MagicMock()
    mock_gemma4_config.gemma4_e2b.return_value = MagicMock()
    mock_gemma4_model = MagicMock()

    mock_gemma4_mod = MagicMock()
    mock_gemma4_mod.Gemma4Config = mock_gemma4_config
    mock_gemma4_mod.Gemma4ForCausalLM = mock_gemma4_model

    with patch.dict(
        sys.modules,
        {
            "jax": mock_jax,
            "jax.numpy": mock_jnp,
            "orbax.checkpoint": mock_ocp,
            "orbax": MagicMock(checkpoint=mock_ocp),
            "flax": mock_flax,
            "flax.nnx": mock_nnx,
            "gemma_4_sql.backends.jax.gemma4": mock_gemma4_mod,
        },
    ):
        yield mock_jax, mock_jnp, mock_ocp, mock_nnx, mock_gemma4_config, mock_gemma4_model, mock_checkpointer


def reload_module():
    """Docstring for reload_module."""
    import gemma_4_sql.backends.jax.export as jax_export

    importlib.reload(jax_export)
    return jax_export


def test_export_model_with_weights(tmp_path):
    """Docstring for test_export_model_with_weights."""
    jax_export = reload_module()

    export_path = str(tmp_path / "export")

    res = jax_export.export_model("test_model", export_path, weights="w")

    assert res["status"] == "exported_with_orbax"
    assert res["model"] == "test_model"
    assert res["format"] == "orbax/saved_model"


def test_export_model_extract_weights(tmp_path):
    """Docstring for test_export_model_extract_weights."""
    jax_export = reload_module()

    export_path = str(tmp_path / "export")

    # "test_model" will hit the test model config branch
    res = jax_export.export_model("test_model", export_path)
    assert res["status"] == "exported_with_orbax"

    # "model" will hit the e2b config branch
    res2 = jax_export.export_model("other_model", export_path)
    assert res2["status"] == "exported_with_orbax"

    # Use config argument
    res3 = jax_export.export_model("test_model", export_path, config="custom_config")
    assert res3["status"] == "exported_with_orbax"


def test_export_model_extract_weights_missing_flax(tmp_path):
    """Docstring for test_export_model_extract_weights_missing_flax."""
    jax_export = reload_module()

    with patch.dict(sys.modules, {"flax": None}), pytest.raises(Exception, match="Flax NNX dependency missing:"):
        # Will raise ImportError when trying to import flax inside function
        jax_export.export_model("test_model", str(tmp_path / "export"))


def test_export_model_extract_weights_error(tmp_path, mock_dependencies):
    """Docstring for test_export_model_extract_weights_error."""
    jax_export = reload_module()

    # Make extraction fail
    _, _, _, mock_nnx, _, _, _ = mock_dependencies
    mock_nnx.state.side_effect = Exception("State Extraction Failed")

    with pytest.raises(Exception, match="Failed to extract state for model 'test_model':"):
        jax_export.export_model("test_model", str(tmp_path / "export"))


def test_export_model_save_error(tmp_path, mock_dependencies):
    """Docstring for test_export_model_save_error."""
    jax_export = reload_module()

    _, _, _, _, _, _, mock_checkpointer = mock_dependencies
    mock_checkpointer.save.side_effect = Exception("Save Error")

    with pytest.raises(Exception, match="Failed to save Orbax checkpoint at"):
        jax_export.export_model("test_model", str(tmp_path / "export"), weights="w")
