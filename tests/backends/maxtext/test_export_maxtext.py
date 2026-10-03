from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.maxtext.export import export_model
from gemma_4_sql.exceptions import DependencyMissingError, ExportError


def test_export_model_success_with_kwargs(tmp_path):
    export_path = tmp_path / "export_dir"
    weights = {"w": 1}

    mock_mngr_instance = MagicMock()
    mock_mngr_class = MagicMock(return_value=mock_mngr_instance)
    mock_mngr_instance.__enter__.return_value = mock_mngr_instance
    mock_pytree_checkpointer = MagicMock()
    mock_checkpoint_manager_options = MagicMock()

    mock_ocp = MagicMock()
    mock_ocp.CheckpointManager = mock_mngr_class
    mock_ocp.PyTreeCheckpointer = mock_pytree_checkpointer
    mock_ocp.CheckpointManagerOptions = mock_checkpoint_manager_options

    mock_jax = MagicMock()
    mock_jnp = MagicMock()

    with patch("gemma_4_sql.backends.maxtext.export.jax", mock_jax), patch("gemma_4_sql.backends.maxtext.export.jnp", mock_jnp), patch("gemma_4_sql.backends.maxtext.export.ocp", mock_ocp):
        result = export_model("test_model", str(export_path), params=weights)

    assert result["backend"] == "maxtext"
    assert result["model"] == "test_model"
    assert result["export_path"] == str(export_path)
    assert result["file_path"] == str(export_path / "maxtext_orbax_ckpt")
    assert result["status"] == "exported_with_maxtext_orbax"
    assert result["format"] == "maxtext/checkpoint"

    mock_mngr_instance.save.assert_called_once_with(0, weights)


def test_export_model_success_without_kwargs(tmp_path):
    export_path = tmp_path / "export_dir"

    mock_mngr_instance = MagicMock()
    mock_mngr_class = MagicMock(return_value=mock_mngr_instance)
    mock_mngr_instance.__enter__.return_value = mock_mngr_instance
    mock_pytree_checkpointer = MagicMock()
    mock_checkpoint_manager_options = MagicMock()

    mock_ocp = MagicMock()
    mock_ocp.CheckpointManager = mock_mngr_class
    mock_ocp.PyTreeCheckpointer = mock_pytree_checkpointer
    mock_ocp.CheckpointManagerOptions = mock_checkpoint_manager_options

    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_jnp.int32 = "int32"
    mock_jnp.zeros.return_value = "zeros"
    mock_jax.random.PRNGKey.return_value = "prngkey"

    mock_model_instance = MagicMock()
    mock_model_instance.init.return_value = {"w": 2}
    mock_Gemma4Model = MagicMock(return_value=mock_model_instance)

    with patch("gemma_4_sql.backends.maxtext.export.jax", mock_jax), patch("gemma_4_sql.backends.maxtext.export.jnp", mock_jnp), patch("gemma_4_sql.backends.maxtext.export.ocp", mock_ocp), patch("gemma_4_sql.backends.maxtext.export.Gemma4Model", mock_Gemma4Model):
        export_model("test_model", str(export_path))

    mock_model_instance.init.assert_called_once_with("prngkey", "zeros")
    mock_mngr_instance.save.assert_called_once_with(0, {"w": 2})


def test_export_model_missing_jax(tmp_path):
    with patch("gemma_4_sql.backends.maxtext.export.jax", None), pytest.raises(DependencyMissingError, match="MaxText export dependencies \\(jax, orbax.checkpoint\\) are missing."):
        export_model("test_model", str(tmp_path))


def test_export_model_missing_gemma4model(tmp_path):
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_ocp = MagicMock()
    with (
        patch("gemma_4_sql.backends.maxtext.export.jax", mock_jax),
        patch("gemma_4_sql.backends.maxtext.export.jnp", mock_jnp),
        patch("gemma_4_sql.backends.maxtext.export.ocp", mock_ocp),
        patch("gemma_4_sql.backends.maxtext.export.Gemma4Model", None),
        pytest.raises(DependencyMissingError, match="MaxText dependency \\(maxtext.models.gemma4.Gemma4Model\\) is missing."),
    ):
        export_model("test_model", str(tmp_path))


def test_export_model_init_error(tmp_path):
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_ocp = MagicMock()
    mock_Gemma4Model = MagicMock(side_effect=Exception("Init failed"))
    with (
        patch("gemma_4_sql.backends.maxtext.export.jax", mock_jax),
        patch("gemma_4_sql.backends.maxtext.export.jnp", mock_jnp),
        patch("gemma_4_sql.backends.maxtext.export.ocp", mock_ocp),
        patch("gemma_4_sql.backends.maxtext.export.Gemma4Model", mock_Gemma4Model),
        pytest.raises(ExportError, match="Failed to initialize MaxText model 'test_model': Init failed"),
    ):
        export_model("test_model", str(tmp_path))


def test_export_model_save_error(tmp_path):
    export_path = tmp_path / "export_dir"

    mock_mngr_class = MagicMock(side_effect=Exception("Save failed"))
    mock_ocp = MagicMock()
    mock_ocp.CheckpointManager = mock_mngr_class
    mock_jax = MagicMock()
    mock_jnp = MagicMock()

    with patch("gemma_4_sql.backends.maxtext.export.jax", mock_jax), patch("gemma_4_sql.backends.maxtext.export.jnp", mock_jnp), patch("gemma_4_sql.backends.maxtext.export.ocp", mock_ocp), pytest.raises(ExportError, match="Failed to save MaxText Orbax checkpoint"):
        export_model("test_model", str(export_path), params={"w": 1})
