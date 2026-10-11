"""Tests for maxtext train."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, ExportError
from gemma_4_sql.type_hints import TrainingConfig


def test_maxtext_train_imports():
    """Test maxtext train imports fallback."""
    import importlib

    with patch.dict(sys.modules, {"jax": None, "maxtext": None}):
        import gemma_4_sql.backends.maxtext.train as train_module

        importlib.reload(train_module)
        assert train_module.jax is None
    importlib.reload(train_module)


def test_maxtext_train_loss_fn():
    """Test _loss_fn."""
    import gemma_4_sql.backends.maxtext.train as train_module

    mock_jnp = MagicMock()
    mock_optax = MagicMock()
    train_module.jnp = mock_jnp
    train_module.optax = mock_optax

    mock_model = MagicMock()
    mock_model.apply.return_value = "logits"
    mock_optax.softmax_cross_entropy_with_integer_labels.return_value = "loss"
    mock_jnp.mean.return_value = "mean_loss"

    params = {"p": 1}
    batch = {"inputs": "inputs", "targets": "targets"}

    res = train_module._loss_fn(mock_model, params, batch)
    assert res == "mean_loss"
    mock_model.apply.assert_called_once_with(params, "inputs")
    mock_optax.softmax_cross_entropy_with_integer_labels.assert_called_once_with("logits", "targets")
    mock_jnp.mean.assert_called_once_with("loss")


def test_get_train_step_fn():
    """Test _get_train_step_fn."""
    import gemma_4_sql.backends.maxtext.train as train_module

    mock_jax = MagicMock()
    train_module.jax = mock_jax
    mock_jax.jit.side_effect = lambda x: x

    mock_model = MagicMock()
    mock_optimizer = MagicMock()
    mock_optimizer.update.return_value = ("updates", "new_opt_state")
    train_module.optax.apply_updates = MagicMock(return_value="new_params")

    def mock_value_and_grad(fn):
        """Docstring for mock_value_and_grad."""

        def wrapper(p, b):
            """Docstring for wrapper."""
            fn(p, b)  # Trigger wrapper
            return ("loss", "grads")

        return wrapper

    mock_jax.value_and_grad.side_effect = mock_value_and_grad

    with patch("gemma_4_sql.backends.maxtext.train._loss_fn") as mock_loss:
        mock_loss.return_value = "loss"
        step_fn = train_module._get_train_step_fn(mock_model, mock_optimizer)
        params = {"p": 1}
        opt_state = {"s": 1}
        batch = {"b": 1}

        new_params, new_opt_state, loss = step_fn(params, opt_state, batch)
        assert new_params == "new_params"
        assert new_opt_state == "new_opt_state"
        assert loss == "loss"

    # test jax is None
    train_module.jax = None
    step_fn2 = train_module._get_train_step_fn(mock_model, mock_optimizer)
    assert step_fn2.__name__ == "train_step"
    train_module.jax = mock_jax


def test_run_training_epochs():
    """Test _run_training_epochs."""
    import gemma_4_sql.backends.maxtext.train as train_module

    state = MagicMock()
    state.params = "params"
    state.opt_state = "opt_state"
    state.epochs = 1
    state.dataloader = [1, 2]

    def train_step(params, opt_state, batch):
        """Docstring for train_step."""
        return "new_params", "new_opt_state", MagicMock(item=lambda: 0.5)

    state.train_step = train_step

    with patch("gemma_4_sql.backends.maxtext.train.generic_run_training_epochs") as mock_run:
        mock_run.return_value = 0.5

        # Test process_batch internally
        def capture_fn(epochs, dataloader, process_batch):
            """Docstring for capture_fn."""
            return process_batch({"batch": 1})

        mock_run.side_effect = capture_fn

        params, opt_state, loss = train_module._run_training_epochs(state)
        assert params == "new_params"
        assert opt_state == "new_opt_state"
        assert loss == 0.5


def test_initialize_jax_distributed():
    """Test _initialize_jax_distributed."""
    import gemma_4_sql.backends.maxtext.train as train_module

    mock_jax = MagicMock()
    train_module.jax = mock_jax

    res = train_module._initialize_jax_distributed(coordinator_address="addr", num_processes=2, process_id=0)
    assert res is True
    mock_jax.distributed.initialize.assert_called_with(coordinator_address="addr", num_processes=2, process_id=0)

    mock_jax.distributed.initialize.side_effect = RuntimeError("error")
    res = train_module._initialize_jax_distributed()
    assert res is False

    train_module.jax = None
    res = train_module._initialize_jax_distributed()
    assert res is False
    train_module.jax = mock_jax


def test_save_maxtext_checkpoint(tmp_path):
    """Test save_maxtext_checkpoint."""
    import gemma_4_sql.backends.maxtext.train as train_module

    mock_ocp = MagicMock()
    train_module.ocp = mock_ocp

    ckpt_dir = tmp_path / "ckpt"
    res = train_module.save_maxtext_checkpoint(ckpt_dir, 1, {"p": 1}, {"s": 1})
    assert res == ckpt_dir.resolve()
    mock_ocp.CheckpointManager.assert_called_once()

    mock_ocp.CheckpointManager.side_effect = Exception("error")
    with pytest.raises(ExportError, match="Failed to persist Orbax checkpoint"):
        train_module.save_maxtext_checkpoint(ckpt_dir, 1, {"p": 1})

    train_module.ocp = None
    with pytest.raises(DependencyMissingError, match="Orbax checkpoint dependency"):
        train_module.save_maxtext_checkpoint(ckpt_dir, 1, {"p": 1})


@patch("gemma_4_sql.backends.maxtext.train.build_dataloader")
def test_execute_train_local(mock_build_dataloader):
    """Test _execute_train local."""
    import gemma_4_sql.backends.maxtext.train as train_module

    train_module.jax = MagicMock()
    train_module.jnp = MagicMock()
    train_module.optax = MagicMock()
    train_module.maxtext_train = MagicMock()
    train_module.Gemma4Model = MagicMock()
    train_module.ocp = MagicMock()

    mock_build_dataloader.return_value = {"loader": [1, 2]}

    config = TrainingConfig(model_name="model", dataset="dataset", epochs=1, batch_size=2)

    with patch("gemma_4_sql.backends.maxtext.train._get_train_step_fn"):
        with patch("gemma_4_sql.backends.maxtext.train._run_training_epochs") as mock_run_epochs:
            mock_run_epochs.return_value = ("params", "opt_state", 0.5)
            status, loss = train_module._execute_train(config, local_step_mode=True, checkpoint_dir="dir")
            assert status == "completed"
            assert loss == 0.5

    # test Gemma4Model missing
    train_module.Gemma4Model = None
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        train_module._execute_train(config, local_step_mode=True)
    train_module.Gemma4Model = MagicMock()

    # test no dataloader
    mock_build_dataloader.return_value = {}
    with pytest.raises(ValueError, match="Invalid dataloader"):
        train_module._execute_train(config, local_step_mode=True)


@patch("gemma_4_sql.backends.maxtext.train.build_dataloader")
@patch("gemma_4_sql.backends.maxtext.config_generator.generate_maxtext_gin_config", return_value="gin")
@patch("gemma_4_sql.backends.maxtext.config_generator.save_maxtext_gin_config", return_value=Path("gin.file"))
@patch("gemma_4_sql.backends.maxtext.config_generator.build_maxtext_cli_args", return_value=["arg"])
def test_execute_train_distributed(mock_build_cli, mock_save_gin, mock_gen_gin, mock_build_dataloader):
    """Test _execute_train distributed."""
    import gemma_4_sql.backends.maxtext.train as train_module

    train_module.jax = MagicMock()
    train_module.jnp = MagicMock()
    train_module.optax = MagicMock()
    train_module.ocp = MagicMock()
    mock_maxtext_train = MagicMock()
    train_module.maxtext_train = mock_maxtext_train

    mock_build_dataloader.return_value = {"loader": [1, 2]}

    config = TrainingConfig(model_name="model", dataset="dataset", epochs=1)
    with patch("gemma_4_sql.backends.maxtext.train.save_maxtext_checkpoint"):
        status, loss = train_module._execute_train(config, local_step_mode=False, checkpoint_dir="dir")
    assert status == "completed"
    assert loss == 0.0
    mock_maxtext_train.main.assert_called_once_with(["arg"])

    # string config
    status, loss = train_module._execute_train("model", local_step_mode=False)
    assert status == "completed"
    assert loss == 0.0

    # check missing deps
    train_module.jax = None
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        train_module._execute_train(config)


def test_train_model():
    """Test train_model."""
    import gemma_4_sql.backends.maxtext.train as train_module

    train_module.jax = MagicMock()
    train_module.jnp = MagicMock()
    train_module.optax = MagicMock()
    train_module.Gemma4Model = MagicMock()

    config = TrainingConfig(model_name="model")
    config.extra_kwargs = {"local_step_mode": True, "checkpoint_dir": "dir2"}

    with patch("gemma_4_sql.backends.maxtext.train._execute_train") as mock_exec:
        mock_exec.return_value = ("completed", 0.5)
        res = train_module.train_model(config, checkpoint_dir="dir")
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5
        assert res["checkpoint_dir"] == "dir"

        # Test without checkpoint_dir
        config.extra_kwargs = {}
        res_no_dir = train_module.train_model(config)
        assert "checkpoint_dir" not in res_no_dir

        mock_exec.side_effect = ValueError("test error")
        res2 = train_module.train_model(config)
        assert "failed: test error" in res2["status"]

    train_module.jax = None
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        train_module.train_model(config)


def test_maxtext_train_successful_imports():
    """Test maxtext train successful imports."""
    import importlib
    import sys
    from unittest.mock import MagicMock

    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_maxtext = MagicMock()
    mock_gemma4 = MagicMock()
    mock_train = MagicMock()
    mock_maxtext.models = MagicMock()
    mock_maxtext.models.gemma4 = mock_gemma4
    mock_maxtext.train = mock_train
    mock_optax = MagicMock()
    mock_ocp = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "jax": mock_jax,
            "jax.numpy": mock_jnp,
            "maxtext": mock_maxtext,
            "maxtext.models": mock_maxtext.models,
            "maxtext.models.gemma4": mock_gemma4,
            "maxtext.train": mock_train,
            "optax": mock_optax,
            "orbax": MagicMock(),
            "orbax.checkpoint": mock_ocp,
        },
    ):
        import gemma_4_sql.backends.maxtext.train as train_module

        importlib.reload(train_module)
        assert train_module.jax is mock_jax
        assert train_module.jnp is not None
        assert train_module.optax is mock_optax
        assert train_module.ocp is not None
    importlib.reload(train_module)


def test_execute_train_no_checkpoint_dir():
    """Test _execute_train without checkpoint_dir to cover branch."""
    import gemma_4_sql.backends.maxtext.train as train_module
    from gemma_4_sql.type_hints import TrainingConfig

    train_module.jax = MagicMock()
    train_module.jnp = MagicMock()
    train_module.optax = MagicMock()
    train_module.Gemma4Model = MagicMock()

    config = TrainingConfig(model_name="model", dataset="dataset", epochs=1, batch_size=2)

    with patch("gemma_4_sql.backends.maxtext.train.build_dataloader") as mock_build:
        mock_build.return_value = {"loader": [1, 2]}
        with patch("gemma_4_sql.backends.maxtext.train._get_train_step_fn"):
            with patch("gemma_4_sql.backends.maxtext.train._run_training_epochs") as mock_run:
                mock_run.return_value = ("params", "opt_state", 0.5)
                # Local step mode without checkpoint_dir
                status, loss = train_module._execute_train(config, local_step_mode=True)
                assert status == "completed"
                assert loss == 0.5

    train_module.maxtext_train = MagicMock()
    with patch("gemma_4_sql.backends.maxtext.train.build_dataloader") as mock_build:
        mock_build.return_value = {"loader": [1, 2]}
        with patch("gemma_4_sql.backends.maxtext.config_generator.generate_maxtext_gin_config"), patch("gemma_4_sql.backends.maxtext.config_generator.save_maxtext_gin_config"), patch("gemma_4_sql.backends.maxtext.config_generator.build_maxtext_cli_args"):
            # Distributed mode without checkpoint_dir
            status, loss = train_module._execute_train(config, local_step_mode=False)
            assert status == "completed"
            assert loss == 0.0
