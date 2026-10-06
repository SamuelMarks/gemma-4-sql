import importlib
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.maxtext import train as maxtext_train_module
from gemma_4_sql.exceptions import DependencyMissingError, ExportError
from gemma_4_sql.type_hints import TrainingConfig


@pytest.fixture(autouse=True)
def mock_deps(monkeypatch):
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_optax = MagicMock()
    mock_maxtext_train = MagicMock()
    mock_gemma4_class = MagicMock()
    mock_ocp = MagicMock()

    def _mean(x):
        return x

    mock_jnp.mean.side_effect = None
    mock_jnp.mean.return_value = "mean_loss"

    def mock_value_and_grad(f):
        def inner(*args, **kwargs):
            try:
                f(*args, **kwargs)
            except Exception:  # noqa: S110, BLE001
                pass
            return ("loss_mock", "grads_mock")

        return inner

    mock_jax.value_and_grad.side_effect = mock_value_and_grad

    mock_optimizer = MagicMock()
    mock_optimizer.update.return_value = ("updates", "new_opt_state")
    mock_optax.adamw.return_value = mock_optimizer
    mock_optax.apply_updates.return_value = "new_params"

    mock_jax.distributed = MagicMock()

    monkeypatch.setattr(maxtext_train_module, "jax", mock_jax)
    monkeypatch.setattr(maxtext_train_module, "jnp", mock_jnp)
    monkeypatch.setattr(maxtext_train_module, "optax", mock_optax)
    monkeypatch.setattr(maxtext_train_module, "maxtext_train", mock_maxtext_train)
    monkeypatch.setattr(maxtext_train_module, "Gemma4Model", mock_gemma4_class)
    monkeypatch.setattr(maxtext_train_module, "ocp", mock_ocp)

    yield {"jax": mock_jax, "jnp": mock_jnp, "optax": mock_optax, "maxtext_train": mock_maxtext_train, "gemma4": mock_gemma4_class, "ocp": mock_ocp, "optimizer": mock_optimizer}


def test_loss_fn(mock_deps):
    mock_model = MagicMock()
    mock_model.apply.return_value = "logits"

    batch = {"inputs": "in", "targets": "tar"}

    mock_optax = mock_deps["optax"]
    mock_optax.softmax_cross_entropy_with_integer_labels.return_value = "loss_val"
    mock_jnp = mock_deps["jnp"]
    mock_jnp.mean.return_value = "mean_loss"

    loss = maxtext_train_module._loss_fn(mock_model, "params", batch)

    assert loss == "mean_loss"


def test_get_train_step_fn(mock_deps):
    mock_jax = mock_deps["jax"]
    mock_jax.jit.side_effect = lambda f: f

    mock_model = MagicMock()
    mock_optimizer = mock_deps["optimizer"]

    train_step = maxtext_train_module._get_train_step_fn(mock_model, mock_optimizer)
    params, _opt_state, _loss = train_step("params", "opt_state", {"batch": 1})
    assert params == "new_params"


def test_get_train_step_fn_no_jit(mock_deps, monkeypatch):
    mock_jax = mock_deps["jax"]
    del mock_jax.jit

    mock_model = MagicMock()
    mock_optimizer = mock_deps["optimizer"]

    train_step = maxtext_train_module._get_train_step_fn(mock_model, mock_optimizer)
    params, _opt_state, _loss = train_step("params", "opt_state", {"batch": 1})
    assert params == "new_params"


def test_get_train_step_fn_no_jax(mock_deps, monkeypatch):
    monkeypatch.setattr(maxtext_train_module, "jax", None)
    mock_model = MagicMock()
    mock_optimizer = mock_deps["optimizer"]
    train_step = maxtext_train_module._get_train_step_fn(mock_model, mock_optimizer)
    assert train_step is not None


def test_run_training_epochs(mock_deps):
    class DummyLoss:
        def item(self):
            return 42.0

    mock_train_step = MagicMock(return_value=("new_p", "new_o", DummyLoss()))

    state = MagicMock()
    state.params = "p0"
    state.opt_state = "o0"
    state.train_step = mock_train_step
    state.epochs = 1
    state.dataloader = [{"a": 1}, {"b": 2}]

    def fake_run(epochs, dataloader, process_batch):
        return process_batch({"batch": 1})

    with patch("gemma_4_sql.backends.maxtext.train.generic_run_training_epochs", side_effect=fake_run):
        p, _o, _loss = maxtext_train_module._run_training_epochs(state)
        assert p == "new_p"


def test_initialize_jax_distributed(mock_deps):
    mock_deps["jax"]
    res = maxtext_train_module._initialize_jax_distributed(coordinator_address="localhost:1234", num_processes=2, process_id=0)
    assert res is True


def test_initialize_jax_distributed_failure(mock_deps):
    mock_jax = mock_deps["jax"]
    mock_jax.distributed.initialize.side_effect = RuntimeError("init failed")
    res = maxtext_train_module._initialize_jax_distributed()
    assert res is False


def test_initialize_jax_distributed_missing_jax(mock_deps, monkeypatch):
    monkeypatch.setattr(maxtext_train_module, "jax", None)
    res = maxtext_train_module._initialize_jax_distributed()
    assert res is False


def test_save_maxtext_checkpoint_missing_ocp(monkeypatch):
    monkeypatch.setattr(maxtext_train_module, "ocp", None)
    with pytest.raises(DependencyMissingError, match="Orbax checkpoint dependency"):
        maxtext_train_module.save_maxtext_checkpoint("dir", 1, {})


def test_save_maxtext_checkpoint_success(mock_deps, tmp_path):
    mock_ocp = mock_deps["ocp"]
    mock_mngr = MagicMock()
    mock_ocp.CheckpointManager.return_value.__enter__.return_value = mock_mngr
    path = maxtext_train_module.save_maxtext_checkpoint(tmp_path, 5, {"p": 1}, {"o": 2})
    assert path == tmp_path


def test_save_maxtext_checkpoint_success_no_opt(mock_deps, tmp_path):
    mock_ocp = mock_deps["ocp"]
    mock_mngr = MagicMock()
    mock_ocp.CheckpointManager.return_value.__enter__.return_value = mock_mngr
    path = maxtext_train_module.save_maxtext_checkpoint(tmp_path, 5, {"p": 1})
    assert path == tmp_path


def test_save_maxtext_checkpoint_failure(mock_deps, tmp_path):
    mock_ocp = mock_deps["ocp"]
    mock_ocp.CheckpointManager.side_effect = Exception("save failed")
    with pytest.raises(ExportError, match="Failed to persist Orbax checkpoint"):
        maxtext_train_module.save_maxtext_checkpoint(tmp_path, 5, {})


def test_execute_train_missing_deps(monkeypatch):
    monkeypatch.setattr(maxtext_train_module, "jax", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing for training"):
        maxtext_train_module._execute_train("gemma-4")


def test_execute_train_invalid_dataloader(mock_deps):
    with patch("gemma_4_sql.backends.maxtext.train.build_dataloader", return_value={}), pytest.raises(ValueError, match="Invalid dataloader"):
        maxtext_train_module._execute_train("gemma-4")


@patch("gemma_4_sql.backends.maxtext.train.build_dataloader")
@patch("gemma_4_sql.backends.maxtext.config_generator.generate_maxtext_gin_config", return_value="gin")
@patch("gemma_4_sql.backends.maxtext.config_generator.save_maxtext_gin_config", return_value="gin_path")
@patch("gemma_4_sql.backends.maxtext.config_generator.build_maxtext_cli_args", return_value=["--arg"])
def test_execute_train_distributed(mock_cli, mock_save, mock_gen, mock_build_dl, mock_deps, tmp_path):
    mock_build_dl.return_value = {"loader": [1, 2, 3]}
    status, _loss = maxtext_train_module._execute_train("gemma-4", local_step_mode=False, checkpoint_dir=str(tmp_path))
    assert status == "completed"


@patch("gemma_4_sql.backends.maxtext.train.build_dataloader")
@patch("gemma_4_sql.backends.maxtext.config_generator.generate_maxtext_gin_config", return_value="gin")
@patch("gemma_4_sql.backends.maxtext.config_generator.save_maxtext_gin_config", return_value="gin_path")
@patch("gemma_4_sql.backends.maxtext.config_generator.build_maxtext_cli_args", return_value=["--arg"])
def test_execute_train_distributed_no_ckpt(mock_cli, mock_save, mock_gen, mock_build_dl, mock_deps, tmp_path):
    mock_build_dl.return_value = {"loader": [1, 2, 3]}
    status, _loss = maxtext_train_module._execute_train("gemma-4", local_step_mode=False)
    assert status == "completed"


@patch("gemma_4_sql.backends.maxtext.train.build_dataloader")
@patch("gemma_4_sql.backends.maxtext.train._run_training_epochs")
def test_execute_train_local(mock_run, mock_build_dl, mock_deps, tmp_path):
    mock_build_dl.return_value = {"loader": [1, 2, 3]}
    mock_run.return_value = ("p", "o", 42.0)

    mock_model_cls = mock_deps["gemma4"]
    mock_model = MagicMock()
    mock_model.init.return_value = "init_params"
    mock_model_cls.return_value = mock_model

    status, _loss = maxtext_train_module._execute_train(TrainingConfig(model_name="gemma-4"), local_step_mode=True, checkpoint_dir=str(tmp_path))
    assert status == "completed"


@patch("gemma_4_sql.backends.maxtext.train.build_dataloader")
@patch("gemma_4_sql.backends.maxtext.train._run_training_epochs")
def test_execute_train_local_no_ckpt(mock_run, mock_build_dl, mock_deps, tmp_path):
    mock_build_dl.return_value = {"loader": [1, 2, 3]}
    mock_run.return_value = ("p", "o", 42.0)
    mock_model_cls = mock_deps["gemma4"]
    mock_model = MagicMock()
    mock_model.init.return_value = "init_params"
    mock_model_cls.return_value = mock_model
    status, _loss = maxtext_train_module._execute_train(TrainingConfig(model_name="gemma-4"), local_step_mode=True)
    assert status == "completed"


def test_execute_train_local_missing_gemma4(mock_deps, monkeypatch):
    monkeypatch.setattr(maxtext_train_module, "Gemma4Model", None)
    with patch("gemma_4_sql.backends.maxtext.train.build_dataloader", return_value={"loader": [1]}), pytest.raises(DependencyMissingError, match="MaxText dependencies are missing for training"):
        maxtext_train_module._execute_train("gemma-4", local_step_mode=True)


def test_train_model_missing_deps(monkeypatch):
    monkeypatch.setattr(maxtext_train_module, "jax", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        maxtext_train_module.train_model(TrainingConfig(model_name="gemma-4"))


def test_train_model_success():
    with patch.object(maxtext_train_module, "_execute_train", return_value=("completed", 10.0)):
        res = maxtext_train_module.train_model(TrainingConfig(model_name="gemma-4"), checkpoint_dir="/tmp/ckpt")
    assert res["status"] == "completed"


def test_train_model_failure():
    with patch.object(maxtext_train_module, "_execute_train", side_effect=ValueError("fail")):
        res = maxtext_train_module.train_model(TrainingConfig(model_name="gemma-4"))
    assert "failed: fail" in res["status"]


def test_train_import_error():
    import sys
    from unittest.mock import patch

    import gemma_4_sql.backends.maxtext.train as train_mod

    with patch.dict(sys.modules, {"optax": None}):
        importlib.reload(train_mod)
        assert train_mod.optax is None
    importlib.reload(train_mod)
