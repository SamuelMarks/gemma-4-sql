"""Unit and behavioral tests for the MaxText training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.maxtext.train as train_module
from gemma_4_sql.exceptions import DependencyMissingError, ExportError
from gemma_4_sql.type_hints import TrainerState, TrainingConfig
from tests.conftest import requires_jax

pytestmark = requires_jax


def test_real_maxtext_loss_and_train_step() -> None:
    """Verify _loss_fn, _get_train_step_fn, and _run_training_epochs execute real JAX/Optax gradient updates."""
    import jax.numpy as jnp
    import optax

    class TinyStatelessModel:
        """Minimal stateless Flax/MaxText-style model."""

        def apply(self, params: dict[str, Any], inputs: Any) -> Any:
            """Project token embeddings to vocabulary logits."""
            emb = jnp.take(params["embed"], inputs, axis=0)
            return jnp.dot(emb, params["head"])

    model = TinyStatelessModel()
    params: dict[str, Any] = {
        "embed": jnp.ones((16, 4), dtype=jnp.float32) * 0.1,
        "head": jnp.ones((4, 16), dtype=jnp.float32) * 0.1,
    }
    batch: dict[str, Any] = {
        "inputs": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "targets": jnp.array([[2, 3, 4]], dtype=jnp.int32),
    }

    loss = train_module._loss_fn(model, params, batch)  # type: ignore[arg-type]
    assert float(loss) > 0.0

    optimizer = optax.adamw(1e-2)
    opt_state = optimizer.init(params)
    step_fn = train_module._get_train_step_fn(model, optimizer)  # type: ignore[arg-type]

    state = TrainerState(
        dataloader=[batch],
        epochs=1,
        train_step=step_fn,
        params=params,
        opt_state=opt_state,
    )
    new_params, _new_opt_state, final_loss = train_module._run_training_epochs(state)
    assert final_loss > 0.0
    assert not jnp.allclose(new_params["head"], params["head"])


def test_get_train_step_fn_without_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_train_step_fn returns un-jitted train_step when jax is None."""
    monkeypatch.setattr(train_module, "jax", None)
    step_fn = train_module._get_train_step_fn(MagicMock(), MagicMock())
    assert getattr(step_fn, "__name__", "") == "train_step"


def test_initialize_jax_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _initialize_jax_distributed handles success, runtime errors, and missing jax."""
    mock_jax = MagicMock()
    monkeypatch.setattr(train_module, "jax", mock_jax)

    res = train_module._initialize_jax_distributed(coordinator_address="addr", num_processes=2, process_id=0)
    assert res is True
    mock_jax.distributed.initialize.assert_called_with(coordinator_address="addr", num_processes=2, process_id=0)

    mock_jax.distributed.initialize.side_effect = RuntimeError("error")
    assert train_module._initialize_jax_distributed() is False

    monkeypatch.setattr(train_module, "jax", None)
    assert train_module._initialize_jax_distributed() is False


def test_save_maxtext_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify save_maxtext_checkpoint persists via Orbax CheckpointManager and handles errors."""
    mock_ocp = MagicMock()
    monkeypatch.setattr(train_module, "ocp", mock_ocp)

    ckpt_dir = tmp_path / "ckpt"
    res = train_module.save_maxtext_checkpoint(ckpt_dir, 1, {"p": 1}, {"s": 1})  # type: ignore[arg-type]
    assert res == ckpt_dir.resolve()
    mock_ocp.CheckpointManager.assert_called_once()

    mock_ocp.CheckpointManager.side_effect = RuntimeError("error")
    with pytest.raises(ExportError, match="Failed to persist Orbax checkpoint"):
        train_module.save_maxtext_checkpoint(ckpt_dir, 1, {"p": 1})  # type: ignore[arg-type]

    monkeypatch.setattr(train_module, "ocp", None)
    with pytest.raises(DependencyMissingError, match="Orbax checkpoint dependency"):
        train_module.save_maxtext_checkpoint(ckpt_dir, 1, {"p": 1})  # type: ignore[arg-type]


def test_execute_train_local_and_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_train supports both local step mode and distributed gin orchestration."""
    monkeypatch.setattr(train_module, "jax", MagicMock())
    monkeypatch.setattr(train_module, "jnp", MagicMock())
    monkeypatch.setattr(train_module, "optax", MagicMock())
    monkeypatch.setattr(train_module, "maxtext_train", MagicMock())
    monkeypatch.setattr(train_module, "Gemma4Model", MagicMock())
    monkeypatch.setattr(train_module, "ocp", MagicMock())

    config = TrainingConfig(model_name="model", dataset="dataset", epochs=1, batch_size=2)

    with (
        patch.object(train_module, "build_dataloader", return_value={"loader": [1, 2]}),
        patch.object(train_module, "_get_train_step_fn"),
        patch.object(train_module, "_run_training_epochs", return_value=("params", "opt_state", 0.5)),
        patch.object(train_module, "save_maxtext_checkpoint"),
    ):
        status, loss = train_module._execute_train(config, local_step_mode=True, checkpoint_dir="dir")
        assert status == "completed"
        assert loss == 0.5

        status_no_ckpt, loss_no_ckpt = train_module._execute_train(config, local_step_mode=True)
        assert status_no_ckpt == "completed"
        assert loss_no_ckpt == 0.5

    # Distributed mode
    with (
        patch.object(train_module, "build_dataloader", return_value={"loader": [1, 2]}),
        patch("gemma_4_sql.backends.maxtext.config_generator.generate_maxtext_gin_config", return_value="gin"),
        patch("gemma_4_sql.backends.maxtext.config_generator.save_maxtext_gin_config", return_value=Path("gin.file")),
        patch("gemma_4_sql.backends.maxtext.config_generator.build_maxtext_cli_args", return_value=["arg"]),
        patch.object(train_module, "save_maxtext_checkpoint"),
    ):
        status_dist, loss_dist = train_module._execute_train(config, local_step_mode=False, checkpoint_dir="dir")
        assert status_dist == "completed"
        assert loss_dist == 0.0

        status_str, loss_str = train_module._execute_train(
            "model",
            local_step_mode=False,
            coordinator_address="127.0.0.1:1234",
            num_processes=2,
            process_id=0,
        )
        assert status_str == "completed"
        assert loss_str == 0.0

    # Invalid dataloader
    with (
        patch.object(train_module, "build_dataloader", return_value={}),
        pytest.raises(ValueError, match="Invalid dataloader"),
    ):
        train_module._execute_train(config, local_step_mode=True)

    # Missing Gemma4Model
    monkeypatch.setattr(train_module, "Gemma4Model", None)
    with (
        patch.object(train_module, "build_dataloader", return_value={"loader": [1, 2]}),
        pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"),
    ):
        train_module._execute_train(config, local_step_mode=True)

    # Missing jax
    monkeypatch.setattr(train_module, "jax", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        train_module._execute_train(config)


def test_train_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train_model returns structured results and handles errors and missing dependencies."""
    monkeypatch.setattr(train_module, "jax", MagicMock())
    monkeypatch.setattr(train_module, "jnp", MagicMock())
    monkeypatch.setattr(train_module, "optax", MagicMock())
    monkeypatch.setattr(train_module, "Gemma4Model", MagicMock())

    config = TrainingConfig(model_name="model")
    config.extra_kwargs = {"local_step_mode": True, "checkpoint_dir": "dir2"}

    with patch.object(train_module, "_execute_train") as mock_exec:
        mock_exec.return_value = ("completed", 0.5)
        res = train_module.train_model(config, checkpoint_dir="dir")
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5
        assert res["checkpoint_dir"] == "dir"

        config.extra_kwargs = {}
        res_no_dir = train_module.train_model(config)
        assert "checkpoint_dir" not in res_no_dir

        mock_exec.side_effect = ValueError("test error")
        res2 = train_module.train_model(config)
        assert "failed: test error" in res2["status"]

    monkeypatch.setattr(train_module, "jax", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        train_module.train_model(config)
