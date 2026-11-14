"""Unit and behavioral tests for the MaxText training pipeline without MagicMock."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch

import jax
import optax
import pytest

import gemma_4_sql.backends.maxtext.inference as inference_module
import gemma_4_sql.backends.maxtext.train as train_module
from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.exceptions import DependencyMissingError, ExportError, TrainingError
from gemma_4_sql.type_hints import MaxTextTrainerState, TrainingConfig
from tests.conftest import requires_jax

pytestmark = requires_jax


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Clear MaxText caches before each training test."""
    inference_module._MODEL_CACHE.clear()
    inference_module._PARAMS_ONLY_CACHE.clear()


class TinyStatelessModel:
    """Minimal stateless Flax/MaxText-style model."""

    def __init__(self, _model_name: str = "tiny") -> None:
        """Initialize stateless model stub."""

    def init(self, _rng: jax.Array | None, _inputs: jax.Array | None) -> dict[str, jax.Array]:
        """Initialize embedding and output head arrays.

        Returns:
            A dictionary with `'embed'` and `'head'` JAX arrays.

        """
        return {
            "embed": jnp.ones((16, 4), dtype=jnp.float32) * 0.1,
            "head": jnp.ones((4, 16), dtype=jnp.float32) * 0.1,
        }

    def apply(self, params: Mapping[str, jax.Array], inputs: jax.Array) -> jax.Array:
        """Project token embeddings to vocabulary logits.

        Returns:
            Vocabulary logits projected from `inputs`.

        """
        emb = jnp.take(params["embed"], inputs, axis=0)
        return jnp.dot(emb, params["head"])


def test_real_maxtext_loss_and_train_step() -> None:
    """Verify `_loss_fn`, `_get_train_step_fn`, and `_run_training_epochs` run real updates with and without mask."""
    model = TinyStatelessModel()
    params = model.init(None, None)
    batch: dict[str, jax.Array] = {
        "inputs": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "targets": jnp.array([[2, 3, 4]], dtype=jnp.int32),
    }
    masked_batch: dict[str, jax.Array] = {
        "inputs": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "targets": jnp.array([[2, 3, 4]], dtype=jnp.int32),
        "loss_mask": jnp.array([[0.0, 1.0, 1.0]], dtype=jnp.float32),
    }

    loss = train_module._loss_fn(model, params, batch)
    masked_loss = train_module._loss_fn(model, params, masked_batch)
    assert float(loss) > 0.0
    assert float(masked_loss) > 0.0

    optimizer = optax.adamw(1e-2)
    opt_state = optimizer.init(params)
    step_fn: train_module.MaxTextStepCallable[dict[str, jax.Array], optax.OptState, dict[str, jax.Array]] = (
        train_module._get_train_step_fn(
            model,
            optimizer,
        )
    )

    state: MaxTextTrainerState[dict[str, jax.Array], optax.OptState, dict[str, jax.Array]] = MaxTextTrainerState(
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
    """Verify `_get_train_step_fn` returns un-jitted `train_step` when `jax` lacks `jit`."""
    model = TinyStatelessModel()
    optimizer = optax.adamw(1e-2)
    monkeypatch.setattr(train_module, "jax", None)
    step_fn = train_module._get_train_step_fn(model, optimizer)
    assert callable(step_fn)


def test_initialize_jax_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_initialize_jax_distributed` handles success, runtime errors, and missing `jax`."""
    recorded_kwargs: list[dict[str, str | int]] = []
    should_fail = False

    class _FakeDistributed:
        """Typed `jax.distributed` double."""

        def initialize(self, **kwargs: str | int) -> None:
            """Record kwargs or raise `RuntimeError`.

            Raises:
                RuntimeError: If `should_fail` is True.

            """
            if should_fail:
                raise RuntimeError("error")
            recorded_kwargs.append(dict(kwargs))

    class _FakeJax:
        """Typed `jax` double with `.distributed`."""

        distributed = _FakeDistributed()

    monkeypatch.setattr(train_module, "jax", _FakeJax())

    res = train_module._initialize_jax_distributed(coordinator_address="addr", num_processes=2, process_id=0)
    assert res is True
    assert recorded_kwargs == [{"coordinator_address": "addr", "num_processes": 2, "process_id": 0}]

    should_fail = True
    assert train_module._initialize_jax_distributed() is False

    monkeypatch.setattr(train_module, "jax", None)
    assert train_module._initialize_jax_distributed() is False


def test_save_maxtext_checkpoint_real_orbax(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `save_maxtext_checkpoint` persists real Orbax checkpoints and handles errors."""
    ckpt_dir = tmp_path / "ckpt"
    p_real = {"head": jnp.array([1.0, 2.0], dtype=jnp.float32)}
    s_real = {"step": jnp.array(1, dtype=jnp.int32)}
    res = train_module.save_maxtext_checkpoint(ckpt_dir, 1, p_real, s_real)
    assert res == ckpt_dir.resolve()
    assert (ckpt_dir / "1").exists()

    # Also save without opt_state
    ckpt_dir2 = tmp_path / "ckpt_no_opt"
    res2 = train_module.save_maxtext_checkpoint(ckpt_dir2, 2, p_real, None)
    assert res2 == ckpt_dir2.resolve()
    assert (ckpt_dir2 / "2").exists()

    with (
        patch.object(train_module.ocp, "CheckpointManager", side_effect=RuntimeError("error")),
        pytest.raises(ExportError, match="Failed to persist Orbax checkpoint"),
    ):
        train_module.save_maxtext_checkpoint(tmp_path / "fail_ckpt", 1, p_real)

    monkeypatch.setattr(train_module, "ocp", None)
    with pytest.raises(DependencyMissingError, match="Orbax checkpoint dependency"):
        train_module.save_maxtext_checkpoint(ckpt_dir, 1, p_real)


def test_execute_train_local_and_distributed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_execute_train` runs real JAX/Optax updates, caches trained params, and supports distributed Gin mode."""

    class _FakeMaxTextTrainModule:
        """Typed `maxtext.train` double."""

        def __init__(self) -> None:
            """Initialize recorded calls list."""
            self.calls: list[Sequence[str]] = []

        def main(self, argv: Sequence[str]) -> float:
            """Record CLI args and return `0.0` loss.

            Returns:
                A deterministic `0.0` loss value.

            """
            self.calls.append(list(argv))
            return 0.0

    fake_mt_train = _FakeMaxTextTrainModule()
    monkeypatch.setattr(train_module, "Gemma4Model", TinyStatelessModel)
    monkeypatch.setattr(train_module, "maxtext_train", fake_mt_train)

    batch: dict[str, jax.Array] = {
        "inputs": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "targets": jnp.array([[2, 3, 4]], dtype=jnp.int32),
    }
    config = TrainingConfig(model_name="model", dataset="dataset", epochs=1, batch_size=1, learning_rate=1e-2)

    with (
        patch.object(train_module, "build_dataloader", return_value={"loader": [batch]}),
        patch.object(train_module, "save_maxtext_checkpoint") as mock_save_ckpt,
    ):
        status, loss = train_module._execute_train(config, local_step_mode=True, checkpoint_dir=str(tmp_path / "dir"))
        assert status == "completed"
        assert loss > 0.0
        mock_save_ckpt.assert_called_once()
        assert inference_module.get_cached_maxtext_params("model") is not None

        status_no_ckpt, loss_no_ckpt = train_module._execute_train(config, local_step_mode=True)
        assert status_no_ckpt == "completed"
        assert loss_no_ckpt > 0.0

    # Distributed mode + synthetic fallback lambda execution
    with (
        patch.object(train_module, "build_dataloader", return_value={"loader": [batch]}),
        patch("gemma_4_sql.backends.maxtext.config_generator.generate_maxtext_gin_config", return_value="gin"),
        patch(
            "gemma_4_sql.backends.maxtext.config_generator.save_maxtext_gin_config", return_value=tmp_path / "gin.file"
        ),
        patch("gemma_4_sql.backends.maxtext.config_generator.build_maxtext_cli_args", return_value=["arg"]),
    ):
        status_dist, loss_dist = train_module._execute_train(config, local_step_mode=False)
        assert status_dist == "completed"
        assert loss_dist == 0.0

        status_str, loss_str = train_module._execute_train(
            "model",
            dataset="synthetic",
            local_step_mode=False,
            coordinator_address="127.0.0.1:1234",
            num_processes=2,
            process_id=0,
        )
        assert status_str == "completed"
        assert loss_str == 0.0

    # Also exercise synthetic fallback lambda when `build_dataloader` raises `DependencyMissingError`
    with patch.object(train_module, "build_dataloader", side_effect=DependencyMissingError("offline")):
        status_syn, loss_syn = train_module._execute_train("model", dataset="synthetic", local_step_mode=True)
        assert status_syn == "completed"
        assert loss_syn > 0.0

    with (
        patch.object(train_module, "build_dataloader", return_value={}),
        pytest.raises(ValueError, match="Invalid dataloader"),
    ):
        train_module._execute_train(config, local_step_mode=True)

    monkeypatch.setattr(train_module, "Gemma4Model", None)
    with (
        patch.object(train_module, "build_dataloader", return_value={"loader": [batch]}),
        pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"),
    ):
        train_module._execute_train(config, local_step_mode=True)

    monkeypatch.setattr(train_module, "jax", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        train_module._execute_train(config)


def test_train_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `train_model` returns structured results and handles errors and missing dependencies."""
    monkeypatch.setattr(train_module, "Gemma4Model", TinyStatelessModel)

    config = TrainingConfig(model_name="model", extra_kwargs={"local_step_mode": True, "checkpoint_dir": "dir2"})

    with patch.object(train_module, "_execute_train") as mock_exec:
        mock_exec.return_value = ("completed", 0.5)
        res = train_module.train_model(config, checkpoint_dir="dir")
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5
        assert res.get("checkpoint_dir") == "dir"

        config_no_dir = TrainingConfig(model_name="model")
        res_no_dir = train_module.train_model(config_no_dir)
        assert "checkpoint_dir" not in res_no_dir

        mock_exec.side_effect = ValueError("test error")
        res2 = train_module.train_model(config_no_dir)
        assert "failed: test error" in str(res2["status"])
        with pytest.raises(TrainingError, match="test error"):
            train_module.train_model(config_no_dir, raise_on_error=True)

    monkeypatch.setattr(train_module, "jax", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        train_module.train_model(config)
