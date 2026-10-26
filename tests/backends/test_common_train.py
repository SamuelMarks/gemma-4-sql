"""Unit and behavioral tests for common train."""

from unittest.mock import MagicMock

from gemma_4_sql.backends.common_train import (
    apply_nnx_optimizer_update,
    create_nnx_optimizer,
    generic_run_training_epochs,
)


def test_nnx_optimizer_helpers() -> None:
    """Verify create_nnx_optimizer and apply_nnx_optimizer_update support both Flax <0.11 and >=0.11 signatures."""
    mock_nnx = MagicMock()
    mock_nnx.Param = "Param"
    mock_nnx.Optimizer.return_value = "opt_with_wrt"
    assert create_nnx_optimizer(mock_nnx, "model", "tx") == "opt_with_wrt"

    def opt_no_wrt(model: str, tx: str, **kwargs: object) -> str:
        """Raise TypeError when wrt is passed."""
        if "wrt" in kwargs:
            raise TypeError("unexpected keyword wrt")
        return f"{model}_{tx}"

    mock_nnx.Optimizer.side_effect = opt_no_wrt
    assert create_nnx_optimizer(mock_nnx, "model", "tx") == "model_tx"

    opt_new = MagicMock()
    apply_nnx_optimizer_update(opt_new, "model", "grads")
    opt_new.update.assert_called_once_with("model", "grads")

    class LegacyOpt:
        """Legacy Flax NNX optimizer accepting only grads."""

        def __init__(self) -> None:
            """Initialize LegacyOpt."""
            self.updated_grads: object = None

        def update(self, grads: object) -> None:
            """Record grads."""
            self.updated_grads = grads

    opt_old = LegacyOpt()
    apply_nnx_optimizer_update(opt_old, "model", "grads")
    assert opt_old.updated_grads == "grads"


def test_generic_run_training_epochs_empty_dataloader() -> None:
    """Test with 0 epochs or empty dataloader."""
    # 0 epochs
    result = generic_run_training_epochs(0, [1, 2, 3], float)
    assert result == 0.0

    # empty dataloader
    result = generic_run_training_epochs(1, [], float)
    assert result == 0.0


def test_generic_run_training_epochs_standard() -> None:
    """Test standard case with multiple epochs."""
    dataloader = [1, 2, 3]
    # In each epoch, sum of loss is 1+2+3=6. Average = 6/3 = 2.0
    result = generic_run_training_epochs(2, dataloader, float)
    assert result == 2.0


def test_generic_run_training_epochs_single_batch() -> None:
    """Test with single batch."""
    dataloader = [42]
    result = generic_run_training_epochs(1, dataloader, float)
    assert result == 42.0


def test_run_train_wrapper() -> None:
    """Verify run_train_wrapper unpacks config, formats optional fields, and handles errors."""
    import pytest

    from gemma_4_sql.backends.common_train import run_train_wrapper
    from gemma_4_sql.exceptions import TrainingError
    from gemma_4_sql.type_hints import TrainingConfig

    cfg = TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1, learning_rate=1e-4)
    res_ok = run_train_wrapper(
        "jax",
        cfg,
        lambda: ("completed", 0.5),
        distributed_strategy="ddp",
        checkpoint_dir="/tmp/ckpt",
    )
    assert res_ok["status"] == "completed"
    assert res_ok["final_loss"] == 0.5
    assert res_ok.get("distributed_strategy") == "ddp"
    assert res_ok.get("checkpoint_dir") == "/tmp/ckpt"

    def _fail() -> tuple[str, float]:
        """Raise RuntimeError for testing error handling."""
        raise RuntimeError("train boom")

    res_fail = run_train_wrapper("jax", cfg, _fail)
    assert res_fail["status"] == "failed: train boom"

    with pytest.raises(TrainingError, match="train boom"):
        run_train_wrapper("jax", cfg, _fail, raise_on_error=True)

    from gemma_4_sql.exceptions import DependencyMissingError

    with pytest.raises(DependencyMissingError, match="missing dep"):
        run_train_wrapper("jax", cfg, lambda: (_ for _ in ()).throw(DependencyMissingError("missing dep")))
