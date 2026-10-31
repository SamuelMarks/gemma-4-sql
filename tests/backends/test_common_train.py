"""Unit and behavioral tests for common train."""

from gemma_4_sql.backends.common_train import (
    apply_nnx_optimizer_update,
    create_nnx_optimizer,
    generic_run_training_epochs,
    resolve_training_dataloader,
)


def test_nnx_optimizer_helpers() -> None:
    """Verify create_nnx_optimizer and apply_nnx_optimizer_update invoke standardized Flax NNX APIs."""

    class _MockNNX:
        """Typed NNX module stub."""

        Param: type = int

        @staticmethod
        def Optimizer(model: str, tx: str, *, wrt: type | str) -> str:
            """Return formatted optimizer identifier."""
            return f"{model}_{tx}_{wrt.__name__ if isinstance(wrt, type) else wrt}"

    assert create_nnx_optimizer(_MockNNX(), "model", "tx") == "model_tx_int"

    calls: list[tuple[str, str]] = []

    class _StandardOpt:
        """Optimizer stub accepting `(model, grads)`."""

        def update(self, model: str, grads: str, /) -> None:
            """Record `(model, grads)` update call."""
            calls.append((model, grads))

    apply_nnx_optimizer_update(_StandardOpt(), "model", "grads")
    assert calls == [("model", "grads")]


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


def test_resolve_training_dataloader() -> None:
    """Verify resolve_training_dataloader handles loaded, synthetic fallback, invalid loader, and strict dataset modes."""
    import pytest

    from gemma_4_sql.exceptions import DependencyMissingError, ETLError
    from gemma_4_sql.type_hints import ETLResultDict

    def _loaded_fn() -> ETLResultDict:
        """Return a valid loaded ETLResultDict."""
        return {"backend": "jax", "status": "loaded", "dataset": "ds", "split": "train", "batch_size": 1, "loader": [{"inputs": [1]}]}

    def _none_loader_fn() -> ETLResultDict:
        """Return an ETLResultDict with loader=None."""
        return {"backend": "jax", "status": "loaded", "dataset": "ds", "split": "train", "batch_size": 1, "loader": None}

    def _err_loader_fn() -> ETLResultDict:
        """Raise ETLError."""
        raise ETLError("bad dataset")

    loader1, src1 = resolve_training_dataloader("real_ds", _loaded_fn, lambda: [{"inputs": [0]}])
    assert list(loader1) == [{"inputs": [1]}]
    assert src1 == "loaded"

    loader2, src2 = resolve_training_dataloader("synthetic", _none_loader_fn, lambda: [{"inputs": [0]}])
    assert list(loader2) == [{"inputs": [0]}]
    assert src2 == "synthetic_fallback"

    with pytest.raises(ValueError, match="Invalid dataloader"):
        resolve_training_dataloader("real_ds", _none_loader_fn, lambda: [{"inputs": [0]}])

    loader3, src3 = resolve_training_dataloader("synthetic", _err_loader_fn, lambda: [{"inputs": [0]}])
    assert list(loader3) == [{"inputs": [0]}]
    assert src3 == "synthetic_fallback"

    with pytest.raises(ETLError, match="bad dataset"):
        resolve_training_dataloader("real_ds", _err_loader_fn, lambda: [{"inputs": [0]}])

    with pytest.raises(ETLError, match="bad dataset"):
        resolve_training_dataloader("synthetic", _err_loader_fn, lambda: [{"inputs": [0]}], strict_dataset=True)

    loader4, src4 = resolve_training_dataloader(
        "real_ds",
        lambda: (_ for _ in ()).throw(DependencyMissingError("no grain")),
        lambda: [{"inputs": [0]}],
        allow_synthetic_fallback=True,
    )
    assert list(loader4) == [{"inputs": [0]}]
    assert src4 == "synthetic_fallback"
