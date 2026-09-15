"""Tests validating true missing dependency error handling across backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import TrainingConfig


def test_missing_jax_dpo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX DPO returns fallback when jnn is missing."""
    import gemma_4_sql.backends.jax.dpo as jdpo

    monkeypatch.setattr(jdpo, "jnn", None)
    assert jdpo.dpo_loss({}, {}, {}, {}) == (0.0, 0.0, 0.0)


def test_missing_jax_train(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX train raises DependencyMissingError when optax is missing."""
    import gemma_4_sql.backends.jax.train as jtrain

    monkeypatch.setattr(jtrain, "optax", None)
    with pytest.raises(DependencyMissingError):
        jtrain.train_model(TrainingConfig(model_name="m", dataset="d"))


def test_missing_pytorch_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch inference raises DependencyMissingError when torch is missing."""
    import gemma_4_sql.backends.pytorch.inference as pinf

    monkeypatch.setattr(pinf, "torch", None)
    with pytest.raises(DependencyMissingError):
        pinf.generate_sql("m", "prompt")


def test_missing_pytorch_peft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch PEFT raises DependencyMissingError when peft is missing."""
    import gemma_4_sql.backends.pytorch.peft as ppeft

    monkeypatch.setattr(ppeft, "peft", None)
    with pytest.raises(DependencyMissingError):
        ppeft.apply_lora("m", ["q_proj"])


def test_missing_mlx_train(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MLX train raises DependencyMissingError when mlx is missing."""
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "mx", None)
    with pytest.raises(DependencyMissingError):
        mtrain.train_model(TrainingConfig(model_name="m", dataset="d"))


def test_missing_mlx_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MLX inference raises DependencyMissingError when mlx is missing."""
    import gemma_4_sql.backends.mlx.inference as minf

    monkeypatch.setattr(minf, "load", None)
    with pytest.raises(DependencyMissingError):
        minf.generate_sql("m", "prompt")


def test_missing_maxtext_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MaxText inference raises DependencyMissingError when Gemma4Model is missing."""
    import gemma_4_sql.backends.maxtext.inference as minf

    monkeypatch.setattr(minf, "Gemma4Model", None)
    with pytest.raises(DependencyMissingError):
        minf.generate_sql("m", "prompt")


def test_missing_maxtext_peft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MaxText PEFT raises DependencyMissingError when jax is missing."""
    import gemma_4_sql.backends.maxtext.peft as mpeft

    monkeypatch.setattr(mpeft, "jax", None)
    with pytest.raises(DependencyMissingError):
        mpeft.apply_lora("m", ["q_proj"])


def test_missing_keras_peft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras PEFT raises DependencyMissingError when keras is missing."""
    import gemma_4_sql.backends.keras.peft as kpeft

    monkeypatch.setattr(kpeft, "keras", None)
    with pytest.raises(DependencyMissingError):
        kpeft.apply_lora("m", ["q_proj"])


def test_missing_keras_quantize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test Keras quantize raises DependencyMissingError when keras is missing."""
    import gemma_4_sql.backends.keras.quantize as kquant

    monkeypatch.setattr(kquant, "keras", None)
    with pytest.raises(DependencyMissingError):
        kquant.quantize_model("m", "int8")
