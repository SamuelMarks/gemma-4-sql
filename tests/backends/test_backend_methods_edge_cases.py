"""Tests for backend edge cases with concrete assertions and no exception swallowing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import TrainingConfig

has_keras = importlib.util.find_spec("keras") is not None
has_tf = importlib.util.find_spec("tensorflow") is not None


def test_jax_dpo_missing_jnn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify JAX `dpo_loss` returns zero tuple when `jnn` is missing."""
    from typing import Any

    import gemma_4_sql.backends.jax.dpo as jdpo

    monkeypatch.setattr(jdpo, "jnn", None)
    empty_arg: Any = {}
    res = jdpo.dpo_loss(empty_arg, empty_arg, empty_arg, empty_arg)
    assert res == (0.0, 0.0, 0.0)


def test_jax_export_missing_ocp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify JAX `export_model` raises `DependencyMissingError` when `ocp` is missing."""
    import gemma_4_sql.backends.jax.export as jexp

    monkeypatch.setattr(jexp, "ocp", None)
    with pytest.raises(DependencyMissingError):
        jexp.export_model("foo", str(tmp_path / "jax_export"))


def test_jax_export_model1(tmp_path: Path) -> None:
    """Verify JAX `export_model` exports a tiny test model (`model1`) with Orbax."""
    import gemma_4_sql.backends.jax.export as jexp

    res = jexp.export_model("model1", str(tmp_path / "jax_export"))
    assert res["status"] == "exported_with_orbax"
    assert res["format"] == "orbax/saved_model"


def test_jax_inference_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify JAX `generate_sql` raises `DependencyMissingError` when `Gemma4ForCausalLM` is missing."""
    import gemma_4_sql.backends.jax.inference as jinf

    monkeypatch.setattr(jinf, "Gemma4ForCausalLM", None)
    with pytest.raises(DependencyMissingError):
        jinf.generate_sql("foo", "bar")


def test_jax_quantize_missing_jnp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify JAX `quantize_model` raises `DependencyMissingError` when `jnp` is missing."""
    import gemma_4_sql.backends.jax.quantize as jquant

    monkeypatch.setattr(jquant, "jnp", None)
    with pytest.raises(DependencyMissingError):
        jquant.quantize_model("foo", "int8")


def test_jax_train_missing_optax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify JAX `train_model` raises `DependencyMissingError` when `optax` is missing."""
    import gemma_4_sql.backends.jax.train as jtrain

    monkeypatch.setattr(jtrain, "optax", None)
    with pytest.raises(DependencyMissingError):
        jtrain.train_model(TrainingConfig(model_name="foo", dataset="bar"))


def test_jax_train_missing_nnx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify JAX `train_model` raises `DependencyMissingError` when `nnx` is missing."""
    import gemma_4_sql.backends.jax.train as jtrain

    monkeypatch.setattr(jtrain, "nnx", None)
    with pytest.raises(DependencyMissingError):
        jtrain.train_model(TrainingConfig(model_name="foo", dataset="bar"))


def test_keras_export_missing_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify Keras `export_model` raises `DependencyMissingError` when `keras` is missing."""
    import gemma_4_sql.backends.keras.export as kexp

    monkeypatch.setattr(kexp, "keras", None)
    with pytest.raises(DependencyMissingError):
        kexp.export_model("foo", str(tmp_path / "keras_export"))


def test_keras_inference_missing_tf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Keras `generate_sql` raises `DependencyMissingError` when `tf` is missing."""
    import gemma_4_sql.backends.keras.inference as kinf

    monkeypatch.setattr(kinf, "tf", None)
    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
        kinf.generate_sql("foo", "bar")


def test_keras_train_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Keras `train_model` raises `DependencyMissingError` when `keras` is missing."""
    import gemma_4_sql.backends.keras.train as ktrain

    monkeypatch.setattr(ktrain, "keras", None)
    with pytest.raises(DependencyMissingError):
        ktrain.train_model(TrainingConfig(model_name="foo", dataset="bar"))


def test_maxtext_export_missing_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify MaxText `export_model` raises `DependencyMissingError` when `ocp` is missing."""
    import gemma_4_sql.backends.maxtext.export as mexp

    monkeypatch.setattr(mexp, "ocp", None)
    with pytest.raises(DependencyMissingError, match="MaxText export dependencies"):
        mexp.export_model("foo", str(tmp_path / "maxtext_export"))


def test_maxtext_train_missing_optax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MaxText `train_model` raises `DependencyMissingError` when `optax` is missing."""
    import gemma_4_sql.backends.maxtext.train as mtrain

    monkeypatch.setattr(mtrain, "optax", None)
    with pytest.raises(DependencyMissingError):
        mtrain.train_model(TrainingConfig(model_name="foo", dataset="bar"))


def test_pytorch_export_missing_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify PyTorch `export_model` raises `DependencyMissingError` when `torch` is missing."""
    import gemma_4_sql.backends.pytorch.export as pexp

    monkeypatch.setattr(pexp, "torch", None)
    with pytest.raises(DependencyMissingError):
        pexp.export_model("foo", str(tmp_path / "pytorch_export"))


def test_pytorch_train_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify PyTorch `train_model` raises `DependencyMissingError` when `torch` is missing."""
    import gemma_4_sql.backends.pytorch.train as ptrain

    monkeypatch.setattr(ptrain, "torch", None)
    with pytest.raises(DependencyMissingError):
        ptrain.train_model(TrainingConfig(model_name="foo", dataset="bar"))


def test_db_engine_missing_postgres_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `LiveDatabaseEngine` raises `DependencyMissingError` (an `ImportError` subclass) when PostgreSQL drivers are missing."""
    import gemma_4_sql.sdk.adapters.postgres_adapter as pga
    from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine

    monkeypatch.setattr(pga, "psycopg2", None)
    monkeypatch.setattr(pga, "asyncpg", None)
    with pytest.raises(DependencyMissingError):
        LiveDatabaseEngine(db_path="postgresql://localhost:5432/db", db_type="postgresql")
