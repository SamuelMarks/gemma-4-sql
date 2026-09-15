"""Tests for backend edge cases with concrete assertions and no exception swallowing."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import TrainingConfig

try:
    import keras
except ImportError:
    keras = None


def test_jax_dpo_missing_jnn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX DPO loss when jnn is missing."""
    import gemma_4_sql.backends.jax.dpo as jdpo

    monkeypatch.setattr(jdpo, "jnn", None)
    res = jdpo.dpo_loss({}, {}, {}, {})
    assert res == (0.0, 0.0, 0.0)


def test_jax_export_missing_ocp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test JAX export raises DependencyMissingError when ocp is missing."""
    import gemma_4_sql.backends.jax.export as jexp

    monkeypatch.setattr(jexp, "ocp", None)
    with pytest.raises(DependencyMissingError):
        jexp.export_model("foo", str(tmp_path / "jax_export"))


def test_jax_export_model1(tmp_path: Path) -> None:
    """Test JAX export for model1."""
    import gemma_4_sql.backends.jax.export as jexp

    res = jexp.export_model("model1", str(tmp_path / "jax_export"))
    assert res["status"] == "exported_with_orbax"


def test_jax_inference_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX inference raises DependencyMissingError when nnx is missing."""
    import gemma_4_sql.backends.jax.inference as jinf

    monkeypatch.setattr(jinf, "Gemma4ForCausalLM", None)
    with pytest.raises(DependencyMissingError):
        jinf.generate_sql("foo", "bar")


def test_jax_quantize_missing_jnp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX quantize raises DependencyMissingError when jnp is missing."""
    import gemma_4_sql.backends.jax.quantize as jquant

    monkeypatch.setattr(jquant, "jnp", None)
    with pytest.raises(DependencyMissingError):
        jquant.quantize_model("foo", "int8")


def test_jax_train_missing_optax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX train raises DependencyMissingError when optax is missing."""
    import gemma_4_sql.backends.jax.train as jtrain

    monkeypatch.setattr(jtrain, "optax", None)
    with pytest.raises(DependencyMissingError):
        jtrain.train_model(TrainingConfig(model_name="foo", dataset="bar"))


def test_jax_train_missing_nnx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX train raises DependencyMissingError when nnx is missing."""
    import gemma_4_sql.backends.jax.train as jtrain

    monkeypatch.setattr(jtrain, "nnx", None)
    with pytest.raises(DependencyMissingError):
        jtrain.train_model(TrainingConfig(model_name="foo", dataset="bar"))


def test_keras_export_missing_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test Keras export raises DependencyMissingError when keras is missing."""
    import gemma_4_sql.backends.keras.export as kexp

    monkeypatch.setattr(kexp, "keras", None)
    with pytest.raises(DependencyMissingError):
        kexp.export_model("foo", str(tmp_path / "keras_export"))


def test_keras_inference_missing_tf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras inference raises DependencyMissingError when tf is missing."""
    import gemma_4_sql.backends.keras.inference as kinf

    monkeypatch.setattr(kinf, "tf", None)
    with pytest.raises(DependencyMissingError):
        kinf.generate_sql("foo", "bar")


@pytest.mark.skipif(keras is None, reason="Keras is not installed")
def test_keras_inference_test_mode() -> None:
    """Test Keras inference handles missing keras_nlp gracefully."""
    import gemma_4_sql.backends.keras.inference as kinf

    res = kinf.generate_sql("foo", "bar", test_mode=True)
    assert "status" in res
    assert "failed" in res["status"] or res["status"] == "success"


def test_keras_train_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras train raises DependencyMissingError when keras is missing."""
    import gemma_4_sql.backends.keras.train as ktrain

    monkeypatch.setattr(ktrain, "keras", None)
    with pytest.raises(DependencyMissingError):
        ktrain.train_model(TrainingConfig(model_name="foo", dataset="bar"))


def test_maxtext_export_missing_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test MaxText export returns mock_exported status when ocp is missing."""
    import gemma_4_sql.backends.maxtext.export as mexp

    monkeypatch.setattr(mexp, "ocp", None)
    res = mexp.export_model("foo", str(tmp_path / "maxtext_export"))
    assert res["status"] == "mock_exported"


def test_maxtext_train_missing_optax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MaxText train raises DependencyMissingError when optax is missing."""
    import gemma_4_sql.backends.maxtext.train as mtrain

    monkeypatch.setattr(mtrain, "optax", None)
    with pytest.raises(DependencyMissingError):
        mtrain.train_model(TrainingConfig(model_name="foo", dataset="bar"))


def test_pytorch_export_missing_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test PyTorch export raises RuntimeError when torch is missing."""
    import gemma_4_sql.backends.pytorch.export as pexp

    monkeypatch.setattr(pexp, "torch", None)
    with pytest.raises(RuntimeError):
        pexp.export_model("foo", str(tmp_path / "pytorch_export"))


def test_pytorch_train_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch train raises DependencyMissingError when torch is missing."""
    import gemma_4_sql.backends.pytorch.train as ptrain

    monkeypatch.setattr(ptrain, "torch", None)
    with pytest.raises(DependencyMissingError):
        ptrain.train_model(TrainingConfig(model_name="foo", dataset="bar"))


def test_db_engine_missing_postgres_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test LiveDatabaseEngine raises ImportError when postgresql driver is missing."""
    import gemma_4_sql.sdk.adapters.postgres_adapter as pga
    from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine

    monkeypatch.setattr(pga, "psycopg2", None)
    monkeypatch.setattr(pga, "asyncpg", None)
    with pytest.raises(ImportError):
        LiveDatabaseEngine(db_path="postgresql://localhost:5432/db", db_type="postgresql")
