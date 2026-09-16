"""Tests for MaxText PEFT / LoRA implementation."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import gemma_4_sql.backends.maxtext.peft as pt
from gemma_4_sql.backends.maxtext.peft import (
    apply_lora,
    count_maxtext_parameters,
    create_maxtext_lora_optimizer,
    load_maxtext_adapters,
    merge_lora_weights,
    save_maxtext_adapters,
    segregate_adapter_params,
    transform_params_to_lora,
)
from gemma_4_sql.exceptions import DependencyMissingError


class MockJnp:
    """Mock JAX numpy for testing error and mock paths."""

    int32 = 1

    @staticmethod
    def zeros(_shape: object, **_kwargs: object) -> object:
        """Return dummy zero list."""
        return [0]


class MockJaxRandom:
    """Mock JAX random module."""

    @staticmethod
    def mock_prngkey(seed: object) -> object:
        """Return dummy PRNGKey."""
        return seed

    PRNGKey = mock_prngkey


class MockJax:
    """Mock JAX module."""

    random = MockJaxRandom()


class MockGemma4Model:
    """Mock Gemma4Model."""

    def __init__(self, name: object) -> None:
        """Initialize mock model."""

    def init(self, _rng: object, _inputs: object) -> object:
        """Return mock parameter string."""
        return "params"


def test_apply_lora_maxtext_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test apply_lora raises DependencyMissingError when JAX is missing."""
    monkeypatch.setattr(pt, "jax", None)
    with pytest.raises(DependencyMissingError):
        pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)


def test_apply_lora_maxtext_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test apply_lora succeeds with mock model."""
    monkeypatch.setattr(pt, "jax", MockJax())
    monkeypatch.setattr(pt, "jnp", MockJnp())
    monkeypatch.setattr(pt, "Gemma4Model", MockGemma4Model)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    assert res["status"] == "completed"


def test_apply_lora_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test apply_lora captures runtime exceptions cleanly."""
    monkeypatch.setattr(pt, "jax", MockJax())
    monkeypatch.setattr(pt, "jnp", MockJnp())
    monkeypatch.setattr(pt, "Gemma4Model", MockGemma4Model)

    def raise_err(*_args: object, **_kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockJnp, "zeros", raise_err)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    assert "failed" in str(res["status"])


def test_transform_params_to_lora_numerical() -> None:
    """Test transform_params_to_lora properly injects adapter matrices."""
    params = {
        "decoder": {
            "layers_0": {
                "q_proj": {"kernel": jnp.ones((8, 16))},
                "v_proj": {"kernel": jnp.ones((8, 16))},
                "out_proj": {"kernel": jnp.ones((16, 8))},
            }
        }
    }
    targets = ["q_proj", "v_proj"]
    transformed, count = transform_params_to_lora(
        params=params,
        target_modules=targets,
        lora_r=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
    )
    assert count == 2

    q_proj = transformed["decoder"]["layers_0"]["q_proj"]
    assert "kernel" in q_proj
    assert "lora_a" in q_proj
    assert "lora_b" in q_proj
    assert "lora_scale" in q_proj
    assert q_proj["lora_a"].shape == (8, 4)
    assert q_proj["lora_b"].shape == (4, 16)
    assert float(q_proj["lora_scale"]) == 2.0

    # Non-target out_proj should not contain LoRA adapters
    out_proj = transformed["decoder"]["layers_0"]["out_proj"]
    assert "lora_a" not in out_proj


def test_transform_params_to_lora_edge_cases() -> None:
    """Test transform_params_to_lora with invalid rank, empty targets, and non-dict params."""
    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        transform_params_to_lora({}, ["q_proj"], lora_r=0)

    # Explicit PRNGKey
    key = jax.random.PRNGKey(42)
    res_key, count_key = transform_params_to_lora({"q_proj": {"kernel": jnp.ones((4, 4))}}, ["q_proj"], rng=key)
    assert count_key == 1
    assert "lora_a" in res_key["q_proj"]

    # Empty target modules
    res, count = transform_params_to_lora({"q_proj": {"kernel": 1}}, [])
    assert count == 0
    assert "kernel" in res["q_proj"]

    # Non-dict parameter input
    res2, count2 = transform_params_to_lora("invalid_params", ["q_proj"])  # type: ignore[arg-type]
    assert count2 == 0
    assert res2 == "invalid_params"


def test_segregate_adapter_params() -> None:
    """Test segregating parameter PyTree into trainable LoRA and frozen base weights."""
    params = {
        "decoder": {
            "q_proj": {
                "kernel": jnp.ones((4, 4)),
                "lora_a": jnp.ones((4, 2)),
                "lora_b": jnp.zeros((2, 4)),
                "lora_scale": jnp.array(2.0),
            },
            "adapters_only": {
                "lora_a": jnp.ones((2, 2)),
            },
            "norm": {"scale": jnp.ones((4,))},
        }
    }
    trainable, frozen = segregate_adapter_params(params)
    assert "lora_a" in trainable["decoder"]["q_proj"]
    assert "lora_b" in trainable["decoder"]["q_proj"]
    assert "adapters_only" in trainable["decoder"]
    assert "kernel" not in trainable["decoder"]["q_proj"]

    assert "kernel" in frozen["decoder"]["q_proj"]
    assert "norm" in frozen["decoder"]
    assert "lora_a" not in frozen["decoder"]["q_proj"]
    assert "adapters_only" not in frozen["decoder"]

    # Non-dict parameter input
    t, f = segregate_adapter_params("not_a_dict")  # type: ignore[arg-type]
    assert t == {}
    assert f == {}


def test_create_maxtext_lora_optimizer() -> None:
    """Test create_maxtext_lora_optimizer updates only adapter matrices and freezes base weights."""
    params = {
        "kernel": jnp.ones((4, 4)),
        "lora_a": jnp.ones((4, 2)),
        "lora_b": jnp.ones((2, 4)),
    }
    # Test with default optimizer
    tx = create_maxtext_lora_optimizer(params)
    opt_state = tx.init(params)

    # Apply all-ones gradient update
    grads = jax.tree.map(lambda x: jnp.ones_like(x), params)
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Base kernel must be completely untouched (frozen)
    assert jnp.all(new_params["kernel"] == 1.0)
    # LoRA matrices must have received updates
    assert not jnp.all(new_params["lora_a"] == 1.0)
    assert not jnp.all(new_params["lora_b"] == 1.0)

    # Test with custom base optimizer
    tx_custom = create_maxtext_lora_optimizer(params, optax.sgd(0.01))
    assert tx_custom is not None


def test_merge_lora_weights() -> None:
    """Test merge_lora_weights correctly folds adapter matrices back into base weights."""
    params = {
        "layer": {
            "kernel": jnp.ones((4, 4)),
            "lora_a": jnp.ones((4, 2)),
            "lora_b": jnp.ones((2, 4)),
            "lora_scale": jnp.array(3.0),
            "bias": jnp.zeros((4,)),
        },
        "other": "static_val",
    }
    # lora_a @ lora_b = sum of two 1.0s = 2.0
    # scale = 3.0 -> 3.0 * 2.0 = 6.0
    # W_merged = 1.0 + 6.0 = 7.0
    merged = merge_lora_weights(params)
    assert "lora_a" not in merged["layer"]
    assert "lora_b" not in merged["layer"]
    assert "lora_scale" not in merged["layer"]
    assert "bias" in merged["layer"]
    assert jnp.allclose(merged["layer"]["kernel"], jnp.full((4, 4), 7.0))
    assert merged["other"] == "static_val"

    # Non-dict test
    assert merge_lora_weights(123) == 123  # type: ignore[arg-type]


def test_save_and_load_maxtext_adapters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test saving and loading MaxText LoRA adapters to/from NPZ files."""
    params = {
        "decoder": {
            "layers_0": {
                "q_proj": {
                    "kernel": jnp.ones((8, 16)),
                    "lora_a": jnp.full((8, 4), 3.5),
                    "lora_b": jnp.full((4, 16), 1.2),
                    "lora_scale": jnp.array(2.0),
                }
            }
        }
    }
    # Save non-dict input (safe no-op saving empty archive)
    save_maxtext_adapters("non_dict_params", tmp_path / "empty.npz")  # type: ignore[arg-type]
    assert (tmp_path / "empty.npz").exists()

    # Save to directory (auto-appends maxtext_lora_adapters.npz)
    save_dir = tmp_path / "adapters_dir"
    save_maxtext_adapters(params, save_dir)
    expected_file = save_dir / "maxtext_lora_adapters.npz"
    assert expected_file.exists()

    # Save to explicit file path
    explicit_file = tmp_path / "explicit" / "custom.npz"
    save_maxtext_adapters(params, explicit_file)
    assert explicit_file.exists()

    # Load adapters back into a fresh base parameter dict
    base_params = {
        "decoder": {
            "layers_0": {
                "q_proj": {
                    "kernel": jnp.ones((8, 16)),
                }
            }
        }
    }
    loaded = load_maxtext_adapters(base_params, expected_file)
    assert "lora_a" in loaded["decoder"]["layers_0"]["q_proj"]
    assert np.allclose(loaded["decoder"]["layers_0"]["q_proj"]["lora_a"], 3.5)
    assert np.allclose(loaded["decoder"]["layers_0"]["q_proj"]["lora_b"], 1.2)

    # Load into an empty dict to test path auto-creation
    loaded_into_empty = load_maxtext_adapters({}, expected_file)
    assert "decoder" in loaded_into_empty

    # Test loading when jnp is None
    monkeypatch.setattr(pt, "jnp", None)
    loaded_no_jnp = load_maxtext_adapters({}, expected_file)
    assert "decoder" in loaded_no_jnp
    monkeypatch.undo()

    # FileNotFoundError on non-existent path
    with pytest.raises(FileNotFoundError, match="Adapter file not found"):
        load_maxtext_adapters(base_params, tmp_path / "missing.npz")


def test_count_maxtext_parameters() -> None:
    """Test count_maxtext_parameters accurately counts total and adapter weights."""
    params = {
        "decoder": {
            "q_proj": {
                "kernel": jnp.ones((8, 16)),  # 128
                "lora_a": jnp.ones((8, 4)),  # 32
                "lora_b": jnp.ones((4, 16)),  # 64
            },
            "norm": jnp.ones((8,)),  # 8
            "scalar_metadata": "static_string",
        }
    }
    total, lora = count_maxtext_parameters(params)
    assert lora == 32 + 64  # 96
    assert total == 128 + 96 + 8  # 232

    # Non-dict parameter input
    t, l = count_maxtext_parameters("none")  # type: ignore[arg-type]
    assert t == 0
    assert l == 0


def test_apply_lora_end_to_end(tmp_path: Path) -> None:
    """Test apply_lora with user params, output_dir saving, and merge folding."""
    params = {
        "layers": {
            "q_proj": {"kernel": jnp.ones((4, 8))},
            "v_proj": {"kernel": jnp.ones((4, 8))},
            "other": {"kernel": jnp.ones((4, 4))},
        }
    }
    out_dir = tmp_path / "peft_output"

    # Test apply_lora with output_dir
    res = apply_lora(
        model_name="test_model",
        target_modules=["q_proj", "v_proj"],
        lora_r=2,
        lora_alpha=4.0,
        lora_dropout=0.05,
        params=params,
        output_dir=str(out_dir),
    )
    assert res["status"] == "completed"
    assert res["backend"] == "maxtext"
    assert res["injected_modules"] == 2
    assert (out_dir / "maxtext_lora_adapters.npz").exists()

    # Test apply_lora with merge=True
    res_merge = apply_lora(
        model_name="test_model",
        target_modules=["q_proj"],
        lora_r=2,
        lora_alpha=4.0,
        params=params,
        merge=True,
    )
    assert res_merge["status"] == "completed"


def test_missing_dependencies_exceptions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test functions raise DependencyMissingError when required modules are absent."""
    monkeypatch.setattr(pt, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX dependencies are missing"):
        transform_params_to_lora({}, ["q_proj"])

    with pytest.raises(DependencyMissingError, match="Optax or JAX dependencies are missing"):
        create_maxtext_lora_optimizer({})

    monkeypatch.setattr(pt, "jax", jax)
    monkeypatch.setattr(pt, "optax", None)
    with pytest.raises(DependencyMissingError, match="Optax or JAX dependencies are missing"):
        create_maxtext_lora_optimizer({})

    monkeypatch.setattr(pt, "np", None)
    with pytest.raises(DependencyMissingError, match="NumPy dependency is missing"):
        save_maxtext_adapters({}, tmp_path / "out.npz")

    with pytest.raises(DependencyMissingError, match="NumPy dependency is missing"):
        load_maxtext_adapters({}, tmp_path / "out.npz")


def test_peft_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test reloading module when dependencies are missing."""
    import importlib
    import sys

    m_peft = sys.modules["gemma_4_sql.backends.maxtext.peft"]
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_peft)
    monkeypatch.undo()
    importlib.reload(m_peft)
