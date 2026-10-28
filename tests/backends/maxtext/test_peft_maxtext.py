"""Unit and behavioral tests for MaxText PEFT / LoRA parameter transformations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.conftest import requires_jax

pytestmark = requires_jax

import jax
import jax.numpy as jnp
import optax
import pytest

from gemma_4_sql.backends.maxtext import peft as peft_module
from gemma_4_sql.backends.maxtext.peft import MaxTextParamDict
from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError, TrainingError


def test_transform_params_to_lora_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify transform_params_to_lora checks JAX availability and positive rank r."""
    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        peft_module.transform_params_to_lora({}, ["q_proj"], lora_r=0)

    monkeypatch.setattr(peft_module, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX dependencies are missing"):
        peft_module.transform_params_to_lora({}, ["q_proj"])


def test_transform_segregate_merge_and_count_real() -> None:
    """Verify real JAX array LoRA injection, forward-pass zero/non-zero B behavior, segregation, counting, and merging."""
    w_q = jnp.ones((4, 6), dtype=jnp.float32)
    w_k = jnp.ones((4, 6), dtype=jnp.float32) * 2.0
    params: MaxTextParamDict = {
        "layer_0": {
            "q_proj": {"kernel": w_q, "bias": jnp.zeros((6,), dtype=jnp.float32)},
            "k_proj": {"kernel": w_k},
            "norm": {"scale": jnp.ones((4,), dtype=jnp.float32)},
            "empty": {},
        },
        "step": 1,
    }

    transformed, count = peft_module.transform_params_to_lora(
        params,
        ["q_proj"],
        lora_r=2,
        lora_alpha=4.0,
        rng=jax.random.PRNGKey(7),
    )
    assert count == 1
    layer0 = transformed["layer_0"]
    assert isinstance(layer0, dict)
    q_dict = layer0["q_proj"]
    k_dict = layer0["k_proj"]
    assert isinstance(q_dict, dict)
    assert isinstance(k_dict, dict)

    q_lora_a = q_dict["lora_a"]
    q_lora_b = q_dict["lora_b"]
    q_kernel = q_dict["kernel"]
    assert isinstance(q_lora_a, jax.Array)
    assert isinstance(q_lora_b, jax.Array)
    assert isinstance(q_kernel, jax.Array)
    assert q_lora_a.shape == (4, 2)
    assert q_lora_b.shape == (2, 6)
    assert jnp.allclose(jnp.asarray(q_lora_b), 0.0)
    assert q_dict["lora_scale"] == 2.0
    assert "lora_a" not in k_dict

    # Verify forward pass: when lora_b == 0, x @ (W + scale * A @ B) == x @ W
    x_input = jnp.ones((2, 4), dtype=jnp.float32)
    y_base = jnp.dot(x_input, w_q)
    y_zero_b = jnp.dot(x_input, q_kernel) + 2.0 * jnp.dot(jnp.dot(x_input, q_lora_a), q_lora_b)
    assert jnp.allclose(y_base, y_zero_b, atol=1e-6)

    # Non-dict input returns unchanged with 0 count
    res_nd, count_nd = peft_module.transform_params_to_lora("not a dict", ["q_proj"])
    assert res_nd == "not a dict"
    assert count_nd == 0
    assert peft_module.segregate_adapter_params("not a dict") == ({}, {})
    assert peft_module.count_maxtext_parameters("not a dict") == (0, 0)
    assert peft_module.merge_lora_weights("not a dict") == "not a dict"

    # Parameter counting: base weights (24 + 6 + 24 + 4 + 1 = 59) + LoRA (8 + 12 = 20) = 79 total, 20 trainable
    total_params, trainable_params = peft_module.count_maxtext_parameters(transformed)
    assert total_params == 79
    assert trainable_params == 20

    # Segregation
    trainable_tree, frozen_tree = peft_module.segregate_adapter_params(transformed)
    tr_l0 = trainable_tree["layer_0"]
    fr_l0 = frozen_tree["layer_0"]
    assert isinstance(tr_l0, dict) and isinstance(fr_l0, dict)
    tr_q = tr_l0["q_proj"]
    fr_q = fr_l0["q_proj"]
    assert isinstance(tr_q, dict) and isinstance(fr_q, dict)
    assert "lora_a" in tr_q
    assert "lora_b" in tr_q
    assert "kernel" not in tr_q
    assert "kernel" in fr_q

    # Set known non-zero lora_a and lora_b and verify forward pass changes and merge math: W + scale * (A @ B)
    q_dict["lora_a"] = jnp.full((4, 2), 0.5, dtype=jnp.float32)
    q_dict["lora_b"] = jnp.full((2, 6), 0.25, dtype=jnp.float32)
    y_nonzero_b = jnp.dot(x_input, q_kernel) + 2.0 * jnp.dot(
        jnp.dot(x_input, jnp.full((4, 2), 0.5, dtype=jnp.float32)),
        jnp.full((2, 6), 0.25, dtype=jnp.float32),
    )
    assert not jnp.allclose(y_base, y_nonzero_b, atol=1e-5)

    merged = peft_module.merge_lora_weights(transformed)
    assert isinstance(merged, dict)
    m_l0 = merged["layer_0"]
    assert isinstance(m_l0, dict)
    m_q = m_l0["q_proj"]
    m_k = m_l0["k_proj"]
    assert isinstance(m_q, dict) and isinstance(m_k, dict)
    m_q_kernel = m_q["kernel"]
    m_k_kernel = m_k["kernel"]
    assert isinstance(m_q_kernel, jax.Array) and isinstance(m_k_kernel, jax.Array)
    expected_w = jnp.ones((4, 6), dtype=jnp.float32) + 2.0 * (jnp.full((4, 2), 0.5) @ jnp.full((2, 6), 0.25))
    assert jnp.allclose(jnp.asarray(m_q_kernel), expected_w, atol=1e-6)
    assert jnp.allclose(jnp.dot(x_input, m_q_kernel), y_nonzero_b, atol=1e-6)
    assert "lora_a" not in m_q
    assert "lora_b" not in m_q
    assert jnp.allclose(jnp.asarray(m_k_kernel), jnp.asarray(w_k))


def test_create_maxtext_lora_optimizer_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify create_maxtext_lora_optimizer updates only LoRA adapters and keeps base weights frozen."""
    params: MaxTextParamDict = {
        "layer_0": {
            "q_proj": {
                "kernel": jnp.ones((2, 2), dtype=jnp.float32),
                "lora_a": jnp.ones((2, 1), dtype=jnp.float32),
                "lora_b": jnp.zeros((1, 2), dtype=jnp.float32),
            }
        }
    }
    opt = peft_module.create_maxtext_lora_optimizer(params, learning_rate=0.1)
    opt_state = opt.init(params)

    grads: MaxTextParamDict = {
        "layer_0": {
            "q_proj": {
                "kernel": jnp.ones((2, 2), dtype=jnp.float32) * 10.0,
                "lora_a": jnp.ones((2, 1), dtype=jnp.float32),
                "lora_b": jnp.ones((1, 2), dtype=jnp.float32),
            }
        }
    }
    updates, _ = opt.update(grads, opt_state, params)
    updated_params: MaxTextParamDict = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)  # pyright: ignore[reportUnknownLambdaType]
    up_l0 = updated_params["layer_0"]
    assert isinstance(up_l0, dict)
    up_q = up_l0["q_proj"]
    assert isinstance(up_q, dict)

    # Frozen kernel must remain untouched (all 1.0), while lora_a and lora_b must change
    assert jnp.allclose(jnp.asarray(up_q["kernel"]), 1.0)
    assert not jnp.allclose(jnp.asarray(up_q["lora_a"]), 1.0)
    assert not jnp.allclose(jnp.asarray(up_q["lora_b"]), 0.0)

    # Custom base optimizer
    custom_opt = peft_module.create_maxtext_lora_optimizer(params, base_optimizer=optax.sgd(0.05))
    assert custom_opt is not None

    monkeypatch.setattr(peft_module, "optax", None)
    with pytest.raises(DependencyMissingError, match="Optax or JAX dependencies are missing"):
        peft_module.create_maxtext_lora_optimizer(params)


def test_save_and_load_maxtext_adapters_real(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify save_maxtext_adapters and load_maxtext_adapters round-trip real .npz files."""
    params: MaxTextParamDict = {
        "layer": {
            "q_proj": {
                "kernel": jnp.ones((2, 2)),
                "lora_a": jnp.full((2, 1), 3.0),
                "lora_b": jnp.full((1, 2), -2.0),
                "lora_scale": 2.0,
            }
        }
    }

    npz_file = tmp_path / "adapters.npz"
    peft_module.save_maxtext_adapters(params, npz_file)
    assert npz_file.exists()

    # Saving to a directory writes maxtext_lora_adapters.npz
    peft_module.save_maxtext_adapters(params, tmp_path)
    assert (tmp_path / "maxtext_lora_adapters.npz").exists()

    # Non-dict input writes empty archive
    peft_module.save_maxtext_adapters("not_dict", tmp_path / "empty.npz")

    target_params: MaxTextParamDict = {"layer": {"q_proj": {"kernel": jnp.ones((2, 2))}}}
    loaded = peft_module.load_maxtext_adapters(target_params, npz_file)
    l_layer = loaded["layer"]
    assert isinstance(l_layer, dict)
    l_q = l_layer["q_proj"]
    assert isinstance(l_q, dict)
    assert jnp.allclose(jnp.asarray(l_q["lora_a"]), 3.0)
    assert jnp.allclose(jnp.asarray(l_q["lora_b"]), -2.0)

    # Also test loading when nested dict path does not yet exist
    loaded_fresh = peft_module.load_maxtext_adapters({}, npz_file)
    lf_layer = loaded_fresh["layer"]
    assert isinstance(lf_layer, dict)
    lf_q = lf_layer["q_proj"]
    assert isinstance(lf_q, dict)
    assert jnp.allclose(jnp.asarray(lf_q["lora_a"]), 3.0)

    with pytest.raises(FileNotFoundError, match="Adapter file not found"):
        peft_module.load_maxtext_adapters({}, tmp_path / "nonexistent.npz")

    monkeypatch.setattr(peft_module, "jnp", None)
    with pytest.raises(DependencyMissingError, match="JAX dependency is missing"):
        peft_module.save_maxtext_adapters(params, npz_file)
    with pytest.raises(DependencyMissingError, match="JAX dependency is missing"):
        peft_module.load_maxtext_adapters(params, npz_file)


def test_apply_lora_maxtext(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify apply_lora executes with explicit params, mocked Gemma4Model, output_dir, merge, and errors."""
    params: MaxTextParamDict = {"layer": {"q_proj": {"kernel": jnp.ones((4, 4), dtype=jnp.float32)}}}

    # Works even when Gemma4Model is None if params dict is provided
    monkeypatch.setattr(peft_module, "Gemma4Model", None)
    res_params = peft_module.apply_lora(
        "gemma-4",
        ["q_proj"],
        lora_r=2,
        params=params,
        output_dir=str(tmp_path),
        merge=True,
    )
    assert res_params["status"] == "completed"
    assert res_params["injected_modules"] == 1
    assert (tmp_path / "maxtext_lora_adapters.npz").exists()

    # Raises DependencyMissingError when Gemma4Model is None and params is not a dict
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        peft_module.apply_lora("gemma-4", ["q_proj"])

    # With Gemma4Model provided
    mock_gemma4 = MagicMock()
    mock_gemma4.return_value.init.return_value = params
    monkeypatch.setattr(peft_module, "Gemma4Model", mock_gemma4)
    res_model = peft_module.apply_lora("gemma-4", ["q_proj"], lora_r=2)
    assert res_model["status"] == "completed"
    assert res_model["injected_modules"] == 1

    # Error handling
    with patch.object(peft_module, "transform_params_to_lora", side_effect=RuntimeError("Fake Error")):
        res_fail = peft_module.apply_lora("gemma-4", ["q_proj"])
        assert "failed: Fake Error" in res_fail["status"]
        with pytest.raises(TrainingError, match="Fake Error"):
            peft_module.apply_lora("gemma-4", ["q_proj"], raise_on_error=True)

    with patch.object(peft_module, "transform_params_to_lora", side_effect=ConfigurationError("cfg fail")), pytest.raises(ConfigurationError, match="cfg fail"):
        peft_module.apply_lora("gemma-4", ["q_proj"])

    monkeypatch.setattr(peft_module, "jax", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        peft_module.apply_lora("gemma-4", ["q_proj"])
