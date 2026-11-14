"""Unit and behavioral tests for MaxText PEFT / LoRA parameter transformations without MagicMock."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

from tests.conftest import requires_jax

pytestmark = requires_jax

import jax
import optax
import pytest

from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.maxtext import inference as inference_module
from gemma_4_sql.backends.maxtext import peft as peft_module
from gemma_4_sql.backends.maxtext.peft import MaxTextParamDict, apply_with_lora
from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import PyTreeNode


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Clear MaxText inference caches before each PEFT test."""
    inference_module._MODEL_CACHE.clear()
    inference_module._PARAMS_ONLY_CACHE.clear()


def test_transform_params_to_lora_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `transform_params_to_lora` checks JAX availability and positive rank `r`."""
    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        peft_module.transform_params_to_lora({}, ["q_proj"], lora_r=0)

    monkeypatch.setattr(peft_module, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX dependencies are missing"):
        peft_module.transform_params_to_lora({}, ["q_proj"])


def test_transform_segregate_apply_with_lora_merge_and_count_real() -> None:
    """Verify JAX LoRA injection, `apply_with_lora` gradients, `lora_scale` freeze parity, counting, and merging."""
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
    q_scale = q_dict["lora_scale"]
    assert isinstance(q_scale, jax.Array)
    assert float(q_scale) == 2.0
    assert "lora_a" not in k_dict

    class _StatelessProjModel:
        """Stateless model computing `inputs @ params['layer_0']['q_proj']['kernel']`."""

        def apply(self, p: Mapping[str, PyTreeNode], x: jax.Array) -> jax.Array:
            """Project `x` by `layer_0.q_proj.kernel`.

            Returns:
                Matrix product `x @ kernel`.

            """
            l0 = p["layer_0"]
            assert isinstance(l0, Mapping)
            qp = l0["q_proj"]
            assert isinstance(qp, Mapping)
            kern = qp["kernel"]
            assert isinstance(kern, jax.Array)
            return jnp.dot(x, kern)

    proj_model = _StatelessProjModel()
    x_input = jnp.ones((2, 4), dtype=jnp.float32)
    y_base = jnp.dot(x_input, w_q)
    y_zero_b = apply_with_lora(proj_model, transformed, x_input)
    assert jnp.allclose(y_base, y_zero_b, atol=1e-6)

    # Differentiate through `apply_with_lora` and verify `lora_b` receives non-zero gradients!
    def _lora_loss(p: MaxTextParamDict) -> jax.Array:
        """Mean squared error toward zero target.

        Returns:
            Scalar mean squared projection output.

        """
        out = apply_with_lora(proj_model, p, x_input)
        return jnp.mean(out**2)

    from collections.abc import Callable

    def _as_leaf_arr(val: peft_module.MaxTextParamLeaf | MaxTextParamDict) -> jax.Array:
        """Convert a numeric `MaxTextParamLeaf` to a `jax.Array`.

        Returns:
            Converted `jax.Array`.

        """
        assert not isinstance(val, (dict, str))
        return jnp.asarray(val)

    grad_builder: Callable[..., Callable[[MaxTextParamDict], MaxTextParamDict]] = vars(jax)["grad"]
    grads = grad_builder(_lora_loss, allow_int=True)(transformed)
    grad_l0 = grads["layer_0"]
    assert isinstance(grad_l0, dict)
    grad_q = grad_l0["q_proj"]
    assert isinstance(grad_q, dict)
    assert not jnp.allclose(_as_leaf_arr(grad_q["lora_b"]), 0.0)

    # Non-dict input returns unchanged with 0 count
    res_nd, count_nd = peft_module.transform_params_to_lora("not a dict", ["q_proj"])
    assert res_nd == "not a dict"
    assert count_nd == 0
    assert peft_module.segregate_adapter_params("not a dict") == ({}, {})
    assert peft_module.count_maxtext_parameters("not a dict") == (0, 0)
    assert peft_module.merge_lora_weights("not a dict") == "not a dict"
    assert peft_module._coerce_maxtext_param_dict({"bad": [1, 2]}) == {}

    # Parameter counting: base weights (24 + 6 + 24 + 4 + 1 = 59) + LoRA (8 + 12 = 20) = 79 total, 20 trainable
    total_params, trainable_params = peft_module.count_maxtext_parameters(transformed)
    assert total_params == 79
    assert trainable_params == 20

    # Segregation: `lora_a` and `lora_b` are trainable; `kernel` AND `lora_scale` are frozen!
    trainable_tree, frozen_tree = peft_module.segregate_adapter_params(transformed)
    tr_l0 = trainable_tree["layer_0"]
    fr_l0 = frozen_tree["layer_0"]
    assert isinstance(tr_l0, dict) and isinstance(fr_l0, dict)
    tr_q = tr_l0["q_proj"]
    fr_q = fr_l0["q_proj"]
    assert isinstance(tr_q, dict) and isinstance(fr_q, dict)
    assert set(tr_q.keys()) == {"lora_a", "lora_b"}
    assert "kernel" in fr_q
    assert "lora_scale" in fr_q

    # Set known non-zero lora_a and lora_b and verify `apply_with_lora` matches `merge_lora_weights`!
    q_dict["lora_a"] = jnp.full((4, 2), 0.5, dtype=jnp.float32)
    q_dict["lora_b"] = jnp.full((2, 6), 0.25, dtype=jnp.float32)
    y_nonzero_b = apply_with_lora(proj_model, transformed, x_input)
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
    """Verify `create_maxtext_lora_optimizer` updates only `lora_a` and `lora_b` while keeping base weights frozen."""
    params: MaxTextParamDict = {
        "layer_0": {
            "q_proj": {
                "kernel": jnp.ones((2, 2), dtype=jnp.float32),
                "lora_a": jnp.ones((2, 1), dtype=jnp.float32),
                "lora_b": jnp.zeros((1, 2), dtype=jnp.float32),
                "lora_scale": jnp.array(2.0, dtype=jnp.float32),
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
                "lora_scale": jnp.array(5.0, dtype=jnp.float32),
            }
        }
    }

    def _add_leaf(p: jax.Array, u: jax.Array) -> jax.Array:
        """Add parameter and update array.

        Returns:
            Elementwise sum `p + u`.

        """
        return p + u

    updates, _ = opt.update(grads, opt_state, params)
    updated_params: MaxTextParamDict = jax.tree_util.tree_map(_add_leaf, params, updates)
    up_l0 = updated_params["layer_0"]
    assert isinstance(up_l0, dict)
    up_q = up_l0["q_proj"]
    assert isinstance(up_q, dict)

    def _leaf(val: peft_module.MaxTextParamLeaf | MaxTextParamDict) -> jax.Array:
        """Narrow numeric `MaxTextParamLeaf` to `jax.Array`.

        Returns:
            Converted `jax.Array`.

        """
        assert not isinstance(val, (dict, str))
        return jnp.asarray(val)

    assert jnp.allclose(_leaf(up_q["kernel"]), 1.0)
    assert jnp.allclose(_leaf(up_q["lora_scale"]), 2.0)
    assert not jnp.allclose(_leaf(up_q["lora_a"]), 1.0)
    assert not jnp.allclose(_leaf(up_q["lora_b"]), 0.0)

    custom_opt = peft_module.create_maxtext_lora_optimizer(params, base_optimizer=optax.sgd(0.05))
    assert custom_opt is not None

    monkeypatch.setattr(peft_module, "optax", None)
    with pytest.raises(DependencyMissingError, match="Optax or JAX dependencies are missing"):
        peft_module.create_maxtext_lora_optimizer(params)


def test_save_and_load_maxtext_adapters_real(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `save_maxtext_adapters` and `load_maxtext_adapters` round-trip real `.npz` files."""
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
    merged_float_scale = peft_module.merge_lora_weights(params)
    assert isinstance(merged_float_scale, dict)
    peft_module.save_maxtext_adapters(params, npz_file)
    assert npz_file.exists()

    peft_module.save_maxtext_adapters(params, tmp_path)
    assert (tmp_path / "maxtext_lora_adapters.npz").exists()

    peft_module.save_maxtext_adapters("not_dict", tmp_path / "empty.npz")

    target_params: MaxTextParamDict = {"layer": {"q_proj": {"kernel": jnp.ones((2, 2))}}}
    loaded = peft_module.load_maxtext_adapters(target_params, npz_file)
    l_layer = loaded["layer"]
    assert isinstance(l_layer, dict)
    l_q = l_layer["q_proj"]
    assert isinstance(l_q, dict)

    def _leaf_s(val: peft_module.MaxTextParamLeaf | MaxTextParamDict) -> jax.Array:
        """Narrow numeric `MaxTextParamLeaf` to `jax.Array`.

        Returns:
            Converted `jax.Array`.

        """
        assert not isinstance(val, (dict, str))
        return jnp.asarray(val)

    assert jnp.allclose(_leaf_s(l_q["lora_a"]), 3.0)
    assert jnp.allclose(_leaf_s(l_q["lora_b"]), -2.0)

    loaded_fresh = peft_module.load_maxtext_adapters({}, npz_file)
    lf_layer = loaded_fresh["layer"]
    assert isinstance(lf_layer, dict)
    lf_q = lf_layer["q_proj"]
    assert isinstance(lf_q, dict)
    assert jnp.allclose(_leaf_s(lf_q["lora_a"]), 3.0)

    with pytest.raises(FileNotFoundError, match="Adapter file not found"):
        peft_module.load_maxtext_adapters({}, tmp_path / "nonexistent.npz")

    monkeypatch.setattr(peft_module, "jnp", None)
    with pytest.raises(DependencyMissingError, match="JAX dependency is missing"):
        peft_module.save_maxtext_adapters(params, npz_file)
    with pytest.raises(DependencyMissingError, match="JAX dependency is missing"):
        peft_module.load_maxtext_adapters(params, npz_file)


def test_apply_lora_maxtext(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `apply_lora` runs with explicit `params`, `Gemma4Model`, caching, `output_dir`, `merge`, and errors."""
    params: MaxTextParamDict = {"layer": {"q_proj": {"kernel": jnp.ones((4, 4), dtype=jnp.float32)}}}

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
    assert res_params.get("injected_modules") == 1
    assert (tmp_path / "maxtext_lora_adapters.npz").exists()
    assert inference_module.get_cached_maxtext_params("gemma-4") is not None

    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        peft_module.apply_lora("gemma-4", ["q_proj"])

    class _TypedGemma4Model:
        """Typed stateless MaxText model double."""

        def __init__(self, _name: str) -> None:
            """No-op init."""

        def init(self, _rng: jax.Array, _inp: jax.Array) -> MaxTextParamDict:
            """Return parameter dictionary."""
            return params

        def apply(self, _p: Mapping[str, PyTreeNode], inp: jax.Array) -> jax.Array:
            """Return input unchanged."""
            return inp

    monkeypatch.setattr(peft_module, "Gemma4Model", _TypedGemma4Model)
    res_model = peft_module.apply_lora("gemma-4-unmerged", ["q_proj"], lora_r=2)
    assert res_model["status"] == "completed"
    assert res_model.get("injected_modules") == 1
    cached_unmerged = inference_module.get_cached_maxtext_params("gemma-4-unmerged")
    assert cached_unmerged is not None

    # Also test `Gemma4Model` class without `.apply` (not `MaxTextCausalLMProtocol`)
    # and `apply_with_lora` non-dict fallback
    class _InitOnlyModel:
        """Model exposing `.init` without `.apply`."""

        def __init__(self, _name: str) -> None:
            """No-op init."""

        def init(self, _rng: jax.Array, _inp: jax.Array) -> MaxTextParamDict:
            """Return parameter dictionary."""
            return params

    monkeypatch.setattr(peft_module, "Gemma4Model", _InitOnlyModel)
    assert peft_module.apply_lora("gemma-4-init-only", ["q_proj"], lora_r=2)["status"] == "completed"

    with patch.object(peft_module, "_coerce_maxtext_param_dict", return_value=None):
        assert jnp.array_equal(apply_with_lora(_TypedGemma4Model("m"), {}, jnp.array([1, 2])), jnp.array([1, 2]))

    with patch.object(peft_module, "transform_params_to_lora", side_effect=RuntimeError("Fake Error")):
        res_fail = peft_module.apply_lora("gemma-4", ["q_proj"])
        assert "failed: Fake Error" in res_fail["status"]
        with pytest.raises(TrainingError, match="Fake Error"):
            peft_module.apply_lora("gemma-4", ["q_proj"], raise_on_error=True)

    with (
        patch.object(peft_module, "transform_params_to_lora", side_effect=ConfigurationError("cfg fail")),
        pytest.raises(ConfigurationError, match="cfg fail"),
    ):
        peft_module.apply_lora("gemma-4", ["q_proj"])

    monkeypatch.setattr(peft_module, "jax", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        peft_module.apply_lora("gemma-4", ["q_proj"])
