"""Unit and behavioral tests for MaxText AQT quantization, artifact storage, and inference round-trip."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import jax
import pytest

import gemma_4_sql.backends.maxtext.inference as inference_mod
import gemma_4_sql.backends.maxtext.quantize as quantize_mod
from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.maxtext.inference import generate_sql
from gemma_4_sql.backends.maxtext.quantize import AQTParamDict, get_quantized_maxtext_artifact
from gemma_4_sql.exceptions import (
    DependencyMissingError,
    UnsupportedQuantizationMethodError,
)
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import PyTreeNode
from tests.conftest import requires_jax

pytestmark = requires_jax


@pytest.fixture(autouse=True)
def _restore_quantize_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure `quantize_mod` attributes and caches are isolated per test."""
    monkeypatch.setattr(quantize_mod, "jax", quantize_mod.jax)
    monkeypatch.setattr(quantize_mod, "jnp", quantize_mod.jnp)
    monkeypatch.setattr(quantize_mod, "aqt", quantize_mod.aqt)
    monkeypatch.setattr(quantize_mod, "Gemma4Model", quantize_mod.Gemma4Model)
    inference_mod._MODEL_CACHE.clear()
    inference_mod._PARAMS_ONLY_CACHE.clear()
    quantize_mod._QUANTIZED_ARTIFACT_CACHE.clear()


def test_quantize_tensor_aqt_missing_jax() -> None:
    """Verify `quantize_tensor_aqt` raises `DependencyMissingError` when JAX is missing."""
    dummy_tensor = jnp.zeros((1, 1), dtype=jnp.float32)
    quantize_mod.jax = None
    quantize_mod.jnp = None
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing\."):
        quantize_mod.quantize_tensor_aqt(dummy_tensor)


def test_quantize_tensor_aqt_negative_bits() -> None:
    """Verify `quantize_tensor_aqt` raises `ValueError` on non-positive bits."""
    tensor = jnp.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Quantization bits must be positive, got 0"):
        quantize_mod.quantize_tensor_aqt(tensor, bits=0)
    with pytest.raises(ValueError, match="Quantization bits must be positive, got -1"):
        quantize_mod.quantize_tensor_aqt(tensor, bits=-1)


def test_quantize_tensor_aqt_int8_and_int16() -> None:
    """Verify `quantize_tensor_aqt` produces accurate `int8` and `int16` quantized arrays on real JAX tensors."""
    tensor = jnp.array([[100.0, 50.0], [-10.0, -100.0]], dtype=jnp.float32)
    quantized, scale = quantize_mod.quantize_tensor_aqt(tensor, bits=8)

    assert quantized.dtype == jnp.int8
    assert quantized.shape == (2, 2)
    assert scale.shape == (2, 1)
    reconstructed = quantized.astype(jnp.float32) * scale
    assert bool(jnp.allclose(jnp.asarray(reconstructed), tensor, atol=1.0))

    tensor16 = jnp.array([[1000.0, 500.0]], dtype=jnp.float32)
    quantized16, _scale16 = quantize_mod.quantize_tensor_aqt(tensor16, bits=16)
    assert quantized16.dtype == jnp.int16


def test_apply_aqt_quantization_missing_jax() -> None:
    """Verify `apply_aqt_quantization` raises `DependencyMissingError` when JAX is missing."""
    quantize_mod.jax = None
    quantize_mod.jnp = None
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing\."):
        quantize_mod.apply_aqt_quantization({})


def test_apply_aqt_quantization_methods() -> None:
    """Verify `apply_aqt_quantization` supports `int8`, `int4`, and `aqt` and rejects unsupported methods."""
    params: AQTParamDict = {}

    _, meta_int8, _ = quantize_mod.apply_aqt_quantization(params, method="int8")
    assert meta_int8["bits"] == 8
    assert meta_int8["memory_reduction_factor"] == 0.5

    _, meta_int4, _ = quantize_mod.apply_aqt_quantization(params, method="int4")
    assert meta_int4["bits"] == 4
    assert meta_int4["memory_reduction_factor"] == 0.75

    _, meta_aqt, _ = quantize_mod.apply_aqt_quantization(params, method="aqt")
    assert meta_aqt["bits"] == 8
    assert meta_aqt["memory_reduction_factor"] == 0.5

    with pytest.raises(UnsupportedQuantizationMethodError):
        quantize_mod.apply_aqt_quantization(params, method="unknown")


def test_apply_aqt_quantization_targets_and_traverse() -> None:
    """Verify `apply_aqt_quantization` traverses nested parameter PyTrees and wires `aqt.config.fully_quantized`."""

    class _FakeAQTConfig:
        """Typed substitute for `aqt.jax.v2.config`."""

        def fully_quantized(self, *, fwd_bits: int, bwd_bits: int) -> str:
            """Return config string."""
            return f"dg_cfg_{fwd_bits}_{bwd_bits}"

    class _FakeAQTModule:
        """Typed substitute for `aqt.jax.v2`."""

        config = _FakeAQTConfig()

    quantize_mod.aqt = _FakeAQTModule()

    tensor = jnp.array([[1.0, -2.0], [0.5, 4.0]], dtype=jnp.float32)
    params: AQTParamDict = {
        "non_dict": "value",
        "nested": {
            "q_proj": {"kernel": tensor, "other_param": 42},
            "unrelated": {"kernel": tensor},
            "k_proj": {"kernel": tensor},
        },
        "deep": {"layer": {"v_proj": {"kernel": tensor}}},
    }

    q_params, meta1, count = quantize_mod.apply_aqt_quantization(params, quant_targets=["q_proj"])
    assert count == 1
    nested_sub = q_params["nested"]
    assert isinstance(nested_sub, dict)
    q_sub = nested_sub["q_proj"]
    k_sub = nested_sub["k_proj"]
    assert isinstance(q_sub, dict) and isinstance(k_sub, dict)
    assert "kernel_scale" in q_sub
    assert q_sub["other_param"] == 42
    assert "kernel_scale" not in k_sub
    deep_sub = q_params["deep"]
    assert isinstance(deep_sub, dict)
    layer_sub = deep_sub["layer"]
    assert isinstance(layer_sub, dict)
    v_sub = layer_sub["v_proj"]
    assert isinstance(v_sub, dict)
    assert "kernel_scale" not in v_sub
    assert q_params["non_dict"] == "value"
    assert meta1["aqt_dg_config"] == "dg_cfg_8_8"

    _q_params2, meta2, count2 = quantize_mod.apply_aqt_quantization(params)
    assert count2 == 3
    assert meta2["aqt_native"] is True


def test_quantize_model_missing_maxtext_deps() -> None:
    """Verify `quantize_model` raises `DependencyMissingError` when JAX or `Gemma4Model` is missing."""
    quantize_mod.Gemma4Model = None
    with pytest.raises(DependencyMissingError, match=r"MaxText dependencies are missing\."):
        quantize_mod.quantize_model("model")

    with pytest.raises(DependencyMissingError, match=r"MaxText dependencies are missing\."):
        quantize_mod.quantize_model("model", params=None)

    quantize_mod.jnp = None
    with pytest.raises(DependencyMissingError, match=r"MaxText dependencies are missing\."):
        quantize_mod.quantize_model("model")

    quantize_mod.jax = None
    with pytest.raises(DependencyMissingError, match=r"MaxText dependencies are missing\."):
        quantize_mod.quantize_model("model")


def test_quantize_model_persists_artifact_and_roundtrips_through_generate_sql(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `quantize_model` stores `QuantizedModelArtifact`, exports to Orbax, and reloads into `generate_sql`."""
    tokenizer = SQLTokenizer()
    target_tok = tokenizer.byte_fallback.byte_to_token_ids(ord("Q"))[0]

    class _QuantizableMaxTextModel:
        """Stateless MaxText model projecting one-hot/constant features via `q_proj.kernel`."""

        def __init__(self, name: str) -> None:
            """Store model name."""
            self.name = name

        def init(self, _rng: jax.Array, _inputs: jax.Array) -> dict[str, PyTreeNode]:
            """Initialize `q_proj.kernel` favoring `'X'` by default.

            Returns:
                A parameter dictionary containing `'q_proj'`.

            """
            x_tok = tokenizer.byte_fallback.byte_to_token_ids(ord("X"))[0]
            k_mat = jnp.zeros((4, 260), dtype=jnp.float32).at[:, x_tok].set(5.0)
            return {"q_proj": {"kernel": k_mat}}

        def apply(self, params: Mapping[str, PyTreeNode], seq: jax.Array) -> jax.Array:
            """Compute logits `ones @ params['q_proj']['kernel']`.

            Returns:
                Vocabulary logits of shape `(1, seq_len, 260)`.

            """
            q_sub = params["q_proj"]
            assert isinstance(q_sub, Mapping)
            kernel = q_sub["kernel"]
            assert isinstance(kernel, jax.Array)
            feats = jnp.ones((1, seq.shape[1], 4), dtype=jnp.float32)
            return jnp.matmul(feats, kernel)

    monkeypatch.setattr(quantize_mod, "Gemma4Model", _QuantizableMaxTextModel)
    monkeypatch.setattr(inference_mod, "Gemma4Model", _QuantizableMaxTextModel)

    # 1. Quantize initialized model directly
    res = quantize_mod.quantize_model("my_model")
    assert res["backend"] == "maxtext"
    assert res["model"] == "my_model"
    assert res["status"] == "quantized_int8"
    artifact = get_quantized_maxtext_artifact("my_model")
    assert artifact is not None
    assert artifact.method == "int8"
    assert artifact.is_quantized is True
    assert artifact.scales["quantized_modules_count"] == 1.0
    q_proj_entry = artifact.model["q_proj"]
    assert isinstance(q_proj_entry, dict)
    assert isinstance(q_proj_entry["kernel"], jax.Array)
    assert q_proj_entry["kernel"].dtype == jnp.int8

    # 2. Second `quantize_model` call reuses cached parameters; also test custom weights favoring 'Q' + Orbax export!
    res_cached = quantize_mod.quantize_model("my_model", method="int4")
    assert res_cached["status"] == "quantized_int4"

    q_kernel = jnp.zeros((4, 260), dtype=jnp.float32).at[:, target_tok].set(8.0)
    export_dir = tmp_path / "q_orbax"
    res_exp = quantize_mod.quantize_model(
        "q_model",
        method="int8",
        params={"q_proj": {"kernel": q_kernel}},
        export_path=str(export_dir),
    )
    assert res_exp["status"] == "quantized_int8"
    assert "export_path" in res_exp

    # Clear in-memory caches and reload quantized Orbax checkpoint into `generate_sql`: dequantization produces 'QQQ'!
    inference_mod._MODEL_CACHE.clear()
    inference_mod._PARAMS_ONLY_CACHE.clear()
    reloaded = generate_sql("q_model", "SELECT", beam_width=1, max_length=3, checkpoint_path=str(export_dir))
    assert "QQQ" in reloaded["sql"]


def test_quantize_model_with_params_kwargs_dict_and_non_dict() -> None:
    """Verify `quantize_model` with explicit `params` dict, nested unquantized dict, and non-dict `params` fallback."""
    tensor = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
    params: AQTParamDict = {"q_proj": {"kernel": tensor}, "meta": {"tag": "v1"}}

    res = quantize_mod.quantize_model("my_model", method="int4", params=params, quant_targets=["q_proj"])
    assert res["status"] == "quantized_int4"
    assert res["memory_reduction_factor"] == 0.75
    meta = res.get("metadata")
    assert isinstance(meta, dict)
    assert meta["bits"] == 4

    assert quantize_mod.apply_aqt_quantization("not_a_dict")[0] == "not_a_dict"
    assert quantize_mod._coerce_aqt_param_dict({"skip_list": [1, 2]}) == {}

    res_non_dict = quantize_mod.quantize_model("my_model", method="int8", params="not_a_dict")
    assert res_non_dict["status"] == "quantized_int8"
    assert res_non_dict["memory_reduction_factor"] == 0.5


def test_quantize_model_exception_handling() -> None:
    """Verify `quantize_model` error handling and `UnsupportedQuantizationMethodError`."""
    params: AQTParamDict = {"q_proj": {"kernel": jnp.array([[1.0]])}}

    with patch.object(quantize_mod, "apply_aqt_quantization", side_effect=RuntimeError("mock error")):
        res = quantize_mod.quantize_model("my_model", params=params)

    assert res["status"] == "failed: mock error"
    assert "metadata" not in res

    with pytest.raises(UnsupportedQuantizationMethodError):
        quantize_mod.quantize_model("my_model", method="fp4", params=params)
