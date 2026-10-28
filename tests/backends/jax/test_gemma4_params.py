"""Unit and behavioral tests for gemma4 params."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from unittest.mock import patch

from tests.conftest import requires_jax

pytestmark = requires_jax

import jax
import jax.numpy as jnp
import pytest
from safetensors.flax import save_file

import gemma_4_sql.backends.jax.gemma4.params as params_mod
from gemma_4_sql.backends.jax.gemma4.config import AudioConfig, ModelConfig, VisionConfig
from gemma_4_sql.backends.jax.gemma4.params import (
    _fix_jax_state_embeddings,
    _get_key_and_transform_mapping,
    _stack_and_assign_expert_tensors,
    create_gemma4_from_pretrained,
    process_standard_tensor,
)
from gemma_4_sql.backends.jax.gemma4.utils_params import KeyMapType, StateNode
from gemma_4_sql.exceptions import DependencyMissingError


def test_stack_expert_tensors_none_key() -> None:
    """Verify stack expert tensors none key behavior."""
    expert_tensors = {0: {"test_proj": {0: jnp.zeros((1,)), 1: jnp.zeros((1,))}}}
    mapping: dict[str, KeyMapType] = {}
    jax_state: StateNode = {}
    _stack_and_assign_expert_tensors(expert_tensors, mapping, jax_state)
    assert jax_state == {}


def test_create_gemma4_from_pretrained(tmp_path: Path) -> None:
    """Verify create gemma4 from pretrained behavior."""
    tensors = {
        "model.embed_tokens.weight": jnp.zeros((10, 8), dtype=jnp.float32),
        "model.norm.weight": jnp.ones((8,), dtype=jnp.float32),
        "lm_head.weight": jnp.zeros((10, 8), dtype=jnp.float32),
        "audio_tower.output_proj.weight": jnp.zeros((8, 8), dtype=jnp.float32),
        "audio_tower.output_proj.bias": jnp.zeros((8,), dtype=jnp.float32),
        "vision_tower.vision_model.post_layernorm.weight": jnp.zeros((8,), dtype=jnp.float32),
        "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight": jnp.zeros((16, 8), dtype=jnp.float32),
        "model.layers.0.block_sparse_moe.experts.1.gate_proj.weight": jnp.zeros((16, 8), dtype=jnp.float32),
        "model.layers.0.block_sparse_moe.experts.2.gate_proj.weight": jnp.zeros((16, 8), dtype=jnp.float32),
        "model.layers.0.block_sparse_moe.experts.3.gate_proj.weight": jnp.zeros((16, 8), dtype=jnp.float32),
    }

    file_path = tmp_path / "model.safetensors"
    save_file(tensors, str(file_path))

    cfg = ModelConfig(
        vocab_size=10,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
        num_experts=4,
        vision_config=VisionConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, patch_size=4),
        audio_config=AudioConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2),
    )

    model = create_gemma4_from_pretrained(str(tmp_path), cfg)
    assert model is not None

    with pytest.raises(ValueError, match="No safetensors found"):
        create_gemma4_from_pretrained(str(tmp_path / "empty"), cfg)


def test_process_standard_tensor_errors() -> None:
    """Verify process standard tensor errors behavior."""
    mapping = _get_key_and_transform_mapping()

    class FakeSF:
        """Test stub for `FakeSF`."""

        def keys(self) -> Iterable[str]:
            """Return empty keys."""
            return ()

        def get_tensor(self, _key: str) -> jax.Array:
            """Provide get tensor test helper."""
            return jnp.zeros((1,))

    jax_state: StateNode = {}
    process_standard_tensor(FakeSF(), "model.embed_tokens.weight", jax_state, mapping)
    process_standard_tensor(FakeSF(), "unknown.key", jax_state, mapping)

    with patch("gemma_4_sql.backends.jax.gemma4.params.assign_weights_from_eval_shape", side_effect=AttributeError("bad")):
        with pytest.raises(AttributeError):
            process_standard_tensor(FakeSF(), "model.embed_tokens.weight", jax_state, mapping)


def test_create_gemma4_from_pretrained_etils_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify create gemma4 from pretrained etils missing behavior via both sys.modules[None] and ImportError."""
    import sys

    cfg = ModelConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=4, intermediate_size=16)

    # Path 1: sys.modules["etils"] = None
    monkeypatch.setitem(sys.modules, "etils", None)
    with pytest.raises(DependencyMissingError):
        create_gemma4_from_pretrained(str(tmp_path), cfg)

    # Path 2: epath is None
    monkeypatch.delitem(sys.modules, "etils", raising=False)
    monkeypatch.setattr(params_mod, "epath", None)
    with pytest.raises(DependencyMissingError):
        create_gemma4_from_pretrained(str(tmp_path), cfg)


def test_fix_jax_state_embeddings_coverage() -> None:
    """Verify fix jax state embeddings coverage behavior."""
    cfg = ModelConfig(
        vocab_size=10,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
        vision_config=VisionConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, patch_size=4),
    )

    jax_state: StateNode = {
        "model": {"embed_scale": jax.ShapeDtypeStruct((), jnp.float32)},
        "vision_tower": {"embeddings": {"position_ids": jax.ShapeDtypeStruct((1, 100), jnp.int32)}},
    }

    _fix_jax_state_embeddings(jax_state, None, cfg)
    model_sub = jax_state["model"]
    assert isinstance(model_sub, dict)
    assert isinstance(model_sub["embed_scale"], jax.Array)
    vt_sub = jax_state["vision_tower"]
    assert isinstance(vt_sub, dict)
    emb_sub = vt_sub["embeddings"]
    assert isinstance(emb_sub, dict)
    assert isinstance(emb_sub["position_ids"], jax.Array)

    cfg2 = ModelConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=4, intermediate_size=16)
    jax_state2: StateNode = {"model": {"embed_scale": 1.0}}
    _fix_jax_state_embeddings(jax_state2, None, cfg2)


def test_create_gemma4_state_dict_fallbacks(tmp_path: Path) -> None:
    """Verify create gemma4 state dict fallbacks behavior."""
    save_file({"lm_head.weight": jnp.zeros((10, 8), dtype=jnp.float32)}, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=4, intermediate_size=16)

    with patch("flax.nnx.split") as mock_split:

        class MockState0:
            """Test stub for `MockState0`."""

            def to_pure_dict(self) -> StateNode:
                """Provide to pure dict test helper."""
                return {"a": 0}

        mock_split.return_value = (None, MockState0())
        with patch("gemma_4_sql.backends.jax.gemma4.params.nnx.merge") as mock_merge:
            mock_merge.return_value = "model0"
            res = create_gemma4_from_pretrained(str(tmp_path), cfg)
            assert res == "model0"

        class MockState1:
            """Test stub for `MockState1`."""

            def to_flat_dict(self) -> StateNode:
                """Provide to flat dict test helper."""
                return {"a": 1}

        mock_split.return_value = (None, MockState1())
        with patch("gemma_4_sql.backends.jax.gemma4.params.nnx.merge") as mock_merge:
            mock_merge.return_value = "model1"
            res = create_gemma4_from_pretrained(str(tmp_path), cfg)
            assert res == "model1"

        class MockState2:
            """Test stub for `MockState2`."""

            def __iter__(self) -> Iterator[tuple[str, int]]:
                """Provide iter test helper."""
                yield from {"b": 2}.items()

        mock_split.return_value = (None, MockState2())
        with patch("gemma_4_sql.backends.jax.gemma4.params.nnx.merge") as mock_merge:
            mock_merge.return_value = "model2"
            res = create_gemma4_from_pretrained(str(tmp_path), cfg)
            assert res == "model2"
