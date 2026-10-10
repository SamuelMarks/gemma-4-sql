"""Module docstring."""

from unittest.mock import patch

import pytest


def test_safetensors_missing():
    """Docstring for test_safetensors_missing."""
    import builtins
    import importlib

    import gemma_4_sql.backends.jax.gemma4.params as mod

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "safetensors":
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(mod)
        assert mod.safetensors is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(mod)


def test_stack_expert_tensors_none_key():
    """Docstring for test_stack_expert_tensors_none_key."""
    import jax.numpy as jnp

    from gemma_4_sql.backends.jax.gemma4.params import _stack_and_assign_expert_tensors

    expert_tensors = {0: {"test_proj": {0: jnp.zeros((1,)), 1: jnp.zeros((1,))}}}

    # mapping that returns (None, None) for jax_key
    class MockMapping:
        """Docstring for MockMapping."""

        @staticmethod
        def __getitem__(key):
            """Docstring for __getitem__."""
            raise KeyError()  # or whatever map_to_jax_key does.

    mapping = {}

    jax_state = {}
    _stack_and_assign_expert_tensors(expert_tensors, mapping, jax_state)
    assert jax_state == {}


def test_create_gemma4_from_pretrained(tmp_path):
    """Docstring for test_create_gemma4_from_pretrained."""
    import numpy as np
    from safetensors.numpy import save_file

    from gemma_4_sql.backends.jax.gemma4.config import AudioConfig, ModelConfig, VisionConfig
    from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained

    # Save a fake safetensors file
    tensors = {
        "model.embed_tokens.weight": np.zeros((10, 8), dtype=np.float32),
        "model.norm.weight": np.ones((8,), dtype=np.float32),
        "lm_head.weight": np.zeros((10, 8), dtype=np.float32),
        # Audio keys
        "audio_tower.output_proj.weight": np.zeros((8, 8), dtype=np.float32),
        "audio_tower.output_proj.bias": np.zeros((8,), dtype=np.float32),
        # Vision keys
        "vision_tower.vision_model.post_layernorm.weight": np.zeros((8,), dtype=np.float32),
        # MOE keys
        "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight": np.zeros((16, 8), dtype=np.float32),
        "model.layers.0.block_sparse_moe.experts.1.gate_proj.weight": np.zeros((16, 8), dtype=np.float32),
        "model.layers.0.block_sparse_moe.experts.2.gate_proj.weight": np.zeros((16, 8), dtype=np.float32),
        "model.layers.0.block_sparse_moe.experts.3.gate_proj.weight": np.zeros((16, 8), dtype=np.float32),
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

    # Test error cases
    with pytest.raises(ValueError, match="No safetensors found"):
        create_gemma4_from_pretrained(str(tmp_path / "empty"), cfg)


def test_process_standard_tensor_errors():
    """Docstring for test_process_standard_tensor_errors."""
    import numpy as np

    from gemma_4_sql.backends.jax.gemma4.params import _get_key_and_transform_mapping, process_standard_tensor

    mapping = _get_key_and_transform_mapping()

    class FakeSF:
        """Docstring for FakeSF."""

        def get_tensor(self, key):
            """Docstring for get_tensor."""
            return np.zeros((1,))

    # Test skipping standard KeyError exception by assign_weights_from_eval_shape
    jax_state = {}
    process_standard_tensor(FakeSF(), "model.embed_tokens.weight", jax_state, mapping)

    # Test jax_key is None
    process_standard_tensor(FakeSF(), "unknown.key", jax_state, mapping)

    # Test raising AttributeError inside try block
    class FakeSF2:
        """Docstring for FakeSF2."""

        def get_tensor(self, key):
            """Docstring for get_tensor."""
            return np.zeros((1,))

    with patch("gemma_4_sql.backends.jax.gemma4.params.assign_weights_from_eval_shape", side_effect=AttributeError("bad")):
        with pytest.raises(AttributeError):
            process_standard_tensor(FakeSF2(), "model.embed_tokens.weight", jax_state, mapping)


def test_create_gemma4_from_pretrained_etils_success(tmp_path, monkeypatch):
    """Docstring for test_create_gemma4_from_pretrained_etils_success."""
    import sys

    import numpy as np
    from safetensors.numpy import save_file

    from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
    from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained

    save_file({"lm_head.weight": np.zeros((10, 8), dtype=np.float32)}, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=4, intermediate_size=16)

    # Remove etils from sys.modules to force import
    if "etils" in sys.modules:
        monkeypatch.delitem(sys.modules, "etils")

    with patch("flax.nnx.split") as mock_split:
        mock_split.return_value = (None, {})
        with patch("gemma_4_sql.backends.jax.gemma4.params.nnx.merge"):
            with patch("gemma_4_sql.backends.jax.gemma4.params.hasattr", side_effect=lambda obj, name: False if name == "State" else hasattr(obj, name)):
                create_gemma4_from_pretrained(str(tmp_path), cfg)


def test_create_gemma4_from_pretrained_etils_missing(tmp_path, monkeypatch):
    """Docstring for test_create_gemma4_from_pretrained_etils_missing."""
    import sys

    from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
    from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained
    from gemma_4_sql.exceptions import DependencyMissingError

    cfg = ModelConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=4, intermediate_size=16)

    # Mock sys.modules to not have etils
    monkeypatch.setitem(sys.modules, "etils", None)

    import builtins

    orig_import = builtins.__import__
    with patch("builtins.__import__") as mock_import:

        def side_effect(name, *args, **kwargs):
            """Docstring for side_effect."""
            if name == "etils":
                raise ImportError("no etils")
            return orig_import(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        with pytest.raises(DependencyMissingError):
            create_gemma4_from_pretrained(str(tmp_path), cfg)


def test_fix_jax_state_embeddings_coverage():
    """Docstring for test_fix_jax_state_embeddings_coverage."""
    import jax
    import jax.numpy as jnp

    from gemma_4_sql.backends.jax.gemma4.config import ModelConfig, VisionConfig
    from gemma_4_sql.backends.jax.gemma4.params import _fix_jax_state_embeddings

    cfg = ModelConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=4, intermediate_size=16, vision_config=VisionConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, patch_size=4))

    # Test when embed_scale is ShapeDtypeStruct
    jax_state = {"model": {"embed_scale": jax.ShapeDtypeStruct((), jnp.float32)}, "vision_tower": {"embeddings": {"position_ids": jax.ShapeDtypeStruct((1, 100), jnp.int32)}}}

    _fix_jax_state_embeddings(jax_state, None, cfg)
    assert isinstance(jax_state["model"]["embed_scale"], jax.Array)
    assert isinstance(jax_state["vision_tower"]["embeddings"]["position_ids"], jax.Array)

    # Test when vision_config is None
    cfg2 = ModelConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=4, intermediate_size=16)
    jax_state2 = {"model": {"embed_scale": 1.0}}
    _fix_jax_state_embeddings(jax_state2, None, cfg2)


def test_create_gemma4_state_dict_fallbacks(tmp_path):
    """Docstring for test_create_gemma4_state_dict_fallbacks."""
    import numpy as np
    from safetensors.numpy import save_file

    from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
    from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained

    save_file({"lm_head.weight": np.zeros((10, 8), dtype=np.float32)}, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=4, intermediate_size=16)

    with patch("flax.nnx.split") as mock_split:
        # Test to_pure_dict
        class MockState0:
            """Docstring for MockState0."""

            def to_pure_dict(self):
                """Docstring for to_pure_dict."""
                return {"a": 0}

        mock_split.return_value = (None, MockState0())

        with patch("gemma_4_sql.backends.jax.gemma4.params.nnx.merge") as mock_merge:
            mock_merge.return_value = "model0"
            res = create_gemma4_from_pretrained(str(tmp_path), cfg)
            assert res == "model0"

        # Test to_flat_dict fallback
        class MockState1:
            """Docstring for MockState1."""

            def to_flat_dict(self):
                """Docstring for to_flat_dict."""
                return {"a": 1}

        mock_split.return_value = (None, MockState1())

        with patch("gemma_4_sql.backends.jax.gemma4.params.nnx.merge") as mock_merge:
            mock_merge.return_value = "model1"
            res = create_gemma4_from_pretrained(str(tmp_path), cfg)
            assert res == "model1"

        # Test dict() fallback
        class MockState2:
            """Docstring for MockState2."""

            def __iter__(self):
                """Docstring for __iter__."""
                yield from {"b": 2}.items()

        mock_split.return_value = (None, MockState2())

        with patch("gemma_4_sql.backends.jax.gemma4.params.nnx.merge") as mock_merge:
            mock_merge.return_value = "model2"
            res = create_gemma4_from_pretrained(str(tmp_path), cfg)
            assert res == "model2"
