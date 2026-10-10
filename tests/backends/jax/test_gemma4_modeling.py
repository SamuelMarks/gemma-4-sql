"""Module docstring."""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.cache import init_cache
from gemma_4_sql.backends.jax.gemma4.config import AudioConfig, ModelConfig, VisionConfig
from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM, Gemma4Model, _default_jit, forward


@pytest.fixture
def tiny_config():
    """Docstring for tiny_config."""
    return ModelConfig(
        vocab_size=10,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
    )


@pytest.fixture
def tiny_ple_config():
    """Docstring for tiny_ple_config."""
    return ModelConfig(
        vocab_size=10,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
        hidden_size_per_layer_input=4,
        vocab_size_per_layer_input=10,
    )


@pytest.fixture
def tiny_multimodal_config():
    """Docstring for tiny_multimodal_config."""
    return ModelConfig(
        vocab_size=10,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
        vision_config=VisionConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, patch_size=4, image_size=16),
        audio_config=AudioConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2),
        final_logit_softcapping=30.0,
    )


def test_gemma4_model_forward(tiny_config):
    """Docstring for test_gemma4_model_forward."""
    rngs = nnx.Rngs(0)
    model = Gemma4Model(tiny_config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])

    # Forward pass
    output = model(input_ids, positions)
    assert output.shape == (1, 3, 8)

    # Forward pass with cache
    cache = init_cache(tiny_config, batch_size=1, max_seq_len=3)
    model(input_ids, positions, cache=cache)


def test_gemma4_model_ple(tiny_ple_config):
    """Docstring for test_gemma4_model_ple."""
    rngs = nnx.Rngs(0)
    model = Gemma4Model(tiny_ple_config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])

    # Forward pass should use PLE automatically
    output = model(input_ids, positions)
    assert output.shape == (1, 3, 8)

    # Explicitly get and project PLE
    ple = model.get_per_layer_inputs(input_ids)
    assert ple.shape == (1, 3, 2, 4)  # batch, seq, layers, ple_hidden

    embeds = model.embed_tokens(input_ids)
    proj = model.project_per_layer_inputs(embeds, ple)
    assert proj.shape == (1, 3, 2, 4)

    # Forward pass with explicit PLE
    model(input_ids, positions, per_layer_inputs=ple)


def test_gemma4_causal_lm(tiny_config):
    """Docstring for test_gemma4_causal_lm."""
    rngs = nnx.Rngs(0)
    model = Gemma4ForCausalLM(tiny_config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])

    logits = model(input_ids, positions)
    assert logits.shape == (1, 3, 10)


def test_gemma4_multimodal(tiny_multimodal_config):
    """Docstring for test_gemma4_multimodal."""
    rngs = nnx.Rngs(0)
    model = Gemma4ForCausalLM(tiny_multimodal_config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])

    pixel_values = jnp.ones((1, 16, 16, 3))
    image_token_mask = jnp.array([[False, True, False]])

    input_features = jnp.ones((1, 100, 128))
    input_features_mask = jnp.ones((1, 100))
    audio_token_mask = jnp.array([[False, False, True]])

    logits = model(
        input_ids,
        positions,
        pixel_values=pixel_values,
        image_token_mask=image_token_mask,
        input_features=input_features,
        input_features_mask=input_features_mask,
        audio_token_mask=audio_token_mask,
    )
    assert logits.shape == (1, 3, 10)


def test_gemma4_multimodal_with_ple(tiny_multimodal_config, tiny_ple_config):
    # Combine configs for multimodal + ple
    """Docstring for test_gemma4_multimodal_with_ple."""
    config = tiny_ple_config
    config.vision_config = VisionConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, patch_size=4, image_size=16)
    config.audio_config = AudioConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2)

    rngs = nnx.Rngs(0)
    model = Gemma4ForCausalLM(config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])

    pixel_values = jnp.ones((1, 16, 16, 3))
    image_token_mask = jnp.array([[False, True, False]])

    logits = model(input_ids, positions, pixel_values=pixel_values, image_token_mask=image_token_mask)
    assert logits.shape == (1, 3, 10)


def test_gemma4_multimodal_missing_masks(tiny_multimodal_config):
    """Docstring for test_gemma4_multimodal_missing_masks."""
    rngs = nnx.Rngs(0)
    model = Gemma4ForCausalLM(tiny_multimodal_config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])

    pixel_values = jnp.ones((1, 16, 16, 3))
    input_features = jnp.ones((1, 100, 128))
    input_features_mask = jnp.ones((1, 100))

    # Missing masks
    logits = model(input_ids, positions, pixel_values=pixel_values, input_features=input_features, input_features_mask=input_features_mask)
    assert logits.shape == (1, 3, 10)


def test_gemma4_model_ple_none(tiny_ple_config):
    """Docstring for test_gemma4_model_ple_none."""
    rngs = nnx.Rngs(0)
    model = Gemma4Model(tiny_ple_config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])

    # Forward pass explicitly without PLE, even if config expects it
    model(input_ids, positions, per_layer_inputs=None)


def test_gemma4_model_ple_none_config(tiny_config):
    """Docstring for test_gemma4_model_ple_none_config."""
    rngs = nnx.Rngs(0)
    model = Gemma4Model(tiny_config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    embeds = model.embed_tokens(input_ids)

    # get_per_layer_inputs with None hidden_size_per_layer_input
    ple = model.get_per_layer_inputs(input_ids)
    assert jnp.array_equal(ple, input_ids)

    # project_per_layer_inputs with None per_layer_inputs
    with pytest.raises(Exception):
        model.project_per_layer_inputs(embeds, per_layer_inputs=None)


def test_project_ple_none(tiny_ple_config):
    """Docstring for test_project_ple_none."""
    rngs = nnx.Rngs(0)
    model = Gemma4Model(tiny_ple_config, rngs=rngs)
    input_ids = jnp.array([[1, 2, 3]])
    embeds = model.embed_tokens(input_ids)
    proj = model.project_per_layer_inputs(embeds, per_layer_inputs=None)
    assert proj.shape == (1, 3, 2, 4)


def test_from_pretrained():
    """Docstring for test_from_pretrained."""
    import sys

    mock_huggingface = MagicMock()
    mock_huggingface.snapshot_download.return_value = "path/to/model"

    with patch.dict(sys.modules, {"huggingface_hub": mock_huggingface}):
        with patch("gemma_4_sql.backends.jax.gemma4.modeling.create_gemma4_from_pretrained") as mock_create:
            mock_create.return_value = "mocked_model"

            # Use known model name
            res = Gemma4ForCausalLM.from_pretrained("google/gemma-4-E2B")
            assert res == "mocked_model"

            # Use unknown model name without config
            with pytest.raises(ValueError, match="Model name 'unknown' is unknown"):
                Gemma4ForCausalLM.from_pretrained("unknown")

            # Use unknown model with config
            res = Gemma4ForCausalLM.from_pretrained("unknown", config=ModelConfig())
            assert res == "mocked_model"


def test_default_jit():
    """Docstring for test_default_jit."""
    assert _default_jit("x") == "x"


def test_forward(tiny_config):
    """Docstring for test_forward."""
    rngs = nnx.Rngs(0)
    model = Gemma4ForCausalLM(tiny_config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])
    cache = init_cache(tiny_config, batch_size=1, max_seq_len=3)

    logits, new_cache = forward(model, cache, input_ids, positions)
    assert logits.shape == (1, 10)
    assert new_cache == cache


def test_gemma4_multimodal_audio_only(tiny_multimodal_config):
    """Docstring for test_gemma4_multimodal_audio_only."""
    import jax.numpy as jnp
    from flax import nnx

    from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM

    rngs = nnx.Rngs(0)
    model = Gemma4ForCausalLM(tiny_multimodal_config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])

    input_features = jnp.ones((1, 100, 128))
    input_features_mask = jnp.ones((1, 100))
    audio_token_mask = jnp.array([[False, True, False]])

    logits = model(
        input_ids,
        positions,
        pixel_values=None,
        image_token_mask=None,
        input_features=input_features,
        input_features_mask=input_features_mask,
        audio_token_mask=audio_token_mask,
    )
    assert logits.shape == (1, 3, 10)
