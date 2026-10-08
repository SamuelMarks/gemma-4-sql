"""Module docstring."""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.config import (
    AudioConfig,
    ModelConfig,
    VisionConfig,
)
from gemma_4_sql.backends.jax.gemma4.modeling import (
    Gemma4ForCausalLM,
    Gemma4Model,
    _download_and_load_pretrained,
    forward,
)


def test_gemma4_model():
    """Docstring for test_gemma4_model."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
    )
    model = Gemma4Model(config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])
    out = model(input_ids, positions)
    assert out.shape == (1, 3, 64)


def test_gemma4_model_per_layer_input():
    """Docstring for test_gemma4_model_per_layer_input."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        hidden_size_per_layer_input=32,
        vocab_size_per_layer_input=500,
    )
    model = Gemma4Model(config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])

    # Check get_per_layer_inputs directly
    pli = model.get_per_layer_inputs(input_ids)
    assert pli.shape == (1, 3, 2, 32)

    # Check project_per_layer_inputs directly
    x = jnp.ones((1, 3, 64))
    proj = model.project_per_layer_inputs(x, pli)
    assert proj.shape == (1, 3, 2, 32)

    # Test full model call
    out = model(input_ids, positions)
    assert out.shape == (1, 3, 64)

    # Test with provided per_layer_inputs
    out2 = model(input_ids, positions, per_layer_inputs=pli)
    assert out2.shape == (1, 3, 64)


@patch("gemma_4_sql.backends.jax.gemma4.modeling.create_gemma4_from_pretrained")
@patch("huggingface_hub.snapshot_download")
def test_download_and_load_pretrained(mock_download, mock_create):
    """Docstring for test_download_and_load_pretrained."""
    mock_download.return_value = "/path/to/model"
    mock_create.return_value = MagicMock()

    _download_and_load_pretrained("google/gemma-4-E2B")
    mock_download.assert_called_once()
    mock_create.assert_called_once()

    with pytest.raises(ValueError, match="Model name 'unknown' is unknown"):
        _download_and_load_pretrained("unknown")


@patch("gemma_4_sql.backends.jax.gemma4.modeling._download_and_load_pretrained")
def test_gemma4_for_causal_lm_from_pretrained(mock_download):
    """Docstring for test_gemma4_for_causal_lm_from_pretrained."""
    mock_download.return_value = MagicMock()
    Gemma4ForCausalLM.from_pretrained("google/gemma-4-E2B")
    mock_download.assert_called_once()


def test_gemma4_for_causal_lm():
    """Docstring for test_gemma4_for_causal_lm."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=1000,
    )
    model = Gemma4ForCausalLM(config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])
    logits = model(input_ids, positions)
    assert logits.shape == (1, 3, 1000)


def test_gemma4_for_causal_lm_softcapping():
    """Docstring for test_gemma4_for_causal_lm_softcapping."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=1000,
        final_logit_softcapping=30.0,
    )
    model = Gemma4ForCausalLM(config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])
    logits = model(input_ids, positions)
    assert logits.shape == (1, 3, 1000)


def test_gemma4_for_causal_lm_multimodal():
    """Docstring for test_gemma4_for_causal_lm_multimodal."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=1000,
        vision_config=VisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=2, image_size=224),
        audio_config=AudioConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=2),
    )
    model = Gemma4ForCausalLM(config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])

    # Fake MultimodalInputs
    pixel_values = jnp.ones((1, 224, 224, 3))
    image_token_mask = jnp.array([[False, True, False]])
    input_features = jnp.ones((1, 10, 128))
    input_features_mask = jnp.ones((1, 10))
    audio_token_mask = jnp.array([[False, False, True]])

    # Try forward with missing some masks to trigger edge cases if any
    logits1 = model(input_ids, positions, pixel_values=pixel_values, image_token_mask=image_token_mask)
    assert logits1.shape == (1, 3, 1000)

    logits2 = model(input_ids, positions, input_features=input_features, input_features_mask=input_features_mask, audio_token_mask=audio_token_mask)
    assert logits2.shape == (1, 3, 1000)

    logits3 = model(input_ids, positions, pixel_values=pixel_values, image_token_mask=image_token_mask, input_features=input_features, input_features_mask=input_features_mask, audio_token_mask=audio_token_mask)
    assert logits3.shape == (1, 3, 1000)


def test_gemma4_for_causal_lm_multimodal_per_layer_inputs():
    """Docstring for test_gemma4_for_causal_lm_multimodal_per_layer_inputs."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=1000,
        vision_config=VisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=2, image_size=224),
        audio_config=AudioConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=2),
        hidden_size_per_layer_input=32,
        vocab_size_per_layer_input=500,
    )
    model = Gemma4ForCausalLM(config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])
    pixel_values = jnp.ones((1, 224, 224, 3))
    image_token_mask = jnp.array([[False, True, False]])

    logits = model(input_ids, positions, pixel_values=pixel_values, image_token_mask=image_token_mask)
    assert logits.shape == (1, 3, 1000)


def test_forward_function():
    """Docstring for test_forward_function."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=1000,
    )
    model = Gemma4ForCausalLM(config, rngs=rngs)

    [MagicMock()] * 2
    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])

    # We only have a fake cache, wait cache might be used in attention.
    # Let's pass cache=None to avoid errors, or just use a valid cache if required.
    # We will pass cache=None since Gemma4Model handles cache=None by passing None to layers.
    # Actually `forward` takes `cache`, we can just pass None.

    out_logits, out_cache = forward(model, None, input_ids, positions)
    assert out_logits.shape == (1, 1000)
    assert out_cache is None


def test_gemma4_model_with_cache():
    """Docstring for test_gemma4_model_with_cache."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
    )
    model = Gemma4Model(config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])

    class DummyCache:
        """Docstring for DummyCache."""

        def __init__(self):
            """Docstring for __init__."""
            import jax.numpy as jnp
            from flax import nnx

            self.k_cache = nnx.Cache(jnp.zeros((1, 10, 2, 16)))
            self.v_cache = nnx.Cache(jnp.zeros((1, 10, 2, 16)))
            self.cur_ind = nnx.Cache(jnp.array(0))
            self.size = 10

    cache = [DummyCache(), DummyCache()]

    print("TYPE IS:", type(cache[0].k_cache))
    out = model(input_ids, positions, cache=cache)
    assert out.shape == (1, 3, 64)


@patch("gemma_4_sql.backends.jax.gemma4.modeling.create_gemma4_from_pretrained")
@patch("huggingface_hub.snapshot_download")
def test_download_and_load_pretrained_with_config(mock_download, mock_create):
    """Docstring for test_download_and_load_pretrained_with_config."""
    mock_download.return_value = "/path/to/model"
    mock_create.return_value = MagicMock()

    config = ModelConfig()
    _download_and_load_pretrained("some-model", config=config)
    mock_download.assert_called_once()
    mock_create.assert_called_once()


def test_gemma4_model_missing_masks():
    """Docstring for test_gemma4_model_missing_masks."""
    import jax.numpy as jnp
    from flax import nnx

    from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
    from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4Model

    rngs = nnx.Rngs(0)
    config = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
    )
    model = Gemma4Model(config, rngs=rngs)

    input_ids = jnp.array([[1, 2, 3]])
    positions = jnp.array([[0, 1, 2]])
    # Call without attention_mask
    out = model(input_ids, positions)
    assert out.shape == (1, 3, 64)
