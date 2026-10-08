"""Module docstring."""

import jax.numpy as jnp

from gemma_4_sql.backends.jax.gemma4.multimodal import MultimodalInputs, batched_merge_modalities


def test_batched_merge_modalities():
    """Docstring for test_batched_merge_modalities."""
    img_emb = jnp.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])
    text_emb = jnp.array([[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], [[40.0, 40.0], [50.0, 50.0], [60.0, 60.0]]])
    token_mask = jnp.array([[True, False, True], [False, True, False]])

    res = batched_merge_modalities(img_emb, text_emb, token_mask)

    assert res.shape == (2, 3, 2)
    assert jnp.allclose(res[0], jnp.array([[1.0, 1.0], [20.0, 20.0], [2.0, 2.0]]))
    assert jnp.allclose(res[1], jnp.array([[40.0, 40.0], [3.0, 3.0], [60.0, 60.0]]))


def test_multimodal_inputs():
    """Docstring for test_multimodal_inputs."""
    inputs = MultimodalInputs(input_ids=jnp.array([1, 2, 3]), pixel_values=jnp.array([1.0]), image_token_mask=jnp.array([True]), input_features=None, input_features_mask=None, audio_token_mask=None, attention_mask=None)
    assert inputs.input_ids.shape == (3,)
    assert inputs.pixel_values.shape == (1,)
    assert inputs.image_token_mask[0]
    assert inputs.input_features is None
