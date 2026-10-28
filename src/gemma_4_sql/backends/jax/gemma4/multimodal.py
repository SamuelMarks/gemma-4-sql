"""Multimodal utilities for Gemma 4."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


def batched_merge_modalities(img_emb: Array, text_emb: Array, token_mask: Array) -> Array:
    """Merge continuous vision/audio feature embeddings into text token embeddings at masked placeholder positions.

    Args:
        img_emb: Continuous modality feature tensor of shape `(batch_size, num_modality_tokens, hidden_size)`.
        text_emb: Text token embedding tensor of shape `(batch_size, seq_len, hidden_size)`.
        token_mask: Boolean mask tensor of shape `(batch_size, seq_len)` indicating placeholder token positions.

    Returns:
        Merged multimodal sequence embedding tensor of shape `(batch_size, seq_len, hidden_size)`.

    """

    def merge_modalities(i_emb: Array, t_emb: Array, mask: Array) -> Array:
        """Merge single-example modality and text embeddings using a boolean token mask.

        Args:
            i_emb: Modality feature embeddings for one batch element.
            t_emb: Text token embeddings for one batch element.
            mask: Boolean mask indicating modality token positions.

        Returns:
            Combined embedding sequence for one batch element.

        """
        img_indices = jnp.cumsum(mask) - 1
        safe_indices = jnp.clip(img_indices, 0, i_emb.shape[0] - 1)
        aligned_images = i_emb[safe_indices]
        return jnp.where(mask[:, None], aligned_images, t_emb)

    res: Array = jax.vmap(merge_modalities)(img_emb, text_emb, token_mask)
    return res


@dataclass
class MultimodalInputs:
    """Container for multimodal input tensors."""

    input_ids: Array
    pixel_values: Array | None = None
    image_token_mask: Array | None = None
    input_features: Array | None = None
    input_features_mask: Array | None = None
    audio_token_mask: Array | None = None
    attention_mask: Array | None = None
