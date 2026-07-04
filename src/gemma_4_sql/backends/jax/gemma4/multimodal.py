# Copyright 2024
"""Multimodal utilities for Gemma 4."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


def batched_merge_modalities(img_emb: Array, text_emb: Array, token_mask: Array) -> Array:
    """Merge image and text embeddings based on a token mask.

    Args:
    ----
        img_emb: Image embeddings (B, Li, D)
        text_emb: Text embeddings (B, Lt, D)
        token_mask: Boolean mask indicating image token positions (B, Lt)

    Returns:
    -------
        Merged embeddings (B, Lt, D)

    """

    def merge_modalities(i_emb: object, t_emb: object, mask: object) -> object:
        """Merge image and text embeddings using the provided token mask.

        Returns:
            object: The resulting output from the operation.

        """
        img_indices = jnp.cumsum(mask) - 1
        safe_indices = jnp.clip(img_indices, 0, i_emb.shape[0] - 1)
        aligned_images = i_emb[safe_indices]
        return jnp.where(mask[:, None], aligned_images, t_emb)

    return jax.vmap(merge_modalities)(img_emb, text_emb, token_mask)


@dataclass
class MultimodalInputs:
    """Container for multimodal input tensors."""

    input_ids: Array
    pixel_values: Array | None = None
    image_token_mask: Array | None = None
    input_features: Array | None = None
    input_features_mask: Array | None = None
    audio_token_mask: Array | None = None
