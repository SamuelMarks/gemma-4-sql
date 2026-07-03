"""Provide module docstring."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .layers import Gemma4RMSNorm, _make_embed, _make_linear

if TYPE_CHECKING:
    from .config import ModelConfig, VisionConfig


class SiglipVisionEmbeddings(nnx.Module):
    """Embeddings for the SigLIP vision model."""

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2
        functools = __import__("functools")
        ki = functools.partial(jax.nn.initializers.lecun_normal())
        bi = functools.partial(jax.nn.initializers.zeros)
        self.patch_embedding = nnx.Conv(config.num_channels, config.hidden_size, kernel_size=(config.patch_size, config.patch_size), strides=(config.patch_size, config.patch_size), padding="valid", kernel_init=ki, bias_init=bi, rngs=rngs)
        self.position_embedding = _make_embed(self.num_patches, config.hidden_size, embedding_metadata={}, rngs=rngs)
        self.position_ids = jnp.expand_dims(jnp.arange(self.num_patches), 0)

    def __call__(self, pixel_values: Array) -> Array:
        """Apply patch and position embeddings to pixel values."""
        patch_embeds = self.patch_embedding(pixel_values)
        (b, h, w, c) = patch_embeds.shape
        embeddings = patch_embeds.reshape((b, h * w, c))
        return embeddings + self.position_embedding(self.position_ids)


class SiglipAttention(nnx.Module):
    """Attention block for SigLIP."""

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        (hs, _shd) = (config.hidden_size, config.shd_cfg)
        km = {}
        bm = {}
        self.q_proj = _make_linear(hs, hs, kernel_metadata=km, bias_metadata=bm, rngs=rngs)
        self.k_proj = _make_linear(hs, hs, kernel_metadata=km, bias_metadata=bm, rngs=rngs)
        self.v_proj = _make_linear(hs, hs, kernel_metadata=km, bias_metadata=bm, rngs=rngs)
        self.proj = _make_linear(hs, hs, kernel_metadata=km, bias_metadata=bm, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """Apply multi-head attention."""
        (b, t, _) = x.shape
        q = self.q_proj(x).reshape((b, t, self.num_heads, self.head_dim))
        k = self.k_proj(x).reshape((b, t, self.num_heads, self.head_dim))
        v = self.v_proj(x).reshape((b, t, self.num_heads, self.head_dim))
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 3, 1))
        v = jnp.transpose(v, (0, 2, 1, 3))
        scores = jnp.matmul(q, k) / jnp.sqrt(self.head_dim)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(attn_weights, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape((b, t, -1))
        return self.proj(out)


class SiglipMLP(nnx.Module):
    """MLP for SigLIP.

    Uses the tanh-approximate GELU (`approximate=True`) to match the
    `gelu_pytorch_tanh` activation used in the HuggingFace reference.
    """

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.config = config
        self.fc1 = _make_linear(config.hidden_size, config.intermediate_size, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        self.fc2 = _make_linear(config.intermediate_size, config.hidden_size, kernel_metadata={}, bias_metadata={}, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """Apply the MLP with tanh-approximate GELU activation."""
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=True)
        return self.fc2(x)


class SiglipEncoderLayer(nnx.Module):
    """A single SigLIP encoder layer.

    Uses Gemma4RMSNorm (matching the HuggingFace reference) rather than LayerNorm.
    """

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.config = config
        shd = config.shd_cfg.layer_norm
        self.layer_norm1 = Gemma4RMSNorm(config.hidden_size, eps=config.layer_norm_eps, _shd=shd, rngs=rngs)
        self.layer_norm2 = Gemma4RMSNorm(config.hidden_size, eps=config.layer_norm_eps, _shd=shd, rngs=rngs)
        self.self_attn = SiglipAttention(config, rngs=rngs)
        self.mlp = SiglipMLP(config, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """Process the encoder layer."""
        hidden = self.layer_norm1(x)
        hidden = self.self_attn(hidden)
        x = x + hidden
        hidden = self.layer_norm2(x)
        hidden = self.mlp(hidden)
        return x + hidden


class Gemma4MultimodalEmbedder(nnx.Module):
    """Embeds multimodal soft tokens (e.g., from audio) into language model space."""

    def __init__(self, multimodal_hidden_size: int, text_hidden_size: int, eps: float, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.embedding_projection = nnx.Linear(multimodal_hidden_size, text_hidden_size, use_bias=False, rngs=rngs)
        self.embedding_pre_projection_norm = Gemma4RMSNorm(multimodal_hidden_size, eps=eps, with_scale=False, rngs=rngs)

    def __call__(self, inputs_embeds: jax.Array) -> jax.Array:
        """Embeds multimodal inputs."""
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)


class SiglipVisionTransformer(nnx.Module):
    """The SigLIP Vision Transformer.

    Uses Gemma4RMSNorm throughout (matching the HuggingFace reference) rather
    than LayerNorm.
    """

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config, rngs=rngs)
        self.layers = nnx.List([SiglipEncoderLayer(config, rngs=rngs) for _ in range(config.num_hidden_layers)])
        shd = config.shd_cfg.layer_norm
        self.post_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.layer_norm_eps, _shd=shd, rngs=rngs)

    def __call__(self, pixel_values: Array) -> Array:
        """Apply the vision transformer to pixel values."""
        x = self.embeddings(pixel_values)
        for layer in self.layers:
            x = layer(x)
        return self.post_layernorm(x)


def _avg_pool_vision_outputs(x: Array, kernel_size: int, num_output_tokens: int, patches_per_img: int, tokens_per_side: int) -> Array:
    """Pools patch tokens into a fixed grid using position-based averaging.

    Each patch is assigned to a kernel bin via floor(position / kernel_size).
    Averaging is done with one-hot weights divided by kernel_size^2, matching
    the HuggingFace reference implementation exactly.

    Args:
    ----
        x: Patch embeddings (B, num_patches, hidden_size).
        kernel_size: Pooling kernel size.
        num_output_tokens: Total output tokens per image.
        patches_per_img: Number of patches along one spatial dimension.
        tokens_per_side: Number of output tokens along one spatial dimension.

    Returns:
    -------
        Pooled embeddings (B, num_output_tokens, hidden_size).

    """
    (_b, num_patches, _hidden) = x.shape
    k_sq = kernel_size * kernel_size
    positions = jnp.arange(num_patches)
    row = positions // patches_per_img
    col = positions % patches_per_img
    kernel_idxs = row // kernel_size * tokens_per_side + col // kernel_size
    weights = jax.nn.one_hot(kernel_idxs, num_output_tokens, dtype=jnp.float32) / k_sq
    return jnp.matmul(weights.T[None], x.astype(jnp.float32))


class Gemma4MultiModalProjector(nnx.Module):
    """Projects vision features into the language model's hidden dimension.

    Pools patch tokens using position-based weighted averaging (matching the
    HuggingFace reference), then projects into the text model's hidden space.

    Attributes
    ----------
        mm_input_projection_weight: Weight matrix (vision_hidden, text_hidden).
        mm_soft_emb_norm: RMSNorm applied to pooled patch embeddings.
        patches_per_img: Number of patches along one spatial dimension.
        tokens_per_side: Number of output tokens along one spatial dimension.
        kernel_size: Pooling kernel size (patches_per_img // tokens_per_side).
        num_output_tokens: Total output tokens per image (tokens_per_side ** 2).

    """

    def __init__(self, text_config: ModelConfig, vision_config: VisionConfig, mm_tokens_per_image: int, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        _ = mm_tokens_per_image
        self.text_config = text_config
        self.vision_config = vision_config
        (vhs, ths) = (vision_config.hidden_size, text_config.hidden_size)
        self.patches_per_img = vision_config.image_size // vision_config.patch_size
        self.tokens_per_side = int(mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_img // self.tokens_per_side
        self.num_output_tokens = self.tokens_per_side * self.tokens_per_side
        self.mm_input_projection_weight = nnx.Param(jnp.zeros((vhs, ths)), rngs=rngs)
        self.mm_soft_emb_norm = Gemma4RMSNorm(vhs, eps=vision_config.layer_norm_eps, dtype=text_config.dtype, rngs=rngs)

    def __call__(self, vision_outputs: Array) -> Array:
        """Projects and pools the vision outputs.

        Args:
        ----
            vision_outputs: Patch embeddings from the vision encoder (B, num_patches, hidden_size).

        Returns:
        -------
            Projected image tokens (B, num_output_tokens, text_hidden_size).

        """
        pooled = _avg_pool_vision_outputs(vision_outputs, self.kernel_size, self.num_output_tokens, self.patches_per_img, self.tokens_per_side)
        pooled = pooled * math.sqrt(self.vision_config.hidden_size)
        pooled = pooled.astype(self.text_config.dtype)
        pooled = self.mm_soft_emb_norm(pooled)
        return jnp.matmul(pooled, self.mm_input_projection_weight[...])
