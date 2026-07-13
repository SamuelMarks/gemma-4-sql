"""PyTorch native gemma 4 implementation."""

from __future__ import annotations

from .attention import Gemma4Attention
from .audio import Gemma4AudioModel
from .audio_attention import Gemma4AudioCrossAttention
from .audio_layers import Gemma4AudioEncoderBlock, Gemma4AudioFeatureExtractor
from .cache import Cache, DynamicCache, StaticCache
from .config import Gemma4AudioConfig, Gemma4Config, Gemma4VisionConfig
from .decoder_layer import Gemma4DecoderLayer
from .layers import Gemma4MLP, Gemma4RMSNorm
from .modeling import Gemma4ForCausalLM
from .moe import Gemma4MoE
from .rope import Gemma4RotaryEmbedding, Gemma4RotaryEmbedding2D, apply_rotary_pos_emb
from .utils_params import translate_jax_to_pytorch
from .vision import Gemma4VisionEmbeddings, Gemma4VisionModel

__all__ = [
    "Cache",
    "DynamicCache",
    "Gemma4Attention",
    "Gemma4AudioConfig",
    "Gemma4AudioCrossAttention",
    "Gemma4AudioEncoderBlock",
    "Gemma4AudioFeatureExtractor",
    "Gemma4AudioModel",
    "Gemma4Config",
    "Gemma4DecoderLayer",
    "Gemma4ForCausalLM",
    "Gemma4MLP",
    "Gemma4MoE",
    "Gemma4RMSNorm",
    "Gemma4RotaryEmbedding",
    "Gemma4RotaryEmbedding2D",
    "Gemma4VisionConfig",
    "Gemma4VisionEmbeddings",
    "Gemma4VisionModel",
    "StaticCache",
    "apply_rotary_pos_emb",
    "translate_jax_to_pytorch",
]
