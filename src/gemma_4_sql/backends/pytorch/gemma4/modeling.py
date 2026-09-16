"""PyTorch native Gemma 4 modeling."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .audio import Gemma4AudioModel
from .cache import Cache, DynamicCache
from .config import Gemma4Config
from .decoder_layer import Gemma4DecoderLayer
from .layers import Gemma4RMSNorm
from .vision import Gemma4VisionModel

__all__ = [
    "Gemma4Config",
    "Gemma4ForCausalLM",
    "Gemma4MultiModalProjector",
    "merge_modality_embeddings",
]


def merge_modality_embeddings(
    modality_features: torch.Tensor,
    text_embeddings: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge projected multimodal feature embeddings into text embeddings at placeholder positions.

    Maintains algorithmic and mathematical parity with the JAX implementation
    (batched_merge_modalities in backends/jax/gemma4/multimodal.py).

    Mathematical Formulation:
        Given text embeddings T in R^{B x L x D}, multimodal features M in R^{B x K x D},
        and binary token mask P in {0, 1}^{B x L}:
            idx_{b, t} = clip(cumsum(P_{b, :})_{t} - 1, 0, K - 1)
            aligned_{b, t} = M_{b, idx_{b, t}}
            output_{b, t} = aligned_{b, t} if P_{b, t} == 1 else T_{b, t}

        If no placeholder tokens are marked in token_mask (or token_mask is None or empty),
        the multimodal features are concatenated with the text embeddings:
            output = cat([modality_features, text_embeddings], dim=1)

    Args:
        modality_features: Projected multimodal feature tensor of shape (B, K, D).
        text_embeddings: Text token embedding tensor of shape (B, L, D).
        token_mask: Optional boolean or integer mask tensor of shape (B, L).

    Returns:
        Tensor of merged embeddings of shape (B, L, D) or (B, K + L, D).
    """
    if token_mask is None or not token_mask.any():
        return torch.cat([modality_features, text_embeddings], dim=1)

    batch_size, seq_len, hidden_dim = text_embeddings.shape
    num_features = modality_features.shape[1]

    mask_long = token_mask.long()
    indices = torch.cumsum(mask_long, dim=1) - 1
    safe_indices = torch.clamp(indices, min=0, max=num_features - 1)

    batch_idx = torch.arange(batch_size, device=modality_features.device).unsqueeze(1).expand(-1, seq_len)
    aligned_features = modality_features[batch_idx, safe_indices]

    mask_expanded = token_mask.bool().unsqueeze(-1).expand(-1, -1, hidden_dim)
    return torch.where(mask_expanded, aligned_features, text_embeddings)


class Gemma4MultiModalProjector(nn.Module):
    """Multimodal projector for Gemma 4."""

    def __init__(self, config: Gemma4Config):
        """Initialize Gemma4MultiModalProjector.

        Args:
            config: Gemma 4 configuration object.
        """
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.hidden_size, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Forward pass for multimodal projector.

        Args:
            image_features: Tensor of vision model outputs.

        Returns:
            Projected multimodal features.
        """
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Gemma4ForCausalLM(nn.Module):
    """Gemma 4 model for causal language modeling."""

    def __init__(self, config: Gemma4Config):
        """Initialize Gemma4ForCausalLM.

        Args:
            config: Gemma 4 configuration object.
        """
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Text embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # Multimodal sub-models
        self.vision_model = Gemma4VisionModel(config.vision_config)
        self.multi_modal_projector = Gemma4MultiModalProjector(config)
        self.audio_model = Gemma4AudioModel(config)

        self.layers = nn.ModuleList([Gemma4DecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embed_tokens.weight = self.lm_head.weight  # Tie weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | Cache | None = None,
        pixel_values: torch.Tensor | None = None,
        audio_values: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
        audio_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...] | Cache | None]:
        """Forward pass of the Gemma 4 multimodal model.

        Args:
            input_ids: Input token ID tensor of shape (batch_size, sequence_length).
            attention_mask: Optional attention mask tensor.
            position_ids: Optional position ID tensor.
            past_key_values: Optional cached key-value states.
            pixel_values: Optional image pixel values tensor.
            audio_values: Optional raw audio waveform tensor.
            image_token_mask: Optional boolean mask for image placeholder tokens.
            audio_token_mask: Optional boolean mask for audio placeholder tokens.

        Returns:
            Tuple containing output logits and updated past key values.
        """
        hidden_states = self.embed_tokens(input_ids)

        if pixel_values is not None:
            vision_outputs = self.vision_model(pixel_values)
            image_features = self.multi_modal_projector(vision_outputs)
            if image_token_mask is None:
                image_token_id = getattr(self.config, "image_token_id", 255999)
                image_token_mask = input_ids == image_token_id

            hidden_states = merge_modality_embeddings(
                modality_features=image_features,
                text_embeddings=hidden_states,
                token_mask=image_token_mask,
            )

        if audio_values is not None:
            audio_features = self.audio_model(audio_values)
            if audio_token_mask is None:
                audio_token_id = getattr(self.config, "audio_token_id", 255998)
                audio_token_mask = input_ids == audio_token_id

            hidden_states = merge_modality_embeddings(
                modality_features=audio_features,
                text_embeddings=hidden_states,
                token_mask=audio_token_mask,
            )

        curr_seq_len = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(0, curr_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)

        if attention_mask is not None and attention_mask.dim() == 2 and attention_mask.shape[1] != curr_seq_len:
            diff = curr_seq_len - attention_mask.shape[1]
            if diff > 0:
                prefix_mask = torch.ones((attention_mask.shape[0], diff), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        next_decoder_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()

        for idx, decoder_layer in enumerate(self.layers):
            layer_past_key_value: tuple[torch.Tensor, torch.Tensor] | Cache | None
            if past_key_values is None:
                layer_past_key_value = None
            elif isinstance(past_key_values, Cache):
                layer_past_key_value = past_key_values
            else:
                layer_past_key_value = past_key_values[idx]

            hidden_states, present_key_value, _router_logits = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past_key_value,
            )
            if present_key_value is not None and not isinstance(present_key_value, Cache):
                next_decoder_cache += (present_key_value,)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if isinstance(past_key_values, Cache):
            return logits, past_key_values

        return logits, next_decoder_cache if len(next_decoder_cache) > 0 else None

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        min_new_tokens: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate text using autoregressive generation with DynamicCache KV caching.

        Args:
            input_ids: Input tensor of token IDs.
            max_new_tokens: Maximum number of tokens to generate.
            min_new_tokens: Minimum number of tokens to generate.
            **kwargs: Extra generation parameters ignored or handled.

        Returns:
            Tensor of generated token IDs including prompt tokens.
        """
        gen_cache: Cache | tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = DynamicCache()
        for i in range(max_new_tokens):
            curr_input = input_ids if i == 0 else input_ids[:, -1:]
            call_res = self(
                curr_input,
                past_key_values=gen_cache,
            )
            logits = call_res[0]
            gen_cache = call_res[1]
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        return input_ids

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Gemma4Config | None = None,
        **kwargs: object,
    ) -> Gemma4ForCausalLM:
        """Load native Gemma 4 model from path or initialize with config.

        Args:
            model_name_or_path: Directory, safetensors path, or model identifier.
            config: Optional Gemma4Config instance.
            **kwargs: Extra arguments.

        Returns:
            Initialized Gemma4ForCausalLM model instance.
        """
        from pathlib import Path

        from safetensors.torch import load_file

        path = Path(str(model_name_or_path))
        if config is None:
            config_json_path = (path if path.is_dir() else path.parent) / "config.json"
            if config_json_path.is_file():
                import json

                with open(config_json_path) as f:
                    config_dict = json.load(f)
                config = Gemma4Config(**config_dict)
            else:
                config = Gemma4Config()

        model = cls(config)
        weights_file: Path | None = None
        if path.is_file():
            weights_file = path
        elif path.is_dir():
            st_path = path / "model.safetensors"
            if st_path.is_file():
                weights_file = st_path

        if weights_file is not None and weights_file.exists():
            try:
                from safetensors import SafetensorError

                state_dict = load_file(str(weights_file))
                model.load_state_dict(state_dict, strict=False)
            except (SafetensorError, ValueError, RuntimeError, OSError, KeyError, AttributeError) as exc:
                import logging

                logging.getLogger(__name__).debug("Failed to load safetensors: %s", exc)

        return model

    def save_pretrained(self, save_directory: str) -> str:
        """Save native model weights to a safetensors file in save_directory.

        Args:
            save_directory: Directory path to save model weights.

        Returns:
            Path to the saved safetensors file.
        """
        from pathlib import Path

        from safetensors.torch import save_file

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / "model.safetensors"
        state_dict = {k: v.clone() if k == "lm_head.weight" else v for k, v in self.state_dict().items()}
        save_file(state_dict, str(file_path))
        return str(file_path)
