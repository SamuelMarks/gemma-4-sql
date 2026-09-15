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


class Gemma4MultiModalProjector(nn.Module):
    """Multimodal projector for Gemma 4."""

    def __init__(self, config: Gemma4Config):
        """Initialize Gemma4MultiModalProjector."""
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.hidden_size, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Forward pass for multimodal projector.

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
        """Initialize Gemma4ForCausalLM."""
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
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...] | Cache | None]:
        """Forward pass of the model.

        Returns:
            Tuple containing output logits and updated past key values.
        """
        hidden_states = self.embed_tokens(input_ids)

        if pixel_values is not None:
            vision_outputs = self.vision_model(pixel_values)
            image_features = self.multi_modal_projector(vision_outputs)

            # Very simplified interleaving: assume image tokens are placed at the end of the sequence for now
            # A real implementation would find the `<image>` token in `input_ids` and splice `image_features` there.
            hidden_states = torch.cat([image_features, hidden_states], dim=1)

        if audio_values is not None:
            audio_features = self.audio_model(audio_values)
            # Very simplified interleaving
            hidden_states = torch.cat([audio_features, hidden_states], dim=1)

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

        cfg = config or Gemma4Config()
        model = cls(cfg)
        path = Path(str(model_name_or_path))
        target_file: Path | None = None
        if path.is_file() and (path.suffix == ".safetensors" or path.name.endswith(".safetensors")):
            target_file = path
        elif path.is_dir():
            st_path = path / "model.safetensors"
            if st_path.is_file():
                target_file = st_path
        if target_file is not None:
            try:
                from safetensors import SafetensorError
                from safetensors.torch import load_file

                state_dict = load_file(str(target_file))
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
