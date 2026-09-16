"""Tests for PyTorch Gemma 4 multimodal token interleaving and positional splicing."""

from __future__ import annotations

import pytest
import torch

from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
from gemma_4_sql.backends.pytorch.gemma4.modeling import (
    Gemma4ForCausalLM,
    merge_modality_embeddings,
)


def test_merge_modality_embeddings_exact_splicing() -> None:
    """Test exact positional splicing of multimodal tokens into text embeddings."""
    batch_size = 2
    seq_len = 5
    hidden_dim = 8
    num_modal_tokens = 2

    text_emb = torch.zeros((batch_size, seq_len, hidden_dim), requires_grad=True)
    text_emb.data.fill_(1.0)

    modal_features = torch.zeros((batch_size, num_modal_tokens, hidden_dim), requires_grad=True)
    modal_features.data[0, 0].fill_(10.0)  # First image token
    modal_features.data[0, 1].fill_(20.0)  # Second image token
    modal_features.data[1, 0].fill_(30.0)
    modal_features.data[1, 1].fill_(40.0)

    # Placeholders at index 1 and index 3
    token_mask = torch.tensor(
        [
            [False, True, False, True, False],
            [False, True, False, True, False],
        ],
        dtype=torch.bool,
    )

    merged = merge_modality_embeddings(modal_features, text_emb, token_mask)
    assert merged.shape == (batch_size, seq_len, hidden_dim)

    # Non-placeholder positions should keep text embeddings (1.0)
    assert torch.all(merged[:, 0, :] == 1.0)
    assert torch.all(merged[:, 2, :] == 1.0)
    assert torch.all(merged[:, 4, :] == 1.0)

    # Placeholder positions must match exact spliced feature tokens
    assert torch.all(merged[0, 1, :] == 10.0)
    assert torch.all(merged[0, 3, :] == 20.0)
    assert torch.all(merged[1, 1, :] == 30.0)
    assert torch.all(merged[1, 3, :] == 40.0)

    # Verify backward pass and gradient flow
    loss = merged.sum()
    loss.backward()
    assert text_emb.grad is not None
    assert modal_features.grad is not None
    assert text_emb.grad.shape == text_emb.shape
    assert modal_features.grad.shape == modal_features.shape
    # Gradient on text_emb should be 0 at placeholder positions (1 and 3)
    assert torch.all(text_emb.grad[:, 1, :] == 0.0)
    assert torch.all(text_emb.grad[:, 3, :] == 0.0)
    # Gradient on text_emb should be 1 at non-placeholder positions (0, 2, 4)
    assert torch.all(text_emb.grad[:, 0, :] == 1.0)


def test_merge_modality_embeddings_fallback_empty_or_none() -> None:
    """Test fallback to concatenation when token mask is None or empty."""
    text_emb = torch.ones((2, 4, 8))
    modal_features = torch.full((2, 3, 8), 5.0)

    # None mask
    merged_none = merge_modality_embeddings(modal_features, text_emb, token_mask=None)
    assert merged_none.shape == (2, 7, 8)
    assert torch.all(merged_none[:, :3, :] == 5.0)
    assert torch.all(merged_none[:, 3:, :] == 1.0)

    # All false mask
    empty_mask = torch.zeros((2, 4), dtype=torch.bool)
    merged_empty = merge_modality_embeddings(modal_features, text_emb, token_mask=empty_mask)
    assert merged_empty.shape == (2, 7, 8)


def test_gemma4_causal_lm_multimodal_interleaving_forward() -> None:
    """Test full Gemma4ForCausalLM forward pass with interleaved image and audio tokens."""
    image_token_id = 90
    audio_token_id = 91

    config = Gemma4Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
        image_token_id=image_token_id,
        audio_token_id=audio_token_id,
        vision_config={"hidden_size": 64, "image_size": 28, "patch_size": 14},
        audio_config={"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2},
    )
    model = Gemma4ForCausalLM(config)

    # Input IDs: (batch=2, seq_len=6)
    # Sequence has text, image placeholder (90), text, audio placeholder (91), text, text
    input_ids = torch.tensor(
        [
            [10, image_token_id, 20, audio_token_id, 30, 40],
            [15, image_token_id, 25, audio_token_id, 35, 45],
        ],
        dtype=torch.long,
    )

    pixel_values = torch.randn(2, 3, 28, 28)
    audio_values = torch.randn(2, 100)

    logits, cache = model(input_ids, pixel_values=pixel_values, audio_values=audio_values)

    assert logits.shape == (2, 6, 100)
    assert cache is not None

    # Test explicit token masks passed
    img_mask = input_ids == image_token_id
    aud_mask = input_ids == audio_token_id
    logits2, _ = model(
        input_ids,
        pixel_values=pixel_values,
        audio_values=audio_values,
        image_token_mask=img_mask,
        audio_token_mask=aud_mask,
    )
    assert logits2.shape == (2, 6, 100)

    # Test backward pass
    loss = logits.sum()
    loss.backward()
    assert model.embed_tokens.weight.grad is not None


def test_gemma4_causal_lm_from_pretrained(tmp_path: pytest.TempPathFactory) -> None:
    """Test Gemma4ForCausalLM.from_pretrained with config=None and safetensors file."""
    import json

    from safetensors.torch import save_file

    cfg_data = {
        "vocab_size": 10,
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "intermediate_size": 64,
        "vision_config": {"hidden_size": 64, "image_size": 28, "patch_size": 14},
        "audio_config": {"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg_data))

    # 1. Config is None and loads from config.json
    m1 = Gemma4ForCausalLM.from_pretrained(str(tmp_path))
    assert isinstance(m1, Gemma4ForCausalLM)

    # 2. Safetensors file exists
    sf_path = tmp_path / "model.safetensors"
    save_file({"lm_head.weight": torch.zeros((10, 64))}, str(sf_path))
    cfg = Gemma4Config(
        vocab_size=10,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=64,
    )
    m2 = Gemma4ForCausalLM.from_pretrained(str(tmp_path), config=cfg)
    assert isinstance(m2, Gemma4ForCausalLM)


def test_gemma4_causal_lm_multimodal_with_2d_attention_masks() -> None:
    """Test Gemma4ForCausalLM forward pass with 2D boolean and integer attention masks."""
    config = Gemma4Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
        sliding_window=4,
    )
    model = Gemma4ForCausalLM(config)

    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)

    # 1. 2D boolean mask
    bool_mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]],
        dtype=torch.bool,
    )
    logits_bool, _ = model(input_ids, attention_mask=bool_mask)
    assert logits_bool.shape == (2, 4, 100)

    # 2. 2D integer mask with q_len > 1
    int_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long)
    logits_int, _ = model(input_ids, attention_mask=int_mask)
    assert logits_int.shape == (2, 4, 100)

    # 3. 2D mask with single query token (q_len == 1)
    single_token = torch.tensor([[1], [5]], dtype=torch.long)
    single_mask = torch.tensor([[1], [1]], dtype=torch.long)
    logits_single, _ = model(single_token, attention_mask=single_mask)
    assert logits_single.shape == (2, 1, 100)


def test_gemma4_causal_lm_multimodal_concat_fallback_attention_mask_padding() -> None:
    """Test attention mask auto-padding when multimodal fallback concatenation expands sequence length."""
    config = Gemma4Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
        image_token_id=999,  # placeholder token not in input_ids
        vision_config={"hidden_size": 64, "image_size": 28, "patch_size": 14},
    )
    model = Gemma4ForCausalLM(config)

    # input_ids has 4 tokens, neither of which is token 999
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    pixel_values = torch.randn(1, 3, 28, 28)

    # Pass 2D attention mask of length 4 (matching input_ids before image feature concatenation)
    attn_mask = torch.ones((1, 4), dtype=torch.long)

    logits, _ = model(input_ids, attention_mask=attn_mask, pixel_values=pixel_values)
    # The concatenated sequence length is 4 (text) + 4 (image patches) = 8
    assert logits.shape[0] == 1
    assert logits.shape[1] > 4
    assert logits.shape[2] == 100


def test_gemma4_causal_lm_from_pretrained_default_config(tmp_path: pytest.TempPathFactory) -> None:
    """Test Gemma4ForCausalLM.from_pretrained fallback when config is None and no config.json exists."""
    from unittest.mock import patch

    empty_dir = tmp_path / "empty_model"
    empty_dir.mkdir(parents=True, exist_ok=True)

    tiny_cfg = Gemma4Config(
        vocab_size=10,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=64,
    )

    with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4Config", return_value=tiny_cfg):
        m = Gemma4ForCausalLM.from_pretrained(str(empty_dir))
        assert isinstance(m, Gemma4ForCausalLM)
