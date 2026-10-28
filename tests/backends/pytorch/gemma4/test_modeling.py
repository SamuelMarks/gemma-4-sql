"""Tests for native PyTorch Gemma 4 modeling."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Never

import pytest
import torch

from gemma_4_sql.backends.pytorch.gemma4 import (
    DynamicCache,
    Gemma4Attention,
    Gemma4Config,
    Gemma4DecoderLayer,
    Gemma4ForCausalLM,
    Gemma4MLP,
    Gemma4RMSNorm,
    Gemma4RotaryEmbedding,
    Gemma4RotaryEmbedding2D,
    apply_rotary_pos_emb,
)
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4AudioConfig, Gemma4VisionConfig


def test_gemma4_config() -> None:
    """Test Gemma4Config."""
    config = Gemma4Config(hidden_size=512, test_kwarg=True, vision_config={"hidden_size": 128, "extra": 1}, audio_config={"hidden_size": 128, "extra": 1})
    assert config.hidden_size == 512
    assert config.vocab_size == 256000
    assert config.extra_attributes["test_kwarg"] is True
    assert config.vision_config.hidden_size == 128
    assert config.vision_config.extra_attributes["extra"] == 1
    assert config.audio_config.hidden_size == 128
    assert config.audio_config.extra_attributes["extra"] == 1

    config_defaults = Gemma4Config(mm_tokens_per_image=[1, 2])
    assert config_defaults.vision_config.hidden_size == 1152
    assert config_defaults.audio_config.hidden_size == 1024
    assert config_defaults.mm_tokens_per_image == 256

    config_with_instances = Gemma4Config(
        vision_config=config_defaults.vision_config,
        audio_config=config_defaults.audio_config,
    )
    assert config_with_instances.vision_config is config_defaults.vision_config
    assert config_with_instances.audio_config is config_defaults.audio_config

    assert Gemma4VisionConfig(extra_v=11).extra_attributes["extra_v"] == 11
    assert Gemma4AudioConfig(extra_a=22).extra_attributes["extra_a"] == 22

    assert Gemma4Config.gemma4_tiny().num_hidden_layers == 1
    assert Gemma4Config.gemma4_e2b().num_hidden_layers == 35
    assert Gemma4Config.gemma4_e4b().num_hidden_layers == 42
    assert Gemma4Config.gemma4_26b_a4b().num_experts == 128
    assert Gemma4Config.gemma4_31b().num_hidden_layers == 60
    assert Gemma4Config.from_preset("gemma-4-tiny").num_hidden_layers == 1
    assert Gemma4Config.from_preset("google/gemma-4-31B").num_hidden_layers == 60
    assert Gemma4Config.from_preset("google/gemma-4-26B-A4B").num_experts == 128
    assert Gemma4Config.from_preset("google/gemma-4-E4B").num_hidden_layers == 42
    assert Gemma4Config.from_preset("google/gemma-4-E2B").num_hidden_layers == 35


def test_gemma4_rmsnorm() -> None:
    """Test Gemma4RMSNorm."""
    norm = Gemma4RMSNorm(dim=512)
    x = torch.randn(2, 4, 512)
    out = norm(x)
    assert out.shape == x.shape


def test_gemma4_mlp() -> None:
    """Test Gemma4MLP."""
    config = Gemma4Config(hidden_size=256, intermediate_size=512)
    mlp = Gemma4MLP(config)
    x = torch.randn(2, 4, 256)
    out = mlp(x)
    assert out.shape == (2, 4, 256)


def test_apply_rotary_pos_emb() -> None:
    """Test apply_rotary_pos_emb mathematical identity and 90-degree rotation."""
    q = torch.tensor([[[1.0, 2.0]]])
    k = torch.tensor([[[3.0, 4.0]]])
    cos = torch.ones(1, 2)
    sin = torch.zeros(1, 2)
    position_ids = torch.zeros(1, dtype=torch.long)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    assert torch.allclose(q_out, q)
    assert torch.allclose(k_out, k)

    # 90-degree rotation: cos=0, sin=1 -> rotate_half([x1, x2]) = [-x2, x1]
    q_rot, k_rot = apply_rotary_pos_emb(q, k, torch.zeros(1, 2), torch.ones(1, 2), position_ids)
    assert torch.allclose(q_rot, torch.tensor([[[-2.0, 1.0]]]))
    assert torch.allclose(k_rot, torch.tensor([[[-4.0, 3.0]]]))

    emb = Gemma4RotaryEmbedding(dim=64, max_position_embeddings=10)
    cos, sin = emb(torch.randn(1, 10, 64), seq_len=15)
    assert cos.shape == (15, 64)
    assert emb.max_seq_len_cached == 15

    emb2d = Gemma4RotaryEmbedding2D(dim=64)
    cos2d, _sin2d = emb2d(torch.randn(1, 10, 64), 14, 14)
    assert cos2d.shape == (196, 64)


def test_gemma4_attention() -> None:
    """Test Gemma4Attention."""
    config = Gemma4Config(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64)
    attn = Gemma4Attention(config, layer_idx=0)
    x = torch.randn(2, 4, 256)
    out, past = attn(x)
    assert out.shape == (2, 4, 256)
    assert past is not None

    # Test with past
    out2, past2 = attn(x, past_key_value=past)
    assert out2.shape == (2, 4, 256)
    assert past2 is not None

    # Test with tuple past
    tuple_past = (torch.randn(2, 2, 4, 64), torch.randn(2, 2, 4, 64))
    out4, _ = attn(x, past_key_value=tuple_past)
    assert out4.shape == (2, 4, 256)

    # Test with mask
    mask = torch.randn(2, 1, 4, 4)
    out3, _ = attn(x, attention_mask=mask)
    assert out3.shape == (2, 4, 256)

    # Test SDPA path
    attn_sdpa = Gemma4Attention(Gemma4Config(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64, sliding_window=None), layer_idx=0)
    out5, _ = attn_sdpa(x)
    assert out5.shape == (2, 4, 256)

    # Test global attention with share_kv_projections, global_head_dim, num_global_key_value_heads, and attn_logits_soft_cap
    cfg_global = Gemma4Config(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=4,
        head_dim=32,
        global_head_dim=64,
        share_kv_projections=True,
        attn_logits_soft_cap=50.0,
        global_attn_layers=[0],
        global_rope_proportion=0.25,
    )
    attn_global = Gemma4Attention(cfg_global, layer_idx=0)
    assert attn_global.v_proj is None
    assert attn_global.head_dim == 64
    assert attn_global.num_key_value_heads == 4
    out_glob, _ = attn_global(x)
    assert out_glob.shape == (2, 4, 256)

    # Test 2D attention mask with q_len == 1
    out_2d_single, _ = attn(x[:, :1, :], attention_mask=torch.ones((2, 1), dtype=torch.long))
    assert out_2d_single.shape == (2, 1, 256)


def test_gemma4_decoder_layer() -> None:
    """Test Gemma4DecoderLayer."""
    config = Gemma4Config(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64, intermediate_size=512, num_experts=1)
    layer = Gemma4DecoderLayer(config, layer_idx=0)
    x = torch.randn(2, 4, 256)
    out, past, router_logits = layer(x)
    assert out.shape == (2, 4, 256)
    assert past is not None
    assert router_logits is None

    config_moe = Gemma4Config(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64, intermediate_size=512, num_experts=4)
    layer_moe = Gemma4DecoderLayer(config_moe, layer_idx=0)
    out_moe, past_moe, router_logits_moe = layer_moe(x)
    assert out_moe.shape == (2, 4, 256)
    assert past_moe is not None
    assert router_logits_moe is not None
    assert router_logits_moe.shape == (8, 4)


def test_gemma4_for_causal_lm() -> None:
    """Test Gemma4ForCausalLM."""
    config = Gemma4Config(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        head_dim=64,
        num_experts=1,
    )
    model = Gemma4ForCausalLM(config)
    input_ids = torch.randint(0, 1000, (2, 4))
    logits, cache = model(input_ids)
    assert logits.shape == (2, 4, 1000)
    assert cache is not None

    logits2, _cache2 = model(input_ids, past_key_values=cache)
    assert logits2.shape == (2, 4, 1000)

    gen = model.generate(input_ids, max_new_tokens=5)
    assert gen.shape == (2, 4 + 5)

    # Test min_new_tokens and eos_token_id early stopping
    model.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=True)
    with torch.no_grad():
        model.lm_head.weight.fill_(0.0)
        model.lm_head.bias.fill_(0.0)
        model.lm_head.bias[7] = 100.0
    gen_eos = model.generate(torch.tensor([[1, 2]], dtype=torch.long), max_new_tokens=5, min_new_tokens=1, eos_token_id=7)
    # Token 0 cannot be 7 due to min_new_tokens=1; token 1 becomes 7 and stops immediately -> length 2 + 2 = 4
    assert gen_eos.shape == (1, 4)
    assert int(gen_eos[0, -1].item()) == 7

    # Test multimodal and cache
    pixel_values = torch.randn(2, 3, 224, 224)
    audio_values = torch.randn(2, 1000)
    logits_mm, cache_mm = model(input_ids, pixel_values=pixel_values, audio_values=audio_values)
    assert logits_mm.shape[0] == 2
    assert cache_mm is not None

    dyn_cache = DynamicCache()
    logits_dyn, dyn_cache = model(input_ids, past_key_values=dyn_cache)
    assert logits_dyn.shape == (2, 4, 1000)
    assert dyn_cache.get_seq_length(0) == 4


def test_attention_edge_cases() -> None:
    """Test attention edge cases functionality."""
    import gemma_4_sql.backends.pytorch.gemma4.attention as attn

    config = Gemma4Config(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        max_position_embeddings=128,
        sliding_window=10,
        is_global=True,
        global_attn_layers=(0,),
        global_rope_max_timescale=None,
    )
    attention = attn.Gemma4Attention(config, layer_idx=0)

    hidden_states = torch.randn(1, 2, 32)
    position_ids = torch.tensor([[0, 1]], dtype=torch.long)

    # attention_mask not None, position_ids provided, is_global=True, sliding_window not None
    # this covers 64->67 (position_ids provided) and 99->105 (not self.is_global is False)
    res, past = attention(hidden_states, attention_mask=torch.ones(1, 1, 2, 2), position_ids=position_ids)
    assert res.shape == (1, 2, 32)
    assert torch.isfinite(res).all()
    assert past is not None


def test_attention_edge_cases2() -> None:
    """Test manual attention fallback with sliding_window=None and local_rope_max_timescale."""
    import gemma_4_sql.backends.pytorch.gemma4.attention as attn

    config = Gemma4Config(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        max_position_embeddings=128,
        sliding_window=None,
        is_global=False,
        global_attn_layers=(1,),
        local_rope_max_timescale=5000,
    )
    attention = attn.Gemma4Attention(config, layer_idx=0)

    hidden_states = torch.randn(1, 2, 32)
    position_ids = torch.tensor([[0, 1]], dtype=torch.long)
    res, past = attention(hidden_states, position_ids=position_ids)
    assert res.shape == (1, 2, 32)
    assert torch.isfinite(res).all()
    assert past is not None


def test_audio_layers_edge_cases() -> None:
    """Test audio layers edge cases functionality."""
    import gemma_4_sql.backends.pytorch.gemma4.audio_layers as al

    config = Gemma4AudioConfig(hidden_size=32)

    layer = al.Gemma4AudioFeatureExtractor(config)
    hidden_states = torch.randn(1, 1, 32)
    res = layer(hidden_states)
    assert res.shape[0] == 1
    assert res.shape[-1] == 32
    assert torch.isfinite(res).all()


def test_gemma4_native_save_load_and_generate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test save_pretrained, from_pretrained, and generate with DynamicCache."""
    config = Gemma4Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
    )
    model = Gemma4ForCausalLM(config)
    save_file = model.save_pretrained(str(tmp_path))
    assert Path(save_file).is_file()

    loaded = Gemma4ForCausalLM.from_pretrained(str(tmp_path), config=config)
    assert loaded is not None
    loaded_direct = Gemma4ForCausalLM.from_pretrained(save_file, config=config)
    assert loaded_direct is not None

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generated = loaded.generate(input_ids, max_new_tokens=2)
    assert generated.shape == (1, 5)

    # Test corrupt safetensors file
    corrupt_file = tmp_path / "corrupt.safetensors"
    corrupt_file.write_bytes(b"not_valid_safetensors")
    model_corrupt = Gemma4ForCausalLM.from_pretrained(str(corrupt_file), config=config)
    assert model_corrupt is not None
    from gemma_4_sql.exceptions import ExportError

    with pytest.raises(ExportError, match="Failed to load safetensors"):
        Gemma4ForCausalLM.from_pretrained(str(corrupt_file), config=config, strict=True)

    # Test empty dir where st_path is not a file
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    model_empty = Gemma4ForCausalLM.from_pretrained(str(empty_dir), config=config)
    assert model_empty is not None

    # Test Exception in load_file
    def mock_load_file_exc(*_args: str, **_kwargs: str) -> Never:
        """Provide mock load file exc test helper."""
        raise RuntimeError("Custom exception")

    monkeypatch.setattr("safetensors.torch.load_file", mock_load_file_exc, raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gemma4.modeling.load_file", mock_load_file_exc, raising=False)
    model_exc = Gemma4ForCausalLM.from_pretrained(save_file, config=config)
    assert model_exc is not None


def test_pytorch_native_pipeline_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test pytorch_native backend integration for train, inference, and export."""
    from gemma_4_sql.backends.pytorch.export import export_model
    from gemma_4_sql.backends.pytorch.inference import generate_sql
    from gemma_4_sql.backends.pytorch.train import train_model
    from gemma_4_sql.type_hints import TrainingConfig

    tiny_cfg = Gemma4Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
    )

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.export.save_file", lambda *_args, **_kwargs: None)  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr("safetensors.torch.save_file", lambda *_args, **_kwargs: None, raising=False)  # pyright: ignore[reportUnknownLambdaType]
    exp_res = export_model("test_model", str(tmp_path), backend_alias="pytorch_native", config=tiny_cfg)
    assert exp_res["status"] == "exported_with_safetensors"
    assert exp_res["backend"] == "pytorch_native"

    torch.manual_seed(42)
    gen_res = generate_sql("test_model", "SELECT 1", max_length=2, backend_alias="pytorch_native", config=tiny_cfg)
    assert gen_res["status"] == "success"
    assert gen_res["backend"] == "pytorch_native"
    assert "sql" in gen_res

    class MockLoader:
        """Test class for MockLoader."""

        def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
            """Initialize __iter__."""
            yield {"inputs": torch.tensor([[1, 2]], dtype=torch.long), "targets": torch.tensor([[2, 3]], dtype=torch.long)}

        def __len__(self) -> int:
            """Initialize __len__."""
            return 1

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.train.build_dataloader", lambda c: {"loader": MockLoader()})  # pyright: ignore[reportUnknownLambdaType]
    train_res = train_model(
        TrainingConfig(action="sft", model_name="test_model", dataset="dummy", epochs=1, backend="pytorch_native"),
        backend_alias="pytorch_native",
        model_config=tiny_cfg,
    )
    assert train_res["status"] == "completed"
    assert train_res["backend"] == "pytorch_native"


def test_modeling_position_ids_and_longer_attention_mask() -> None:
    """Verify Gemma4ForCausalLM handles explicit position_ids and longer attention_mask with KV cache."""
    config = Gemma4Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=64,
        vocab_size=100,
    )
    model = Gemma4ForCausalLM(config)
    prefill_ids = torch.randint(0, 100, (1, 5))
    _, past_cache = model(input_ids=prefill_ids, past_key_values=DynamicCache())

    input_ids = torch.randint(0, 100, (1, 5))
    position_ids = torch.arange(5, 10).unsqueeze(0)
    attention_mask = torch.ones((1, 10))
    logits, updated_cache = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_cache,
    )
    assert logits.shape == (1, 5, 100)
    assert updated_cache is not None


def test_gemma4_ple_post_norms_kv_sharing_and_softcapping() -> None:
    """Verify PyTorch Gemma4ForCausalLM supports PLE, use_post_norms, KV layer sharing, and final_logit_softcapping."""
    cfg = Gemma4Config(
        vocab_size=64,
        vocab_size_per_layer_input=64,
        hidden_size=32,
        hidden_size_per_layer_input=16,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_experts=2,
        num_experts_per_tok=1,
        num_kv_shared_layers=1,
        use_post_norms=True,
        final_logit_softcapping=30.0,
        attn_logits_soft_cap=50.0,
    )
    model = Gemma4ForCausalLM(cfg)
    input_ids = torch.randint(0, 64, (2, 4))
    logits, cache = model(input_ids)
    assert logits.shape == (2, 4, 64)
    assert torch.all(logits.abs() <= 30.0 + 1e-4)
    assert cache is not None
    assert model.last_aux_loss is not None
    assert float(model.last_aux_loss.item()) >= 0.0

    # Test second step with tuple cache and KV sharing
    logits2, cache2 = model(input_ids[:, :1], past_key_values=cache)
    assert logits2.shape == (2, 1, 64)
    assert cache2 is not None

    # Test project_per_layer_inputs when per_layer_inputs is None
    embeds = model.embed_tokens(input_ids) * model.embed_scale
    proj_only = model.project_per_layer_inputs(embeds, per_layer_inputs=None)
    assert proj_only is not None
    assert proj_only.shape == (2, 4, 3, 16)

    # Test scale-free RMSNorm directly
    norm_no_scale = Gemma4RMSNorm(16, with_scale=False)
    assert norm_no_scale.weight is None
    x_norm = norm_no_scale(torch.randn(2, 16))
    assert x_norm.shape == (2, 16)
