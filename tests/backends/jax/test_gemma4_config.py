from jax.sharding import PartitionSpec

from gemma_4_sql.backends.jax.gemma4.config import (
    AttentionType,
    AudioConfig,
    ModelConfig,
    ModelConfigPresets,
    ShardConfig,
    ShardMode,
    VisionConfig,
    VisionShardConfig,
    _make_spec,
)


def test_vision_shard_config():
    cfg = VisionShardConfig.no_sharding()
    assert isinstance(cfg, VisionShardConfig)
    assert cfg.attn_kernel is None


def test_vision_config():
    cfg = VisionConfig()
    assert cfg.hidden_size == 1152
    assert cfg.image_size == 896


def test_attention_type():
    assert AttentionType.LOCAL_SLIDING.value == "local_sliding"
    assert AttentionType.GLOBAL.value == "global"


def test_shard_mode():
    assert ShardMode.FSDP.value == "fsdp"
    assert ShardMode.TP.value == "tp"


def test_make_spec():
    spec = _make_spec(None, "fsdp")
    assert isinstance(spec, PartitionSpec)
    assert spec == PartitionSpec(None, "fsdp")


def test_shard_config():
    no_shard = ShardConfig.no_sharding()
    assert no_shard.attn_kernel is None

    cfg_def = ShardConfig.default(use_fsdp=False, use_tp=False)
    assert cfg_def.attn_kernel == PartitionSpec(None, None)

    cfg_fsdp = ShardConfig.default(use_fsdp=True, use_tp=False)
    assert cfg_fsdp.attn_kernel == PartitionSpec(None, "fsdp")

    cfg_tp = ShardConfig.default(use_fsdp=False, use_tp=True)
    assert cfg_tp.attn_kernel == PartitionSpec("tp", None)

    cfg_both = ShardConfig.default(use_fsdp=True, use_tp=True)
    assert cfg_both.attn_kernel == PartitionSpec("tp", "fsdp")


def test_audio_config():
    cfg = AudioConfig()
    assert cfg.hidden_size == 1024


def test_model_config_presets():
    cfg1 = ModelConfigPresets.gemma4_base()
    assert isinstance(cfg1, ModelConfigPresets)

    cfg2 = ModelConfigPresets.gemma4_base(use_fsdp=True)
    assert isinstance(cfg2, ModelConfigPresets)

    cfg3 = ModelConfigPresets.gemma4_e2b()
    assert cfg3.num_hidden_layers == 35

    cfg4 = ModelConfigPresets.gemma4_e2b(use_fsdp=True)
    assert cfg4.num_hidden_layers == 35

    cfg5 = ModelConfigPresets.gemma4_e4b()
    assert cfg5.num_hidden_layers == 42

    cfg6 = ModelConfigPresets.gemma4_e4b(use_fsdp=True)
    assert cfg6.num_hidden_layers == 42

    cfg7 = ModelConfigPresets.gemma4_26b_a4b()
    assert cfg7.num_hidden_layers == 30

    cfg8 = ModelConfigPresets.gemma4_26b_a4b(use_tp=True)
    assert cfg8.num_hidden_layers == 30

    cfg9 = ModelConfigPresets.gemma4_31b()
    assert cfg9.num_hidden_layers == 60

    cfg10 = ModelConfigPresets.gemma4_31b(use_fsdp=True, use_tp=True)
    assert cfg10.num_hidden_layers == 60


def test_model_config():
    cfg = ModelConfig()
    assert cfg.vocab_size == 256000
    assert cfg.hidden_size == 2048


def test_model_config_presets_2():
    from gemma_4_sql.backends.jax.gemma4.config import ModelConfigPresets

    assert ModelConfigPresets.gemma4_e2b(use_fsdp=True).num_hidden_layers == 35
    assert ModelConfigPresets.gemma4_e4b(use_tp=True).num_hidden_layers == 42
    assert ModelConfigPresets.gemma4_26b_a4b().num_hidden_layers == 30
    assert ModelConfigPresets.gemma4_31b().num_hidden_layers == 60
