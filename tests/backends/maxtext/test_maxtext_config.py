from unittest.mock import MagicMock

import pytest

from gemma_4_sql.backends.maxtext.config_generator import (
    _to_float,
    _to_int,
    build_maxtext_cli_args,
    build_maxtext_hyperparameters,
    generate_maxtext_gin_config,
    normalize_model_architecture,
    save_maxtext_gin_config,
)


class MockTrainingConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_to_float():
    assert _to_float(None, 2.0) == 2.0
    assert _to_float("3.14", 2.0) == 3.14
    assert _to_float(10, 2.0) == 10.0
    assert _to_float(MagicMock(), 2.0) == 2.0


def test_to_int():
    assert _to_int(None, 2) == 2
    assert _to_int("3", 2) == 3
    assert _to_int(10.5, 2) == 10
    assert _to_int(MagicMock(), 2) == 2


def test_normalize_model_architecture():
    assert normalize_model_architecture("gemma-4") == "gemma4_2b"
    assert normalize_model_architecture("gemma4_2b") == "gemma4_2b"
    assert normalize_model_architecture("gemma-4-7b") == "gemma4_7b"
    assert normalize_model_architecture("7B_model") == "gemma4_7b"
    with pytest.raises(ValueError, match="must be a non-empty string"):
        normalize_model_architecture("")


def test_build_maxtext_hyperparameters_defaults():
    config = MockTrainingConfig()
    params = build_maxtext_hyperparameters(config)
    assert params.model_architecture == "gemma4_2b"
    assert params.per_device_batch_size == 2.0
    assert params.global_batch_size == 8
    assert params.learning_rate == 1e-4


def test_build_maxtext_hyperparameters_overrides():
    config = MockTrainingConfig(model_name="gemma-4-7b", batch_size=4.0, learning_rate=2e-4, epochs=2, dataset="test_data", extra_kwargs={"opt_type": "adam"})
    params = build_maxtext_hyperparameters(config, steps=2000)
    assert params.model_architecture == "gemma4_7b"
    assert params.per_device_batch_size == 4.0
    assert params.global_batch_size == 16
    assert params.learning_rate == 2e-4
    assert params.opt_type == "adam"
    assert params.steps == 2000
    assert params.dataset_name == "test_data"


def test_build_maxtext_hyperparameters_validation_errors():
    config = MockTrainingConfig()
    with pytest.raises(ValueError, match="per_device_batch_size must be positive"):
        build_maxtext_hyperparameters(config, per_device_batch_size=-1)

    with pytest.raises(ValueError, match="global_batch_size must be positive"):
        build_maxtext_hyperparameters(config, global_batch_size=0)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        build_maxtext_hyperparameters(config, learning_rate=0)

    with pytest.raises(ValueError, match="steps must be positive"):
        build_maxtext_hyperparameters(config, steps=-5)

    with pytest.raises(ValueError, match="learning_rate_schedule_steps must be positive"):
        build_maxtext_hyperparameters(config, learning_rate_schedule_steps=0)

    with pytest.raises(ValueError, match="warmup_steps_fraction must be between 0.0 and 1.0"):
        build_maxtext_hyperparameters(config, warmup_steps_fraction=1.5)

    with pytest.raises(ValueError, match="opt_type cannot be empty"):
        build_maxtext_hyperparameters(config, opt_type="")

    with pytest.raises(ValueError, match="mesh_axes must be a list of strings"):
        build_maxtext_hyperparameters(config, mesh_axes="invalid")

    with pytest.raises(ValueError, match="checkpoint_period must be positive"):
        build_maxtext_hyperparameters(config, checkpoint_period=0)


def test_generate_maxtext_gin_config():
    config = MockTrainingConfig()
    gin_str = generate_maxtext_gin_config(config)
    assert "gemma4_2b" in gin_str
    assert "mesh_axes = ['data', 'fsdp', 'tensor']" in gin_str


def test_save_maxtext_gin_config(tmp_path):
    gin_content = "test_content"

    # Test with output path
    out_path = tmp_path / "test.gin"
    res_path = save_maxtext_gin_config(gin_content, out_path)
    assert res_path == out_path
    assert res_path.read_text() == gin_content

    # Test without output path
    res_path2 = save_maxtext_gin_config(gin_content)
    assert res_path2.exists()
    assert res_path2.read_text() == gin_content

    # Test empty
    with pytest.raises(ValueError, match="cannot be empty"):
        save_maxtext_gin_config("")


def test_build_maxtext_cli_args(tmp_path):
    config = MockTrainingConfig()

    # Test without gin_config_path
    args = build_maxtext_cli_args(config)
    assert len(args) == 2
    assert args[0] == "train.py"

    # Test with gin_config_path
    gin_file = tmp_path / "custom.gin"
    gin_file.touch()
    args2 = build_maxtext_cli_args(config, gin_config_path=gin_file)
    assert args2[1] == str(gin_file.resolve())
