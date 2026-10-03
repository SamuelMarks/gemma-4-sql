import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gemma_4_sql.backends.maxtext.config_generator import (
    MaxTextHyperparameters,
    _to_float,
    _to_int,
    build_maxtext_cli_args,
    build_maxtext_hyperparameters,
    generate_maxtext_gin_config,
    normalize_model_architecture,
    save_maxtext_gin_config,
)


def test_maxtext_hyperparameters_defaults():
    hp = MaxTextHyperparameters()
    assert hp.model_architecture == "gemma4_2b"
    assert hp.per_device_batch_size == 2.0
    assert hp.global_batch_size == 8
    assert hp.learning_rate == 1e-4
    assert hp.learning_rate_schedule_steps == 1000
    assert hp.warmup_steps_fraction == 0.1
    assert hp.opt_type == "adamw"
    assert hp.mesh_axes == ["data", "fsdp", "tensor"]
    assert hp.base_output_directory == "./maxtext_output"
    assert hp.run_name == "gemma4_sql_run"
    assert hp.steps == 1000
    assert hp.dataset_name == "dummy"
    assert hp.checkpoint_period == 500


def test_to_float():
    assert _to_float(None, 1.5) == 1.5
    assert _to_float(2, 1.5) == 2.0
    assert _to_float(2.5, 1.5) == 2.5
    assert _to_float("3.5", 1.5) == 3.5
    assert _to_float([], 1.5) == 1.5
    assert _to_float({}, 1.5) == 1.5


def test_to_int():
    assert _to_int(None, 5) == 5
    assert _to_int(2, 5) == 2
    assert _to_int(2.5, 5) == 2
    assert _to_int("3", 5) == 3
    assert _to_int([], 5) == 5
    assert _to_int({}, 5) == 5


def test_normalize_model_architecture():
    assert normalize_model_architecture("gemma-4") == "gemma4_2b"
    assert normalize_model_architecture("gemma-4-7b") == "gemma4_7b"
    assert normalize_model_architecture("  7b  ") == "gemma4_7b"
    assert normalize_model_architecture("gemma4_2b") == "gemma4_2b"

    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        normalize_model_architecture("")

    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        normalize_model_architecture(None)


def test_build_maxtext_hyperparameters_defaults():
    config = MagicMock()
    config.extra_kwargs = None
    config.model_name = "gemma4_2b"
    config.batch_size = 2.0
    config.learning_rate = 1e-4
    config.epochs = 1
    config.dataset = "dummy"

    hp = build_maxtext_hyperparameters(config)
    assert hp.model_architecture == "gemma4_2b"
    assert hp.per_device_batch_size == 2.0
    assert hp.global_batch_size == 8
    assert hp.learning_rate == 1e-4
    assert hp.steps == 1000
    assert hp.learning_rate_schedule_steps == 1000
    assert hp.warmup_steps_fraction == 0.1
    assert hp.opt_type == "adamw"
    assert hp.mesh_axes == ["data", "fsdp", "tensor"]
    assert hp.base_output_directory == "./maxtext_output"
    assert hp.run_name == "gemma4_2b_dummy"
    assert hp.dataset_name == "dummy"
    assert hp.checkpoint_period == 500


def test_build_maxtext_hyperparameters_overrides():
    config = MagicMock()
    config.extra_kwargs = {"run_name": "custom_run", "warmup_steps_fraction": 0.2}
    config.model_name = "gemma-4-7b"
    config.batch_size = 4.0
    config.learning_rate = 2e-4
    config.epochs = 2
    config.dataset = "my_dataset"

    kwargs = {
        "opt_type": "adam",
        "mesh_axes": ["fsdp", "tensor"],
        "checkpoint_period": 100,
        "global_batch_size": 16,
    }

    hp = build_maxtext_hyperparameters(config, **kwargs)
    assert hp.model_architecture == "gemma4_7b"
    assert hp.per_device_batch_size == 4.0
    assert hp.global_batch_size == 16
    assert hp.learning_rate == 2e-4
    assert hp.steps == 2000
    assert hp.warmup_steps_fraction == 0.2
    assert hp.opt_type == "adam"
    assert hp.mesh_axes == ["fsdp", "tensor"]
    assert hp.run_name == "custom_run"
    assert hp.dataset_name == "my_dataset"
    assert hp.checkpoint_period == 100


def test_build_maxtext_hyperparameters_validation_errors():
    config = MagicMock()
    config.extra_kwargs = {}

    # per_device_batch_size <= 0
    with pytest.raises(ValueError, match="per_device_batch_size must be positive"):
        build_maxtext_hyperparameters(config, per_device_batch_size=0)

    # global_batch_size <= 0
    with pytest.raises(ValueError, match="global_batch_size must be positive"):
        build_maxtext_hyperparameters(config, global_batch_size=-1)

    # learning_rate <= 0
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        build_maxtext_hyperparameters(config, learning_rate=0.0)

    # steps <= 0
    with pytest.raises(ValueError, match="steps must be positive"):
        build_maxtext_hyperparameters(config, steps=0)

    # learning_rate_schedule_steps <= 0
    with pytest.raises(ValueError, match="learning_rate_schedule_steps must be positive"):
        build_maxtext_hyperparameters(config, learning_rate_schedule_steps=-5)

    # warmup_steps_fraction not in [0, 1]
    with pytest.raises(ValueError, match="warmup_steps_fraction must be between 0.0 and 1.0"):
        build_maxtext_hyperparameters(config, warmup_steps_fraction=1.5)
    with pytest.raises(ValueError, match="warmup_steps_fraction must be between 0.0 and 1.0"):
        build_maxtext_hyperparameters(config, warmup_steps_fraction=-0.1)

    # opt_type empty
    with pytest.raises(ValueError, match="opt_type cannot be empty"):
        build_maxtext_hyperparameters(config, opt_type="   ")
    with pytest.raises(ValueError, match="opt_type cannot be empty"):
        build_maxtext_hyperparameters(config, opt_type="")

    # mesh_axes not list of strings
    with pytest.raises(ValueError, match="mesh_axes must be a list of strings"):
        build_maxtext_hyperparameters(config, mesh_axes="data")
    with pytest.raises(ValueError, match="mesh_axes must be a list of strings"):
        build_maxtext_hyperparameters(config, mesh_axes=["data", 1])

    # checkpoint_period <= 0
    with pytest.raises(ValueError, match="checkpoint_period must be positive"):
        build_maxtext_hyperparameters(config, checkpoint_period=0)


def test_generate_maxtext_gin_config():
    config = MagicMock()
    config.extra_kwargs = {}
    config.model_name = "gemma4_2b"
    config.batch_size = 2.0
    config.learning_rate = 1e-4
    config.epochs = 1
    config.dataset = "dummy"

    gin_str = generate_maxtext_gin_config(config)

    assert 'model_name = "gemma4_2b"' in gin_str
    assert "per_device_batch_size = 2.0" in gin_str
    assert "global_batch_size = 8" in gin_str
    assert "mesh_axes = ['data', 'fsdp', 'tensor']" in gin_str
    assert "learning_rate = 0.0001" in gin_str
    assert 'opt_type = "adamw"' in gin_str
    assert "steps = 1000" in gin_str
    assert "learning_rate_schedule_steps = 1000" in gin_str
    assert "warmup_steps_fraction = 0.1" in gin_str
    assert 'dataset_name = "dummy"' in gin_str
    assert "checkpoint_period = 500" in gin_str


def test_save_maxtext_gin_config_with_path(tmp_path):
    output_path = tmp_path / "custom" / "test.gin"
    gin_content = "some_config = 1\n"

    result_path = save_maxtext_gin_config(gin_content, output_path=output_path)

    assert result_path == output_path.resolve()
    assert result_path.exists()
    assert result_path.read_text(encoding="utf-8") == gin_content


def test_save_maxtext_gin_config_without_path():
    gin_content = "some_config = 1\n"
    result_path = save_maxtext_gin_config(gin_content)

    try:
        assert result_path.exists()
        assert result_path.read_text(encoding="utf-8") == gin_content
    finally:
        os.remove(result_path)


def test_save_maxtext_gin_config_empty_content():
    with pytest.raises(ValueError, match="gin_content cannot be empty"):
        save_maxtext_gin_config("")

    with pytest.raises(ValueError, match="gin_content cannot be empty"):
        save_maxtext_gin_config("   \n")


def test_build_maxtext_cli_args_without_gin_path():
    config = MagicMock()
    config.extra_kwargs = {}
    config.model_name = "gemma4_2b"
    config.batch_size = 2.0
    config.learning_rate = 1e-4
    config.epochs = 1
    config.dataset = "dummy"

    args = build_maxtext_cli_args(config)
    assert len(args) == 2
    assert args[0] == "train.py"

    path = Path(args[1])
    assert path.exists()
    assert path.suffix == ".gin"
    os.remove(path)


def test_build_maxtext_cli_args_with_gin_path(tmp_path):
    config = MagicMock()
    gin_path = tmp_path / "my_config.gin"
    gin_path.write_text("test = 1")

    args = build_maxtext_cli_args(config, gin_config_path=gin_path)
    assert len(args) == 2
    assert args[0] == "train.py"
    assert args[1] == str(gin_path.resolve())
