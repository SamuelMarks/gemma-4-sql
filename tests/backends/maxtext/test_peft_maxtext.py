"""Unit and behavioral tests for peft maxtext."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from gemma_4_sql.backends.maxtext import peft as peft_module
from gemma_4_sql.exceptions import DependencyMissingError


@pytest.fixture(autouse=True)
def mock_deps(monkeypatch):
    """Execute and verify `mock_deps` behavior and invariants."""
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_np = MagicMock()
    mock_optax = MagicMock()
    mock_gemma4 = MagicMock()

    mock_jnp.float32 = "float32"
    mock_jnp.int32 = "int32"

    def mock_zeros(shape, dtype=None):
        """Execute and verify `mock_zeros` behavior and invariants."""
        m = MagicMock()
        m.shape = shape
        m.dtype = dtype
        return m

    mock_jnp.zeros.side_effect = mock_zeros
    mock_jnp.asarray.side_effect = lambda val: val
    mock_np.asarray.side_effect = lambda val: val
    mock_jnp.array.side_effect = lambda val, dtype=None: val

    def mock_uniform(rng, shape, dtype=None, minval=None, maxval=None):
        """Execute and verify `mock_uniform` behavior and invariants."""
        m = MagicMock()
        m.shape = shape
        m.dtype = dtype
        return m

    mock_jax.random.uniform.side_effect = mock_uniform

    def mock_split(rng):
        """Execute and verify `mock_split` behavior and invariants."""
        return rng, MagicMock()

    mock_jax.random.split.side_effect = mock_split

    monkeypatch.setattr(peft_module, "jax", mock_jax)
    monkeypatch.setattr(peft_module, "jnp", mock_jnp)
    monkeypatch.setattr(peft_module, "np", mock_np)
    monkeypatch.setattr(peft_module, "optax", mock_optax)
    monkeypatch.setattr(peft_module, "Gemma4Model", mock_gemma4)

    yield {"jax": mock_jax, "jnp": mock_jnp, "np": mock_np, "optax": mock_optax, "gemma4": mock_gemma4}


def test_transform_params_to_lora_missing_jax(monkeypatch):
    """Execute and verify `test_transform_params_to_lora_missing_jax` behavior and invariants."""
    monkeypatch.setattr(peft_module, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX dependencies are missing"):
        peft_module.transform_params_to_lora({}, ["q_proj"])


def test_transform_params_to_lora_invalid_rank():
    """Execute and verify `test_transform_params_to_lora_invalid_rank` behavior and invariants."""
    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        peft_module.transform_params_to_lora({}, ["q_proj"], lora_r=0)


def test_transform_params_to_lora_success():
    """Execute and verify `test_transform_params_to_lora_success` behavior and invariants."""
    kernel = MagicMock()
    kernel.shape = (10, 20)
    kernel.dtype = "float32"

    params = {"layer_1": {"q_proj": {"kernel": kernel, "other": 1}, "k_proj": {"kernel": kernel}, "no_kernel": {"a": 1}}, "scalar": 42}

    _transformed, count = peft_module.transform_params_to_lora(params, ["q_proj"], lora_r=8)

    assert count == 1


def test_transform_params_to_lora_with_rng():
    """Execute and verify `test_transform_params_to_lora_with_rng` behavior and invariants."""
    kernel = MagicMock()
    kernel.shape = (10, 20)
    params = {"layer": {"q_proj": {"kernel": kernel}}}
    rng = MagicMock()
    peft_module.transform_params_to_lora(params, ["q_proj"], lora_r=8, rng=rng)


def test_transform_params_to_lora_no_dict():
    """Execute and verify `test_transform_params_to_lora_no_dict` behavior and invariants."""
    res, count = peft_module.transform_params_to_lora("not a dict", ["q_proj"])
    assert res == "not a dict"
    assert count == 0


def test_segregate_adapter_params():
    """Execute and verify `test_segregate_adapter_params` behavior and invariants."""
    params = {"layer": {"q_proj": {"kernel": "frozen_kernel", "lora_a": "trainable_a", "lora_b": "trainable_b", "lora_scale": "trainable_scale", "other": "frozen_other"}}, "scalar": "frozen_scalar"}

    trainable, _frozen = peft_module.segregate_adapter_params(params)
    assert trainable["layer"]["q_proj"]["lora_a"] == "trainable_a"


def test_segregate_adapter_params_empty_sub_t_sub_f():
    """Execute and verify `test_segregate_adapter_params_empty_sub_t_sub_f` behavior and invariants."""
    params = {"layer": {"empty": {}}}
    t, f = peft_module.segregate_adapter_params(params)
    assert t == {}
    assert f == {}


def test_segregate_adapter_params_not_dict():
    """Execute and verify `test_segregate_adapter_params_not_dict` behavior and invariants."""
    trainable, frozen = peft_module.segregate_adapter_params("not dict")
    assert trainable == {}
    assert frozen == {}


def test_create_maxtext_lora_optimizer_missing_deps(monkeypatch):
    """Execute and verify `test_create_maxtext_lora_optimizer_missing_deps` behavior and invariants."""
    monkeypatch.setattr(peft_module, "optax", None)
    with pytest.raises(DependencyMissingError, match="Optax or JAX dependencies are missing"):
        peft_module.create_maxtext_lora_optimizer({})


def test_create_maxtext_lora_optimizer_success(mock_deps):
    """Execute and verify `test_create_maxtext_lora_optimizer_success` behavior and invariants."""
    mock_optax = mock_deps["optax"]
    mock_optax.adam.return_value = "adam_opt"
    mock_optax.set_to_zero.return_value = "zero_opt"
    mock_optax.multi_transform.return_value = "multi_opt"

    def mock_tree_map_with_path(f, tree):
        """Execute and verify `mock_tree_map_with_path` behavior and invariants."""

        class PathElement:
            """Execute and verify `PathElement` behavior and invariants."""

            def __init__(self, key):
                """Execute and verify `__init__` behavior and invariants."""
                self.key = key

        class PathElementStr:
            """Execute and verify `PathElementStr` behavior and invariants."""

            def __str__(self):
                """Execute and verify `__str__` behavior and invariants."""
                return "str_key"

        assert f([PathElement("lora_a")], None) == "trainable"
        assert f([PathElement("lora_b")], None) == "trainable"
        assert f([PathElement("kernel")], None) == "frozen"
        assert f([PathElementStr()], None) == "frozen"

        return "mapped_tree"

    mock_deps["jax"].tree_util.tree_map_with_path.side_effect = mock_tree_map_with_path

    opt = peft_module.create_maxtext_lora_optimizer({"fake": "params"})
    assert opt == "multi_opt"


def test_create_maxtext_lora_optimizer_custom_base(mock_deps):
    """Execute and verify `test_create_maxtext_lora_optimizer_custom_base` behavior and invariants."""
    peft_module.create_maxtext_lora_optimizer({"fake": "params"}, base_optimizer="custom")
    mock_optax = mock_deps["optax"]
    mock_optax.multi_transform.assert_called_once_with({"trainable": "custom", "frozen": mock_optax.set_to_zero.return_value}, ANY)


def test_merge_lora_weights_better():
    """Execute and verify `test_merge_lora_weights_better` behavior and invariants."""

    class DummyTensor:
        """Execute and verify `DummyTensor` behavior and invariants."""

        def __init__(self, name):
            """Execute and verify `__init__` behavior and invariants."""
            self.name = name

        def __matmul__(self, other):
            """Execute and verify `__matmul__` behavior and invariants."""
            return DummyTensor(f"{self.name}@{other.name}")

        def __rmul__(self, scalar):
            """Execute and verify `__rmul__` behavior and invariants."""
            return DummyTensor(f"{scalar}*{self.name}")

        def __add__(self, other):
            """Execute and verify `__add__` behavior and invariants."""
            return DummyTensor(f"{self.name}+{other.name}")

    params = {"layer": {"q_proj": {"kernel": DummyTensor("W"), "lora_a": DummyTensor("A"), "lora_b": DummyTensor("B"), "lora_scale": 2.0, "other": "kept"}, "no_lora": {"kernel": "just_kernel"}}, "scalar": 42}
    merged = peft_module.merge_lora_weights(params)
    assert merged["scalar"] == 42


def test_merge_lora_weights_not_dict():
    """Execute and verify `test_merge_lora_weights_not_dict` behavior and invariants."""
    assert peft_module.merge_lora_weights("not dict") == "not dict"


def test_save_maxtext_adapters_missing_deps(monkeypatch):
    """Execute and verify `test_save_maxtext_adapters_missing_deps` behavior and invariants."""
    monkeypatch.setattr(peft_module, "np", None)
    with pytest.raises(DependencyMissingError, match="NumPy dependency is missing"):
        peft_module.save_maxtext_adapters({}, "path")


def test_save_maxtext_adapters_success(mock_deps, tmp_path):
    """Execute and verify `test_save_maxtext_adapters_success` behavior and invariants."""
    mock_np = mock_deps["np"]
    params = {"layer": {"q_proj": {"lora_a": "array_a", "lora_b": "array_b", "lora_scale": "scale_val", "kernel": "array_kernel"}, "other": "value"}}

    save_path = tmp_path / "adapters.npz"
    peft_module.save_maxtext_adapters(params, save_path)
    mock_np.savez.assert_called_once()


def test_save_maxtext_adapters_not_dict(mock_deps, tmp_path):
    """Execute and verify `test_save_maxtext_adapters_not_dict` behavior and invariants."""
    mock_np = mock_deps["np"]
    peft_module.save_maxtext_adapters("not dict", tmp_path)
    mock_np.savez.assert_called_once_with(ANY)


def test_save_maxtext_adapters_is_dir(mock_deps, tmp_path):
    """Execute and verify `test_save_maxtext_adapters_is_dir` behavior and invariants."""
    mock_np = mock_deps["np"]
    peft_module.save_maxtext_adapters({"lora_a": "1"}, tmp_path)
    expected_path = tmp_path / "maxtext_lora_adapters.npz"
    mock_np.savez.assert_called_once_with(expected_path, lora_a="1")


def test_load_maxtext_adapters_missing_deps(monkeypatch):
    """Execute and verify `test_load_maxtext_adapters_missing_deps` behavior and invariants."""
    monkeypatch.setattr(peft_module, "np", None)
    with pytest.raises(DependencyMissingError, match="NumPy dependency is missing"):
        peft_module.load_maxtext_adapters({}, "path")


def test_load_maxtext_adapters_not_found():
    """Execute and verify `test_load_maxtext_adapters_not_found` behavior and invariants."""
    with pytest.raises(FileNotFoundError, match="Adapter file not found"):
        peft_module.load_maxtext_adapters({}, "nonexistent.npz")


def test_load_maxtext_adapters_success(mock_deps, tmp_path):
    """Execute and verify `test_load_maxtext_adapters_success` behavior and invariants."""
    mock_np = mock_deps["np"]
    mock_np.load.return_value = {"layer.q_proj.lora_a": "loaded_a", "layer.q_proj.lora_b": "loaded_b", "new_layer.lora_a": "new_a"}
    save_path = tmp_path / "adapters.npz"
    save_path.touch()

    params = {"layer": {"q_proj": {"kernel": "W"}}}
    loaded = peft_module.load_maxtext_adapters(params, save_path)
    assert loaded["layer"]["q_proj"]["lora_a"] == "loaded_a"


def test_load_maxtext_adapters_no_jnp(mock_deps, tmp_path, monkeypatch):
    """Execute and verify `test_load_maxtext_adapters_no_jnp` behavior and invariants."""
    monkeypatch.setattr(peft_module, "jnp", None)
    mock_np = mock_deps["np"]
    mock_np.load.return_value = {"a.b": "val"}
    save_path = tmp_path / "adapters.npz"
    save_path.touch()
    loaded = peft_module.load_maxtext_adapters({}, save_path)
    assert loaded["a"]["b"] == "val"


def test_count_maxtext_parameters():
    """Execute and verify `test_count_maxtext_parameters` behavior and invariants."""

    class Sized:
        """Execute and verify `Sized` behavior and invariants."""

        def __init__(self, size):
            """Execute and verify `__init__` behavior and invariants."""
            self.size = size

    params = {"layer": {"kernel": Sized(100), "lora_a": Sized(10), "lora_b": Sized(20)}, "not_sized": "scalar", "scalar_size": Sized(5)}

    total, trainable = peft_module.count_maxtext_parameters(params)
    assert total == 135
    assert trainable == 30


def test_count_maxtext_parameters_not_dict():
    """Execute and verify `test_count_maxtext_parameters_not_dict` behavior and invariants."""
    total, trainable = peft_module.count_maxtext_parameters("not dict")
    assert total == 0
    assert trainable == 0


def test_apply_lora_missing_deps(monkeypatch):
    """Execute and verify `test_apply_lora_missing_deps` behavior and invariants."""
    monkeypatch.setattr(peft_module, "jax", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        peft_module.apply_lora("gemma-4", ["q_proj"])

    monkeypatch.setattr(peft_module, "jax", MagicMock())
    monkeypatch.setattr(peft_module, "jnp", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        peft_module.apply_lora("gemma-4", ["q_proj"])

    monkeypatch.setattr(peft_module, "jnp", MagicMock())
    monkeypatch.setattr(peft_module, "Gemma4Model", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        peft_module.apply_lora("gemma-4", ["q_proj"])


def test_apply_lora_missing_gemma4_but_has_params(mock_deps, monkeypatch):
    """Execute and verify `test_apply_lora_missing_gemma4_but_has_params` behavior and invariants."""
    monkeypatch.setattr(peft_module, "Gemma4Model", None)
    with patch.object(peft_module, "transform_params_to_lora", return_value=({}, 0)):
        res = peft_module.apply_lora("gemma-4", ["q_proj"], params={})
    assert res["status"] == "completed"


def test_apply_lora_success(mock_deps, tmp_path):
    """Execute and verify `test_apply_lora_success` behavior and invariants."""
    mock_gemma4_class = mock_deps["gemma4"]
    mock_model = MagicMock()
    mock_gemma4_class.return_value = mock_model

    mock_model.init.return_value = {"original": "params"}

    def fake_transform(params, target_modules, lora_r, lora_alpha, lora_dropout):
        """Execute and verify `fake_transform` behavior and invariants."""
        return {"transformed": "params"}, 5

    with patch.object(peft_module, "transform_params_to_lora", side_effect=fake_transform), patch.object(peft_module, "save_maxtext_adapters"), patch.object(peft_module, "merge_lora_weights", return_value={"merged": "params"}):
        res = peft_module.apply_lora("gemma-4", ["q_proj"], lora_r=16, output_dir=tmp_path, merge=True)
        assert res["status"] == "completed"


def test_apply_lora_success_no_output_dir_no_merge(mock_deps):
    """Execute and verify `test_apply_lora_success_no_output_dir_no_merge` behavior and invariants."""
    with patch.object(peft_module, "transform_params_to_lora", return_value=({}, 5)):
        res = peft_module.apply_lora("gemma-4", ["q_proj"])
        assert res["status"] == "completed"


def test_apply_lora_failure(mock_deps):
    """Execute and verify `test_apply_lora_failure` behavior and invariants."""

    def fake_transform(*args, **kwargs):
        """Execute and verify `fake_transform` behavior and invariants."""
        raise RuntimeError("Fake Error")

    with patch.object(peft_module, "transform_params_to_lora", side_effect=fake_transform):
        res = peft_module.apply_lora("gemma-4", ["q_proj"])
        assert "failed: Fake Error" in res["status"]


def test_apply_lora_missing_dependency_inside(mock_deps, monkeypatch):
    """Execute and verify `test_apply_lora_missing_dependency_inside` behavior and invariants."""
    monkeypatch.setattr(peft_module, "Gemma4Model", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependency missing."):
        peft_module.apply_lora("gemma-4", ["q_proj"], params="not_a_dict")


def test_transform_and_merge_extra_branches():
    """Verify transform_params_to_lora and merge_lora_weights traverse non-matching nested dicts."""
    import jax.numpy as jnp

    params = {"a": {"b": jnp.zeros((2, 2))}, "c": jnp.zeros((2,))}
    _, count = peft_module.transform_params_to_lora(params, ["nonexistent"])
    assert count == 0

    lora_params = {
        "a": {
            "kernel": jnp.zeros((2, 2)),
            "lora_a": jnp.zeros((2, 2)),
            "lora_b": jnp.zeros((2, 2)),
            "lora_scale": 1.0,
            "extra_key": jnp.zeros((1,)),
        }
    }
    merged = peft_module.merge_lora_weights(lora_params)
    assert "extra_key" in merged["a"]
