"""Module docstring."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from gemma_4_sql.backends.maxtext import peft as peft_module
from gemma_4_sql.exceptions import DependencyMissingError


@pytest.fixture(autouse=True)
def mock_deps(monkeypatch):
    """Docstring for mock_deps."""
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_np = MagicMock()
    mock_optax = MagicMock()
    mock_gemma4 = MagicMock()

    mock_jnp.float32 = "float32"
    mock_jnp.int32 = "int32"

    def mock_zeros(shape, dtype=None):
        """Docstring for mock_zeros."""
        m = MagicMock()
        m.shape = shape
        m.dtype = dtype
        return m

    mock_jnp.zeros.side_effect = mock_zeros
    mock_jnp.asarray.side_effect = lambda val: val
    mock_np.asarray.side_effect = lambda val: val
    mock_jnp.array.side_effect = lambda val, dtype=None: val

    def mock_uniform(rng, shape, dtype=None, minval=None, maxval=None):
        """Docstring for mock_uniform."""
        m = MagicMock()
        m.shape = shape
        m.dtype = dtype
        return m

    mock_jax.random.uniform.side_effect = mock_uniform

    def mock_split(rng):
        """Docstring for mock_split."""
        return rng, MagicMock()

    mock_jax.random.split.side_effect = mock_split

    monkeypatch.setattr(peft_module, "jax", mock_jax)
    monkeypatch.setattr(peft_module, "jnp", mock_jnp)
    monkeypatch.setattr(peft_module, "np", mock_np)
    monkeypatch.setattr(peft_module, "optax", mock_optax)
    monkeypatch.setattr(peft_module, "Gemma4Model", mock_gemma4)

    yield {"jax": mock_jax, "jnp": mock_jnp, "np": mock_np, "optax": mock_optax, "gemma4": mock_gemma4}


def test_transform_params_to_lora_missing_jax(monkeypatch):
    """Docstring for test_transform_params_to_lora_missing_jax."""
    monkeypatch.setattr(peft_module, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX dependencies are missing"):
        peft_module.transform_params_to_lora({}, ["q_proj"])


def test_transform_params_to_lora_invalid_rank():
    """Docstring for test_transform_params_to_lora_invalid_rank."""
    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        peft_module.transform_params_to_lora({}, ["q_proj"], lora_r=0)


def test_transform_params_to_lora_success():
    """Docstring for test_transform_params_to_lora_success."""
    kernel = MagicMock()
    kernel.shape = (10, 20)
    kernel.dtype = "float32"

    params = {"layer_1": {"q_proj": {"kernel": kernel, "other": 1}, "k_proj": {"kernel": kernel}, "no_kernel": {"a": 1}}, "scalar": 42}

    _transformed, count = peft_module.transform_params_to_lora(params, ["q_proj"], lora_r=8)

    assert count == 1


def test_transform_params_to_lora_with_rng():
    """Docstring for test_transform_params_to_lora_with_rng."""
    kernel = MagicMock()
    kernel.shape = (10, 20)
    params = {"layer": {"q_proj": {"kernel": kernel}}}
    rng = MagicMock()
    peft_module.transform_params_to_lora(params, ["q_proj"], lora_r=8, rng=rng)


def test_transform_params_to_lora_no_dict():
    """Docstring for test_transform_params_to_lora_no_dict."""
    res, count = peft_module.transform_params_to_lora("not a dict", ["q_proj"])
    assert res == "not a dict"
    assert count == 0


def test_segregate_adapter_params():
    """Docstring for test_segregate_adapter_params."""
    params = {"layer": {"q_proj": {"kernel": "frozen_kernel", "lora_a": "trainable_a", "lora_b": "trainable_b", "lora_scale": "trainable_scale", "other": "frozen_other"}}, "scalar": "frozen_scalar"}

    trainable, _frozen = peft_module.segregate_adapter_params(params)
    assert trainable["layer"]["q_proj"]["lora_a"] == "trainable_a"


def test_segregate_adapter_params_empty_sub_t_sub_f():
    """Docstring for test_segregate_adapter_params_empty_sub_t_sub_f."""
    params = {"layer": {"empty": {}}}
    t, f = peft_module.segregate_adapter_params(params)
    assert t == {}
    assert f == {}


def test_segregate_adapter_params_not_dict():
    """Docstring for test_segregate_adapter_params_not_dict."""
    trainable, frozen = peft_module.segregate_adapter_params("not dict")
    assert trainable == {}
    assert frozen == {}


def test_create_maxtext_lora_optimizer_missing_deps(monkeypatch):
    """Docstring for test_create_maxtext_lora_optimizer_missing_deps."""
    monkeypatch.setattr(peft_module, "optax", None)
    with pytest.raises(DependencyMissingError, match="Optax or JAX dependencies are missing"):
        peft_module.create_maxtext_lora_optimizer({})


def test_create_maxtext_lora_optimizer_success(mock_deps):
    """Docstring for test_create_maxtext_lora_optimizer_success."""
    mock_optax = mock_deps["optax"]
    mock_optax.adam.return_value = "adam_opt"
    mock_optax.set_to_zero.return_value = "zero_opt"
    mock_optax.multi_transform.return_value = "multi_opt"

    def mock_tree_map_with_path(f, tree):
        """Docstring for mock_tree_map_with_path."""

        class PathElement:
            """Docstring for PathElement."""

            def __init__(self, key):
                """Docstring for __init__."""
                self.key = key

        class PathElementStr:
            """Docstring for PathElementStr."""

            def __str__(self):
                """Docstring for __str__."""
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
    """Docstring for test_create_maxtext_lora_optimizer_custom_base."""
    peft_module.create_maxtext_lora_optimizer({"fake": "params"}, base_optimizer="custom")
    mock_optax = mock_deps["optax"]
    mock_optax.multi_transform.assert_called_once_with({"trainable": "custom", "frozen": mock_optax.set_to_zero.return_value}, ANY)


def test_merge_lora_weights_better():
    """Docstring for test_merge_lora_weights_better."""

    class DummyTensor:
        """Docstring for DummyTensor."""

        def __init__(self, name):
            """Docstring for __init__."""
            self.name = name

        def __matmul__(self, other):
            """Docstring for __matmul__."""
            return DummyTensor(f"{self.name}@{other.name}")

        def __rmul__(self, scalar):
            """Docstring for __rmul__."""
            return DummyTensor(f"{scalar}*{self.name}")

        def __add__(self, other):
            """Docstring for __add__."""
            return DummyTensor(f"{self.name}+{other.name}")

    params = {"layer": {"q_proj": {"kernel": DummyTensor("W"), "lora_a": DummyTensor("A"), "lora_b": DummyTensor("B"), "lora_scale": 2.0, "other": "kept"}, "no_lora": {"kernel": "just_kernel"}}, "scalar": 42}
    merged = peft_module.merge_lora_weights(params)
    assert merged["scalar"] == 42


def test_merge_lora_weights_not_dict():
    """Docstring for test_merge_lora_weights_not_dict."""
    assert peft_module.merge_lora_weights("not dict") == "not dict"


def test_save_maxtext_adapters_missing_deps(monkeypatch):
    """Docstring for test_save_maxtext_adapters_missing_deps."""
    monkeypatch.setattr(peft_module, "np", None)
    with pytest.raises(DependencyMissingError, match="NumPy dependency is missing"):
        peft_module.save_maxtext_adapters({}, "path")


def test_save_maxtext_adapters_success(mock_deps, tmp_path):
    """Docstring for test_save_maxtext_adapters_success."""
    mock_np = mock_deps["np"]
    params = {"layer": {"q_proj": {"lora_a": "array_a", "lora_b": "array_b", "lora_scale": "scale_val", "kernel": "array_kernel"}, "other": "value"}}

    save_path = tmp_path / "adapters.npz"
    peft_module.save_maxtext_adapters(params, save_path)
    mock_np.savez.assert_called_once()


def test_save_maxtext_adapters_not_dict(mock_deps, tmp_path):
    """Docstring for test_save_maxtext_adapters_not_dict."""
    mock_np = mock_deps["np"]
    peft_module.save_maxtext_adapters("not dict", tmp_path)
    mock_np.savez.assert_called_once_with(ANY)


def test_save_maxtext_adapters_is_dir(mock_deps, tmp_path):
    """Docstring for test_save_maxtext_adapters_is_dir."""
    mock_np = mock_deps["np"]
    peft_module.save_maxtext_adapters({"lora_a": "1"}, tmp_path)
    expected_path = tmp_path / "maxtext_lora_adapters.npz"
    mock_np.savez.assert_called_once_with(expected_path, lora_a="1")


def test_load_maxtext_adapters_missing_deps(monkeypatch):
    """Docstring for test_load_maxtext_adapters_missing_deps."""
    monkeypatch.setattr(peft_module, "np", None)
    with pytest.raises(DependencyMissingError, match="NumPy dependency is missing"):
        peft_module.load_maxtext_adapters({}, "path")


def test_load_maxtext_adapters_not_found():
    """Docstring for test_load_maxtext_adapters_not_found."""
    with pytest.raises(FileNotFoundError, match="Adapter file not found"):
        peft_module.load_maxtext_adapters({}, "nonexistent.npz")


def test_load_maxtext_adapters_success(mock_deps, tmp_path):
    """Docstring for test_load_maxtext_adapters_success."""
    mock_np = mock_deps["np"]
    mock_np.load.return_value = {"layer.q_proj.lora_a": "loaded_a", "layer.q_proj.lora_b": "loaded_b", "new_layer.lora_a": "new_a"}
    save_path = tmp_path / "adapters.npz"
    save_path.touch()

    params = {"layer": {"q_proj": {"kernel": "W"}}}
    loaded = peft_module.load_maxtext_adapters(params, save_path)
    assert loaded["layer"]["q_proj"]["lora_a"] == "loaded_a"


def test_load_maxtext_adapters_no_jnp(mock_deps, tmp_path, monkeypatch):
    """Docstring for test_load_maxtext_adapters_no_jnp."""
    monkeypatch.setattr(peft_module, "jnp", None)
    mock_np = mock_deps["np"]
    mock_np.load.return_value = {"a.b": "val"}
    save_path = tmp_path / "adapters.npz"
    save_path.touch()
    loaded = peft_module.load_maxtext_adapters({}, save_path)
    assert loaded["a"]["b"] == "val"


def test_count_maxtext_parameters():
    """Docstring for test_count_maxtext_parameters."""

    class Sized:
        """Docstring for Sized."""

        def __init__(self, size):
            """Docstring for __init__."""
            self.size = size

    params = {"layer": {"kernel": Sized(100), "lora_a": Sized(10), "lora_b": Sized(20)}, "not_sized": "scalar", "scalar_size": Sized(5)}

    total, trainable = peft_module.count_maxtext_parameters(params)
    assert total == 135
    assert trainable == 30


def test_count_maxtext_parameters_not_dict():
    """Docstring for test_count_maxtext_parameters_not_dict."""
    total, trainable = peft_module.count_maxtext_parameters("not dict")
    assert total == 0
    assert trainable == 0


def test_apply_lora_missing_deps(monkeypatch):
    """Docstring for test_apply_lora_missing_deps."""
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
    """Docstring for test_apply_lora_missing_gemma4_but_has_params."""
    monkeypatch.setattr(peft_module, "Gemma4Model", None)
    with patch.object(peft_module, "transform_params_to_lora", return_value=({}, 0)):
        res = peft_module.apply_lora("gemma-4", ["q_proj"], params={})
    assert res["status"] == "completed"


def test_apply_lora_success(mock_deps, tmp_path):
    """Docstring for test_apply_lora_success."""
    mock_gemma4_class = mock_deps["gemma4"]
    mock_model = MagicMock()
    mock_gemma4_class.return_value = mock_model

    mock_model.init.return_value = {"original": "params"}

    def fake_transform(params, target_modules, lora_r, lora_alpha, lora_dropout):
        """Docstring for fake_transform."""
        return {"transformed": "params"}, 5

    with patch.object(peft_module, "transform_params_to_lora", side_effect=fake_transform), patch.object(peft_module, "save_maxtext_adapters"), patch.object(peft_module, "merge_lora_weights", return_value={"merged": "params"}):
        res = peft_module.apply_lora("gemma-4", ["q_proj"], lora_r=16, output_dir=tmp_path, merge=True)
        assert res["status"] == "completed"


def test_apply_lora_success_no_output_dir_no_merge(mock_deps):
    """Docstring for test_apply_lora_success_no_output_dir_no_merge."""
    with patch.object(peft_module, "transform_params_to_lora", return_value=({}, 5)):
        res = peft_module.apply_lora("gemma-4", ["q_proj"])
        assert res["status"] == "completed"


def test_apply_lora_failure(mock_deps):
    """Docstring for test_apply_lora_failure."""

    def fake_transform(*args, **kwargs):
        """Docstring for fake_transform."""
        raise RuntimeError("Fake Error")

    with patch.object(peft_module, "transform_params_to_lora", side_effect=fake_transform):
        res = peft_module.apply_lora("gemma-4", ["q_proj"])
        assert "failed: Fake Error" in res["status"]


def test_apply_lora_missing_dependency_inside(mock_deps, monkeypatch):
    """Docstring for test_apply_lora_missing_dependency_inside."""
    monkeypatch.setattr(peft_module, "Gemma4Model", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependency missing."):
        peft_module.apply_lora("gemma-4", ["q_proj"], params="not_a_dict")


def test_module_reload_for_coverage():
    """Docstring for test_module_reload_for_coverage."""
    import importlib
    import sys
    from unittest.mock import MagicMock

    mock_gemma4 = MagicMock()
    mock_gemma4.Gemma4Model = "mocked_model"
    sys.modules["maxtext"] = MagicMock()
    sys.modules["maxtext.models"] = MagicMock()
    sys.modules["maxtext.models.gemma4"] = mock_gemma4

    from gemma_4_sql.backends.maxtext import peft

    importlib.reload(peft)
    assert peft.Gemma4Model is not None

    del sys.modules["maxtext.models.gemma4"]
    peft.Gemma4Model = None
