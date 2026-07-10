import pytest

from gemma_4_sql.backends.lazy_loader import LazyLoader, catch_optional_imports


def test_catch_optional_imports():
    with catch_optional_imports():
        pass

    with pytest.raises(ValueError), catch_optional_imports():
        raise ValueError("Should not be caught")


def test_lazy_loader(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "my_fake_module", type("Fake", (), {}))

    loader = LazyLoader("my_fake_module")
    assert not loader._loaded
    mod = loader.get_module()
    assert mod is not None
    assert loader._loaded

    # Check repeated calls
    mod2 = loader.get_module()
    assert mod2 is mod


def test_lazy_loader_missing():
    loader = LazyLoader("this_module_does_not_exist")
    assert not loader._loaded
    mod = loader.get_module()
    assert mod is None
    assert loader._loaded
    assert not loader.is_available


def test_lazy_loader_get_modules():
    import gemma_4_sql.backends.lazy_loader as ll

    ll.get_jax()
    ll.get_jnp()
    ll.get_flax_nnx()
    ll.get_tensorflow()
    ll.get_keras()
    ll.get_torch()
    ll.get_mlx()
    ll.get_duckdb()
    ll.get_transformers()
    ll.get_safetensors()
    ll.get_maxtext_gemma4()
