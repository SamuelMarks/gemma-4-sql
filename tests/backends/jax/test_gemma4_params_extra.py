import jax
import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained


def test_assign_weight_type_error(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.jax.gemma4.params as m_params

    def raise_type_error(*args, **kwargs):
        msg = "err"
        raise TypeError(msg)

    monkeypatch.setattr(m_params, "assign_weights_from_eval_shape", raise_type_error)
    with pytest.raises(TypeError):
        m_params._process_standard_tensor(type("SF", (), {"get_tensor": lambda self, k: 1})(), "a", {}, {"a": ("a", type("MockTransform", (), {"value": None})())})


def test_create_gemma4_from_pretrained_missing_nnx_state(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import gemma_4_sql.backends.jax.gemma4.params as m_params

    class MockConfig:
        hidden_size = 128
        vision_config = True

    class MockModelObj:
        vision_tower = type("VT", (), {"embeddings": type("E", (), {"num_patches": 10})()})()

    class MockNNX:
        def eval_shape(self, fn):
            return MockModelObj()

        def split(self, x):
            return None, type("State", (), {"to_pure_dict": lambda self: {"model": {"embed_scale": jax.ShapeDtypeStruct((1,), jnp.bfloat16)}, "vision_tower": {"embeddings": {"position_ids": jax.ShapeDtypeStruct((1,), jnp.int32)}}}})()

        def merge(self, graph_def, state):
            return state

        class Rngs:
            def __init__(self, *args, **kwargs):
                pass

    monkeypatch.setattr(m_params, "nnx", MockNNX())

    class MockSt:
        def safe_open(self, *args, **kwargs):
            return type("CM", (), {"__enter__": lambda self: self, "__exit__": lambda self, *a: None, "keys": lambda self: [], "get_tensor": lambda self, k: jnp.array(1)})()

    monkeypatch.setattr(m_params, "safetensors", MockSt())

    # Touch a file to bypass glob check
    (tmp_path / "model.safetensors").touch()

    res = create_gemma4_from_pretrained(str(tmp_path), MockConfig())
    assert "embed_scale" in res["model"]
    assert "position_ids" in res["vision_tower"]["embeddings"]
