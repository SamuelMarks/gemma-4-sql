"""Tests for test gemma4 params module."""

from pathlib import Path

import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights


def test_assign_weights_key_error():
    """Test assign weights key error functionality."""
    state = {"a": jnp.zeros((1,))}
    with pytest.raises(KeyError):
        assign_weights(["b"], jnp.ones((1,)), state, "st_key", None)


def test_assign_weights_permute():
    """Test assign weights permute functionality."""
    state = {"a": jnp.zeros((2, 1))}
    assign_weights(["a"], jnp.ones((1, 2)), state, "st_key", transform=((1, 0), None, False))
    assert state["a"].shape == (2, 1)


from typing import Any
from typing import NoReturn as Never

import jax

from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained


def test_assign_weight_type_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    m_params = __import__("gemma_4_sql.backends.jax.gemma4.params", fromlist=[""])

    def raise_type_error(*_args: object, **_kwargs: object) -> Never:
        """Execute function.

        Raises:
            TypeError: Description.

        """
        msg = "err"
        raise TypeError(msg)

    monkeypatch.setattr(m_params, "assign_weights_from_eval_shape", raise_type_error)
    with pytest.raises(TypeError):
        m_params.process_standard_tensor(type("SF", (), {"get_tensor": lambda _self, _k: 1})(), "a", {}, {"a": ("a", type("MockTransform", (), {"value": None})())})


class MockConfig:
    """Provide class docstring."""

    hidden_size = 128
    vision_config = True


class MockModelObj:
    """Provide class docstring."""

    vision_tower = type("VT", (), {"embeddings": type("E", (), {"num_patches": 10})()})()


class MockNNX:
    """Provide class docstring."""

    def eval_shape(self, _fn: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockModelObj()

    def split(self, _x: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return (None, type("State", (), {"to_pure_dict": lambda _self: {"model": {"embed_scale": jax.ShapeDtypeStruct((1,), jnp.bfloat16)}, "vision_tower": {"embeddings": {"position_ids": jax.ShapeDtypeStruct((1,), jnp.int32)}}}})())

    def merge(self, _graph_def: object, state: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return state

    class Rngs:
        """Provide class docstring."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Execute function."""


class MockSt:
    """Provide class docstring."""

    def safe_open(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return type("CM", (), {"__enter__": lambda _self: _self, "__exit__": lambda _self, *_a: None, "__iter__": lambda _self: iter([]), "keys": lambda _self: [], "get_tensor": lambda _self, _k: jnp.array(1)})()


def test_create_gemma4_from_pretrained_missing_nnx_state(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_params = __import__("gemma_4_sql.backends.jax.gemma4.params", fromlist=[""])
    monkeypatch.setattr(m_params, "nnx", MockNNX())
    monkeypatch.setattr(m_params, "safetensors", MockSt())
    (tmp_path / "model.safetensors").touch()
    res = create_gemma4_from_pretrained(str(tmp_path), MockConfig())
    if "embed_scale" not in res["model"]:
        raise AssertionError
    if "position_ids" not in res["vision_tower"]["embeddings"]:
        raise AssertionError


def test_create_gemma4_from_pretrained_flat_and_plain_dict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test state dictionary extraction using to_flat_dict and plain dict fallbacks.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    m_params = __import__("gemma_4_sql.backends.jax.gemma4.params", fromlist=[""])
    (tmp_path / "model.safetensors").touch()

    # 1. State with to_flat_dict
    class MockNNXFlat(MockNNX):
        """Mock NNX with to_flat_dict."""

        def split(self, _x: object) -> object:
            """Return state with to_flat_dict."""
            return (None, type("StateFlat", (), {"to_flat_dict": lambda _self: {"model": {"embed_scale": jax.ShapeDtypeStruct((1,), jnp.bfloat16)}, "vision_tower": {"embeddings": {"position_ids": jax.ShapeDtypeStruct((1,), jnp.int32)}}}})())

    monkeypatch.setattr(m_params, "nnx", MockNNXFlat())
    monkeypatch.setattr(m_params, "safetensors", MockSt())
    res_flat = create_gemma4_from_pretrained(str(tmp_path), MockConfig())
    assert "embed_scale" in res_flat["model"]

    # 2. State as plain dict
    class MockNNXDict(MockNNX):
        """Mock NNX with plain dict state."""

        def split(self, _x: object) -> object:
            """Return plain dict state."""
            return (None, {"model": {"embed_scale": jax.ShapeDtypeStruct((1,), jnp.bfloat16)}, "vision_tower": {"embeddings": {"position_ids": jax.ShapeDtypeStruct((1,), jnp.int32)}}})

    monkeypatch.setattr(m_params, "nnx", MockNNXDict())
    res_dict = create_gemma4_from_pretrained(str(tmp_path), MockConfig())
    assert "embed_scale" in res_dict["model"]


import re
from unittest.mock import MagicMock

from gemma_4_sql.backends.jax.gemma4.params import _process_safetensors_file
from gemma_4_sql.backends.jax.gemma4.utils_params import _load_weights_from_safetensors_file


def test_load_weights_from_safetensors_file_key_error(monkeypatch):
    """Test load weights from safetensors file key error functionality."""

    class MockFile:
        """Test class for MockFile."""

        def __iter__(self):
            """Initialize __iter__."""
            yield "model.layer.weight"

        def get_tensor(self, key):
            """Execute get tensor helper."""
            return jnp.ones((1,))

    mock_safe_open = MagicMock()
    mock_safe_open.return_value.__enter__.return_value = MockFile()
    monkeypatch.setattr("gemma_4_sql.backends.jax.gemma4.utils_params.safe_open", mock_safe_open)
    state = {}
    key_mapping = {"model.layer.weight": ("layer.weight", None)}
    _load_weights_from_safetensors_file("test.safetensors", state, key_mapping)


def test_process_safetensors_file_jax_key_not_none(monkeypatch):
    """Test process safetensors file jax key not none functionality."""

    class MockFile:
        """Test class for MockFile."""

        def __init__(self):
            """Initialize __init__."""
            self.keys_list = ["model.layers.0.mlp.routed_experts.w1.weight"]

        def keys(self):
            """Execute keys helper."""
            return self.keys_list

        def get_tensor(self, key):
            """Execute get tensor helper."""
            return jnp.ones((1,))

    mock_safe_open = MagicMock()
    mock_safe_open.return_value.__enter__.return_value = MockFile()
    monkeypatch.setattr("gemma_4_sql.backends.jax.gemma4.params.safetensors.safe_open", mock_safe_open)

    expert_tensors = {}
    jax_state = {"model": {"layers": {"0": {"mlp": {"routed_experts": {"w1": {"weight": MagicMock()}}}}}}}
    mapping = {"model.layers.0.mlp.routed_experts.w1.weight": ("model.layers.0.mlp.routed_experts.w1.weight", None)}
    moe_pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(w1|w2|w3)\.weight")
    _process_safetensors_file("test.safetensors", moe_pattern, expert_tensors, jax_state, mapping)


def test_create_gemma4_vision_pos_ids():
    """Test create gemma4 vision pos ids functionality."""

    class MockConfig:
        """Test class for MockConfig."""

        vision_config = True
        audio_config = False
        hidden_size = 64
        intermediate_size = 128
        num_hidden_layers = 1
        num_attention_heads = 4
        num_key_value_heads = 2
        head_dim = 16
        shd_cfg = MagicMock()
        dtype = jnp.float32
        vocab_size = 100

    cfg = MockConfig()

    mock_model = MagicMock()
    mock_model.vision_tower.embeddings.num_patches = 14

    import jax

    from gemma_4_sql.backends.jax.gemma4.params import _fix_jax_state_embeddings

    mock_state = {"model": {"embed_scale": jax.ShapeDtypeStruct((), jnp.float32)}, "vision_tower": {"embeddings": {"position_ids": jax.ShapeDtypeStruct((1, 14), jnp.int32)}}}
    _fix_jax_state_embeddings(mock_state, mock_model, cfg)
    assert mock_state["vision_tower"]["embeddings"]["position_ids"].shape == (1, 14)


def test_map_to_jax_key_multiple_mappings() -> None:
    """Test map_to_jax_key raises ValueError when multiple mappings match."""
    from gemma_4_sql.backends.jax.gemma4.utils_params import map_to_jax_key

    mapping = {
        r"layer\.(.*)": ("l1", None),
        r"layer\.weight": ("l2", None),
    }
    with pytest.raises(ValueError, match="Multiple mappings found"):
        map_to_jax_key(mapping, "layer.weight")


def test_assign_weights_with_sharding_and_int_keys() -> None:
    """Test assign_weights with sharding_dict and int/string resolved keys."""
    from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights, assign_weights_from_eval_shape

    class ParamContainer:
        def __init__(self, arr: Any) -> None:
            self.value = arr

    container = ParamContainer(jnp.zeros((2,)))
    state_dict = {"0": container}
    shd = {"0": None}
    assign_weights([0], jnp.ones((2,)), state_dict, "key", None, sharding_dict=shd)
    assert (container.value == jnp.ones((2,))).all()

    # Test shape mismatch in assign_weights
    with pytest.raises(ValueError, match="Shape mismatch"):
        assign_weights(["0"], jnp.ones((5,)), state_dict, "key", None)

    # Test int resolved_key to str in assign_weights_from_eval_shape
    state_dict2 = {"1": jnp.zeros((2,))}
    assign_weights_from_eval_shape([1], jnp.ones((2,)), state_dict2, "key", None)
    assert (state_dict2["1"] == jnp.ones((2,))).all()

    # Test str digit resolved_key to int in assign_weights_from_eval_shape
    state_dict3 = {0: jnp.zeros((2,))}
    assign_weights_from_eval_shape(["0"], jnp.ones((2,)), state_dict3, "key", None)
    assert (state_dict3[0] == jnp.ones((2,))).all()


def test_process_moe_tensor_and_stacking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test processing MoE expert tensors and stacking them into jax_state."""
    import re
    from unittest.mock import MagicMock

    from gemma_4_sql.backends.jax.gemma4.params import _process_safetensors_file, _stack_and_assign_expert_tensors

    class MockFile:
        def keys(self) -> list[str]:
            return [
                "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight",
                "model.layers.0.block_sparse_moe.experts.1.gate_proj.weight",
            ]

        def get_tensor(self, _key: str) -> Any:
            return jnp.ones((4, 4))

    mock_safe_open = MagicMock()
    mock_safe_open.return_value.__enter__.return_value = MockFile()
    monkeypatch.setattr("gemma_4_sql.backends.jax.gemma4.params.safetensors.safe_open", mock_safe_open)

    moe_pattern = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    expert_tensors: dict[int, dict[str, dict[int, jax.Array]]] = {}
    jax_state = {"model": {"layers": {"0": {"mlp": {"routed_experts": {"gate_proj": {"weight": jnp.zeros((2, 4, 4))}}}}}}}
    mapping = {r"model\.layers\.(\d+)\.mlp\.routed_experts\.(.*)\.weight": (r"model\.layers\.\1\.mlp\.routed_experts\.\2\.weight", None)}

    _process_safetensors_file("dummy.safetensors", moe_pattern, expert_tensors, jax_state, mapping)
    assert 0 in expert_tensors
    assert "gate_proj" in expert_tensors[0]
    assert len(expert_tensors[0]["gate_proj"]) == 2

    _stack_and_assign_expert_tensors(expert_tensors, mapping, jax_state)

    from gemma_4_sql.backends.jax.gemma4.params import process_standard_tensor

    process_standard_tensor(MockFile(), "unmapped.key", jax_state, {})


def test_create_gemma4_from_pretrained_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Test create_gemma4_from_pretrained full flow with mock files."""
    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config
    from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir(parents=True, exist_ok=True)
    cfg = Gemma4Config(vocab_size=10, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8, intermediate_size=16)
    with pytest.raises(ValueError, match="No safetensors found"):
        create_gemma4_from_pretrained(str(empty_dir), cfg)

    (tmp_path / "model.safetensors").write_bytes(b"mock")
    monkeypatch.setattr("gemma_4_sql.backends.jax.gemma4.params._process_safetensors_file", lambda *a, **k: None)
    res = create_gemma4_from_pretrained(str(tmp_path), cfg)
    assert res is not None


def test_gemma4_config_presets_fsdp_tp() -> None:
    """Test Gemma4Config presets with use_fsdp and use_tp enabled."""
    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config
    from gemma_4_sql.backends.jax.gemma4.config import VisionShardConfig

    assert VisionShardConfig.no_sharding() is not None

    cfg1 = Gemma4Config.gemma4_base(use_fsdp=False, use_tp=False)
    assert hasattr(cfg1, "hidden_size")
    cfg2 = Gemma4Config.gemma4_base(use_fsdp=True, use_tp=False)
    assert hasattr(cfg2, "shd_cfg")
    cfg3 = Gemma4Config.gemma4_base(use_fsdp=False, use_tp=True)
    assert hasattr(cfg3, "shd_cfg")

    assert Gemma4Config.gemma4_e2b(use_fsdp=False, use_tp=False) is not None
    assert Gemma4Config.gemma4_e2b(use_fsdp=True, use_tp=True) is not None
    assert Gemma4Config.gemma4_e4b(use_fsdp=True, use_tp=True) is not None
    assert Gemma4Config.gemma4_26b_a4b(use_fsdp=True, use_tp=True) is not None
    assert Gemma4Config.gemma4_31b(use_fsdp=True, use_tp=True) is not None
