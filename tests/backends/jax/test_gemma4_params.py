import sys
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.jax.gemma4.params import (
    _fix_jax_state_embeddings,
    _get_audio_mappings,
    _get_key_and_transform_mapping,
    _get_text_mappings,
    _get_vision_mappings,
    _process_moe_tensor,
    _process_safetensors_file,
    _stack_and_assign_expert_tensors,
    create_gemma4_from_pretrained,
    process_standard_tensor,
)
from gemma_4_sql.exceptions import DependencyMissingError


class DummyTransform:
    DEFAULT = None
    BIAS = None
    LINEAR = ((1, 0), None, False)
    CONV2D = ((2, 3, 1, 0), None, False)
    EMBED = None
    LINEAR_3D = ((0, 2, 1), None, False)


def test_mappings():
    t_maps = _get_text_mappings(DummyTransform)
    a_maps = _get_audio_mappings(DummyTransform)
    v_maps = _get_vision_mappings(DummyTransform)
    all_maps = _get_key_and_transform_mapping()

    assert r"^model\.embed_tokens\.weight$" in t_maps
    assert r"^audio_tower\.output_proj\.bias$" in a_maps
    assert r"^vision_tower\.vision_model\.post_layernorm\.weight$" in v_maps
    assert r"^model\.embed_tokens\.weight$" in all_maps
    assert r"^audio_tower\.output_proj\.bias$" in all_maps
    assert r"^vision_tower\.vision_model\.post_layernorm\.weight$" in all_maps


def test_process_moe_tensor():
    import re

    moe_pattern = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    match = moe_pattern.match("model.layers.0.block_sparse_moe.experts.1.gate_proj.weight")

    mock_sf = MagicMock()
    mock_sf.get_tensor.return_value = jnp.array([1.0])
    expert_tensors = {}

    _process_moe_tensor(match, mock_sf, "model.layers.0.block_sparse_moe.experts.1.gate_proj.weight", expert_tensors)

    assert 0 in expert_tensors
    assert "gate_proj" in expert_tensors[0]
    assert 1 in expert_tensors[0]["gate_proj"]
    assert jnp.allclose(expert_tensors[0]["gate_proj"][1], jnp.array([1.0]))


@patch("gemma_4_sql.backends.jax.gemma4.params.assign_weights_from_eval_shape")
def test_process_standard_tensor(mock_assign):
    mock_sf = MagicMock()
    mock_sf.get_tensor.return_value = jnp.array([1.0])

    mapping = {r"^model\.embed\.weight$": (r"model\.embed", DummyTransform.EMBED)}

    process_standard_tensor(mock_sf, "model.embed.weight", {}, mapping)
    mock_assign.assert_called_once()

    mock_assign.reset_mock()
    process_standard_tensor(mock_sf, "unmatched.weight", {}, mapping)
    mock_assign.assert_not_called()

    mock_assign.side_effect = KeyError("Test error")
    process_standard_tensor(mock_sf, "model.embed.weight", {}, mapping)

    mock_assign.side_effect = TypeError("Test error")
    with pytest.raises(TypeError):
        process_standard_tensor(mock_sf, "model.embed.weight", {}, mapping)


@patch("gemma_4_sql.backends.jax.gemma4.params.assign_weights_from_eval_shape")
def test_stack_and_assign_expert_tensors(mock_assign):
    expert_tensors = {0: {"gate_proj": {0: jnp.array([1.0]), 1: jnp.array([2.0])}}}
    mapping = {r"^model\.layers\.(\d+)\.mlp\.routed_experts\.(gate_proj)\.weight$": ("model\\.layers\\.\1\\.mlp\\.routed_experts\\.\2_kernel", DummyTransform.LINEAR_3D)}
    jax_state = {}

    _stack_and_assign_expert_tensors(expert_tensors, mapping, jax_state)
    mock_assign.assert_called_once()


@patch("gemma_4_sql.backends.jax.gemma4.params.safetensors")
@patch("gemma_4_sql.backends.jax.gemma4.params._process_moe_tensor")
@patch("gemma_4_sql.backends.jax.gemma4.params.process_standard_tensor")
def test_process_safetensors_file(mock_standard, mock_moe, mock_safetensors):
    mock_sf = MagicMock()
    mock_sf.keys.return_value = ["model.layers.0.block_sparse_moe.experts.1.gate_proj.weight", "model.embed.weight"]
    mock_safetensors.safe_open.return_value.__enter__.return_value = mock_sf

    import re

    moe_pattern = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    expert_tensors = {}
    jax_state = {}
    mapping = {}

    _process_safetensors_file("dummy.safetensors", moe_pattern, expert_tensors, jax_state, mapping)

    mock_moe.assert_called_once()
    mock_standard.assert_called_once()


def test_fix_jax_state_embeddings():
    class DummyShapeDtypeStruct:
        pass

    class DummyConfig:
        hidden_size = 4
        vision_config = MagicMock(num_patches=256)

    cfg = DummyConfig()

    jax_state = {"model": {"embed_scale": DummyShapeDtypeStruct()}}
    with patch("gemma_4_sql.backends.jax.gemma4.params.jax.ShapeDtypeStruct", DummyShapeDtypeStruct, create=True):
        _fix_jax_state_embeddings(jax_state, None, cfg)

    assert isinstance(jax_state["model"]["embed_scale"], jax.Array)

    jax_state = {"vision_tower": {"embeddings": {"position_ids": DummyShapeDtypeStruct()}}}
    with patch("gemma_4_sql.backends.jax.gemma4.params.jax.ShapeDtypeStruct", DummyShapeDtypeStruct, create=True):
        _fix_jax_state_embeddings(jax_state, None, cfg)

    assert isinstance(jax_state["vision_tower"]["embeddings"]["position_ids"], jax.Array)

    jax_state = {}
    _fix_jax_state_embeddings(jax_state, None, cfg)


@patch("gemma_4_sql.backends.jax.gemma4.params.nnx")
@patch("gemma_4_sql.backends.jax.gemma4.params._process_safetensors_file")
@patch("gemma_4_sql.backends.jax.gemma4.params._stack_and_assign_expert_tensors")
@patch("gemma_4_sql.backends.jax.gemma4.params._fix_jax_state_embeddings")
def test_create_gemma4_from_pretrained(mock_fix, mock_stack, mock_process, mock_nnx):
    mock_epath = MagicMock()
    mock_epath.epath = mock_epath
    mock_epath.Path.return_value.expanduser.return_value.glob.return_value = ["file1.safetensors"]

    cfg = MagicMock()

    mock_graph_def = MagicMock()
    mock_state = MagicMock()
    mock_state.to_pure_dict.return_value = {"a": 1}
    mock_nnx.eval_shape.return_value = "dummy"
    mock_nnx.split.return_value = (mock_graph_def, mock_state)
    mock_nnx.merge.return_value = "merged_model"

    with patch.dict("sys.modules", {"etils": MagicMock(epath=mock_epath)}):
        res = create_gemma4_from_pretrained("dummy_dir", cfg)
        assert res == "merged_model"
        mock_process.assert_called_once()
        mock_stack.assert_called_once()
        mock_fix.assert_called_once()

        mock_state_2 = MagicMock()
        del mock_state_2.to_pure_dict
        mock_state_2.to_flat_dict.return_value = {"a": 1}
        mock_nnx.split.return_value = (mock_graph_def, mock_state_2)
        create_gemma4_from_pretrained("dummy_dir", cfg)

        mock_state_3 = {"a": 1}
        mock_nnx.split.return_value = (mock_graph_def, mock_state_3)
        create_gemma4_from_pretrained("dummy_dir", cfg)

        mock_epath.Path.return_value.expanduser.return_value.glob.return_value = []
        with pytest.raises(ValueError, match="No safetensors found"):
            create_gemma4_from_pretrained("dummy_dir", cfg)

    with patch.dict("sys.modules", {"etils": None}), pytest.raises(DependencyMissingError):
        create_gemma4_from_pretrained("dummy_dir", cfg)


def test_process_moe_tensor_branch_coverage():
    from unittest.mock import MagicMock

    import jax.numpy as jnp

    from gemma_4_sql.backends.jax.gemma4.params import _process_moe_tensor

    sf = MagicMock()
    sf.get_tensor.return_value = jnp.ones((2, 2))
    match = MagicMock()
    match.groups.return_value = ("0", "1", "gate_proj")
    expert_tensors = {}
    _process_moe_tensor(match, sf, "key1", expert_tensors)

    match2 = MagicMock()
    match2.groups.return_value = ("0", "2", "up_proj")
    _process_moe_tensor(match2, sf, "key2", expert_tensors)

    match3 = MagicMock()
    match3.groups.return_value = ("0", "3", "gate_proj")
    _process_moe_tensor(match3, sf, "key3", expert_tensors)


def test_fix_jax_state_embeddings_coverage():
    import jax

    from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
    from gemma_4_sql.backends.jax.gemma4.params import _fix_jax_state_embeddings

    cfg = ModelConfig()

    state = {"model": {"embed_scale": jax.ShapeDtypeStruct((), jax.numpy.float32)}}
    _fix_jax_state_embeddings(state, None, cfg)


def test_params_missing_moe_branches():
    from unittest.mock import MagicMock

    import jax.numpy as jnp

    from gemma_4_sql.backends.jax.gemma4.params import _process_moe_tensor

    sf = MagicMock()
    sf.get_tensor.return_value = jnp.ones((2, 2))

    match = MagicMock()
    match.groups.return_value = ("0", "1", "gate_proj")
    expert_tensors = {}
    _process_moe_tensor(match, sf, "key1", expert_tensors)

    match2 = MagicMock()
    match2.groups.return_value = ("0", "2", "up_proj")
    _process_moe_tensor(match2, sf, "key2", expert_tensors)

    match3 = MagicMock()
    match3.groups.return_value = ("0", "3", "gate_proj")
    _process_moe_tensor(match3, sf, "key3", expert_tensors)


def test_params_missing_fix_jax_state_embeddings():
    import jax

    from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
    from gemma_4_sql.backends.jax.gemma4.params import _fix_jax_state_embeddings

    cfg = ModelConfig()

    state = {"model": {"embed_scale": jax.ShapeDtypeStruct((), jax.numpy.float32)}}
    _fix_jax_state_embeddings(state, None, cfg)


def test_create_gemma4_from_pretrained_etils_missing():
    from unittest.mock import MagicMock, patch

    from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained
    from gemma_4_sql.exceptions import DependencyMissingError

    # Mock sys.modules to remove etils
    with patch.dict(sys.modules, {"etils": None, "etils.epath": None}), pytest.raises(DependencyMissingError):
        create_gemma4_from_pretrained("/fake/dir", MagicMock())


def test_create_gemma4_from_pretrained_etils_success_and_state():
    from importlib.abc import Loader, MetaPathFinder
    from importlib.machinery import ModuleSpec
    from unittest.mock import MagicMock, patch

    from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained

    class EtilsFinder(MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "etils":
                return ModuleSpec("etils", EtilsLoader())
            return None

    class EtilsLoader(Loader):
        def create_module(self, spec):
            mock_etils = MagicMock()
            mock_epath = MagicMock()
            mock_epath.Path.return_value.expanduser.return_value.glob.return_value = ["file.safetensors"]
            mock_etils.epath = mock_epath
            return mock_etils

        def exec_module(self, module):
            pass

    sys.modules.pop("etils", None)
    finder = EtilsFinder()
    sys.meta_path.insert(0, finder)
    try:
        mock_nnx = MagicMock()
        mock_graph_def = MagicMock()
        mock_state = MagicMock()
        mock_state.to_pure_dict.return_value = {"a": 1}
        mock_nnx.eval_shape.return_value = "dummy"
        mock_nnx.split.return_value = (mock_graph_def, mock_state)
        mock_nnx.merge.return_value = "merged_model"

        # We need to simulate that nnx does not have "State"
        # Since we patch params.nnx with mock_nnx, we must ensure it doesn't have 'State'
        # MagicMock has everything by default, so we delete it.
        del mock_nnx.State

        cfg = MagicMock()
        with patch("gemma_4_sql.backends.jax.gemma4.params.nnx", mock_nnx), patch("gemma_4_sql.backends.jax.gemma4.params._process_safetensors_file"), patch("gemma_4_sql.backends.jax.gemma4.params._stack_and_assign_expert_tensors"), patch("gemma_4_sql.backends.jax.gemma4.params._fix_jax_state_embeddings"):
            res = create_gemma4_from_pretrained("dummy_dir", cfg)
            assert res == "merged_model"

    finally:
        sys.meta_path.remove(finder)


def test_create_gemma4_from_pretrained_missing_etils():
    from unittest.mock import patch

    from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
    from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained
    from gemma_4_sql.exceptions import DependencyMissingError

    with patch.dict("sys.modules", {"etils": None}), pytest.raises(DependencyMissingError):
        create_gemma4_from_pretrained("/fake/dir", ModelConfig())


def test_create_gemma4_from_pretrained_no_safetensors():
    from unittest.mock import MagicMock, patch

    from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
    from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained

    mock_epath = MagicMock()
    mock_epath.Path.return_value.expanduser.return_value.glob.return_value = []
    with patch.dict("sys.modules", {"etils": mock_epath, "etils.epath": mock_epath}), pytest.raises(ValueError):
        create_gemma4_from_pretrained("/fake/dir", ModelConfig())


def test_create_gemma4_from_pretrained_success():
    from unittest.mock import MagicMock, patch

    from gemma_4_sql.backends.jax.gemma4.config import ModelConfig
    from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained

    mock_epath = MagicMock()
    mock_epath.epath = mock_epath
    mock_epath.Path.return_value.expanduser.return_value.glob.return_value = ["file1.safetensors"]

    with (
        patch.dict("sys.modules", {"etils": mock_epath, "etils.epath": mock_epath}),
        patch("gemma_4_sql.backends.jax.gemma4.params.nnx.eval_shape") as mock_eval,
        patch("gemma_4_sql.backends.jax.gemma4.params.nnx.split") as mock_split,
        patch("gemma_4_sql.backends.jax.gemma4.params.nnx.merge") as mock_merge,
        patch("gemma_4_sql.backends.jax.gemma4.params._process_safetensors_file"),
        patch("gemma_4_sql.backends.jax.gemma4.params._stack_and_assign_expert_tensors"),
        patch("gemma_4_sql.backends.jax.gemma4.params._fix_jax_state_embeddings"),
    ):
        from flax import nnx

        if hasattr(nnx, "State"):
            del nnx.State

        mock_eval.return_value = "gemma4"
        mock_split.return_value = ("graph_def", {})
        mock_merge.return_value = "merged_model"

        model = create_gemma4_from_pretrained("/fake/dir", ModelConfig())
        assert model == "merged_model"
