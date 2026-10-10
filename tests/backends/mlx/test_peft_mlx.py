"""Tests for mlx peft."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError


def test_mlx_peft_imports():
    """Test mlx peft imports fallback."""
    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None, "mlx.nn": None, "mlx_lm": None}):
        if "gemma_4_sql.backends.mlx.peft" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.peft"]
        import gemma_4_sql.backends.mlx.peft as peft_module

        assert peft_module.mx is None
        assert peft_module.nn is None


def test_mlx_lora_linear_init():
    """Test MLXLoRALinear initialization."""
    import gemma_4_sql.backends.mlx.peft as peft_module

    peft_module.nn = MagicMock()
    peft_module.mx = MagicMock()

    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        peft_module.MLXLoRALinear(10, 10, r=0)

    layer = peft_module.MLXLoRALinear(10, 20, bias=True)
    assert layer.in_features == 10
    assert layer.out_features == 20
    assert layer.bias is not None
    assert layer.W is layer.weight
    assert layer.A is layer.lora_a
    assert layer.B is layer.lora_b

    layer.W = "w2"
    assert layer.weight == "w2"
    layer.A = "a2"
    assert layer.lora_a == "a2"
    layer.B = "b2"
    assert layer.lora_b == "b2"
    layer.other = "o"
    assert layer.other == "o"

    peft_module.MLXLoRALinear(10, 20, lora_dropout=0.0)

    peft_module.nn = None
    with pytest.raises(DependencyMissingError):
        peft_module.MLXLoRALinear(10, 10)


def test_mlx_lora_linear_from_linear():
    """Test MLXLoRALinear from_linear."""
    import gemma_4_sql.backends.mlx.peft as peft_module

    peft_module.nn = MagicMock()
    peft_module.mx = MagicMock()

    linear = MagicMock()
    linear.weight.shape = (20, 10)
    linear.bias = "bias"

    layer = peft_module.MLXLoRALinear.from_linear(linear)
    assert layer.weight == linear.weight
    assert layer.bias == "bias"

    linear.bias = None
    peft_module.MLXLoRALinear.from_linear(linear)


def test_mlx_lora_linear_call():
    """Test MLXLoRALinear call."""
    import gemma_4_sql.backends.mlx.peft as peft_module

    peft_module.nn = MagicMock()
    peft_module.mx = MagicMock()

    layer = peft_module.MLXLoRALinear(10, 20)

    x = MagicMock()
    w_t = MagicMock()
    layer.weight.T = w_t
    base = MagicMock()
    x.__matmul__.return_value = base

    layer(x)

    layer.bias = MagicMock()
    layer(x)


def test_mlx_lora_linear_save_load(tmp_path):
    """Test save/load adapters."""
    import gemma_4_sql.backends.mlx.peft as peft_module

    peft_module.nn = MagicMock()
    peft_module.mx = MagicMock()

    layer = peft_module.MLXLoRALinear(10, 20)

    path = tmp_path / "adapters.safetensors"
    layer.save_adapters(path)
    peft_module.mx.save_safetensors.assert_called_once()

    peft_module.mx.load.return_value = {"lora_a": "a", "lora_b": "b"}
    layer.load_adapters(path)
    assert layer.lora_a == "a"
    assert layer.lora_b == "b"

    peft_module.mx.load.return_value = {"lora_a": "a"}
    with pytest.raises(KeyError):
        layer.load_adapters(path)

    peft_module.mx = None
    with pytest.raises(DependencyMissingError):
        layer.save_adapters(path)
    with pytest.raises(DependencyMissingError):
        layer.load_adapters(path)


def test_inject_lora():
    """Test inject_lora."""
    import gemma_4_sql.backends.mlx.peft as peft_module

    peft_module.nn = MagicMock()
    peft_module.mx = MagicMock()

    class MockLinear:
        """Docstring for MockLinear."""

    peft_module.nn.Linear = MockLinear

    model = MagicMock()
    # Mock model WITHOUT freeze
    del model.freeze

    # 1: matches, 2: not matching, 3: not linear, 4: list part
    model.named_modules.return_value = [("l1.q_proj", MockLinear()), ("q_proj", MockLinear()), ("other", MockLinear()), ("not_lin", MagicMock())]

    with patch.object(peft_module, "MLXLoRALinear") as mock_lora:
        mock_lora.from_linear.return_value = MagicMock()
        model_out, count = peft_module.inject_lora(model, ["q_proj"])
        assert count == 2

        # list index part
        model2 = [MockLinear()]
        model2[0].named_modules = MagicMock(return_value=[("0.q_proj", MockLinear()), ("0.1", MockLinear())])

        # We need model2 to be a list containing an object with named_modules that yields paths starting with 0.
        # Wait, if name is '0.1', parts is ['0', '1'].
        # curr is model2 (a list).
        # part '0' is digit and curr is list -> curr = curr[0].
        # next, part '1' is last. last='1'. isdigit and curr (now model2[0]) is NOT list -> setattr.
        # Wait, to hit 324 `curr[int(last)] = lora_module`, `last` must be digit and `curr` must be list.
        # So name = '1', curr = [0, MockLinear()].
        class ListModel(list):
            """Docstring for ListModel."""

            def named_modules(self):
                """Docstring for named_modules."""
                return [("0.q_proj", MockLinear()), ("1", MockLinear())]

        model_list = ListModel([MagicMock(), MockLinear()])
        model_out, count = peft_module.inject_lora(model_list, ["q_proj", "1"])
        assert count == 2

    peft_module.nn = None
    with pytest.raises(DependencyMissingError):
        peft_module.inject_lora(model, ["q_proj"])


def test_save_load_adapter_weights(tmp_path):
    """Test save/load adapter weights."""
    import gemma_4_sql.backends.mlx.peft as peft_module

    peft_module.nn = MagicMock()
    peft_module.mx = MagicMock()

    model = MagicMock()

    with patch("builtins.__import__") as mock_import:
        mock_tree = MagicMock()
        mock_tree.tree_flatten.return_value = [("a", 1)]
        mock_import.return_value = mock_tree
        peft_module.save_adapter_weights(model, tmp_path)
        peft_module.save_adapter_weights(model, tmp_path / "test.safetensors")
        peft_module.save_adapter_weights(model, tmp_path / "test.txt")  # hits path.suffix != .safetensors

    # Create mock file
    (tmp_path / "test.safetensors").touch()

    peft_module.load_adapter_weights(model, tmp_path / "test.safetensors")
    model.load_weights.assert_called_once()

    # Test without load_weights attr
    del model.load_weights
    peft_module.load_adapter_weights(model, tmp_path / "test.safetensors")

    with pytest.raises(FileNotFoundError):
        peft_module.load_adapter_weights(model, tmp_path / "not_found.safetensors")

    peft_module.mx = None
    with pytest.raises(DependencyMissingError):
        peft_module.save_adapter_weights(model, tmp_path)
    with pytest.raises(DependencyMissingError):
        peft_module.load_adapter_weights(model, tmp_path / "test.safetensors")


def test_apply_lora():
    """Test apply_lora."""
    import gemma_4_sql.backends.mlx.peft as peft_module

    peft_module.nn = MagicMock()
    peft_module.mx = MagicMock()
    peft_module.load = MagicMock(return_value=(MagicMock(), None))

    with patch.object(peft_module, "inject_lora") as mock_inject:
        mock_inject.return_value = (MagicMock(), 1)
        with patch.object(peft_module, "save_adapter_weights"):
            res = peft_module.apply_lora("model", ["q_proj"], output_dir="dir")
            assert res["status"] == "completed"

            res2 = peft_module.apply_lora("model", ["q_proj"], model="model_obj")
            assert res2["status"] == "completed"

            mock_inject.side_effect = RuntimeError("error")
            res3 = peft_module.apply_lora("model", ["q_proj"])
            assert "failed: error" in res3["status"]

    peft_module.nn = None
    with pytest.raises(DependencyMissingError):
        peft_module.apply_lora("model", ["q_proj"])


def test_inject_lora_coverage():
    """Test inject_lora missing branches."""
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.mlx.peft as peft_module

    peft_module.nn = MagicMock()
    peft_module.mx = MagicMock()

    model = MagicMock()
    model.freeze = MagicMock()

    # Empty target_modules
    out, count = peft_module.inject_lora(model, [])
    assert count == 0
    model.freeze.assert_called_once()

    # Missing named_modules
    model2 = MagicMock()
    del model2.named_modules
    out2, count2 = peft_module.inject_lora(model2, ["q"])
    assert count2 == 0


def test_mlx_lm_import_success_exec():
    """Docstring for test_mlx_lm_import_success_exec."""
    import gemma_4_sql.backends.mlx.peft as q

    with open(q.__file__) as f:
        code = f.read()

    import builtins

    orig_import = builtins.__import__
    from unittest.mock import MagicMock

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "mlx_lm":
            mock_mlxlm = MagicMock()
            mock_mlxlm.load = "MockLoad"
            return mock_mlxlm
        return orig_import(name, *args, **kwargs)

    namespace = {"__name__": "mock_peft", "__builtins__": dict(builtins.__dict__)}
    namespace["__builtins__"]["__import__"] = mock_import

    exec(code, namespace)  # noqa: S102

    assert namespace.get("load") == "MockLoad"


def test_mlx_lm_import_success_real(monkeypatch):
    """Docstring for test_mlx_lm_import_success_real."""
    import sys
    from unittest.mock import MagicMock

    mock_mlxlm = MagicMock()
    mock_mlxlm.load = "MockLoad"

    with patch.dict(sys.modules, {"mlx_lm": mock_mlxlm}):
        if "gemma_4_sql.backends.mlx.peft" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.peft"]

        import gemma_4_sql.backends.mlx.peft as peft_module

        assert peft_module.load == "MockLoad"
