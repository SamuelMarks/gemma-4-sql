"""Tests for test mlx coverage module."""

import importlib

import pytest

from gemma_4_sql.type_hints import DPOConfig, TrainingConfig


def test_mlx_export_fail(monkeypatch):
    """Test mlx export fail functionality."""
    import gemma_4_sql.backends.mlx.export as mexp

    def mock_load(n):
        """Execute mock load helper."""
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        """Execute mock load helper."""
        return (None, None)

    def mock_import(name, *a, **k):
        """Execute mock import helper."""
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    monkeypatch.setattr(mexp, "mx", type("MX", (), {}))
    with pytest.raises(ValueError):
        mexp.export_model("model", "path")


def test_mlx_inference_fail(monkeypatch):
    """Test mlx inference fail functionality."""
    import gemma_4_sql.backends.mlx.inference as minf

    def mock_load_err(n):
        """Execute mock load err helper."""
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load_err, "generate": lambda *a, **k: "a"}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        """Execute mock import helper."""
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load_err})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    monkeypatch.setattr(minf, "load", mock_load_err)
    res = minf.generate_sql("model", "prompt")
    assert "failed" in str(res) or isinstance(res, tuple)


def test_mlx_inference_test_mode(monkeypatch):
    """Test mlx inference test mode functionality."""
    import gemma_4_sql.backends.mlx.inference as minf
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(minf, "load", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing"):
        minf.generate_sql("model", "prompt")

    monkeypatch.setattr(minf, "load", lambda n: (None, None))
    monkeypatch.setattr(minf, "generate", lambda *a, **k: "")
    res = minf.generate_sql("model", "prompt", test_mode=True)
    assert res["status"] == "success"


def test_mlx_peft_fail(monkeypatch):
    """Test mlx peft fail functionality."""
    import gemma_4_sql.backends.mlx.peft as mpeft

    def mock_load_err(n):
        """Execute mock load err helper."""
        raise ValueError("err")

    monkeypatch.setattr(mpeft, "load", mock_load_err)
    res = mpeft.apply_lora("m", [], lora_r=8, lora_alpha=16, lora_dropout=0.1)
    assert "failed" in str(res) or isinstance(res, tuple)


def test_mlx_quantize_mock(monkeypatch):
    """Test mlx quantize mock functionality."""
    import gemma_4_sql.backends.mlx.quantize as mquant

    monkeypatch.setattr(mquant, "quantize_model_wrapper", lambda **k: k["apply_fn"]())

    def mock_load(n):
        """Execute mock load helper."""
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        """Execute mock load helper."""
        return (None, None)

    def mock_import(name, *a, **k):
        """Execute mock import helper."""
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    res = mquant.quantize_model("m")
    assert isinstance(res, tuple)

    # Test BitsAndBytesConfig not None branch
    from unittest.mock import MagicMock

    monkeypatch.setattr(builtins, "__import__", orig_import)
    monkeypatch.setattr(mquant, "BitsAndBytesConfig", MagicMock())
    monkeypatch.setattr(mquant, "apply_bits_and_bytes_quantization", lambda method, cfg, dt: (0.75, "bnb_quantized"))
    res_bnb = mquant.quantize_model("m")
    assert res_bnb == (0.75, "bnb_quantized")

    # Test nn without quantize and BitsAndBytesConfig is None
    import sys

    monkeypatch.setitem(sys.modules, "mlx.nn", type("NN", (), {})())
    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda n: (type("M", (), {})(), None)}))
    monkeypatch.setattr(mquant, "BitsAndBytesConfig", None)
    res_noquant = mquant.quantize_model("m", "int8")
    assert res_noquant == (0.5, "quantized_int8")


def test_mlx_train_fail(monkeypatch):
    """Test mlx train fail functionality."""
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "_execute_train", lambda a, b, c, d: ("failed", 1.0))
    res = mtrain.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1))
    assert res["status"] == "failed"


def test_mlx_train_exception(monkeypatch):
    """Test mlx train exception functionality."""
    import gemma_4_sql.backends.mlx.train as mtrain

    def mock_exec(*a, **k):
        """Execute mock exec helper."""
        raise ValueError("err")

    monkeypatch.setattr(mtrain, "_execute_train", mock_exec)
    res = mtrain.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1))
    assert "failed" in str(res) or isinstance(res, tuple)


def test_mlx_train_inner_fail(monkeypatch):
    """Test mlx train inner fail functionality."""
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "build_dataloader", lambda c: {"loader": None})

    def mock_load_err(n):
        """Execute mock load err helper."""
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load_err}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        """Execute mock import helper."""
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load_err})
        if name == "mlx.optimizers":
            return type("Opt", (), {"AdamW": lambda **kw: None})
        if name == "mlx":
            return type("MLX", (), {"nn": type("NN", (), {"losses": type("L", (), {"cross_entropy": lambda *args: None})})})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    monkeypatch.setattr(mtrain, "load", mock_load_err)

    with pytest.raises(ValueError):
        mtrain._execute_train("m", "d", 1, 0.1)


def test_mlx_train_inner_success(monkeypatch):
    """Test mlx train inner success functionality."""
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "build_dataloader", lambda c: {"loader": [1]})
    monkeypatch.setattr(mtrain, "_run_training_epochs", lambda s: 1.0)

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda n: (None, None)}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        """Execute mock load helper."""
        return (None, None)

    def mock_import(name, *a, **k):
        """Execute mock import helper."""
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load})
        if name == "mlx.optimizers":
            return type("Opt", (), {"AdamW": lambda **kw: None})
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load})
        if name == "mlx":
            return type("MLX", (), {"nn": type("NN", (), {"losses": type("L", (), {"cross_entropy": lambda *args: None})})})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    monkeypatch.setattr(mtrain, "load", lambda n: (None, None))
    monkeypatch.setattr(mtrain, "optim", type("Opt", (), {"AdamW": lambda **k: None}))
    monkeypatch.setattr(mtrain, "nn", type("NN", (), {"value_and_grad": lambda *a, **k: lambda *a, **k: (None, None)}))
    res = mtrain._execute_train("m", "d", 1, 0.1)
    assert res[0] == "completed"


def test_mlx_dpo_fail(monkeypatch):
    """Test mlx dpo fail functionality."""
    import builtins

    import gemma_4_sql.backends.mlx.dpo as mdpo

    orig_import = builtins.__import__

    def mock_load(n):
        """Execute mock load helper."""
        return (None, None)

    def mock_import(name, *a, **k):
        """Execute mock import helper."""
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load})
        if name == "mlx.optimizers":
            return type("Opt", (), {"AdamW": lambda **kw: None})
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    monkeypatch.setattr(mdpo, "build_dataloader", lambda c: {"loader": None})
    from gemma_4_sql.exceptions import DependencyMissingError

    try:
        res = mdpo.run_dpo(DPOConfig(model_name="m", dataset="d"))
        assert "failed" in str(res) or isinstance(res, tuple)
    except DependencyMissingError:
        pass


def test_mlx_dpo_missing_import(monkeypatch):
    """Test mlx dpo missing import functionality."""
    import sys

    monkeypatch.setitem(sys.modules, "mlx.nn", type("NN", (), {"functional": None}))
    import gemma_4_sql.backends.mlx.dpo as mdpo

    importlib.reload(mdpo)


def test_mlx_peft_full(monkeypatch):
    """Test mlx peft full functionality."""
    import gemma_4_sql.backends.mlx.peft as mpeft

    monkeypatch.setattr(mpeft, "nn", type("NN", (), {}))

    class MockModel:
        """Test class for MockModel."""

        def parameters(self):
            """Execute parameters helper."""
            return {"a": 1}

    def mock_load(n):
        """Execute mock load helper."""
        return (MockModel(), None)

    monkeypatch.setattr(mpeft, "load", mock_load)
    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load}))
    monkeypatch.setitem(sys.modules, "mlx.utils", type("MLXU", (), {"tree_map": lambda f, p: f(p)}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        """Execute mock import helper."""
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        if name == "mlx.utils":
            return sys.modules["mlx.utils"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    res = mpeft.apply_lora("m", [], lora_r=8, lora_alpha=16, lora_dropout=0.1)
    assert res["status"] == "completed"


def test_mlx_quantize_full(monkeypatch):
    """Test mlx quantize full functionality."""
    import gemma_4_sql.backends.mlx.quantize as mquant

    monkeypatch.setattr(mquant, "quantize_model_wrapper", lambda **k: k["apply_fn"]())
    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda n: (None, None)}))
    monkeypatch.setitem(sys.modules, "transformers", type("Transformers", (), {"BitsAndBytesConfig": lambda **k: None, "AutoModelForCausalLM": type("Auto", (), {"from_pretrained": lambda *a, **k: None})}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        """Execute mock load helper."""
        return (None, None)

    def mock_import(name, *a, **k):
        """Execute mock import helper."""
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        if name == "transformers":
            return sys.modules["transformers"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    res = mquant.quantize_model("m")
    assert "quantized" in str(res)


def test_mlx_inference_full(monkeypatch):
    """Test mlx inference full functionality."""
    import sys

    import gemma_4_sql.backends.mlx.inference as minf

    monkeypatch.setattr(minf, "load", lambda n: (None, None))
    monkeypatch.setattr(minf, "generate", lambda *a, **k: "select 1")
    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda n: (None, None), "generate": lambda *a, **k: "select 1"}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        """Execute mock load helper."""
        return (None, None)

    def mock_import(name, *a, **k):
        """Execute mock import helper."""
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    res = minf.generate_sql("model", "prompt")
    assert res["status"] == "success"


def test_mlx_train_full_exec(monkeypatch):
    """Test mlx train full exec functionality."""
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "nn", type("NN", (), {}))
    monkeypatch.setattr(mtrain, "optim", type("Optim", (), {}))
    monkeypatch.setattr(mtrain, "load", lambda n: None)
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "_execute_train", lambda a, b, c, d: ("completed", 1.0))
    res = mtrain.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1))
    assert res["status"] == "completed"

    monkeypatch.setattr(mtrain, "_execute_train", lambda a, b, c, d: ("completed", 1.0))

    def mock_fail(*a, **k):
        """Execute mock fail helper."""
        raise ValueError("err")

    monkeypatch.setattr(mtrain, "_execute_train", mock_fail)
    res_fail = mtrain.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1))
    assert "failed" in res_fail["status"]


def test_mlx_peft_error(monkeypatch):
    """Test mlx peft error functionality."""
    import gemma_4_sql.backends.mlx.peft as mpeft

    def mock_load(n):
        """Execute mock load helper."""
        raise ValueError("err")

    monkeypatch.setattr(mpeft, "load", mock_load)

    res = mpeft.apply_lora("m", [], lora_r=8, lora_alpha=16, lora_dropout=0.1)
    assert "failed" in str(res) or isinstance(res, tuple)


def test_mlx_train_missing_import(monkeypatch):
    """Test mlx train missing import functionality."""
    import gemma_4_sql.backends.mlx.train as mtrain
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(mtrain, "mx", None)
    with pytest.raises(DependencyMissingError):
        mtrain.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1))


def test_mlx_quantize_error(monkeypatch):
    """Test mlx quantize error functionality."""
    import gemma_4_sql.backends.mlx.quantize as mquant

    monkeypatch.setattr(mquant, "quantize_model_wrapper", lambda **k: k["apply_fn"]())

    def mock_load(n):
        """Execute mock load helper."""
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        """Execute mock load helper."""
        return (None, None)

    def mock_import(name, *a, **k):
        """Execute mock import helper."""
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load})
        if name == "transformers":
            return type("Transformers", (), {"BitsAndBytesConfig": lambda **k: None, "AutoModelForCausalLM": type("Auto", (), {"from_pretrained": lambda *a, **k: None})})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    res = mquant.quantize_model("m")
    assert "failed" in str(res) or isinstance(res, tuple)


def test_mlx_train_missing(monkeypatch):
    """Test mlx train missing functionality."""
    import gemma_4_sql.backends.mlx.train as mtrain
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(mtrain, "mx", None)
    with pytest.raises(DependencyMissingError):
        mtrain.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1))
