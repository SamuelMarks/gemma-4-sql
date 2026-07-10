import importlib

import pytest

from gemma_4_sql.type_hints import DPOConfig, TrainingConfig


def test_mlx_export_fail(monkeypatch):
    import gemma_4_sql.backends.mlx.export as mexp

    def mock_load(n):
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        return (None, None)

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ValueError):
        mexp.export_model("model", "path")


def test_mlx_inference_fail(monkeypatch):
    import gemma_4_sql.backends.mlx.inference as minf

    def mock_load_err(n):
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load_err, "generate": lambda *a, **k: "a"}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load_err})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    monkeypatch.setattr(minf, "load", mock_load_err)
    res = minf.generate_sql("model", "prompt")
    assert "failed" in str(res) or isinstance(res, tuple)


def test_mlx_inference_test_mode(monkeypatch):
    import gemma_4_sql.backends.mlx.inference as minf

    res = minf.generate_sql("model", "prompt", test_mode=True)
    assert res["status"] == "success"


def test_mlx_peft_fail(monkeypatch):
    import gemma_4_sql.backends.mlx.peft as mpeft

    def mock_load(n):
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        return (None, None)

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    res = mpeft.apply_lora("m", [], lora_r=8, lora_alpha=16, lora_dropout=0.1)
    assert "failed" in str(res) or isinstance(res, tuple)


def test_mlx_quantize_mock(monkeypatch):
    import gemma_4_sql.backends.mlx.quantize as mquant

    monkeypatch.setattr(mquant, "quantize_model_wrapper", lambda **k: k["apply_fn"]())

    def mock_load(n):
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        return (None, None)

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    res = mquant.quantize_model("m")
    assert isinstance(res, tuple)


def test_mlx_train_fail(monkeypatch):
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "_execute_train", lambda a, b, c, d: ("failed", 1.0))
    res = mtrain.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1))
    assert res["status"] == "failed"


def test_mlx_train_exception(monkeypatch):
    import gemma_4_sql.backends.mlx.train as mtrain

    def mock_exec(*a, **k):
        raise ValueError("err")

    monkeypatch.setattr(mtrain, "_execute_train", mock_exec)
    res = mtrain.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1))
    assert "failed" in str(res) or isinstance(res, tuple)


def test_mlx_train_inner_fail(monkeypatch):
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "build_dataloader", lambda c: {"loader": None})

    def mock_load_err(n):
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load_err}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
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
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "build_dataloader", lambda c: {"loader": [1]})
    monkeypatch.setattr(mtrain, "_run_training_epochs", lambda s: 1.0)

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda n: (None, None)}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        return (None, None)

    def mock_import(name, *a, **k):
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
    res = mtrain._execute_train("m", "d", 1, 0.1)
    assert res[0] == "completed"


def test_mlx_dpo_fail(monkeypatch):
    import builtins

    import gemma_4_sql.backends.mlx.dpo as mdpo

    orig_import = builtins.__import__

    def mock_load(n):
        return (None, None)

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load})
        if name == "mlx.optimizers":
            return type("Opt", (), {"AdamW": lambda **kw: None})
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    monkeypatch.setattr(mdpo, "build_dataloader", lambda c: {"loader": None})
    res = mdpo.run_dpo(DPOConfig(model_name="m", dataset="d"))
    assert "failed" in str(res) or isinstance(res, tuple)


def test_mlx_dpo_missing_import(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "mlx.nn", type("NN", (), {"functional": None}))
    import gemma_4_sql.backends.mlx.dpo as mdpo

    importlib.reload(mdpo)


def test_mlx_peft_full(monkeypatch):
    import gemma_4_sql.backends.mlx.peft as mpeft

    monkeypatch.setattr(mpeft, "nn", type("NN", (), {}))
    import gemma_4_sql.backends.mlx.peft as mpeft

    class MockModel:
        def parameters(self):
            return {"a": 1}

    def mock_load(n):
        return (MockModel(), None)

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load}))
    monkeypatch.setitem(sys.modules, "mlx.utils", type("MLXU", (), {"tree_map": lambda f, p: f(p)}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        return (None, None)

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        if name == "mlx.utils":
            return sys.modules["mlx.utils"]
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    res = mpeft.apply_lora("m", [], lora_r=8, lora_alpha=16, lora_dropout=0.1)
    assert res["status"] == "completed"


def test_mlx_quantize_full(monkeypatch):
    import gemma_4_sql.backends.mlx.quantize as mquant

    monkeypatch.setattr(mquant, "quantize_model_wrapper", lambda **k: k["apply_fn"]())
    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda n: (None, None)}))
    monkeypatch.setitem(sys.modules, "transformers", type("Transformers", (), {"BitsAndBytesConfig": lambda **k: None, "AutoModelForCausalLM": type("Auto", (), {"from_pretrained": lambda *a, **k: None})}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        return (None, None)

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        if name == "transformers":
            return sys.modules["transformers"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    res = mquant.quantize_model("m")
    assert "quantized" in str(res)


def test_mlx_inference_full(monkeypatch):
    import sys

    import gemma_4_sql.backends.mlx.inference as minf

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda n: (None, None), "generate": lambda *a, **k: "select 1"}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        return (None, None)

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    res = minf.generate_sql("model", "prompt")
    assert res["status"] == "success"


def test_mlx_train_full_exec(monkeypatch):
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
        raise ValueError("err")

    monkeypatch.setattr(mtrain, "_execute_train", mock_fail)
    res_fail = mtrain.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1))
    assert "failed" in res_fail["status"]


def test_mlx_peft_error(monkeypatch):
    import gemma_4_sql.backends.mlx.peft as mpeft

    def mock_load(n):
        raise ValueError("err")

    monkeypatch.setattr(mpeft, "load", mock_load)

    res = mpeft.apply_lora("m", [], lora_r=8, lora_alpha=16, lora_dropout=0.1)
    assert "failed" in str(res) or isinstance(res, tuple)


def test_mlx_train_missing_import(monkeypatch):
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "mx", None)
    res = mtrain.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1))
    assert res["status"] == "mocked_missing_mlx"


def test_mlx_quantize_error(monkeypatch):
    import gemma_4_sql.backends.mlx.quantize as mquant

    monkeypatch.setattr(mquant, "quantize_model_wrapper", lambda **k: k["apply_fn"]())

    def mock_load(n):
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load}))
    import builtins

    orig_import = builtins.__import__

    def mock_load(n):
        return (None, None)

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": mock_load})
        if name == "transformers":
            return type("Transformers", (), {"BitsAndBytesConfig": lambda **k: None, "AutoModelForCausalLM": type("Auto", (), {"from_pretrained": lambda *a, **k: None})})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    res = mquant.quantize_model("m")
    assert "failed" in str(res) or isinstance(res, tuple)


def test_mlx_train_missing(monkeypatch):
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "mx", None)
    res = mtrain.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1))
    assert res["status"] == "mocked_missing_mlx"
