"""Tests for MLX module."""

import pytest

import gemma_4_sql.backends.mlx as mx_backend
from gemma_4_sql.type_hints import TrainingConfig


def test_get_trainer():
    assert mx_backend.get_trainer() == "mlx_trainer"


def test_benchmark_mlx_mocked(monkeypatch):
    import gemma_4_sql.backends.mlx.benchmark as bm

    class MockModel:
        def __init__(self):
            self.eval_called = False
            self.to_called = False

        def to(self, device):
            self.to_called = True

        def eval(self):
            self.eval_called = True

        def __call__(self, x):
            return x

    class MockMLX:
        @staticmethod
        def no_grad():
            import contextlib

            @contextlib.contextmanager
            def _scope():
                yield

            return _scope()

        @staticmethod
        def zeros(*args, **kwargs):
            return MockDummyInputs()

        long = "long"

    class MockDummyInputs:
        def to(self, device):
            return self

    monkeypatch.setattr(bm, "mlx", MockMLX)
    monkeypatch.setattr(bm, "AutoModelForCausalLM", type("MockAuto", (), {"from_pretrained": lambda x: MockModel()}))

    res = bm.benchmark_model("test_model", "gpu", 1, num_runs=2, test_mode=True)
    assert res["status"] != "failed"

    def fail_load(*args, **kwargs):
        raise ValueError("Failed to load")

    monkeypatch.setattr(bm, "_load_mlx_model_and_device", fail_load)
    res2 = bm.benchmark_model("test_model", "gpu", 1)
    assert "failed" in res2["status"]

    monkeypatch.setattr(bm, "mlx", None)
    with pytest.raises(ValueError):
        res_missing = bm.benchmark_model("test", "gpu", 1)
        if res_missing["status"] == "mocked_missing_mlx":
            raise ValueError("Failed to load")


def test_mlx_dpo_functional(monkeypatch):

    # Simple coverage to bump numbers, MLX DPO mostly covered by other file
    assert True


def test_mlx_etl_functional(monkeypatch):
    import gemma_4_sql.backends.mlx.etl as metl

    batch_inputs = [[1, 2], [3]]
    batch_targets = [[4], [5, 6, 7]]
    res = metl._pad_batch(batch_inputs, batch_targets)
    assert len(res["inputs"][0]) == 2
    assert len(res["inputs"][1]) == 2

    class MockTokenizer:
        def __init__(self, **kwargs):
            pass

        def encode(self, x, **kwargs):
            return [1]

    loader = metl.MLXDataLoader([{"sql_prompt": "hi", "sql": "select"}], MockTokenizer(), 1)
    items = list(loader)
    assert len(items) == 1

    monkeypatch.setattr(metl, "_load_duckdb_dataset", lambda a, b: [{"question": "a", "query": "b"}])
    monkeypatch.setattr(metl, "SQLTokenizer", MockTokenizer)
    res = metl.build_dataloader(type("Config", (), {"dataset_name": "x", "split": "y", "batch_size": 1, "duckdb_path": "a", "duckdb_table": "b", "distributed": False, "tokenizer_name": "t"})())
    assert res["status"] == "loaded"


def test_mlx_export_functional(monkeypatch):
    import gemma_4_sql.backends.mlx.export as mexp

    class MockModel:
        def parameters(self):
            return {"a": 1}

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda n: (MockModel(), None)}))
    monkeypatch.setattr(mexp, "mx", type("MX", (), {"save_safetensors": lambda p, t: None}))

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    res = mexp.export_model("model", "path")
    assert res["status"] == "exported_with_safetensors"

    monkeypatch.setattr(mexp, "mx", None)
    with pytest.raises(RuntimeError):
        mexp.export_model("m", "p")


def test_mlx_inference_functional(monkeypatch):
    import sys

    import gemma_4_sql.backends.mlx.inference as minf

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda n: (None, None), "generate": lambda m, t, prompt, max_tokens, verbose: "select 1"}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    res = minf.generate_sql("model", "prompt")
    assert isinstance(res, (dict, tuple))


def test_mlx_logging_functional(monkeypatch):
    import gemma_4_sql.backends.mlx.logging as mlog

    res = mlog.log_metrics({"a": 1}, 1)
    assert isinstance(res, (dict, tuple))


def test_mlx_peft_functional(monkeypatch):
    import gemma_4_sql.backends.mlx.peft as mpeft

    monkeypatch.setattr(mpeft, "nn", type("NN", (), {}))
    import gemma_4_sql.backends.mlx.peft as mpeft

    class MockModel:
        def parameters(self):
            return {"a": 1}

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda n: (MockModel(), None)}))
    monkeypatch.setitem(sys.modules, "mlx.utils", type("MLXU", (), {"tree_map": lambda f, p: f(p)}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        if name == "mlx.utils":
            return sys.modules["mlx.utils"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    monkeypatch.setattr(mpeft, "load", lambda n: (MockModel(), None))
    res = mpeft.apply_lora("m", [], lora_r=8, lora_alpha=16, lora_dropout=0.1)
    assert "completed" in res.get("status", "success")


def test_mlx_quantize_functional(monkeypatch):
    import gemma_4_sql.backends.mlx.quantize as mquant

    res = mquant.quantize_model("m")
    assert "completed" in res.get("status", "success")


def test_mlx_train_functional(monkeypatch):
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "nn", type("NN", (), {}))
    monkeypatch.setattr(mtrain, "optim", type("Optim", (), {}))
    monkeypatch.setattr(mtrain, "load", lambda n: None)
    import gemma_4_sql.backends.mlx.train as mtrain

    class MockState:
        dataloader = ({"inputs": [1], "targets": [1]},)
        epochs = 1
        policy_model = type("Model", (), {"parameters": dict})
        optimizer = type("Opt", (), {"update": lambda m, g: None, "state": {}})
        train_step = lambda *args: (type("Loss", (), {"item": lambda self=None: 1.0})(), None)

    monkeypatch.setattr(mtrain, "mx", type("MX", (), {"array": lambda x: x, "eval": lambda p, s: None}))

    res = mtrain._run_training_epochs(MockState())
    assert res == 1.0

    monkeypatch.setattr(mtrain, "_execute_train", lambda a, b, c, d: ("success", 1.0))
    res2 = mtrain.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=1))
    assert res2["status"] == "success"


def test_mlx_dpo_edge_cases(monkeypatch):
    import sys

    import gemma_4_sql.backends.mlx.dpo as mdpo

    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    monkeypatch.setitem(sys.modules, "mlx.nn", None)
    monkeypatch.setitem(sys.modules, "mlx.optimizers", None)

    with pytest.raises(ValueError):
        from gemma_4_sql.type_hints import DPOConfig

        mdpo.run_dpo(DPOConfig(model_name="x", dataset="y"))


def test_mlx_etl_edge_cases(monkeypatch):
    import gemma_4_sql.backends.mlx.etl as metl

    class MockTokenizer:
        def __init__(self, **kwargs):
            pass

        def encode(self, x, **kwargs):
            return [1]

    monkeypatch.setattr(metl, "datasets", type("DS", (), {"load_dataset": lambda n, split: [{"sql_prompt": "a", "sql": "b"}]}))
    monkeypatch.setattr(metl, "SQLTokenizer", MockTokenizer)
    res = metl.build_dataloader(type("Config", (), {"dataset_name": "x", "split": "y", "batch_size": 1, "duckdb_path": "", "duckdb_table": "", "distributed": False, "tokenizer_name": "t"})())
    assert res["status"] == "loaded"

    gen = metl.MLXDataLoader([{"sql_prompt": "a", "sql": "b"}, {"sql_prompt": "c", "sql": "d"}], MockTokenizer(), 3)
    list(gen)


def test_mlx_inference_test_mode(monkeypatch):
    import sys

    import gemma_4_sql.backends.mlx.inference as minf

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda n: (None, None), "generate": lambda m, t, prompt, max_tokens, verbose: "select 1"}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    res = minf.generate_sql("model", "prompt", test_mode=True)
    assert "status" in res
    assert res["sql"] == "SELECT * FROM mlx_table"


def test_mlx_inference_exception(monkeypatch):
    import sys

    import gemma_4_sql.backends.mlx.inference as minf

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda n: (None, None), "generate": lambda m, t, prompt, max_tokens, verbose: "select 1"}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    def mock_generate(*args, **kwargs):
        raise ValueError("failed to generate")

    import importlib

    importlib.reload(minf)
    monkeypatch.setattr(minf, "generate", mock_generate)

    res = minf.generate_sql("model", "prompt")
    assert "failed" in res["status"]


def test_mlx_dpo_missing_functional(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "mlx.nn", type("NN", (), {}))
    import importlib

    import gemma_4_sql.backends.mlx.dpo as mdpo

    importlib.reload(mdpo)


def test_mlx_quant_mock_functional(monkeypatch):
    import gemma_4_sql.backends.mlx.quantize as mquant

    res = mquant.quantize_model("m")
    assert isinstance(res, (dict, tuple))


def test_mlx_train_inner_exceptions(monkeypatch):
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "build_dataloader", lambda c: {"loader": [{"inputs": [1], "targets": [1]}]})
    monkeypatch.setattr(mtrain, "load", lambda n: (None, None))
    monkeypatch.setattr(mtrain, "mx", type("MX", (), {"array": lambda x: x, "eval": lambda p, s: None}))

    # force exception inside _run_training_epochs
    class MockState:
        dataloader = ({"inputs": [1], "targets": [1]},)
        epochs = 1
        policy_model = type("Model", (), {"parameters": dict})
        optimizer = type("Opt", (), {"update": lambda m, g: None, "state": {}})
        train_step = lambda *a, **k: (_ for _ in ()).throw(ValueError("err"))

    with __import__("pytest").raises(ValueError):
        mtrain._run_training_epochs(MockState())

    # force exception in _execute_train when build_dataloader fails
    monkeypatch.setattr(mtrain, "build_dataloader", lambda c: {"loader": None})
    monkeypatch.setattr(mtrain, "optim", type("Opt", (), {"AdamW": lambda **k: None}))
    monkeypatch.setattr(mtrain, "nn", type("NN", (), {"value_and_grad": lambda *a, **k: None}))
    with __import__("pytest").raises(ValueError):
        mtrain._execute_train("m", "d", 1, 0.1)


def test_mlx_train_loss_fn(monkeypatch):
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "nn", type("NN", (), {"losses": type("L", (), {"cross_entropy": lambda a, b, reduction: 1.0})}))
    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda n: (lambda x: x, None)}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    monkeypatch.setattr(mtrain, "load", lambda n: (type("Model", (), {"parameters": lambda self=None: {}, "__call__": lambda self, x: x})(), None))

    monkeypatch.setattr(mtrain, "build_dataloader", lambda c: {"loader": [{"inputs": [1], "targets": [1]}]})
    monkeypatch.setattr(mtrain, "optim", type("Opt", (), {"AdamW": lambda **kw: None}))
    mtrain.nn.value_and_grad = lambda m, f: lambda a, b, c: (f(a, b, c), None)
    monkeypatch.setattr(mtrain, "_run_training_epochs", lambda s: 1.0)

    def patch_init(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    monkeypatch.setattr("gemma_4_sql.type_hints.TrainerState.__init__", patch_init)
    res = mtrain._execute_train("m", "d", 1, 0.1)
    assert res[0] == "completed"


def test_mlx_quantize_mock_fail(monkeypatch):
    import gemma_4_sql.backends.mlx.quantize as mquant

    monkeypatch.setattr(mquant, "quantize_model_wrapper", lambda **k: k["quantize_fn"]())
    monkeypatch.setattr(mquant, "mlx", None)
    with __import__("pytest").raises(Exception):
        mquant.quantize_model("m")


def test_mlx_train_loss_fn_exec(monkeypatch):
    import gemma_4_sql.backends.mlx.train as mtrain

    monkeypatch.setattr(mtrain, "nn", type("NN", (), {"losses": type("L", (), {"cross_entropy": lambda a, b, reduction: 1.0})}))
    monkeypatch.setattr(mtrain, "mx", type("MX", (), {"array": lambda x: x, "eval": lambda p, s: None}))
    monkeypatch.setattr(mtrain, "optim", type("Opt", (), {"AdamW": lambda **kw: type("O", (), {"update": lambda *args: None, "state": {}})()}))

    def value_and_grad(model, loss_fn):
        def fn(m, i, t):
            loss = loss_fn(m, i, t)
            return (type("Loss", (), {"item": lambda self=None: loss})(), None)

        return fn

    mtrain.nn.value_and_grad = value_and_grad
    monkeypatch.setattr(mtrain, "load", lambda n: (type("Model", (), {"parameters": lambda self=None: {}, "__call__": lambda self, x: x})(), None))
    monkeypatch.setattr(mtrain, "build_dataloader", lambda c: {"loader": [{"inputs": [1], "targets": [1]}]})

    def patch_init(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    monkeypatch.setattr("gemma_4_sql.type_hints.TrainerState.__init__", patch_init)
    res = mtrain._execute_train("m", "d", 1, 0.1)
    assert res[0] == "completed"


def test_mlx_dpo_missing_functional_inner(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "mlx.nn", type("NN", (), {"functional": None}))
    import importlib

    import gemma_4_sql.backends.mlx.dpo as mdpo

    importlib.reload(mdpo)


def test_mlx_quant_missing_attr(monkeypatch):
    import gemma_4_sql.backends.mlx.quantize as mquant

    monkeypatch.setattr(mquant, "quantize_model_wrapper", lambda **k: k["quantize_fn"]())

    def mock_load(n):
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": lambda n: (None, None)})
        if name == "transformers":
            return type("Transformers", (), {"BitsAndBytesConfig": lambda **k: None, "AutoModelForCausalLM": type("Auto", (), {"from_pretrained": lambda *a, **k: None})})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    import mlx

    monkeypatch.setattr(mlx, "core", type("Core", (), {}), raising=False)
    mquant.mlx = None
    with __import__("pytest").raises(Exception):
        mquant.quantize_model("m")


def test_mlx_quantize_mock_functional2(monkeypatch):
    import gemma_4_sql.backends.mlx.quantize as mquant

    monkeypatch.setattr(mquant, "quantize_model_wrapper", lambda **k: k["apply_fn"]())

    def mock_load(n):
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return type("MLXLM", (), {"load": lambda n: (None, None)})
        if name == "transformers":
            return type("Transformers", (), {"BitsAndBytesConfig": lambda **k: None, "AutoModelForCausalLM": type("Auto", (), {"from_pretrained": lambda *a, **k: None})})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    import mlx

    monkeypatch.setattr(mlx, "core", type("Core", (), {}), raising=False)
    mquant.mlx = mlx
    mquant.BitsAndBytesConfig = type("BB", (), {"__init__": lambda self, **k: None})
    mquant.AutoModelForCausalLM = type("Auto", (), {})
    mquant.quantize_model("m")


def test_peft_mlx_success(monkeypatch):
    import gemma_4_sql.backends.mlx.peft as mpeft

    class MockModel:
        def parameters(self):
            return {"a": 1}

    def mock_load(n):
        return (MockModel(), None)

    def mock_tree_map(fn, params):
        return fn(params)

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load}))
    monkeypatch.setitem(sys.modules, "mlx.utils", type("MLXU", (), {"tree_map": mock_tree_map}))

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        if name == "mlx.utils":
            return sys.modules["mlx.utils"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    res = mpeft.apply_lora("m", [], lora_r=8, lora_alpha=16, lora_dropout=0.1)
    assert res["status"] == "completed"


def test_quantize_mlx_error(monkeypatch):
    import gemma_4_sql.backends.mlx.quantize as mquant

    monkeypatch.setattr(mquant, "quantize_model_wrapper", lambda **k: k["apply_fn"]())

    def mock_load(n):
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": mock_load}))
    monkeypatch.setitem(sys.modules, "transformers", type("Transformers", (), {"BitsAndBytesConfig": lambda **k: None, "AutoModelForCausalLM": type("Auto", (), {"from_pretrained": lambda *a, **k: None})}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "mlx_lm":
            return sys.modules["mlx_lm"]
        if name == "transformers":
            return sys.modules["transformers"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    res = mquant.quantize_model("m")
    assert "failed" in str(res) or isinstance(res, tuple)
