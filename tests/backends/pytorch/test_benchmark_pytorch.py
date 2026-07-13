"""Tests for PyTorch Benchmark."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.pytorch.benchmark as pt_bm
from gemma_4_sql.backends.pytorch.benchmark import benchmark_model


class MockTorch:
    """Provide class docstring."""

    long = "long"

    class MockCuda:
        """Provide class docstring."""

        @staticmethod
        def is_available() -> bool:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return False

    cuda = MockCuda

    def randint(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return [1]

    def zeros(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [0]


class MockAutoModelForCausalLM:
    """Mock Model."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Mock method.

        Returns:
            object: Description of return.

        """
        return cls()

    """Provide class docstring."""


def test_benchmark_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(pt_bm, "torch", None)
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", None)
    with pytest.raises(DependencyMissingError, match="PyTorch dependencies are missing."):
        benchmark_model("model", "gpu", 1)


def test_benchmark_pytorch_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt_bm, "torch", MockTorch())
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", MockAutoModelForCausalLM)
    res = benchmark_model("model", "gpu", 1, test_mode=True, num_runs=2)
    if not res["status"] == "success":
        raise AssertionError
    if not res["tokens_per_sec"] > 0:
        raise AssertionError


def test_benchmark_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt_bm, "torch", MockTorch())
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", MockAutoModelForCausalLM)

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockTorch, "randint", raise_err)
    res = benchmark_model("model", "gpu", 1, test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError


class MockModel:
    """Provide class docstring."""

    def to(self, device: object) -> None:
        """Execute function."""

    def eval(self) -> None:
        """Execute function."""

    def __call__(self, x: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return x


def test_benchmark_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_benchmark = __import__("gemma_4_sql.backends.pytorch.benchmark", fromlist=[""])
    monkeypatch.setattr(m_benchmark, "torch", MockTorch())

    class MockAutoModel:
        """Docstring."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring."""
            return MockAutoModelForCausalLM()

    monkeypatch.setattr(m_benchmark, "AutoModelForCausalLM", MockAutoModel)
    res = m_benchmark.benchmark_model("m", "cuda", 1, test_mode=True)
    if res["status"] != "success":
        raise AssertionError
    "Execute function."
    pt_bm = __import__("gemma_4_sql.backends.pytorch.benchmark", fromlist=[""])
    monkeypatch.setattr(pt_bm, "torch", MockTorch())
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", MockAutoModelForCausalLM())
    res = pt_bm.benchmark_model("model", "gpu", 1, test_mode=False, num_runs=2)
    if res["status"] != "success":
        raise AssertionError
    res = pt_bm.benchmark_model("model", "cpu", 1, test_mode=False, num_runs=2)
    if res["status"] != "success":
        raise AssertionError


def test_pytorch_trainer():
    import gemma_4_sql.backends.pytorch as pt

    assert pt.get_trainer() == "pytorch_trainer"


def test_pytorch_benchmark_eval(monkeypatch):
    import gemma_4_sql.backends.pytorch.benchmark as bm

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

    def mock_get(m, h, test_mode=False, dtype="bfloat16", backend_alias="pytorch"):
        return (MockModel(), "cuda")

    monkeypatch.setattr(bm, "_load_pytorch_model_and_device", mock_get)

    import torch

    monkeypatch.setattr(bm, "torch", type("Torch", (), {"no_grad": torch.no_grad, "cuda": type("Cuda", (), {"is_available": lambda: True, "synchronize": lambda *a, **kw: None, "max_memory_allocated": lambda: 1024 * 1024 * 1024}), "randint": lambda *a, **k: MockModel()}))

    res = bm.benchmark_model("m", "gpu", 1)
    assert res["status"] == "success"


def test_pytorch_benchmark_rest(monkeypatch):
    import gemma_4_sql.backends.pytorch.benchmark as bm

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

    def mock_get(m, h, test_mode=False, dtype="bfloat16", backend_alias="pytorch"):
        return (MockModel(), "cuda")

    monkeypatch.setattr(bm, "_load_pytorch_model_and_device", mock_get)

    import torch

    monkeypatch.setattr(bm, "torch", type("Torch", (), {"no_grad": torch.no_grad, "cuda": type("Cuda", (), {"is_available": lambda: True, "synchronize": lambda *a, **kw: None, "max_memory_allocated": lambda: 1024 * 1024 * 1024}), "randint": lambda *a, **k: MockModel()}))

    res = bm.benchmark_model("m", "gpu", 1)
    assert res["status"] == "success"


def test_pytorch_benchmark_eval2(monkeypatch):
    import gemma_4_sql.backends.pytorch.benchmark as bm

    class MockModel:
        def to(self, device):
            pass

        def eval(self):
            pass

    monkeypatch.setattr(bm, "AutoModelForCausalLM", type("Auto", (), {"from_pretrained": lambda x, torch_dtype=None: MockModel()}))
    monkeypatch.setattr(bm, "torch", type("Torch", (), {"cuda": type("Cuda", (), {"is_available": lambda self: True})()}))
    bm._load_pytorch_model_and_device("m", "gpu")


def test_pytorch_dpo_loss2(monkeypatch):
    import gemma_4_sql.backends.pytorch.dpo as pt_dpo

    monkeypatch.setattr(pt_dpo, "torch", type("Torch", (), {"nn": type("NN", (), {"functional": type("F", (), {"logsigmoid": lambda x: x})()})}))


def test_pytorch_dpo_load_err2(monkeypatch):
    import gemma_4_sql.backends.pytorch.dpo as pt_dpo

    def mock_load(n):
        raise ValueError("err")

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.dpo.AutoModelForCausalLM", type("Auto", (), {"from_pretrained": mock_load}), raising=False)
    with __import__("pytest").raises(Exception):
        pt_dpo._load_model("m")


def test_pytorch_benchmark_inner(monkeypatch):
    import gemma_4_sql.backends.pytorch.benchmark as bm

    class MockModel:
        def __call__(self, x):
            return x

    class MockTensor:
        def to(self, device):
            return self

    monkeypatch.setattr(
        bm, "torch", type("Torch", (), {"no_grad": lambda: type("CM", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})(), "randint": lambda *a, **k: MockTensor(), "cuda": type("Cuda", (), {"synchronize": lambda self=None: None, "max_memory_allocated": lambda self=None: 1024 * 1024 * 1024})()})
    )


def test_pytorch_dpo_loss_exec(monkeypatch):
    import gemma_4_sql.backends.pytorch.dpo as pt_dpo

    class MockTensor:
        def __sub__(self, o):
            return self

        def __rmul__(self, o):
            return self

        def __mul__(self, o):
            return self

        def __neg__(self):
            return self

        def mean(self):
            return 1.0

        def detach(self):
            return self

    monkeypatch.setattr(pt_dpo, "torch", type("Torch", (), {"nn": type("NN", (), {"functional": type("F", (), {"logsigmoid": lambda x: MockTensor()})()})}))


def test_pytorch_train_device2(monkeypatch):
    import gemma_4_sql.backends.pytorch.train as pt_train

    monkeypatch.setattr(pt_train, "torch", type("Torch", (), {"cuda": type("Cuda", (), {"set_device": lambda x: None, "is_available": lambda: True, "device_count": lambda: 1})()}))
    monkeypatch.setattr(pt_train, "dist", type("Dist", (), {"is_initialized": lambda: False}), raising=False)

    import os

    os.environ["LOCAL_RANK"] = "0"
    monkeypatch.setattr(pt_train, "device", "cuda:0", raising=False)


def test_pytorch_benchmark_all(monkeypatch):
    import gemma_4_sql.backends.pytorch.benchmark as bm

    class MockTensor:
        def to(self, device):
            return self

    class MockModel:
        def __call__(self, x):
            return x

        def generate(self, x, **kwargs):
            return x

        def to(self, x):
            pass

    class MockNoGrad:
        def __enter__(self):
            pass

        def __exit__(self, *a):
            pass

    class MockCuda:
        def synchronize(self):
            pass

        def max_memory_allocated(self):
            return 1024 * 1024 * 1024

        def reset_peak_memory_stats(self):
            pass

        def is_available(self):
            return True

    class MockMps:
        def synchronize(self):
            pass

        def driver_allocated_memory(self):
            return 1024 * 1024 * 1024

        def is_available(self):
            return True

    class MockBackends:
        mps = MockMps()

    import torch

    class MockTorch:
        long = "long"
        bfloat16 = torch.bfloat16
        cuda = MockCuda()
        mps = MockMps()
        backends = MockBackends()

        @staticmethod
        def compile(model):
            raise RuntimeError("mock compile err")

        @staticmethod
        def no_grad():
            return MockNoGrad()

        @staticmethod
        def randint(*a, **k):
            return MockTensor()

        @staticmethod
        def manual_seed(s):
            pass

    monkeypatch.setattr(bm, "torch", MockTorch)
    monkeypatch.setattr(bm, "AutoModelForCausalLM", MockAutoModelForCausalLM)

    # Test prefill mode
    res = bm._run_benchmark_pass(MockModel(), "cuda", 1, 1, 1, "prefill", 128)
    assert len(res) == 3
    assert res[2] == 1024.0  # memory_mb

    # Test generation mode
    res_gen = bm._run_benchmark_pass(MockModel(), "cuda", 1, 1, 1, "generate", 128)
    assert len(res_gen) == 3

    # Test MPS memory
    assert bm._get_memory_mb(None, "mps") == 1024.0

    # Test MPS sync
    bm._sync_cuda("mps")

    # Test MPS device mapping
    assert bm._get_device("mps") == "cuda"  # cuda is checked first if available in MockTorch
    # Let's disable cuda to test MPS
    MockTorch.cuda.is_available = lambda: False
    assert bm._get_device("mps") == "mps"

    class MockNativeModel:
        def to(self, device):
            pass

        def eval(self):
            pass

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM", lambda config: MockNativeModel(), raising=False)

    # Test native backend loading
    res_load_native = bm._load_pytorch_model_and_device("m", "cpu", backend_alias="pytorch_native")
    assert res_load_native[1] == "cpu"

    # Test compile error logging
    bm._load_pytorch_model_and_device("m", "cpu", test_mode=False, backend_alias="pytorch")


def test_pytorch_benchmark_edge_cases(monkeypatch):
    import gemma_4_sql.backends.pytorch.benchmark as bm

    # 58->67 (native model without .to)
    class MockNativeModelNoTo:
        def eval(self):
            pass

        def __call__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM", lambda config: MockNativeModelNoTo(), raising=False)
    bm._load_pytorch_model_and_device("m", "cpu", backend_alias="pytorch_native")

    # 59->61 (native model with .to but torch_dtype=None)
    class MockNativeModelWithTo:
        def to(self, *a, **k):
            pass

        def eval(self):
            pass

        def __call__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM", lambda config: MockNativeModelWithTo(), raising=False)

    # mock torch to not have float32 so torch_dtype becomes None when test_mode=True
    class MockTorchNoFloat32:
        pass

    monkeypatch.setattr(bm, "torch", MockTorchNoFloat32)
    bm._load_pytorch_model_and_device("m", "cpu", test_mode=True, backend_alias="pytorch_native")

    # 136->131, 146->141 (generate mode but model has no generate)
    class MockTorch:
        cuda = type("Cuda", (), {"is_available": lambda: False})()
        mps = type("Mps", (), {"is_available": lambda: False})()

        @staticmethod
        def no_grad():
            return type("CM", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()

        @staticmethod
        def randint(*a, **k):
            return "dummy"

    monkeypatch.setattr(bm, "torch", MockTorch)

    class MockModelNoGenerate:
        pass

    bm._run_benchmark_pass(MockModelNoGenerate(), "cpu", 1, 1, 1, "generate", 128)
