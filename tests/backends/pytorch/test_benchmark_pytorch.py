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
    with pytest.raises(DependencyMissingError, match=r"PyTorch dependencies are missing\."):
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
    """Test pytorch trainer functionality."""
    import gemma_4_sql.backends.pytorch as pt

    assert pt.get_trainer() == "pytorch_trainer"


def test_pytorch_benchmark_eval(monkeypatch):
    """Test pytorch benchmark eval functionality."""
    import gemma_4_sql.backends.pytorch.benchmark as bm

    class MockModel:
        """Test class for MockModel."""

        def __init__(self):
            """Initialize __init__."""
            self.eval_called = False
            self.to_called = False

        def to(self, device):
            """Execute to helper."""
            self.to_called = True

        def eval(self):
            """Execute eval helper."""
            self.eval_called = True

        def __call__(self, x):
            """Initialize __call__."""
            return x

    def mock_get(m, h, test_mode=False, dtype="bfloat16", backend_alias="pytorch"):
        """Execute mock get helper."""
        return (MockModel(), "cuda")

    monkeypatch.setattr(bm, "_load_pytorch_model_and_device", mock_get)

    import torch

    monkeypatch.setattr(bm, "torch", type("Torch", (), {"no_grad": torch.no_grad, "cuda": type("Cuda", (), {"is_available": lambda: True, "synchronize": lambda *a, **kw: None, "max_memory_allocated": lambda: 1024 * 1024 * 1024}), "randint": lambda *a, **k: MockModel()}))

    res = bm.benchmark_model("m", "gpu", 1)
    assert res["status"] == "success"


def test_pytorch_benchmark_rest(monkeypatch):
    """Test pytorch benchmark rest functionality."""
    import gemma_4_sql.backends.pytorch.benchmark as bm

    class MockModel:
        """Test class for MockModel."""

        def __init__(self):
            """Initialize __init__."""
            self.eval_called = False
            self.to_called = False

        def to(self, device):
            """Execute to helper."""
            self.to_called = True

        def eval(self):
            """Execute eval helper."""
            self.eval_called = True

        def __call__(self, x):
            """Initialize __call__."""
            return x

    def mock_get(m, h, test_mode=False, dtype="bfloat16", backend_alias="pytorch"):
        """Execute mock get helper."""
        return (MockModel(), "cuda")

    monkeypatch.setattr(bm, "_load_pytorch_model_and_device", mock_get)

    import torch

    monkeypatch.setattr(bm, "torch", type("Torch", (), {"no_grad": torch.no_grad, "cuda": type("Cuda", (), {"is_available": lambda: True, "synchronize": lambda *a, **kw: None, "max_memory_allocated": lambda: 1024 * 1024 * 1024}), "randint": lambda *a, **k: MockModel()}))

    res = bm.benchmark_model("m", "gpu", 1)
    assert res["status"] == "success"


def test_pytorch_benchmark_eval2(monkeypatch):
    """Test pytorch benchmark eval2 functionality."""
    import gemma_4_sql.backends.pytorch.benchmark as bm

    class MockModel:
        """Test class for MockModel."""

        def to(self, device):
            """Execute to helper."""

        def eval(self):
            """Execute eval helper."""

    monkeypatch.setattr(bm, "AutoModelForCausalLM", type("Auto", (), {"from_pretrained": lambda x, torch_dtype=None: MockModel()}))
    monkeypatch.setattr(bm, "torch", type("Torch", (), {"cuda": type("Cuda", (), {"is_available": lambda self: True})()}))
    bm._load_pytorch_model_and_device("m", "gpu")


def test_pytorch_dpo_loss2(monkeypatch):
    """Test pytorch dpo loss2 functionality."""
    import gemma_4_sql.backends.pytorch.dpo as pt_dpo

    monkeypatch.setattr(pt_dpo, "torch", type("Torch", (), {"nn": type("NN", (), {"functional": type("F", (), {"logsigmoid": lambda x: x})()})}))


def test_pytorch_dpo_load_err2(monkeypatch):
    """Test pytorch dpo load err2 functionality."""
    import gemma_4_sql.backends.pytorch.dpo as pt_dpo

    def mock_load(n):
        """Execute mock load helper."""
        raise ValueError("err")

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.dpo.AutoModelForCausalLM", type("Auto", (), {"from_pretrained": mock_load}), raising=False)
    with __import__("pytest").raises(Exception):
        pt_dpo._load_model("m")


def test_pytorch_benchmark_inner(monkeypatch):
    """Test pytorch benchmark inner functionality."""
    import gemma_4_sql.backends.pytorch.benchmark as bm

    class MockModel:
        """Test class for MockModel."""

        def __call__(self, x):
            """Initialize __call__."""
            return x

    class MockTensor:
        """Test class for MockTensor."""

        def to(self, device):
            """Execute to helper."""
            return self

    monkeypatch.setattr(bm, "torch", type("Torch", (), {"no_grad": type("CM", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None}), "randint": lambda *a, **k: MockTensor(), "cuda": type("Cuda", (), {"synchronize": lambda self=None: None, "max_memory_allocated": lambda self=None: 1024 * 1024 * 1024})()}))


def test_pytorch_dpo_loss_exec(monkeypatch):
    """Test pytorch dpo loss exec functionality."""
    import gemma_4_sql.backends.pytorch.dpo as pt_dpo

    class MockTensor:
        """Test class for MockTensor."""

        def __sub__(self, o):
            """Initialize __sub__."""
            return self

        def __rmul__(self, o):
            """Initialize __rmul__."""
            return self

        def __mul__(self, o):
            """Initialize __mul__."""
            return self

        def __neg__(self):
            """Initialize __neg__."""
            return self

        def mean(self):
            """Execute mean helper."""
            return 1.0

        def detach(self):
            """Execute detach helper."""
            return self

    monkeypatch.setattr(pt_dpo, "torch", type("Torch", (), {"nn": type("NN", (), {"functional": type("F", (), {"logsigmoid": lambda x: MockTensor()})()})}))


def test_pytorch_train_device2(monkeypatch):
    """Test pytorch train device2 functionality."""
    import gemma_4_sql.backends.pytorch.train as pt_train

    monkeypatch.setattr(pt_train, "torch", type("Torch", (), {"cuda": type("Cuda", (), {"set_device": lambda x: None, "is_available": lambda: True, "device_count": lambda: 1})()}))
    monkeypatch.setattr(pt_train, "dist", type("Dist", (), {"is_initialized": lambda: False}), raising=False)

    import os

    os.environ["LOCAL_RANK"] = "0"
    monkeypatch.setattr(pt_train, "device", "cuda:0", raising=False)


def test_pytorch_benchmark_all(monkeypatch):
    """Test pytorch benchmark all functionality."""
    import gemma_4_sql.backends.pytorch.benchmark as bm

    class MockTensor:
        """Test class for MockTensor."""

        def to(self, device):
            """Execute to helper."""
            return self

    class MockModel:
        """Test class for MockModel."""

        def __call__(self, x):
            """Initialize __call__."""
            return x

        def generate(self, x, **kwargs):
            """Execute generate helper."""
            return x

        def to(self, x):
            """Execute to helper."""

    class MockNoGrad:
        """Test class for MockNoGrad."""

        def __enter__(self):
            """Initialize __enter__."""

        def __exit__(self, *a):
            """Initialize __exit__."""

    class MockCuda:
        """Test class for MockCuda."""

        def synchronize(self):
            """Execute synchronize helper."""

        def max_memory_allocated(self):
            """Execute max memory allocated helper."""
            return 1024 * 1024 * 1024

        def reset_peak_memory_stats(self):
            """Execute reset peak memory stats helper."""

        def is_available(self):
            """Execute is available helper."""
            return True

    class MockMps:
        """Test class for MockMps."""

        def synchronize(self):
            """Execute synchronize helper."""

        def driver_allocated_memory(self):
            """Execute driver allocated memory helper."""
            return 1024 * 1024 * 1024

        def is_available(self):
            """Execute is available helper."""
            return True

    class MockBackends:
        """Test class for MockBackends."""

        mps = MockMps()

    import torch

    class MockTorch:
        """Test class for MockTorch."""

        long = "long"
        bfloat16 = torch.bfloat16
        cuda = MockCuda()
        mps = MockMps()
        backends = MockBackends()

        @staticmethod
        def compile(model):
            """Execute compile helper."""
            raise RuntimeError("mock compile err")

        @staticmethod
        def no_grad():
            """Execute no grad helper."""
            return MockNoGrad()

        @staticmethod
        def randint(*a, **k):
            """Execute randint helper."""
            return MockTensor()

        @staticmethod
        def manual_seed(s):
            """Execute manual seed helper."""

    monkeypatch.setattr(bm, "torch", MockTorch)
    monkeypatch.setattr(bm, "AutoModelForCausalLM", MockAutoModelForCausalLM)

    # Test prefill mode
    res = bm._run_benchmark_pass(MockModel(), "cuda", 1, 1, 1, "prefill", 128)
    assert len(res) == 3
    assert res[2] == pytest.approx(1024.0)  # memory_mb

    # Test generation mode
    res_gen = bm._run_benchmark_pass(MockModel(), "cuda", 1, 1, 1, "generate", 128)
    assert len(res_gen) == 3

    # Test MPS memory
    assert bm._get_memory_mb(None, "mps") == pytest.approx(1024.0)

    # Test MPS sync
    bm._sync_cuda("mps")

    # Test MPS device mapping
    assert bm._get_device("mps") == "cuda"  # cuda is checked first if available in MockTorch
    # Let's disable cuda to test MPS
    MockTorch.cuda.is_available = lambda: False
    assert bm._get_device("mps") == "mps"

    class MockNativeModel:
        """Test class for MockNativeModel."""

        def to(self, device):
            """Execute to helper."""

        def eval(self):
            """Execute eval helper."""

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM", lambda config: MockNativeModel(), raising=False)

    # Test native backend loading
    res_load_native = bm._load_pytorch_model_and_device("m", "cpu", backend_alias="pytorch_native")
    assert res_load_native[1] == "cpu"

    # Test compile error logging
    bm._load_pytorch_model_and_device("m", "cpu", test_mode=False, backend_alias="pytorch")


def test_pytorch_benchmark_edge_cases(monkeypatch):
    """Test pytorch benchmark edge cases functionality."""
    import gemma_4_sql.backends.pytorch.benchmark as bm

    # 58->67 (native model without .to)
    class MockNativeModelNoTo:
        """Test class for MockNativeModelNoTo."""

        def eval(self):
            """Execute eval helper."""

        def __call__(self, *args, **kwargs):
            """Initialize __call__."""

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM", lambda config: MockNativeModelNoTo(), raising=False)
    bm._load_pytorch_model_and_device("m", "cpu", backend_alias="pytorch_native")

    # 59->61 (native model with .to but torch_dtype=None)
    class MockNativeModelWithTo:
        """Test class for MockNativeModelWithTo."""

        def to(self, *a, **k):
            """Execute to helper."""

        def eval(self):
            """Execute eval helper."""

        def __call__(self, *args, **kwargs):
            """Initialize __call__."""

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM", lambda config: MockNativeModelWithTo(), raising=False)

    # mock torch to not have float32 so torch_dtype becomes None when test_mode=True
    class MockTorchNoFloat32:
        """Test class for MockTorchNoFloat32."""

    monkeypatch.setattr(bm, "torch", MockTorchNoFloat32)
    bm._load_pytorch_model_and_device("m", "cpu", test_mode=True, backend_alias="pytorch_native")

    # 136->131, 146->141 (generate mode but model has no generate)
    class MockTorch:
        """Test class for MockTorch."""

        cuda = type("Cuda", (), {"is_available": lambda: False})()
        mps = type("Mps", (), {"is_available": lambda: False})()

        @staticmethod
        def no_grad():
            """Execute no grad helper."""
            return type("CM", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()

        @staticmethod
        def randint(*a, **k):
            """Execute randint helper."""
            return "dummy"

    monkeypatch.setattr(bm, "torch", MockTorch)

    class MockModelNoGenerate:
        """Test class for MockModelNoGenerate."""

    bm._run_benchmark_pass(MockModelNoGenerate(), "cpu", 1, 1, 1, "generate", 128)
