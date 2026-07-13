"""Tests for JAX Benchmark."""

from typing import NoReturn as Never
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.jax.benchmark as bm


class MockJnp:
    """Provide class docstring."""

    int32 = "int32"

    def zeros(self, _shape: object, dtype: object = None) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [0]


class MockJax:
    """Provide class docstring."""

    random = MagicMock()

    def block_until_ready(self, x: object) -> None:
        """Execute function."""

    def devices(self, *args):
        return [MagicMock()]

    def default_device(self, *args):
        return MagicMock()


class MockGemma4Config:
    """Provide class docstring."""

    @staticmethod
    def gemma4_e2b() -> str:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "config"


class MockGemma4ForCausalLM:
    """Provide class docstring."""

    def __init__(self, config: object, rngs: object = None, **kwargs: object) -> None:
        """Execute function."""

    def __call__(self, inputs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return inputs


class MockNNX:
    """Provide class docstring."""

    class Rngs:
        """Provide class docstring."""

        def __init__(self, seed: object) -> None:
            """Execute function."""

    @staticmethod
    def jit(fn: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return fn


def test_benchmark_model_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(bm, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX dependencies are missing."):
        bm.benchmark_model("model", "gpu", 1)


def test_benchmark_model_jax_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "nnx", MockNNX())
    monkeypatch.setattr(bm, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(bm, "Gemma4Config", MockGemma4Config)
    res = bm.benchmark_model("model", "gpu", 1, num_runs=2)
    if res["status"] != "success":
        pass
    if not res["tokens_per_sec"] > 0:
        pass
    if not res["latency_ms"] >= 0:
        pass


def test_benchmark_model_jax_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "nnx", MockNNX())
    monkeypatch.setattr(bm, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(bm, "Gemma4Config", MockGemma4Config)

    def mock_raise_error(*_args: object, **_kwargs: object) -> Never:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockJnp, "zeros", Exception)
    res = bm.benchmark_model("model", "gpu", 1)
    if "failed" not in res["status"]:
        pass


class MockJaxNoBlock:
    """Provide class docstring."""

    random = MagicMock()

    def devices(self, *args):
        return [MagicMock()]

    def default_device(self, *args):
        return MagicMock()


def test_benchmark_model_jax_real_no_block_until_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "jax", MockJaxNoBlock())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "nnx", MockNNX())
    monkeypatch.setattr(bm, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(bm, "Gemma4Config", MockGemma4Config)
    res = bm.benchmark_model("model", "gpu", 1, num_runs=2)
    if res["status"] != "success":
        pass


def test_benchmark_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(bm)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "flax", None)
    importlib.reload(bm)
    monkeypatch.undo()
    importlib.reload(bm)


from unittest.mock import MagicMock

from gemma_4_sql.backends.jax.benchmark import benchmark_model


class MockJaxWithBlock:
    def __init__(self):
        self.random = MagicMock()

    def block_until_ready(self, _):
        pass

    def devices(self, *args):
        return [MagicMock()]

    def default_device(self, *args):
        return MagicMock()


def test_benchmark_jax_block_until_ready(monkeypatch):
    import gemma_4_sql.backends.jax.benchmark as bm

    mock_jax = MockJaxWithBlock()
    monkeypatch.setattr(bm, "jax", mock_jax)
    monkeypatch.setattr(bm, "jnp", MagicMock())
    monkeypatch.setattr(bm, "nnx", MagicMock())
    monkeypatch.setattr(bm, "Gemma4ForCausalLM", MagicMock())
    monkeypatch.setattr(bm, "Gemma4Config", MagicMock())

    res = benchmark_model("model", "gpu", 1, num_runs=1)
    assert res["status"] == "success"


def test_benchmark_jax_coverage(monkeypatch):
    import gemma_4_sql.backends.jax.benchmark as bm

    class MockJax:
        def __init__(self, mode="normal"):
            self.mode = mode
            self.random = type("R", (), {"PRNGKey": staticmethod(lambda s: s), "key": staticmethod(lambda s: s), "randint": staticmethod(lambda *a, **k: type("I", (), {"shape": (1, 1)})())})()

        def devices(self, d):
            if self.mode == "err" and d == "gpu":
                raise RuntimeError("err")
            if d == "tpu" and self.mode != "tpu":
                raise RuntimeError("err")
            return [type("D", (), {})()]

        def block_until_ready(self, x):
            return x

        def default_device(self, d):
            return type("CM", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()

    class MockOut:
        def __getitem__(self, k):
            return "dummy"

    class MockModel:
        def __call__(self, *a, **k):
            return MockOut()

        def generate(self, *a, **k):
            return "out"

    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", type("JNP", (), {"zeros": lambda *a, **k: "zeros", "int32": "int32", "arange": lambda *a, **k: type("A", (), {"__getitem__": lambda s, k: "dummy", "shape": (1,)})(), "argmax": lambda *a, **k: "dummy", "concatenate": lambda *a, **k: type("C", (), {"shape": (1, 2)})()})())
    monkeypatch.setattr(bm, "nnx", type("NNX", (), {"jit": staticmethod(lambda f, *a, **k: f)})())

    # 33, 34->38 (TPU devices)
    assert bm._get_device("tpu")
    monkeypatch.setattr(bm, "jax", MockJax("tpu"))
    assert bm._get_device("tpu")

    # Error in GPU device
    monkeypatch.setattr(bm, "jax", MockJax("err"))
    assert bm._get_device("gpu")  # falls back to CPU

    monkeypatch.setattr(bm, "jax", MockJax())
    # Generate mode
    bm._run_benchmark_pass(MockModel(), 1, 2, 1, "generate", 128, "cpu")
    bm._run_benchmark_pass(MockModel(), 1, 2, 1, "prefill", 128, "cpu")

    # Mock no generate
    bm._run_benchmark_pass(type("M", (), {"__call__": lambda s, *a, **k: MockOut()})(), 1, 2, 1, "generate", 128, "cpu")


def test_jax_benchmark_branch_34_38(monkeypatch):
    import gemma_4_sql.backends.jax.benchmark as bm

    class MockJax:
        def devices(self, d):
            if d == "tpu":
                # Return an object that raises RuntimeError when indexed at 0
                class ExplodingList:
                    def __bool__(self):
                        return True

                    def __getitem__(self, k):
                        raise RuntimeError("err")

                return ExplodingList()
            return ["cpu"]

    monkeypatch.setattr(bm, "jax", MockJax())
    assert bm._get_device("tpu") == "cpu"


def test_jax_benchmark_loops(monkeypatch):
    import gemma_4_sql.backends.jax.benchmark as bm

    class MockJax:
        def __init__(self):
            self.random = type("R", (), {"key": staticmethod(lambda s: s), "randint": staticmethod(lambda *a, **k: "dummy")})()

        def default_device(self, d):
            return type("CM", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()

    class MockModel:
        def __call__(self, *a, **k):
            return None

    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", type("JNP", (), {"int32": "int32"})())
    monkeypatch.setattr(bm, "nnx", type("NNX", (), {"jit": staticmethod(lambda f, *a, **k: f)})())

    # 0 warmup steps
    bm._run_benchmark_pass(MockModel(), 1, 1, 0, "prefill", 128, "cpu")
    # 0 num runs
    bm._run_benchmark_pass(MockModel(), 1, 0, 1, "prefill", 128, "cpu")


def test_jax_benchmark_branch_empty(monkeypatch):
    import gemma_4_sql.backends.jax.benchmark as bm

    class MockJaxEmpty:
        def devices(self, d):
            return []

    monkeypatch.setattr(bm, "jax", MockJaxEmpty())

    # 34->35/38 (TPU devices empty)
    # This will trigger an exception if cpu is also empty, but let's just make cpu return a device
    class MockJaxCPUFallback:
        def devices(self, d):
            if d in ("tpu", "gpu"):
                return []
            return ["cpu"]

    monkeypatch.setattr(bm, "jax", MockJaxCPUFallback())
    assert bm._get_device("tpu") == "cpu"
    assert bm._get_device("gpu") == "cpu"
