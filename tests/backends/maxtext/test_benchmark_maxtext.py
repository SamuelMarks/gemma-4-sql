"""Tests for MaxText Benchmark."""

from __future__ import annotations

import typing

import gemma_4_sql.backends.maxtext.benchmark as bm
from gemma_4_sql.backends.maxtext.benchmark import benchmark_model

if typing.TYPE_CHECKING:
    import pytest


class MockJnp:
    """Provide class docstring."""

    int32 = 1

    @staticmethod
    def zeros(_shape: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [0]


class MockJaxRandom:
    """Provide class docstring."""

    @staticmethod
    def randint(*args: object, **kwargs: object) -> object:
        """Execute function."""
        return [0]

    @staticmethod
    def mock_prngkey(seed: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return seed

    PRNGKey = mock_prngkey


class MockJax:
    """Provide class docstring."""

    random = MockJaxRandom()

    def devices(self, *args):
        """Execute devices helper."""
        from unittest.mock import MagicMock

        return [MagicMock()]

    def default_device(self, *args):
        """Execute default device helper."""
        from unittest.mock import MagicMock

        return MagicMock()

    @staticmethod
    def jit(fn: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return fn

    def block_until_ready(self, x: object) -> None:
        """Execute function."""


class MockGemma4Model:
    """Provide class docstring."""

    def __init__(self, name: object) -> None:
        """Execute function."""

    def init(self, _rng: object, _inputs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "params"

    def apply(self, _params: object, inputs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return inputs


def test_benchmark_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "jax", None)
    res = benchmark_model("model", "gpu", 1)
    if not res["status"] == "mocked_missing_maxtext":
        raise AssertionError


def test_benchmark_maxtext_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "Gemma4Model", MockGemma4Model)
    res = benchmark_model("model", "gpu", 1, num_runs=2, test_mode=True)
    if not res["status"] == "success":
        raise AssertionError
    if not res["tokens_per_sec"] > 0:
        raise AssertionError


def test_benchmark_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "Gemma4Model", MockGemma4Model)

    def mock_raise_error(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(bm.jax.random, "randint", Exception)
    res = benchmark_model("model", "gpu", 1, test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError


class MockModel:
    """Provide class docstring."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""

    def init(self, *_args: object, **_kwargs: object) -> str:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "params"

    def apply(self, *_args: object, **_kwargs: object) -> str:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "out"


def test_benchmark_maxtext_real_no_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(
        bm,
        "jax",
        type(
            "MockJax",
            (),
            {
                "random": type("MockRng", (), {"PRNGKey": lambda x: x, "randint": lambda *a, **k: [0]}),
                "jit": lambda x: x,
                "distributed": type("MockDist", (), {"initialize": lambda: None}),
                "devices": lambda *a: [type("MockDevice", (), {})()],
                "default_device": lambda *a: type("MockContextManager", (), {"__enter__": lambda self: None, "__exit__": lambda self, *args: None})(),
            },
        ),
    )
    monkeypatch.setattr(bm, "jnp", type("MockJnp", (), {"int32": "int32", "zeros": lambda *args, **_kwargs: args}))
    monkeypatch.setattr(bm, "Gemma4Model", MockModel)
    res = bm.benchmark_model("model", "tpu", 1)
    if res["status"] != "success":
        raise AssertionError


def test_benchmark_maxtext_real_no_test_mode_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """

    def raise_err() -> typing.Never:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(
        bm,
        "jax",
        type(
            "MockJax",
            (),
            {
                "random": type("MockRng", (), {"PRNGKey": lambda x: x, "randint": lambda *a, **k: [0]}),
                "jit": lambda x: x,
                "distributed": type("MockDist", (), {"initialize": raise_err}),
                "devices": lambda *a: [type("MockDevice", (), {})()],
                "default_device": lambda *a: type("MockContextManager", (), {"__enter__": lambda self: None, "__exit__": lambda self, *args: None})(),
            },
        ),
    )
    monkeypatch.setattr(bm, "jnp", type("MockJnp", (), {"int32": "int32", "zeros": lambda *args, **_kwargs: args}))
    monkeypatch.setattr(bm, "Gemma4Model", MockModel)
    res = bm.benchmark_model("model", "tpu", 1)
    if res["status"] != "success":
        raise AssertionError


def test_benchmark_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    m_benchmark = __import__("gemma_4_sql.backends.maxtext.benchmark", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_benchmark)
    monkeypatch.undo()
    importlib.reload(m_benchmark)


def test_benchmark_maxtext_coverage(monkeypatch):
    """Test benchmark maxtext coverage functionality."""
    import gemma_4_sql.backends.maxtext.benchmark as bm

    class MockJax:
        """Test class for MockJax."""

        def __init__(self, mode="normal"):
            """Initialize __init__."""
            self.mode = mode
            self.random = type("R", (), {"PRNGKey": staticmethod(lambda s: s), "randint": staticmethod(lambda *a, **k: "dummy")})()

        def devices(self, d):
            """Execute devices helper."""
            if self.mode == "err" and d == "gpu":
                raise RuntimeError("err")
            if d == "tpu" and self.mode != "tpu":
                raise RuntimeError("err")
            return [type("D", (), {})()]

        def block_until_ready(self, x):
            """Execute block until ready helper."""
            return x

        def jit(self, f, *a, **k):
            """Execute jit helper."""
            return f

        def default_device(self, d):
            """Execute default device helper."""
            return type("CM", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()

    monkeypatch.setattr(bm, "jnp", type("JNP", (), {"zeros": lambda *a, **k: "zeros", "int32": "int32", "argmax": lambda *a, **k: "dummy", "concatenate": lambda *a, **k: "dummy"})())

    class MockOut:
        """Test class for MockOut."""

        def __getitem__(self, k):
            """Initialize __getitem__."""
            return "dummy"

    class MockModel:
        """Test class for MockModel."""

        def init(self, rng, inputs):
            """Execute init helper."""
            return "params"

        def apply(self, params, inputs):
            """Execute apply helper."""
            return MockOut()

        def generate(self, *a, **k):
            """Execute generate helper."""
            return MockOut()

        def __call__(self, x):
            """Initialize __call__."""
            return x

    monkeypatch.setattr(bm, "Gemma4Model", lambda name: MockModel())
    monkeypatch.setattr(bm, "jax", MockJax())

    # 32-34 (TPU devices)
    assert bm._get_device("tpu")
    monkeypatch.setattr(bm, "jax", MockJax("tpu"))
    assert bm._get_device("tpu")

    # Error in GPU device
    monkeypatch.setattr(bm, "jax", MockJax("err"))
    assert bm._get_device("gpu")  # falls back to CPU

    monkeypatch.setattr(bm, "jax", MockJax())
    # Generate mode
    bm._run_benchmark_pass(MockModel(), "params", 1, 2, 1, "generate", 128, "cpu")
    bm._run_benchmark_pass(MockModel(), "params", 1, 2, 1, "prefill", 128, "cpu")

    # Mock no generate
    bm._run_benchmark_pass(type("M", (), {"apply": lambda *a, **k: MockOut()})(), "params", 1, 2, 1, "generate", 128, "cpu")

    # Mock jax without jit
    class MockJaxNoJit:
        """Test class for MockJaxNoJit."""

        def __init__(self):
            """Initialize __init__."""
            self.random = type("R", (), {"PRNGKey": staticmethod(lambda s: s), "randint": staticmethod(lambda *a, **k: "dummy")})()

        def default_device(self, d):
            """Execute default device helper."""
            return type("CM", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()

        def block_until_ready(self, x):
            """Execute block until ready helper."""
            return x

    mock_jax_nojit = MockJaxNoJit()
    monkeypatch.setattr(bm, "jax", mock_jax_nojit)
    bm._run_benchmark_pass(MockModel(), "params", 1, 1, 0, "prefill", 16, "cpu")


def test_maxtext_benchmark_branch_empty(monkeypatch):
    """Test maxtext benchmark branch empty functionality."""
    import gemma_4_sql.backends.maxtext.benchmark as bm

    class MockJaxEmpty:
        """Test class for MockJaxEmpty."""

        def devices(self, d):
            """Execute devices helper."""
            return []

    monkeypatch.setattr(bm, "jax", MockJaxEmpty())

    # 34->35/38 (TPU devices empty)
    # This will trigger an exception if cpu is also empty, but let's just make cpu return a device
    class MockJaxCPUFallback:
        """Test class for MockJaxCPUFallback."""

        def devices(self, d):
            """Execute devices helper."""
            if d in ("tpu", "gpu"):
                return []
            return ["cpu"]

    monkeypatch.setattr(bm, "jax", MockJaxCPUFallback())
    assert bm._get_device("tpu") == "cpu"
    assert bm._get_device("gpu") == "cpu"
