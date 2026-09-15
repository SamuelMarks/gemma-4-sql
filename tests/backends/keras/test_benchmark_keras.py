"""Tests for Keras Benchmark."""

import pytest

import gemma_4_sql.backends.keras.benchmark as bm


class MockTfTensor:
    """Provide class docstring."""

    def numpy(self) -> float:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return 0.0


class MockTf:
    """Provide class docstring."""

    int32 = "int32"

    class MockRandom:
        """Provide class docstring."""

        @staticmethod
        def set_seed(*_args: object, **_kwargs: object) -> None:
            """Execute set seed helper."""

        @staticmethod
        def uniform(*_args: object, **_kwargs: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return MockTfTensor()

    random = MockRandom

    def zeros(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockTfTensor()

    def function(self, fn: object = None, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        if fn is None:
            return lambda x: x
        return fn

    class MockConfig:
        """Provide class docstring."""

        @staticmethod
        def list_physical_devices(_d: str) -> list:
            """Execute list physical devices helper."""
            return [1]

        class MockExperimental:
            """Provide class docstring."""

            @staticmethod
            def get_memory_info(_device: str) -> dict:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return {"current": 1024 * 1024 * 100}

            @staticmethod
            def reset_memory_stats(_device: str) -> None:
                """Execute reset memory stats helper."""

        experimental = MockExperimental

    config = MockConfig

    class device:
        """Test class for device."""

        def __init__(self, d):
            """Initialize __init__."""
            self.d = d

        def __enter__(self):
            """Initialize __enter__."""

        def __exit__(self, *a):
            """Initialize __exit__."""


def test_benchmark_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "keras", None)
    from gemma_4_sql.exceptions import DependencyMissingError

    with pytest.raises(DependencyMissingError, match=r"Keras dependencies are missing\."):
        bm.benchmark_model("model", "gpu", 1)


def test_benchmark_keras_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "tf", MockTf())
    monkeypatch.setattr(bm, "keras", MockKeras())
    res = bm.benchmark_model("model", "gpu", 1, test_mode=True, num_runs=2)
    if "failed" not in res["status"]:
        raise AssertionError


def test_benchmark_keras_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "tf", MockTf())
    monkeypatch.setattr(bm, "keras", object())

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockTf, "zeros", raise_err)
    res = bm.benchmark_model("model", "gpu", 1, test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError


class MockKeras:
    """Provide class docstring."""

    class MockKerasConfig:
        """Test class for MockKerasConfig."""

        @staticmethod
        def set_floatx(dtype):
            """Execute set floatx helper."""

    config = MockKerasConfig

    def mock_input(*_args: object, **_kwargs: object) -> None:
        """Execute function."""
        return

    Input = mock_input

    class MockLayers:
        """Provide class docstring."""

        class Embedding:
            """Provide class docstring."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Execute function."""

            def __call__(self, x: object) -> object:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return x

        class Dense:
            """Provide class docstring."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Execute function."""

            def __call__(self, x: object) -> object:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return x

    layers = MockLayers

    def mock_model(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return lambda x: x

    Model = mock_model


def test_benchmark_keras_real_no_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    import sys

    monkeypatch.setitem(sys.modules, "keras_nlp", type("MockKerasNLP", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("MockModels", (), {"GemmaCausalLM": type("MockGemma", (), {"from_preset": lambda *a, **k: MockKeras.Model()})}))

    builtins = __import__("builtins", fromlist=[""])
    orig_import = builtins.__import__

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Test function."""
        if name == "keras_nlp.models":
            return sys.modules["keras_nlp.models"]
        return orig_import(name, _globals, _locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    monkeypatch.setattr(bm, "tf", MockTf())
    monkeypatch.setattr(bm, "keras", MockKeras())
    res = bm.benchmark_model("model", "gpu", 1)
    if "failed" in str(res.get("status", "")):
        raise AssertionError(f"Expected success, got {res}")


def test_benchmark_keras_real_mem(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    import sys

    monkeypatch.setitem(sys.modules, "keras_nlp", type("MockKerasNLP", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("MockModels", (), {"GemmaCausalLM": type("MockGemma", (), {"from_preset": lambda *a, **k: MockKeras.Model()})}))

    builtins = __import__("builtins", fromlist=[""])
    orig_import = builtins.__import__

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Test function."""
        if name == "keras_nlp.models":
            return sys.modules["keras_nlp.models"]
        return orig_import(name, _globals, _locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    mock_tf = MockTf()

    def mock_get_memory_info(_x: str) -> dict:
        """Test function."""
        msg = "No memory info"
        raise ValueError(msg)

    mock_tf.config = type("MockConfig", (), {"experimental": type("MockExp", (), {"get_memory_info": mock_get_memory_info})})()
    monkeypatch.setattr(bm, "tf", mock_tf)
    monkeypatch.setattr(bm, "keras", MockKeras())
    bm.benchmark_model("model", "gpu", 1)


def test_benchmark_keras_coverage(monkeypatch):
    """Test benchmark keras coverage functionality."""
    import gemma_4_sql.backends.keras.benchmark as bm

    class MockOut:
        """Test class for MockOut."""

        def numpy(self):
            """Execute numpy helper."""

    class MockModel:
        """Test class for MockModel."""

        def __call__(self, x):
            """Initialize __call__."""
            return MockOut()

        def generate(self, *a, **k):
            """Execute generate helper."""
            return MockOut()

    class MockTF:
        """Test class for MockTF."""

        class random:
            """Test class for random."""

            @staticmethod
            def set_seed(s):
                """Execute set seed helper."""

            @staticmethod
            def uniform(*a, **k):
                """Execute uniform helper."""
                return "dummy"

        int32 = "int32"

        class device:
            """Test class for device."""

            def __init__(self, d):
                """Initialize __init__."""
                self.d = d

            def __enter__(self):
                """Initialize __enter__."""

            def __exit__(self, *a):
                """Initialize __exit__."""

        @staticmethod
        def function(*a, **k):
            """Execute function helper."""
            return lambda f: f

        class config:
            """Test class for config."""

            @staticmethod
            def list_physical_devices(d):
                """Execute list physical devices helper."""
                return [1] if d == "GPU" else []

            class experimental:
                """Test class for experimental."""

                @staticmethod
                def reset_memory_stats(d):
                    """Execute reset memory stats helper."""
                    if "err" in d:
                        raise ValueError()

                @staticmethod
                def get_memory_info(d):
                    """Execute get memory info helper."""
                    if "err" in d:
                        raise ValueError()
                    return {"peak": 1024 * 1024 * 100}

        class Tensor:
            """Test class for Tensor."""

    monkeypatch.setattr(bm, "tf", MockTF)
    monkeypatch.setattr(bm, "keras", type("Keras", (), {"KerasTensor": MockTF.Tensor}))

    bm._run_benchmark_pass(MockModel(), 1, 1, 1, "prefill", 128, "cpu")
    bm._run_benchmark_pass(MockModel(), 1, 1, 1, "generate", 128, "gpu")

    # cover exceptions in GPU memory
    bm._run_benchmark_pass(MockModel(), 1, 1, 1, "generate", 128, "errGPU")


def test_benchmark_keras_coverage2(monkeypatch):
    """Test benchmark keras coverage2 functionality."""
    import gemma_4_sql.backends.keras.benchmark as bm

    class MockModelNoNumpy:
        """Test class for MockModelNoNumpy."""

        def __call__(self, x):
            """Initialize __call__."""
            return x

        def generate(self, *a, **k):
            """Execute generate helper."""
            return "out"

    class MockTF:
        """Test class for MockTF."""

        class random:
            """Test class for random."""

            @staticmethod
            def set_seed(s):
                """Execute set seed helper."""

            @staticmethod
            def uniform(*a, **k):
                """Execute uniform helper."""
                return "dummy"

        int32 = "int32"

        class device:
            """Test class for device."""

            def __init__(self, d):
                """Initialize __init__."""
                self.d = d

            def __enter__(self):
                """Initialize __enter__."""

            def __exit__(self, *a):
                """Initialize __exit__."""

        @staticmethod
        def function(*a, **k):
            """Execute function helper."""
            return lambda f: f

        class config:
            """Test class for config."""

            @staticmethod
            def list_physical_devices(d):
                """Execute list physical devices helper."""
                return []

            class experimental:
                """Test class for experimental."""

                @staticmethod
                def reset_memory_stats(d):
                    """Execute reset memory stats helper."""

                @staticmethod
                def get_memory_info(d):
                    """Execute get memory info helper."""
                    return {"peak": 10}

        class Tensor:
            """Test class for Tensor."""

    monkeypatch.setattr(bm, "tf", MockTF)
    monkeypatch.setattr(bm, "keras", type("Keras", (), {"KerasTensor": MockTF.Tensor}))

    # cpu string
    assert bm._get_device_str("cpu")
    # missing deps
    monkeypatch.setattr(bm, "keras", None)
    import pytest

    from gemma_4_sql.exceptions import DependencyMissingError

    with pytest.raises(DependencyMissingError):
        bm.benchmark_model("m", "cpu", 1)

    monkeypatch.setattr(bm, "keras", type("Keras", (), {"KerasTensor": MockTF.Tensor}))
    # Cover False branch of hasattr(out, 'numpy')
    bm._run_benchmark_pass(MockModelNoNumpy(), 1, 2, 1, "prefill", 128, "cpu")
    bm._run_benchmark_pass(MockModelNoNumpy(), 1, 2, 1, "generate", 128, "cpu")

    # Cover False branch of hasattr(model, 'generate')
    class MockModelNoGenerate:
        """Test class for MockModelNoGenerate."""

        def __call__(self, x):
            """Initialize __call__."""
            return x

    bm._run_benchmark_pass(MockModelNoGenerate(), 1, 2, 1, "generate", 128, "cpu")


def test_benchmark_keras_coverage3(monkeypatch):
    """Test benchmark keras coverage3 functionality."""
    import gemma_4_sql.backends.keras.benchmark as bm

    class config:
        """Test class for config."""

        @staticmethod
        def list_physical_devices(d):
            """Execute list physical devices helper."""
            return [1] if d in ("GPU", "TPU") else []

        class experimental:
            """Test class for experimental."""

            @staticmethod
            def reset_memory_stats(d):
                """Execute reset memory stats helper."""

            @staticmethod
            def get_memory_info(d):
                """Execute get memory info helper."""
                return {"peak": 10}

    monkeypatch.setattr(bm, "tf", type("MockTF", (), {"config": config()}))
    assert bm._get_device_str("tpu") == "/TPU:0"

    # 51 -> 161 (DependencyMissingError)
    monkeypatch.setattr(bm, "keras", None)
    import pytest

    from gemma_4_sql.exceptions import DependencyMissingError

    with pytest.raises(DependencyMissingError):
        bm.benchmark_model("m", "gpu", 1)


def test_keras_benchmark_126_127(monkeypatch):
    """Test keras benchmark 126 127 functionality."""
    import gemma_4_sql.backends.keras.benchmark as bm

    class MockModel:
        """Test class for MockModel."""

        def __call__(self, x):
            """Initialize __call__."""
            return x

    class MockTF:
        """Test class for MockTF."""

        class random:
            """Test class for random."""

            @staticmethod
            def set_seed(s):
                """Execute set seed helper."""

            @staticmethod
            def uniform(*a, **k):
                """Execute uniform helper."""
                return "dummy"

        int32 = "int32"

        class device:
            """Test class for device."""

            def __init__(self, d):
                """Initialize __init__."""
                self.d = d

            def __enter__(self):
                """Initialize __enter__."""

            def __exit__(self, *a):
                """Initialize __exit__."""

        @staticmethod
        def function(*a, **k):
            """Execute function helper."""
            return lambda f: f

        class config:
            """Test class for config."""

            @staticmethod
            def list_physical_devices(d):
                """Execute list physical devices helper."""
                return [1]

            class experimental:
                """Test class for experimental."""

                @staticmethod
                def reset_memory_stats(d):
                    """Execute reset memory stats helper."""
                    raise ValueError("dummy")

                @staticmethod
                def get_memory_info(d):
                    """Execute get memory info helper."""
                    raise ValueError("dummy")

    monkeypatch.setattr(bm, "tf", MockTF)
    monkeypatch.setattr(bm, "keras", type("Keras", (), {}))

    # This should trigger the ValueError
    res = bm._run_benchmark_pass(MockModel(), 1, 1, 1, "prefill", 128, "gpu")
    assert res[2] == pytest.approx(6000.0)


def test_keras_benchmark_missing_deps(monkeypatch):
    """Test keras benchmark missing deps functionality."""
    import gemma_4_sql.backends.keras.benchmark as bm

    monkeypatch.setattr(bm, "keras", None)
    import pytest

    from gemma_4_sql.exceptions import DependencyMissingError

    with pytest.raises(DependencyMissingError):
        bm.benchmark_model("m", "cpu", 1)
