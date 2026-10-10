"""Test coverage for Keras benchmark."""

"""Test coverage for Keras benchmark."""

from unittest.mock import MagicMock

import pytest


def test_load_keras_model_missing(monkeypatch):
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.keras.benchmark as bm
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(bm, "keras", None)
    with pytest.raises(DependencyMissingError):
        bm._load_keras_model("name", "bfloat16")


def test_load_keras_model_error(monkeypatch):
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.keras.benchmark as bm

    mock_keras = MagicMock()
    monkeypatch.setattr(bm, "keras", mock_keras)

    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "keras_nlp.models":
            raise ValueError("simulated error")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ValueError):
        bm._load_keras_model("name", "bfloat16")


def test_get_device_str(monkeypatch):
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.keras.benchmark as bm

    # tf is none
    monkeypatch.setattr(bm, "tf", None)
    assert bm._get_device_str("gpu") == "/CPU:0"

    mock_tf = MagicMock()
    mock_tf.config.list_physical_devices.return_value = True
    monkeypatch.setattr(bm, "tf", mock_tf)

    assert bm._get_device_str("cpu") == "/CPU:0"
    assert bm._get_device_str("gpu") == "/GPU:0"
    assert bm._get_device_str("tpu") == "/TPU:0"
    assert bm._get_device_str("unknown") == "/CPU:0"


def test_run_benchmark_pass_missing(monkeypatch):
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.keras.benchmark as bm
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(bm, "tf", None)
    with pytest.raises(DependencyMissingError):
        bm._run_benchmark_pass(MagicMock(), 1, 1, 1, "prefill", 1, "cpu")


def test_run_benchmark_pass(monkeypatch):
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.keras.benchmark as bm

    mock_tf = MagicMock()
    # mock decorator tf.function
    mock_tf.function = lambda *args, **kwargs: lambda fn: fn
    mock_tf.config.list_physical_devices.return_value = True

    # Context manager mock for tf.device
    class MockDevice:
        """Docstring for MockDevice."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

        def __enter__(self):
            """Docstring for __enter__."""

        def __exit__(self, *args):
            """Docstring for __exit__."""

    mock_tf.device = MockDevice

    mock_tf.config.experimental.get_memory_info.return_value = {"peak": 1048576}
    monkeypatch.setattr(bm, "tf", mock_tf)

    # 1. GPU + prefill
    mock_model = MagicMock()
    mock_out = MagicMock()
    mock_out.numpy.return_value = 1
    mock_model.return_value = mock_out

    res = bm._run_benchmark_pass(mock_model, 1, 1, 1, "prefill", 1, "gpu")
    assert res[2] == 1.0  # 1048576 / (1024*1024)

    # 2. CPU + generate
    mock_model_gen = MagicMock()
    mock_model_gen.generate.return_value = mock_out

    res2 = bm._run_benchmark_pass(mock_model_gen, 1, 1, 1, "generate", 1, "cpu")
    assert res2[2] == 6000.0  # CPU default


def test_run_benchmark_pass_exceptions(monkeypatch):
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.keras.benchmark as bm

    mock_tf = MagicMock()
    mock_tf.function = lambda *args, **kwargs: lambda fn: fn
    mock_tf.config.list_physical_devices.return_value = True

    class MockDevice:
        """Docstring for MockDevice."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

        def __enter__(self):
            """Docstring for __enter__."""

        def __exit__(self, *args):
            """Docstring for __exit__."""

    mock_tf.device = MockDevice

    # exceptions in memory stats / get memory info
    def raise_value_error(*args):
        """Docstring for raise_value_error."""
        raise ValueError("sim")

    mock_tf.config.experimental.reset_memory_stats.side_effect = raise_value_error
    mock_tf.config.experimental.get_memory_info.side_effect = raise_value_error

    monkeypatch.setattr(bm, "tf", mock_tf)
    res = bm._run_benchmark_pass(MagicMock(), 1, 1, 1, "prefill", 1, "gpu")
    assert res[2] == 6000.0  # fallback


def test_benchmark_model(monkeypatch):
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.keras.benchmark as bm
    from gemma_4_sql.exceptions import DependencyMissingError

    # test missing
    monkeypatch.setattr(bm, "keras", None)
    with pytest.raises(DependencyMissingError):
        bm.benchmark_model("m", "h", 1)

    monkeypatch.setattr(bm, "keras", MagicMock())
    monkeypatch.setattr(bm, "tf", MagicMock())

    # test actual execution through wrapper
    def mock_run_wrapper(*args, **kwargs):
        # execute the _run
        """Docstring for mock_run_wrapper."""
        kwargs["benchmark_fn"]()
        return {"status": "completed"}

    monkeypatch.setattr(bm, "run_benchmark_wrapper", mock_run_wrapper)

    monkeypatch.setattr(bm, "_load_keras_model", MagicMock())
    monkeypatch.setattr(bm, "_run_benchmark_pass", MagicMock())

    res = bm.benchmark_model("m", "cpu", 1, num_runs=5)
    assert res["status"] == "completed"


def test_run_benchmark_pass_missing_attrs(monkeypatch):
    """Test function."""
    """Test function."""
    """Test missing attrs."""
    import gemma_4_sql.backends.keras.benchmark as bm

    mock_tf = MagicMock()
    mock_tf.function = lambda *args, **kwargs: lambda fn: fn
    mock_tf.config.list_physical_devices.return_value = True

    class MockDevice:
        """Mock context."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

        def __enter__(self):
            """Docstring for __enter__."""

        def __exit__(self, *args):
            """Docstring for __exit__."""

    mock_tf.device = MockDevice
    monkeypatch.setattr(bm, "tf", mock_tf)

    # 1. No numpy on out, prefill
    mock_model = MagicMock()
    mock_out = object()
    mock_model.return_value = mock_out

    bm._run_benchmark_pass(mock_model, 1, 1, 1, "prefill", 1, "cpu")

    # 2. No generate on model, generate mode
    mock_model_no_gen = object()
    bm._run_benchmark_pass(mock_model_no_gen, 1, 1, 1, "generate", 1, "cpu")

    # 3. No numpy on out, generate mode
    mock_model_gen = MagicMock()
    mock_model_gen.generate.return_value = mock_out
    bm._run_benchmark_pass(mock_model_gen, 1, 1, 1, "generate", 1, "cpu")
