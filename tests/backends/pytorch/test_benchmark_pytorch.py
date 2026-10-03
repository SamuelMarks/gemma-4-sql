from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.pytorch.benchmark as bm
from gemma_4_sql.exceptions import DependencyMissingError


def test_get_device():
    # cpu
    assert bm._get_device("cpu") == "cpu"

    # cuda
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch):
        assert bm._get_device("gpu") == "cuda"

    # mps
    mock_torch = MagicMock()
    mock_torch.cuda = None
    mock_torch.backends.mps.is_available.return_value = True
    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch):
        assert bm._get_device("gpu") == "mps"

    # fallback to cpu
    mock_torch = MagicMock()
    mock_torch.cuda = None
    mock_torch.backends = None
    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch):
        assert bm._get_device("gpu") == "cpu"


def test_load_pytorch_model_and_device_native():
    mock_torch = MagicMock()
    mock_torch.bfloat16 = "mock_bfloat16"
    mock_torch.compile = MagicMock(side_effect=RuntimeError("compile failed"))

    class MockGemma4Config:
        pass

    class MockGemma4ForCausalLM:
        def __init__(self, config):
            self.config = config

        def to(self, device_or_dtype):
            pass

        def eval(self):
            pass

    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch), patch("gemma_4_sql.backends.pytorch.benchmark._get_device", return_value="cpu"), patch.dict("sys.modules", {"gemma_4_sql.backends.pytorch.gemma4.modeling": MagicMock(Gemma4Config=MockGemma4Config, Gemma4ForCausalLM=MockGemma4ForCausalLM)}):
        model, device = bm._load_pytorch_model_and_device("dummy", "cpu", backend_alias="pytorch_native")
        assert device == "cpu"
        # Since torch.compile failed, it should return original model
        assert isinstance(model, MockGemma4ForCausalLM)

    # test without dtype
    mock_torch_no_dtype = MagicMock()
    del mock_torch_no_dtype.bfloat16
    mock_torch_no_dtype.float32 = "mock_float32"
    with (
        patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch_no_dtype),
        patch("gemma_4_sql.backends.pytorch.benchmark._get_device", return_value="cpu"),
        patch.dict("sys.modules", {"gemma_4_sql.backends.pytorch.gemma4.modeling": MagicMock(Gemma4Config=MockGemma4Config, Gemma4ForCausalLM=MockGemma4ForCausalLM)}),
    ):
        model, device = bm._load_pytorch_model_and_device("dummy", "cpu", backend_alias="pytorch_native")

    # test where model lacks 'to' and 'eval', and torch_dtype is fully None
    mock_torch_none_dtype = MagicMock()
    del mock_torch_none_dtype.bfloat16
    del mock_torch_none_dtype.float32

    class MockGemma4ForCausalLMNoTo:
        def __init__(self, config):
            self.config = config

    with (
        patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch_none_dtype),
        patch("gemma_4_sql.backends.pytorch.benchmark._get_device", return_value="cpu"),
        patch.dict("sys.modules", {"gemma_4_sql.backends.pytorch.gemma4.modeling": MagicMock(Gemma4Config=MockGemma4Config, Gemma4ForCausalLM=MockGemma4ForCausalLMNoTo)}),
    ):
        model, device = bm._load_pytorch_model_and_device("dummy", "cpu", backend_alias="pytorch_native")

    # test where torch_dtype is None but model HAS 'to'
    with (
        patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch_none_dtype),
        patch("gemma_4_sql.backends.pytorch.benchmark._get_device", return_value="cpu"),
        patch.dict("sys.modules", {"gemma_4_sql.backends.pytorch.gemma4.modeling": MagicMock(Gemma4Config=MockGemma4Config, Gemma4ForCausalLM=MockGemma4ForCausalLM)}),
    ):
        model, device = bm._load_pytorch_model_and_device("dummy", "cpu", backend_alias="pytorch_native")


def test_load_pytorch_model_and_device_hf():
    mock_torch = MagicMock()
    mock_torch.bfloat16 = "mock_bfloat16"
    mock_torch.compile = MagicMock(return_value="compiled_model")

    mock_auto_model = MagicMock()
    mock_model_instance = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model_instance

    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch), patch("gemma_4_sql.backends.pytorch.benchmark.AutoModelForCausalLM", mock_auto_model), patch("gemma_4_sql.backends.pytorch.benchmark._get_device", return_value="cuda"):
        model, device = bm._load_pytorch_model_and_device("dummy", "gpu", backend_alias="pytorch_hf")
        assert device == "cuda"
        assert model == "compiled_model"
        mock_model_instance.to.assert_called_with("cuda")
        mock_model_instance.eval.assert_called_once()

    # test torch.compile not available
    mock_torch_no_compile = MagicMock()
    del mock_torch_no_compile.compile
    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch_no_compile), patch("gemma_4_sql.backends.pytorch.benchmark.AutoModelForCausalLM", mock_auto_model), patch("gemma_4_sql.backends.pytorch.benchmark._get_device", return_value="cuda"):
        model, device = bm._load_pytorch_model_and_device("dummy", "gpu", backend_alias="pytorch_hf")
        assert model == mock_model_instance

    # test hf without to or eval
    class MockModelNoToNoEval:
        pass

    mock_auto_model_2 = MagicMock()
    mock_auto_model_2.from_pretrained.return_value = MockModelNoToNoEval()

    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch_no_compile), patch("gemma_4_sql.backends.pytorch.benchmark.AutoModelForCausalLM", mock_auto_model_2), patch("gemma_4_sql.backends.pytorch.benchmark._get_device", return_value="cuda"):
        model, device = bm._load_pytorch_model_and_device("dummy", "gpu", backend_alias="pytorch_hf")


def test_sync_cuda():
    mock_torch = MagicMock()
    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch):
        bm._sync_cuda("cuda")
        mock_torch.cuda.synchronize.assert_called_once()

        bm._sync_cuda("mps")
        mock_torch.mps.synchronize.assert_called_once()

        bm._sync_cuda("cpu")


def test_get_memory_mb():
    mock_torch = MagicMock()
    mock_torch.cuda.max_memory_allocated.return_value = 1024 * 1024 * 5
    mock_torch.mps.driver_allocated_memory.return_value = 1024 * 1024 * 10

    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch):
        assert bm._get_memory_mb(None, "cuda") == 5.0
        assert bm._get_memory_mb(None, "mps") == 10.0
        assert bm._get_memory_mb(None, "cpu") == 8192.0


def test_run_benchmark_pass():
    mock_torch = MagicMock()
    mock_tensor = MagicMock()
    mock_torch.randint.return_value = mock_tensor
    mock_tensor.to.return_value = mock_tensor

    class MockModelContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    mock_torch.no_grad.return_value = MockModelContext()

    mock_model = MagicMock()

    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch), patch("gemma_4_sql.backends.pytorch.benchmark._sync_cuda"), patch("time.time", side_effect=[0.0, 1.0]):  # 1 second duration
        # Test prefill
        tokens_per_sec, latency_ms, memory_mb = bm._run_benchmark_pass(mock_model, "cuda", 2, 5, 2, "prefill", 100)
        assert mock_model.call_count == 7  # 2 warmup + 5 runs
        assert latency_ms == (1000.0 / 5)  # 200 ms per run
        assert tokens_per_sec == 32 * 2 * 5 / 1.0

    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch), patch("gemma_4_sql.backends.pytorch.benchmark._sync_cuda"), patch("time.time", side_effect=[0.0, 2.0]):  # 2 second duration
        mock_model.reset_mock()
        # Test generation
        tokens_per_sec, latency_ms, _memory_mb = bm._run_benchmark_pass(mock_model, "cpu", 2, 5, 2, "generation", 100)
        assert mock_model.generate.call_count == 7
        assert tokens_per_sec == 100 * 2 * 5 / 2.0

    # test lacking all optional attributes but keeping no_grad
    mock_torch_no_attr = MagicMock()
    del mock_torch_no_attr.manual_seed
    del mock_torch_no_attr.cuda
    mock_torch_no_attr.no_grad.return_value = MockModelContext()

    class MockTensorNoTo:
        pass

    mock_torch_no_attr.randint.return_value = MockTensorNoTo()

    class MockModelNoGenerate:
        def __call__(self, *args, **kwargs):
            pass

    mock_model_no_gen = MockModelNoGenerate()

    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch_no_attr), patch("time.time", side_effect=[0.0, 1.0]):
        # test prefill mode
        bm._run_benchmark_pass(mock_model_no_gen, "cpu", 2, 1, 1, "prefill", 100)

    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch_no_attr), patch("time.time", side_effect=[0.0, 1.0]):
        # test generation mode without generate method
        bm._run_benchmark_pass(mock_model_no_gen, "cpu", 2, 1, 1, "generation", 100)

    # test completely lacking no_grad
    mock_torch_absolutely_no_grad = MagicMock()
    del mock_torch_absolutely_no_grad.no_grad

    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch_absolutely_no_grad), patch("time.time", side_effect=[0.0, 1.0]):
        bm._run_benchmark_pass(mock_model_no_gen, "cpu", 2, 1, 1, "generation", 100)


def test_benchmark_model():
    # Test dependencies missing
    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", None), pytest.raises(DependencyMissingError):
        bm.benchmark_model("model", "cpu", 1)

    # Test successful execution (via wrapper mock)
    mock_torch = MagicMock()
    mock_auto_model = MagicMock()
    with patch("gemma_4_sql.backends.pytorch.benchmark.torch", mock_torch), patch("gemma_4_sql.backends.pytorch.benchmark.AutoModelForCausalLM", mock_auto_model), patch("gemma_4_sql.backends.pytorch.benchmark.run_benchmark_wrapper") as mock_wrapper:

        def fake_wrapper(*args, **kwargs):
            # Extract and run the benchmark fn
            run = kwargs["benchmark_fn"]
            return run()

        mock_wrapper.side_effect = fake_wrapper

        with patch("gemma_4_sql.backends.pytorch.benchmark._load_pytorch_model_and_device") as mock_load, patch("gemma_4_sql.backends.pytorch.benchmark._run_benchmark_pass") as mock_run_pass:
            mock_load.return_value = (MagicMock(), "cpu")
            mock_run_pass.return_value = (100.0, 10.0, 500.0)

            result = bm.benchmark_model("model", "cpu", 2, num_runs=10)
            assert result == (100.0, 10.0, 500.0)
