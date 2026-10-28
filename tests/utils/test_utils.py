"""Unit tests and shared test doubles (`RecordingBackendDouble`, `FakeDBConnection`) for `tests/utils`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from gemma_4_sql.sdk.protocols import BackendProtocol
from gemma_4_sql.type_hints import (
    BenchmarkResultDict,
    DataloaderResultDict,
    DPOConfig,
    DPOResultDict,
    ETLConfig,
    ExportResultDict,
    InferenceResultDict,
    JSONDict,
    JSONPrimitive,
    JSONValue,
    LogMetricsResultDict,
    PEFTResultDict,
    QuantizeResultDict,
    ServeResultDict,
    TrainingConfig,
    TrainResultDict,
)
from tests.utils import (
    assert_sequences_close,
    compute_snr,
    create_dummy_weight_matrix,
    create_in_memory_sqlite,
    get_sample_ddl,
    insert_sample_users,
)


class RecordingBackendDouble(BackendProtocol):
    """Stateful test double implementing `BackendProtocol` that validates call contracts and records invocations."""

    def __init__(
        self,
        backend_name: str = "jax",
        *,
        error_to_raise: Exception | None = None,
    ) -> None:
        """Initialize the recording backend double.

        Args:
            backend_name: Canonical backend name reported in returned result dicts.
            error_to_raise: Optional exception raised on any method call to test error propagation.

        """
        self.backend_name: str = backend_name
        self.error_to_raise: Exception | None = error_to_raise
        self.train_configs: list[TrainingConfig] = []
        self.train_kwargs: list[JSONDict] = []
        self.dpo_configs: list[DPOConfig] = []
        self.dpo_kwargs: list[JSONDict] = []
        self.etl_configs: list[ETLConfig] = []
        self.etl_kwargs: list[JSONDict] = []
        self.export_calls: list[tuple[str, str, JSONDict]] = []
        self.log_calls: list[tuple[dict[str, float], int, str, JSONDict]] = []
        self.lora_calls: list[tuple[str, list[str], int, int, float, JSONDict]] = []
        self.quantize_calls: list[tuple[str, str, JSONDict]] = []
        self.generate_calls: list[tuple[str, str, int, int, JSONDict]] = []
        self.serve_calls: list[tuple[str, int, int, JSONDict]] = []
        self.benchmark_calls: list[tuple[str, str, int, JSONDict]] = []

    def _check_error(self) -> None:
        """Raise `self.error_to_raise` if configured."""
        if self.error_to_raise is not None:
            raise self.error_to_raise

    def train_model(self, config: TrainingConfig, **kwargs: JSONValue) -> TrainResultDict:
        """Validate and record a `train_model` invocation."""
        self._check_error()
        if config.epochs <= 0:
            raise ValueError("epochs must be positive")
        self.train_configs.append(config)
        self.train_kwargs.append(dict(kwargs))
        return {
            "backend": self.backend_name,
            "action": config.action or "sft",
            "model": config.model_name,
            "dataset": config.dataset,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "status": "completed",
            "final_loss": round(1.0 / (config.epochs + 1), 4),
        }

    def run_dpo(self, config: DPOConfig, **kwargs: JSONValue) -> DPOResultDict:
        """Validate and record a `run_dpo` invocation."""
        self._check_error()
        if config.beta <= 0.0:
            raise ValueError("beta must be positive")
        self.dpo_configs.append(config)
        self.dpo_kwargs.append(dict(kwargs))
        return {
            "backend": self.backend_name,
            "action": "dpo",
            "model": config.model_name,
            "dataset": config.dataset,
            "beta": config.beta,
            "epochs": config.epochs,
            "status": "completed",
            "final_loss": round(0.6931 / max(1, config.epochs), 4),
        }

    def build_dataloader(self, config: ETLConfig, **kwargs: JSONValue) -> DataloaderResultDict:
        """Validate and record a `build_dataloader` invocation."""
        self._check_error()
        self.etl_configs.append(config)
        self.etl_kwargs.append(dict(kwargs))
        return {
            "backend": self.backend_name,
            "dataset": config.dataset_name,
            "split": config.split,
            "batch_size": config.batch_size,
            "distributed": config.distributed,
            "status": "ready",
            "loader": [{"inputs": [1, 2], "targets": [2, 3]}],
        }

    def export_model(self, model_name: str, export_path: str, **kwargs: JSONValue) -> ExportResultDict:
        """Validate and record an `export_model` invocation."""
        self._check_error()
        self.export_calls.append((model_name, export_path, dict(kwargs)))
        return {
            "backend": self.backend_name,
            "model": model_name,
            "export_path": export_path,
            "file_path": f"{export_path}/model.safetensors",
            "format": "safetensors",
            "status": "exported",
        }

    def log_metrics(self, metrics: dict[str, float], step: int, log_dir: str = "logs", **kwargs: JSONValue) -> LogMetricsResultDict:
        """Validate and record a `log_metrics` invocation."""
        self._check_error()
        self.log_calls.append((dict(metrics), step, log_dir, dict(kwargs)))
        return {
            "backend": self.backend_name,
            "status": "logged",
            "step": step,
            "logged_step": step,
            "log_dir": log_dir,
            "metrics": dict(metrics),
        }

    def apply_lora(
        self,
        model_name: str,
        target_modules: list[str],
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        **kwargs: JSONValue,
    ) -> PEFTResultDict:
        """Validate and record an `apply_lora` invocation."""
        self._check_error()
        self.lora_calls.append((model_name, list(target_modules), lora_r, lora_alpha, lora_dropout, dict(kwargs)))
        return {
            "backend": self.backend_name,
            "action": "lora",
            "model": model_name,
            "target_modules": list(target_modules),
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "injected_modules": len(target_modules),
            "status": "lora_applied",
        }

    def quantize_model(self, model_name: str, method: str = "int8", **kwargs: JSONValue) -> QuantizeResultDict:
        """Validate and record a `quantize_model` invocation."""
        self._check_error()
        self.quantize_calls.append((model_name, method, dict(kwargs)))
        return {
            "backend": self.backend_name,
            "model": model_name,
            "method": method,
            "memory_reduction_factor": 2.0 if method == "int8" else 4.0,
            "status": f"quantized_{method}",
        }

    def get_trainer(self, **_kwargs: JSONValue) -> str:
        """Return canonical trainer name for `self.backend_name`."""
        self._check_error()
        return f"{self.backend_name}_trainer"

    def generate_sql(
        self,
        model_name: str,
        prompt: str,
        beam_width: int = 3,
        max_length: int = 50,
        **kwargs: JSONValue,
    ) -> InferenceResultDict:
        """Validate and record a `generate_sql` invocation."""
        self._check_error()
        self.generate_calls.append((model_name, prompt, beam_width, max_length, dict(kwargs)))
        return {
            "backend": self.backend_name,
            "model": model_name,
            "prompt": prompt,
            "sql": f"SELECT * FROM {self.backend_name}_table;",
            "beam_width": beam_width,
            "confidence_score": 0.95,
            "status": "success",
        }

    def serve_model(
        self,
        model_name: str,
        port: int = 8000,
        max_batch_size: int = 32,
        **kwargs: JSONValue,
    ) -> ServeResultDict:
        """Validate and record a `serve_model` invocation."""
        self._check_error()
        self.serve_calls.append((model_name, port, max_batch_size, dict(kwargs)))
        return {
            "backend": self.backend_name,
            "model": model_name,
            "port": port,
            "max_batch_size": max_batch_size,
            "mode": "continuous_batching",
            "status": "serving",
        }

    def benchmark_model(
        self,
        model_name: str,
        hardware: str = "tpu-v5p",
        batch_size: int = 32,
        **kwargs: JSONValue,
    ) -> BenchmarkResultDict:
        """Validate and record a `benchmark_model` invocation."""
        self._check_error()
        self.benchmark_calls.append((model_name, hardware, batch_size, dict(kwargs)))
        return {
            "backend": self.backend_name,
            "model": model_name,
            "hardware": hardware,
            "batch_size": batch_size,
            "tokens_per_sec": float(batch_size * 100),
            "latency_ms": round(1000.0 / max(1, batch_size * 10), 2),
            "memory_mb": 128.0,
            "status": "completed",
        }


def test_db_helpers() -> None:
    """Test db helper functions."""
    ddl = get_sample_ddl()
    assert "CREATE TABLE users" in ddl
    conn = create_in_memory_sqlite(ddl)
    insert_sample_users(conn, [(1, "Alice", "alice@example.com")])
    cur = conn.cursor()
    res = cur.execute("SELECT name FROM users WHERE id=1;").fetchone()
    assert res[0] == "Alice"
    cur.close()
    conn.close()


def test_tensor_helpers() -> None:
    """Test tensor comparison helpers."""
    mat = create_dummy_weight_matrix(2, 3)
    assert len(mat) == 2
    assert len(mat[0]) == 3

    assert_sequences_close([1.0, 2.0], [1.0001, 1.9999], atol=1e-3)

    with pytest.raises(AssertionError, match="Length mismatch"):
        assert_sequences_close([1.0], [1.0, 2.0])

    with pytest.raises(AssertionError, match="Divergence at index"):
        assert_sequences_close([1.0], [2.0], atol=1e-5)

    snr_val = compute_snr([1.0, 2.0], [1.0, 2.0])
    assert snr_val == float("inf")

    snr_noise = compute_snr([10.0, 10.0], [9.9, 10.1])
    assert snr_noise > 0.0

    snr_zero = compute_snr([0.0, 0.0], [1.0, 1.0])
    assert snr_zero == 0.0

    with pytest.raises(ValueError, match="Sequence lengths must match"):
        compute_snr([1.0], [1.0, 2.0])


class FakeDBCursor:
    """Fake DB-API 2.0 cursor for testing."""

    def __init__(self, connection: FakeDBConnection) -> None:
        """Initialize fake cursor.

        Args:
            connection: The parent connection.

        """
        self.connection = connection
        self.arraysize = 1
        self.rowcount = -1
        self.description: bool | None = None
        self._results: list[tuple[JSONPrimitive, ...]] = []
        self._closed = False

    def execute(
        self,
        operation: str,
        parameters: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = None,
    ) -> FakeDBCursor:
        """Fake execute method.

        Args:
            operation: SQL query.
            parameters: Query parameters.

        Returns:
            Self.

        Raises:
            RuntimeError: If the cursor is closed or the query matches the failure trigger.

        """
        _ = parameters
        if self._closed:
            raise RuntimeError("Cursor closed")
        if operation == self.connection._fail_query:
            raise RuntimeError("Simulated execution failure")
        self._results = self.connection._results_queue.pop(0) if self.connection._results_queue else []
        self.rowcount = len(self._results)
        if self._results:
            self.description = True
        return self

    def fetchall(self) -> list[tuple[JSONPrimitive, ...]]:
        """Fetch all results.

        Returns:
            List of result rows.

        """
        if self._closed:
            raise RuntimeError("Cursor closed")
        return self._results

    def close(self) -> None:
        """Close the cursor."""
        self._closed = True


class FakeDBConnection:
    """Fake DB-API 2.0 connection for testing."""

    def __init__(self) -> None:
        """Initialize fake connection."""
        self._closed = False
        self._transactions_committed = 0
        self._transactions_rolled_back = 0
        self._results_queue: list[list[tuple[JSONPrimitive, ...]]] = []
        self._fail_query: str | None = None

    def cursor(self) -> FakeDBCursor:
        """Create a new fake cursor.

        Returns:
            A FakeDBCursor instance.

        """
        if self._closed:
            raise RuntimeError("Connection closed")
        return FakeDBCursor(self)

    def commit(self) -> None:
        """Commit the current transaction."""
        self._transactions_committed += 1

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self._transactions_rolled_back += 1

    def close(self) -> None:
        """Close the connection."""
        self._closed = True

    def set_results(self, results: list[list[tuple[JSONPrimitive, ...]]]) -> None:
        """Set results for subsequent queries.

        Args:
            results: List of lists containing rows to return.

        """
        self._results_queue.extend(results)

    def set_fail_query(self, query: str) -> None:
        """Configure the connection to fail when executing a specific query.

        Args:
            query: The exact SQL query that should trigger an exception.

        """
        self._fail_query = query


def test_seed_table() -> None:
    """Test seed_table."""
    from tests.utils.db_helpers import seed_table

    conn = create_in_memory_sqlite("CREATE TABLE t (a INT, b TEXT);")
    seed_table(conn, "t", ["a", "b"], [(1, "A"), (2, "B")])
    cur = conn.cursor()
    assert len(cur.execute("SELECT * FROM t").fetchall()) == 2
    cur.close()
    conn.close()


def test_fake_db() -> None:
    """Test fake DB classes."""
    conn = FakeDBConnection()
    conn.set_results([[("row1",)], [("row2",)]])
    cur = conn.cursor()
    cur.execute("SELECT 1")
    assert cur.fetchall() == [("row1",)]
    assert cur.rowcount == 1
    conn.commit()
    conn.rollback()
    assert conn._transactions_committed == 1
    assert conn._transactions_rolled_back == 1

    conn.set_fail_query("FAIL")
    with pytest.raises(RuntimeError, match="Simulated execution failure"):
        cur.execute("FAIL")

    cur.close()
    with pytest.raises(RuntimeError, match="Cursor closed"):
        cur.execute("SELECT 1")
    with pytest.raises(RuntimeError, match="Cursor closed"):
        cur.fetchall()

    conn.close()
    with pytest.raises(RuntimeError, match="Connection closed"):
        conn.cursor()


def test_multi_dialect_ddl() -> None:
    """Test multi-dialect DDL."""
    assert "SERIAL" in get_sample_ddl("postgres")
    assert "UINTEGER" in get_sample_ddl("duckdb")


def test_assert_tensors_close() -> None:
    """Test assert_tensors_close."""
    from tests.utils.tensor_helpers import assert_tensors_close

    class DummyTensor:
        """Dummy tensor for testing assert_tensors_close."""

        def __init__(self, data: list[float]) -> None:
            """Initialize DummyTensor."""
            self.data = data
            self.shape = (len(data),)

        def flatten(self) -> DummyTensor:
            """Return flattened tensor."""
            return self

        def tolist(self) -> list[float]:
            """Return tensor elements as list."""
            return self.data

    t1 = DummyTensor([1.0, 2.0])
    t2 = DummyTensor([1.0001, 1.9999])
    assert_tensors_close(t1, t2, backend="pytorch")

    with pytest.raises(ValueError, match="Unknown backend"):
        assert_tensors_close(t1, t2, backend="unknown")


def test_assert_tensors_close_jax_mlx() -> None:
    """Test assert_tensors_close for jax, mlx, and pytorch sequence backends."""
    from tests.utils.tensor_helpers import assert_tensors_close

    t1 = [[1.0, 2.0], [3.0, 4.0]]
    t2 = [[1.0, 2.0], [3.0, 4.0]]

    assert_tensors_close(t1, t2, backend="jax")
    assert_tensors_close(t1, t2, backend="mlx")
    assert_tensors_close(t1, t2, backend="pytorch")
    assert_tensors_close([1.0, 2.0], [1.0, 2.0], backend="jax")


def test_recording_backend_double() -> None:
    """Test RecordingBackendDouble conforms to BackendProtocol and records calls."""
    double = RecordingBackendDouble("jax")
    assert isinstance(double, BackendProtocol)
    assert double.get_trainer() == "jax_trainer"
    assert double.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=2))["status"] == "completed"
    with pytest.raises(ValueError, match="epochs must be positive"):
        double.train_model(TrainingConfig(action="sft", model_name="m", dataset="d", epochs=0))
    assert double.run_dpo(DPOConfig(model_name="m", dataset="d", beta=0.1))["status"] == "completed"
    with pytest.raises(ValueError, match="beta must be positive"):
        double.run_dpo(DPOConfig(model_name="m", dataset="d", beta=0.0))
    assert double.build_dataloader(ETLConfig(dataset_name="d", split="train"))["status"] == "ready"
    assert double.export_model("m", "/tmp/out")["status"] == "exported"
    assert double.log_metrics({"loss": 0.5}, 1)["status"] == "logged"
    assert double.apply_lora("m", ["q_proj"])["injected_modules"] == 1
    assert double.quantize_model("m", "int8")["memory_reduction_factor"] == 2.0
    assert double.generate_sql("m", "SELECT 1")["status"] == "success"
    assert double.serve_model("m", 8080, 16)["port"] == 8080
    assert double.benchmark_model("m", "cpu", 4)["tokens_per_sec"] == 400.0

    err_double = RecordingBackendDouble("jax", error_to_raise=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        err_double.get_trainer()
