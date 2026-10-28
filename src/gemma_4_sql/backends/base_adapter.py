"""Unified module-delegating backend adapter implementing BackendProtocol."""

from __future__ import annotations

from gemma_4_sql.sdk.protocols import BackendProtocol
from gemma_4_sql.type_hints import (
    BenchmarkResultDict,
    BenchmarkSubmoduleProtocol,
    DataloaderResultDict,
    DPOConfig,
    DPOResultDict,
    DPOSubmoduleProtocol,
    ETLConfig,
    ETLSubmoduleProtocol,
    ExportResultDict,
    ExportSubmoduleProtocol,
    InferenceResultDict,
    InferenceSubmoduleProtocol,
    JSONValue,
    LoggingSubmoduleProtocol,
    LogMetricsResultDict,
    PEFTResultDict,
    PEFTSubmoduleProtocol,
    QuantizeResultDict,
    QuantizeSubmoduleProtocol,
    ServeResultDict,
    ServeSubmoduleProtocol,
    TrainingConfig,
    TrainResultDict,
    TrainSubmoduleProtocol,
)


class ModuleDelegatedBackend(BackendProtocol):
    """Base backend adapter that delegates BackendProtocol calls to typed package submodules."""

    train_module: TrainSubmoduleProtocol
    dpo_module: DPOSubmoduleProtocol
    etl_module: ETLSubmoduleProtocol
    export_module: ExportSubmoduleProtocol
    logging_module: LoggingSubmoduleProtocol
    peft_module: PEFTSubmoduleProtocol
    quantize_module: QuantizeSubmoduleProtocol
    inference_module: InferenceSubmoduleProtocol
    serve_module: ServeSubmoduleProtocol
    benchmark_module: BenchmarkSubmoduleProtocol
    trainer_name: str = ""

    def __init__(self, backend_alias: str | None = None) -> None:
        """Initialize the backend adapter with an optional backend alias.

        Args:
            backend_alias: Optional alias (e.g., `'pytorch_native'`, `'pytorch_hf'`) forwarded in kwargs.

        """
        self.backend_alias = backend_alias

    def _inject_alias(self, kwargs: dict[str, JSONValue]) -> dict[str, JSONValue]:
        """Inject `backend_alias` into `kwargs` if configured and not already present.

        Args:
            kwargs: Original keyword arguments dictionary.

        Returns:
            Updated keyword arguments dictionary.

        """
        if self.backend_alias is not None and "backend_alias" not in kwargs:
            merged: dict[str, JSONValue] = dict(kwargs)
            merged["backend_alias"] = self.backend_alias
            return merged
        return kwargs

    def train_model(self, config: TrainingConfig, **kwargs: JSONValue) -> TrainResultDict:
        """Train a model according to training configuration.

        Args:
            config: TrainingConfig specifying dataset, epochs, learning rate, and strategy.
            **kwargs: Backend-specific execution overrides.

        Returns:
            Dictionary containing training status and final metrics.

        """
        return self.train_module.train_model(config, **self._inject_alias(kwargs))

    def run_dpo(self, config: DPOConfig, **kwargs: JSONValue) -> DPOResultDict:
        """Run Direct Preference Optimization (DPO).

        Args:
            config: DPOConfig specifying dataset, model, beta, epochs, and learning rate.
            **kwargs: Backend-specific execution overrides.

        Returns:
            Dictionary containing DPO training status and final metrics.

        """
        return self.dpo_module.run_dpo(config, **self._inject_alias(kwargs))

    def build_dataloader(self, config: ETLConfig, **kwargs: JSONValue) -> DataloaderResultDict:
        """Construct ETL dataloader pipeline.

        Args:
            config: ETLConfig specifying dataset name, split, batch size, and storage options.
            **kwargs: Overrides for ETL configuration.

        Returns:
            Dictionary containing dataloader objects and metadata.

        """
        return self.etl_module.build_dataloader(config, **self._inject_alias(kwargs))

    def export_model(self, model_name: str, export_path: str, **kwargs: JSONValue) -> ExportResultDict:
        """Export model weights and architecture definition.

        Args:
            model_name: Target model identifier.
            export_path: Destination path for exported artifacts.
            **kwargs: Backend-specific export options.

        Returns:
            Dictionary containing export status and target path.

        """
        return self.export_module.export_model(model_name, export_path, **self._inject_alias(kwargs))

    def log_metrics(self, metrics: dict[str, float], step: int, log_dir: str = "logs", **kwargs: JSONValue) -> LogMetricsResultDict:
        """Log training and evaluation metrics.

        Args:
            metrics: Dictionary of metric names and numeric values.
            step: Training or evaluation step index.
            log_dir: Directory path for event logs.
            **kwargs: Backend-specific logger options.

        Returns:
            Dictionary containing logging confirmation and status.

        """
        return self.logging_module.log_metrics(metrics, step, log_dir, **self._inject_alias(kwargs))

    def apply_lora(
        self,
        model_name: str,
        target_modules: list[str],
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        **kwargs: JSONValue,
    ) -> PEFTResultDict:
        """Apply Parameter-Efficient Fine-Tuning (LoRA) adapters.

        Args:
            model_name: Target model identifier.
            target_modules: Layer names to apply LoRA rank decomposition on.
            lora_r: Rank dimension of LoRA adapters.
            lora_alpha: Scaling alpha parameter.
            lora_dropout: Dropout probability for LoRA layers.
            **kwargs: Backend-specific LoRA parameters.

        Returns:
            Dictionary containing adapter configuration status.

        """
        return self.peft_module.apply_lora(model_name, target_modules, lora_r, lora_alpha, lora_dropout, **self._inject_alias(kwargs))

    def quantize_model(self, model_name: str, method: str = "int8", **kwargs: JSONValue) -> QuantizeResultDict:
        """Quantize model weights for reduced memory footprint.

        Args:
            model_name: Target model identifier.
            method: Quantization method (e.g., 'int8', 'awq', 'gptq').
            **kwargs: Backend-specific quantization options.

        Returns:
            Dictionary containing quantization status and reduction factor.

        """
        return self.quantize_module.quantize_model(model_name, method, **self._inject_alias(kwargs))

    def generate_sql(
        self,
        model_name: str,
        prompt: str,
        beam_width: int = 3,
        max_length: int = 50,
        **kwargs: JSONValue,
    ) -> InferenceResultDict:
        """Generate a SQL query from natural language input.

        Args:
            model_name: Target model identifier.
            prompt: Input text prompt or schema context.
            beam_width: Number of beams in beam search.
            max_length: Maximum generation token length.
            **kwargs: Backend-specific generation options.

        Returns:
            Dictionary containing generated SQL and confidence metrics.

        """
        return self.inference_module.generate_sql(model_name, prompt, beam_width, max_length, **self._inject_alias(kwargs))

    def serve_model(
        self,
        model_name: str,
        port: int = 8000,
        max_batch_size: int = 32,
        **kwargs: JSONValue,
    ) -> ServeResultDict:
        """Serve model via continuous batching API endpoint.

        Args:
            model_name: Target model identifier.
            port: Network port to host the server on.
            max_batch_size: Maximum continuous batch size.
            **kwargs: Underlying server and backend-specific configuration options.

        Returns:
            Dictionary containing server status and instance details.

        """
        return self.serve_module.serve_model(model_name, port, max_batch_size, **self._inject_alias(kwargs))

    def benchmark_model(
        self,
        model_name: str,
        hardware: str = "tpu-v5p",
        batch_size: int = 32,
        **kwargs: JSONValue,
    ) -> BenchmarkResultDict:
        """Run latency, throughput, and memory benchmarking.

        Args:
            model_name: Target model identifier.
            hardware: Target hardware device ('gpu', 'tpu-v5p', 'cpu').
            batch_size: Batch size for benchmarking.
            **kwargs: Additional benchmark options like warmup_steps or num_runs.

        Returns:
            Dictionary containing throughput (tokens/sec), latency (ms), and memory (MB).

        """
        return self.benchmark_module.benchmark_model(model_name, hardware, batch_size, **self._inject_alias(kwargs))

    def get_trainer(self, **_kwargs: JSONValue) -> str:
        """Return the canonical trainer identifier for this backend.

        Args:
            **_kwargs: Optional backend-specific keyword arguments.

        Returns:
            Trainer identifier string (e.g., `'jax_trainer'`).

        """
        return self.trainer_name


__all__ = ["ModuleDelegatedBackend"]
