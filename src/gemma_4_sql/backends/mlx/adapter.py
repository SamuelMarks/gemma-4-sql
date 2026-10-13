"""MLX backend adapter."""

from typing import Any

from gemma_4_sql.sdk.protocols import BackendProtocol
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, JSONDict, JSONValue, TrainingConfig

from . import benchmark, dpo, etl, export, inference, logging, peft, quantize, serve, train


class MLXBackend(BackendProtocol):
    """MLX implementation of BackendProtocol."""

    def train_model(self, config: TrainingConfig, **kwargs: JSONValue) -> JSONDict:
        """Train a model."""
        return train.train_model(config, **kwargs)

    def run_dpo(self, config: DPOConfig, **kwargs: JSONValue) -> JSONDict:
        """Run DPO."""
        return dpo.run_dpo(config, **kwargs)

    def build_dataloader(self, config: ETLConfig, **kwargs: JSONValue) -> JSONDict:
        """Build dataloader."""
        return etl.build_dataloader(config, **kwargs)

    def export_model(self, model_name: str, export_path: str, **kwargs: JSONValue) -> JSONDict:
        """Export model."""
        return export.export_model(model_name, export_path, **kwargs)

    def log_metrics(self, metrics: dict[str, float], step: int, log_dir: str = "logs", **kwargs: JSONValue) -> JSONDict:
        """Log metrics."""
        return logging.log_metrics(metrics, step, log_dir, **kwargs)

    def apply_lora(self, model_name: str, target_modules: list[str], lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05, **kwargs: JSONValue) -> JSONDict:
        """Apply LoRA."""
        return peft.apply_lora(model_name, target_modules, lora_r, lora_alpha, lora_dropout, **kwargs)

    def quantize_model(self, model_name: str, method: str = "int8", **kwargs: JSONValue) -> JSONDict:
        """Quantize model."""
        return quantize.quantize_model(model_name, method, **kwargs)

    def generate_sql(self, model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50, **kwargs: JSONValue) -> JSONDict:
        """Generate SQL."""
        return inference.generate_sql(model_name, prompt, beam_width, max_length, **kwargs)

    def serve_model(self, model_name: str, port: int = 8000, max_batch_size: int = 32, **kwargs: JSONValue) -> JSONDict:
        """Serve model."""
        return serve.serve_model(model_name, port, max_batch_size, **kwargs)

    def benchmark_model(self, model_name: str, hardware: str = "tpu-v5p", batch_size: int = 32, **kwargs: Any) -> JSONDict:
        """Benchmark model."""
        return benchmark.benchmark_model(model_name, hardware, batch_size, **kwargs)
