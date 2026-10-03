from gemma_4_sql.backends.jax import __all__, get_trainer


def test_get_trainer():
    assert get_trainer() == "jax_trainer"


def test_all_exports():
    expected_exports = ["apply_lora", "benchmark_model", "build_dataloader", "export_model", "generate_sql", "get_trainer", "log_metrics", "quantize_model", "run_dpo", "serve_model", "train_model"]
    assert __all__ == expected_exports
