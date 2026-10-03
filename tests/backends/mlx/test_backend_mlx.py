from gemma_4_sql.backends.mlx import __all__, get_trainer


def test_get_trainer():
    assert get_trainer() == "mlx_trainer"


def test_all_exports():
    expected_exports = ["apply_lora", "benchmark_model", "build_dataloader", "export_model", "generate_sql", "get_trainer", "log_metrics", "quantize_model", "run_dpo", "serve_model", "train_model"]
    for exp in expected_exports:
        assert exp in __all__
