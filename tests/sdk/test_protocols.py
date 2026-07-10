def test_protocols():
    from gemma_4_sql.sdk.protocols import BackendProtocol

    class MockBackend(BackendProtocol):
        def train_model(self, action, model_name, dataset_name):
            return {}

        def run_dpo(self, model_name, dataset_name, beta):
            return {}

        def build_dataloader(self, dataset_name, split, batch_size, **kwargs):
            return {}

        def export_model(self, model_name, output_path):
            return {}

        def log_metrics(self, metrics, step, log_dir="logs"):
            return {}

        def apply_lora(self, model_name, target_modules, **kwargs):
            return {}

        def quantize_model(self, model_name, method="int8"):
            return {}

        def generate_sql(self, model_name, prompt, beam_width=3, max_length=50):
            return {}

        def serve_model(self, model_name, port=8000, max_batch_size=32, **kwargs):
            return {}

        def benchmark_model(self, model_name, hardware="tpu-v5p", batch_size=32):
            return {}

    b = MockBackend()
    b.train_model("", "", "")
