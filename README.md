gemma-4-sql
===========

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0) <!-- badges --> ![Test coverage](https://img.shields.io/badge/Test%20coverage-100%25-brightgreen) ![Doc coverage](https://img.shields.io/badge/Doc%20coverage-100%25-brightgreen) <!-- /badges -->

Natural text to SQL with Gemma 4; with DuckDB support and swappable-backends: PyTorch; Keras ; JAX / Bonsai; JAX / MaxText.

**Documentation:**
- [Extending / Custom Backends](EXTENDING.md)
- [Deployment & CI/CD](DEPLOY.md)
- [DuckDB Support](DUCKDB_SUPPORT.md)
- [SQL Dataset Analysis](SQL_DATASET_ANALYSIS.md)
- [Architecture Details](ARCHITECTURE.md)
- [Usage Guide](USAGE.md)


`gemma-4-sql` is a specialized SDK and CLI tool designed for orchestrating Text-to-SQL training pipelines. It provides an end-to-end framework capable of ingesting diverse Text-to-SQL datasets, transforming them using Google's `grain` library into consistent multidimensional formats, and preparing them for modern AI-Hypercomputer workloads.

We explicitly integrate with and support the following Gemma 4 model architectures across different ecosystems:
*   **PyTorch**: Directly imports and uses `Gemma4ForCausalLM` from **[Hugging Face Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4)**;
*   **MaxText**: Directly imports and uses `Gemma4Model` from **[AI-Hypercomputer MaxText](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/models/gemma4.py)**;
*   **JAX**: Directly imports and uses `BonsaiModel` from **[bonsai](https://github.com/jax-ml/bonsai)** (specifically at my [fork with a brand new Gemma 4 implementation](https://github.com/SamuelMarks/bonsai/tree/gemma4));
*   **Keras**: Supports generic [Keras 3](https://keras.io) workflows.

### Feature Support Matrix

| Feature | PyTorch Backend | Keras 3 Backend | JAX (Bonsai) | MaxText |
| :--- | :--- | :--- | :--- | :--- |
| **ETL (Data Loading)** | ✅ Native `DataLoader` | ✅ Grain + `KerasTupleTransform` | ✅ Grain + `JAXFormatTransform` | ✅ Grain + `MaxTextFormatTransform` |
| **Training (Fit/JIT)** | ✅ `Gemma4ForCausalLM` | ✅ `keras.Model.fit()` | ✅ `@nnx.jit` loop | ✅ `@jax.jit` loop |
| **PEFT / LoRA** | ✅ `peft` | ✅ Native Keras | ✅ `optax` | ✅ Native JAX |
| **Inference (Beam)** | ✅ Tensor-based Search | ✅ TF Native Search | ✅ Compiled `argsort` | ✅ Compiled `argsort` |
| **Evaluation (DB)** | ✅ Live `sqlite3` Loop | ✅ Live `sqlite3` Loop | ✅ Live `sqlite3` Loop | ✅ Live `sqlite3` Loop |
| **Export (Ckpt)** | ✅ `safetensors` | ✅ `.keras` v3 format | ✅ `orbax` Checkpointer | ✅ `orbax` Checkpointer |
| **Agentic Loop** | ✅ Self-Correction | ✅ Self-Correction | ✅ Self-Correction | ✅ Self-Correction |

*Note on ETL differences:* Both JAX and MaxText leverage Google's `grain` library with `JAXDistributedSharding`. However, JAX uses `JAXFormatTransform` (yielding standard `inputs` and `targets`), while MaxText uses `MaxTextFormatTransform` to inject additional Seq2Seq features like `segment_ids` and `positions` expected by the MaxText architecture.

## Documentation & Usage

For full instructions on Installation, Development, ETL, Training, Inference, and other workflows, please see the **[Usage Guide](USAGE.md)**.

For a comprehensive guide on running these training scripts across distributed infrastructure (like Google Cloud TPU VMs) using MaxText or JAX, please refer to the **[DEPLOY.md](./DEPLOY.md)** file.

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
