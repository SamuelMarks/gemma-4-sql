gemma-4-sql
===========

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0) <!-- badges --> ![Test coverage](https://img.shields.io/badge/Test%20coverage-100%25-brightgreen) ![Doc coverage](https://img.shields.io/badge/Doc%20coverage-100%25-brightgreen) <!-- /badges -->
[![CI](https://github.com/SamuelMarks/gemma-4-sql/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/gemma-4-sql/actions/workflows/ci.yml)

Natural text to SQL with Gemma 4; with [DuckDB](https://duckdb.org/) support and swappable backends: [PyTorch](https://pytorch.org/) (HF); [PyTorch](https://pytorch.org/) (Native); [Keras](https://keras.io/); [JAX](https://jax.readthedocs.io/); [JAX](https://jax.readthedocs.io/) / [MaxText](https://github.com/AI-Hypercomputer/maxtext); [MLX](https://github.com/ml-explore/mlx) (Apple Silicon).

**Documentation:**
- [Extending / Custom Backends](EXTENDING.md)
- [Deployment & CI/CD](DEPLOY.md)
- [DuckDB Support](DUCKDB_SUPPORT.md)
- [SQL Dataset Analysis](SQL_DATASET_ANALYSIS.md)
- [Architecture Details](ARCHITECTURE.md)
- [Usage Guide](USAGE.md)


`gemma-4-sql` is a comprehensive SDK and CLI framework designed for the end-to-end Text-to-SQL lifecycle with Gemma 4. It unifies dataset ingestion (via Google's [`grain`](https://github.com/google/grain) and native data loaders), pre-training, supervised fine-tuning (SFT), and preference optimization (DPO) across six swappable backends—spanning local Apple Silicon ([MLX](https://github.com/ml-explore/mlx)), [PyTorch](https://pytorch.org/), [Keras 3](https://keras.io/), and distributed AI-Hypercomputer TPUs ([JAX](https://jax.readthedocs.io/) and [MaxText](https://github.com/AI-Hypercomputer/maxtext)). Beyond training, it delivers production-ready inference and continuous serving, schema-aware RAG, an agentic self-correcting execution loop against live databases ([SQLite](https://docs.python.org/3/library/sqlite3.html), [PostgreSQL](https://www.postgresql.org/docs/), [Snowflake](https://docs.snowflake.com/), [DuckDB](https://duckdb.org/docs/)), and an embedded DuckDB UDF extension for in-database analytics.

## System Architecture

```mermaid
flowchart TD
    subgraph UI["Interfaces & Entry Points"]
        CLI["CLI Tooling (`gemma-4-sql`)"]
        SDK["Python SDK (`gemma_4_sql.sdk`)"]
        DUCK_UDF["DuckDB UDF (`ask_gemma`)"]
    end

    subgraph CORE["Core Orchestration & Pipelines"]
        ETL["ETL & Tokenization<br/>(Google Grain / Native Loaders)"]
        RAG["Schema RAG & Few-Shot<br/>(DDL Parsing & Linking)"]
        AGENT["Agentic Self-Correction Loop<br/>(Iterative Error Feedback)"]
        SERVE["Continuous Serving & Inference<br/>(FastAPI / Dynamic Caching / Beam)"]
    end

    subgraph DISPATCH["Unified Dispatcher Layer"]
        DISPATCHER["Backend Dispatcher<br/>(`models.py`)"]
    end

    subgraph BACKENDS["Swappable Execution Backends"]
        direction TB
        PT_HF["PyTorch (HF Transformers)"]
        PT_NAT["PyTorch (Native DynamicCache)"]
        KERAS["Keras 3 Backend"]
        JAX_NNX["JAX (Flax NNX)"]
        MAXTEXT["MaxText (AI-Hypercomputer TPU)"]
        MLX["MLX (Apple Silicon)"]
    end

    subgraph DB["Live Database Engine & Validation"]
        DB_ENGINE["LiveDatabaseEngine"]
        SQLITE[("SQLite")]
        PG[("PostgreSQL")]
        SNOW[("Snowflake")]
        DUCKDB[("DuckDB")]
    end

    UI --> CORE
    CORE --> DISPATCHER
    DISPATCHER --> BACKENDS

    AGENT <--> DB_ENGINE
    DB_ENGINE --> SQLITE
    DB_ENGINE --> PG
    DB_ENGINE --> SNOW
    DB_ENGINE --> DUCKDB
```

We explicitly integrate with and support the following Gemma 4 model architectures across different ecosystems:
*   **PyTorch (HF)**: Directly imports and uses [`Gemma4ForCausalLM`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4) from **[Hugging Face Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4)**;
*   **PyTorch (Native)**: A custom, from-scratch implementation of Gemma 4 built with **Native PyTorch** ([`torch.nn`](https://pytorch.org/docs/stable/nn.html)) featuring [`DynamicCache`](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.DynamicCache) KV generation and direct [`safetensors`](https://huggingface.co/docs/safetensors/index) serialization;
*   **MaxText**: Integrates with Google's **[AI-Hypercomputer MaxText](https://github.com/AI-Hypercomputer/maxtext)** framework; dynamically generates [Gin configs](https://github.com/google/gin-config), orchestrates multi-host TPU Pod training ([`maxtext.train`](https://github.com/AI-Hypercomputer/maxtext)) over multi-dimensional device meshes, uses Google **AQT** ([`aqt.jax.v2`](https://github.com/google/aqt)) for INT8/INT4 quantization, and operates on functional [Linen](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/index.html)/[JAX](https://jax.readthedocs.io/en/latest/) parameter [PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html);
*   **JAX**: A custom, from-scratch implementation of the full Gemma 4 architecture built with **[Flax NNX](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/index.html)** featuring [`@nnx.jit`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.jit) execution, [`nnx.value_and_grad`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.value_and_grad), dynamic KV caching, [`nnx.LoRALinear`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html) adapters, and native [Activation-aware Weight Quantization (AWQ)](https://github.com/casper-hansen/AutoAWQ) and uniform INT8;
*   **Keras**: Directly imports and uses [`GemmaCausalLM`](https://keras.io/api/keras_nlp/models/gemma/gemma_causal_lm/) from **[KerasNLP](https://keras.io/keras_nlp/)**;
*   **MLX**: Native Apple Silicon implementation for optimized training, DPO, quantization, and continuous serving on macOS using [MLX](https://ml-explore.github.io/mlx/build/html/index.html).

### Feature Support Matrix

| Feature | PyTorch (HF) | PyTorch (Native) | Keras 3 Backend | JAX | MaxText | MLX |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ETL (Data Loading)** | ✅ Native [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) | ✅ Native [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) | ✅ [Grain](https://github.com/google/grain) + `BaseFormatTransform` | ✅ [Grain](https://github.com/google/grain) + `BaseFormatTransform` | ✅ [Grain](https://github.com/google/grain) + `MaxTextFormatTransform` | ✅ Native Batching |
| **Training (Fit/JIT)** | ✅ [`Gemma4ForCausalLM`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4) | ✅ Native [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) | ✅ [`keras.Model.fit()`](https://keras.io/api/models/model_training_apis/#fit-method) | ✅ [`@nnx.jit`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.jit) loop | ✅ [`@jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) / [`maxtext.train`](https://github.com/AI-Hypercomputer/maxtext) | ✅ [`mx.compile`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html) |
| **PEFT / LoRA** | ✅ [`peft`](https://huggingface.co/docs/peft/index) | ✅ [`peft`](https://huggingface.co/docs/peft/index) | ✅ Native [Keras LoRA](https://keras.io/api/keras_nlp/models/gemma/gemma_causal_lm/#enable_lora-method) | ✅ Flax NNX [`LoRALinear`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html) | ✅ Functional [PyTree](https://jax.readthedocs.io/en/latest/pytrees.html) LoRA | ✅ [MLX LoRA](https://github.com/ml-explore/mlx-examples/tree/main/lora) |
| **Quantization** | ✅ [AWQ](https://github.com/casper-hansen/AutoAWQ) / [BitsAndBytes](https://huggingface.co/docs/bitsandbytes/main/en/index) | ✅ Native INT8 / [AWQ](https://github.com/casper-hansen/AutoAWQ) | ✅ [Keras INT8](https://keras.io/api/quantizers/) | ✅ Native [AWQ](https://github.com/casper-hansen/AutoAWQ) / INT8 | ✅ Google [AQT](https://github.com/google/aqt) (INT8/INT4) | ✅ 4-bit / 8-bit Group |
| **Inference (Beam)** | ✅ Tensor-based Search | ✅ [`DynamicCache`](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.DynamicCache) Search | ✅ TF Native Search | ✅ Compiled [`argsort`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.argsort.html) | ✅ Compiled [`argsort`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.argsort.html) | ✅ MLX Greed/Beam |
| **Evaluation (DB)** | ✅ Live [`sqlite3`](https://docs.python.org/3/library/sqlite3.html) Loop | ✅ Live [`sqlite3`](https://docs.python.org/3/library/sqlite3.html) Loop | ✅ Live [`sqlite3`](https://docs.python.org/3/library/sqlite3.html) Loop | ✅ Live [`sqlite3`](https://docs.python.org/3/library/sqlite3.html) Loop | ✅ Live [`sqlite3`](https://docs.python.org/3/library/sqlite3.html) Loop | ✅ Live [`sqlite3`](https://docs.python.org/3/library/sqlite3.html) Loop |
| **Export (Ckpt)** | ✅ [`safetensors`](https://huggingface.co/docs/safetensors/index) | ✅ [`safetensors`](https://huggingface.co/docs/safetensors/index) | ✅ [`.keras` v3 format](https://keras.io/api/models/model_saving_apis/) | ✅ [`orbax`](https://orbax.readthedocs.io/en/latest/) Checkpointer | ✅ [`orbax`](https://orbax.readthedocs.io/en/latest/) Checkpointer | ✅ [`safetensors`](https://huggingface.co/docs/safetensors/index) |
| **Agentic Loop** | ✅ Self-Correction | ✅ Self-Correction | ✅ Self-Correction | ✅ Self-Correction | ✅ Self-Correction | ✅ Self-Correction |

*Note on ETL differences:* JAX, MaxText, and Keras all leverage Google's [`grain`](https://github.com/google/grain) library. While JAX and Keras use a shared `BaseFormatTransform` yielding standard `inputs` and `targets`, MaxText uses `MaxTextFormatTransform` to inject additional Seq2Seq features like `segment_ids` and `positions` expected by the MaxText architecture. Distributed environments use [`JAXDistributedSharding`](https://github.com/google/grain). All training commands support configurable `--batch-size` arguments.

### CLI Output & Interaction

All CLI commands support direct standard output (stdout) reporting:
*   `gemma-4-sql generate --prompt "..."`: Prints clean SQL directly to stdout.
*   `gemma-4-sql agent --prompt "..."`: Emits JSON execution status, retry count, and final verified SQL.
*   `gemma-4-sql chat --prompt "..."`: Prints multi-turn conversational SQL assistant response.
*   `gemma-4-sql evaluate --dataset "..."`: Displays formatted evaluation metrics table.
*   `gemma-4-sql tokenize --encode "..."`: Emits JSON token IDs array.
*   `gemma-4-sql execute --query "..."`: Outputs query execution results and error status.

## Documentation & Usage

For full instructions on Installation, Development, ETL, Training, Inference, and other workflows, please see the **[Usage Guide](USAGE.md)**.

For a comprehensive guide on running these training scripts across distributed infrastructure (like Google Cloud TPU VMs) using MaxText or JAX, please refer to the **[DEPLOY_TO_TPU.md](./DEPLOY_TO_TPU.md)** file.

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
