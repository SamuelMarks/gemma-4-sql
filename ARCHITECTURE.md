# `gemma-4-sql` Architecture

`gemma-4-sql` is built with a highly modular and backend-agnostic architecture. Its core philosophy is to provide a single, unified interface for building state-of-the-art Text-to-SQL models while seamlessly delegating the heavy lifting (training, execution, distributed computing) to specialized backends.

This architecture is optimized for environments spanning from local experimentation (PyTorch, Keras, Apple Silicon MLX) to massive supercomputer scale (JAX, MaxText on Google Cloud TPUs provisioned dynamically via libscript).

---

## 1. System Components

The system is composed of several key modules:

### 1.1 CLI & SDK Layer (`src/gemma_4_sql/cli.py`, `cli_*.py`, `cli_parsers/`, and `sdk/`)
This is the entry point. The CLI maps shell commands directly to Python SDK functions. The SDK exposes high-level orchestration abstractions:
*   `train_from_scratch`, `pretrain_model`, `sft_model`, `posttrain_model`, `run_dpo`
*   `evaluate`, `export_model`, `generate` (and `generate_sql`), `benchmark`
*   `run_agentic_loop`, `chat_turn`, `serve_model`, `apply_peft`, `quantize_model`, `log_metrics`
*   `build_rag_prompt`, `extract_schema_entities`, `retrieve_relevant_schema`, `build_few_shot_prompt`, `generate_few_shot_sql`, `etl_pretrain`, `etl_sft`, `etl_posttrain`, `embed_in_duckdb`

### 1.2 The Backend Registry & Dispatcher (`sdk/registry.py`, `sdk/models.py`, and `backends/base_adapter.py`)
All SDK operations route through `get_backend(name)` in `src/gemma_4_sql/sdk/registry.py`. Based on the `--backend` flag or configuration (`jax`, `keras`, `maxtext`, `pytorch`, `pytorch_hf`, `pytorch_native`, `mlx`), the registry resolves the backend's `adapter.py` module (`ADAPTER_CLASS: type[BackendProtocol]`), instantiates its `ModuleDelegatedBackend` adapter, and caches the validated `BackendProtocol` instance thread-safely.

### 1.3 Swappable Execution Backends (`backends/`)
Each backend folder (`jax`, `keras`, `maxtext`, `pytorch`, `mlx`) implements `BackendProtocol` across training, DPO, ETL, PEFT, quantization, exporting, logging, benchmarking, serving, and inference. Shared orchestration, numerical calibration, continuous batching, and configuration schemas live in `backends/common_*.py` (`common_train.py`, `common_dpo.py`, `common_data.py`, `common_inference.py`, `common_quantize.py`, `common_serve.py`, `common_benchmark.py`, `common_export.py`, `common_logging.py`, `common_multimodal.py`, `common_peft.py`, `common_gemma4_config.py`), while framework-idiomatic execution remains native to each backend:

*   **JAX (`backends/jax`)**: Built on the modern **Flax NNX** object-oriented paradigm. Houses a custom, from-scratch implementation of Gemma 4 (`backends/jax/gemma4/`) across text, vision, audio, and MoE architectures. Employs `@nnx.jit` compiled loops, `nnx.value_and_grad`, dynamic KV caching, `nnx.LoRALinear` adapters, and native Activation-aware Weight Quantization (AWQ) and uniform INT8.
*   **MaxText (`backends/maxtext`)**: Integrates directly with Google's **MaxText** AI-Hypercomputer distributed training framework (`maxtext.train` and `maxtext.models.gemma4`). Rather than Flax NNX, it operates on functional Linen/JAX parameter PyTrees, generates Gin configuration files dynamically, orchestrates multi-host TPU Pod training across multi-dimensional device meshes (`data`, `fsdp`, `tensor`), uses `MaxTextFormatTransform`, and applies Google **AQT** (`aqt.jax.v2`) for INT8/INT4 quantization.
*   **Keras (`backends/keras`)**: Integrates with `keras_nlp` (using `GemmaCausalLM`) and leverages Keras 3 core abstractions (`keras.Model.fit`, `keras.ops`, dtype policies, and `.keras` v3 serialization).
*   **PyTorch (`backends/pytorch`)**: Supports both the standard Hugging Face `transformers` stack (`pytorch` / `pytorch_hf` via `Gemma4ForCausalLM`) and a standalone pure Native PyTorch implementation (`pytorch_native` in `backends/pytorch/gemma4/` covering text, vision, audio, MoE, RoPE, and `DynamicCache` KV caching), plus native GGUF binary serialization (`backends/pytorch/gguf.py`) and `safetensors` export.
*   **MLX (`backends/mlx`)**: Tailored for Apple Silicon hardware, providing compiled Metal-accelerated training (`mx.compile`), DPO, LoRA (`MLXLoRALinear`), 4-bit/8-bit/AWQ/GPTQ quantization, and continuous batching serving via FastAPI.

### 1.4 MaxText Exclusive vs. JAX (Flax NNX) Stack

While both backends run on top of JAX and XLA, they represent distinct architectural paradigms:

| Architectural Dimension | JAX Backend (`backends/jax`) | MaxText Backend (`backends/maxtext`) |
| :--- | :--- | :--- |
| **Paradigm / Framework** | **Flax NNX** (modern object-oriented stateful/functional) | **Google MaxText** + classic functional Linen JAX |
| **Model Source** | From-scratch internal implementation (`backends/jax/gemma4/`) | Upstream `maxtext.models.gemma4.Gemma4Model` |
| **Modality Support** | Text, Audio (Conformer/Subsample), Vision (SigLIP), MoE | Core Text-to-SQL Causal LM via MaxText |
| **Configuration** | Typed Python dataclasses (`ModelConfig`, `TrainingConfig`) | Dynamic Gin configuration generator (`.gin` files & CLI args) |
| **Cluster Orchestration** | Local accelerators or JAX distributed host initialization | Multi-host TPU Pod orchestration via `maxtext.train.main` & device meshes |
| **Device Mesh Topology** | JAX default device placement / sharding | Multi-dimensional mesh axes (`data`, `fsdp`, `tensor`) |
| **ETL Format Transform** | Grain `BaseFormatTransform` (`inputs`, `targets`) | `MaxTextFormatTransform` (injects Seq2Seq segment IDs & positions) |
| **PEFT / LoRA** | Object-oriented `nnx.LoRALinear` module adaptation | Functional PyTree surgery on raw parameter dictionaries (`"kernel"`) |
| **Quantization** | Native JAX uniform INT8 and Activation-aware Weight Quantization (AWQ) | Google **AQT** (`aqt.jax.v2`) targeting attention & MLP projection matrices |
| **Checkpoints & Export** | Orbax checkpointing over `nnx.state(model)` | Orbax checkpointing over MaxText parameter PyTrees |

---

## 2. Core Pipelines

### 2.1 ETL Pipeline (The `grain` Integration)
Data loading at scale is a massive bottleneck. We use **Google Grain**, a high-performance dataloader built for distributed training, alongside native framework loaders.
1.  **Datasets:** Connects to Hugging Face datasets (e.g., `my-custom-dataset`, `my-custom-dpo-dataset`) or directly to DuckDB tables (`duckdb_path`, `duckdb_table`).
2.  **Transformations:** Datasets pass through composable `BaseFormatTransform` and `MaxTextFormatTransform` pipelines in `backends/common_data.py`.
3.  **Target Formats:** The pipeline normalizes the text (and optional multimodal image/audio inputs) into integer sequences and produces dataset shards tailored to the specific backend:
    *   **JAX / Keras:** Utilizes Grain's `BaseFormatTransform` to yield standard `inputs` and `targets` dictionaries.
    *   **MaxText:** Utilizes `MaxTextFormatTransform` to inject additional Seq2Seq features like `segment_ids` and `positions` expected by the MaxText architecture, with distributed sharding via `JAXDistributedSharding`.
    *   **PyTorch / MLX:** Yields standard `inputs` and `targets` dictionaries via native PyTorch `DataLoader` or MLX batching streams.
    *   **Configurable Batch Size:** Every training and DPO pipeline accepts unified `batch_size` settings across CLI and SDK configurations.

### 2.2 Live Database Execution Engine (`sdk/db_engine.py` & `sdk/adapters/`)
Unlike standard NLP generation where BLEU/ROUGE are sufficient, Text-to-SQL must be measured by **Execution Accuracy (EX)**. We developed `LiveDatabaseEngine` backed by pluggable synchronous and asynchronous adapters in `src/gemma_4_sql/sdk/adapters/`:
*   **Multi-Dialect Sync & Async Support:**
    *   **SQLite (`sqlite_adapter.py`):** Synchronous `sqlite3` and asynchronous `aiosqlite` with thread/async locking for shared in-memory URIs.
    *   **PostgreSQL (`postgres_adapter.py`):** Synchronous `psycopg2` and asynchronous `asyncpg`.
    *   **Snowflake (`snowflake_adapter.py`):** `snowflake-connector-python` with async worker offloading.
    *   **DuckDB (`duckdb_adapter.py`):** In-process `duckdb` with thread-safe execution.
*   **Read-Only Lexer Validation (`sdk/sql_lexer.py`):** Validates queries in read-only mode to block destructive statements, writable CTEs (`WITH ... DELETE/UPDATE/INSERT`), `SELECT ... INTO`, and file/extension commands (`COPY`, `EXPORT`, `INSTALL`, `LOAD`).
*   **Execution with Feedback:** Executes SQL dynamically and captures database exceptions (e.g., `Syntax Error`, `Missing Column`) for self-correction.

### 2.3 Agentic Loop / Self-Correction (`sdk/agent.py`)
Because models hallucinate schema names or misapply joins, `run_agentic_loop` (and `run_agentic_loop_async`) utilizes `LiveDatabaseEngine` iteratively:
1.  Model generates SQL (incorporating text prompt, DDL, and optional vision/audio context).
2.  `LiveDatabaseEngine` attempts execution.
3.  If an error occurs, the exact SQL exception is appended to the agent's prompt history.
4.  The model regenerates the SQL. This repeats until success or `max_retries` is reached.

### 2.4 RAG Contextualization & Few-Shot Building (`sdk/rag.py`, `sdk/few_shot.py`)
Providing pure prompts is insufficient:
*   **RAG Engine:** Parses SQL `DDL` (Data Definition Language) strings, extracts table structures, column names, and types, and selects the relevant schema elements based on token overlap with the user's prompt.
*   **Few-Shot Builder:** Dynamically selects and embeds relevant `(input -> SQL)` examples into the model's instruction prompt prior to generation.

### 2.5 DuckDB UDF Support (`sdk/duckdb_extension.py`)
A unique architectural feature of `gemma-4-sql` is its ability to embed the LLM natively into an analytics database pipeline.
By utilizing DuckDB's Python UDF (User Defined Function) bindings, we expose `ask_gemma()` directly inside the DuckDB process, allowing queries like:
```sql
SELECT ask_gemma('Show total revenue for 2024');
```
This enables seamless, in-process AI query generation and execution without external network calls.

---

## 3. Parameter-Efficient Training (PEFT) and Quantization
To fit large models (like Gemma 4) on compute-constrained or consumer hardware, we abstract LoRA and numerical quantization:
*   **PEFT / LoRA:**
    *   **JAX:** Handled via Flax NNX module replacement using `nnx.LoRALinear` layers with configurable `lora_r` rank and scaling alpha.
    *   **MaxText:** Handled via functional parameter PyTree surgery (`transform_params_to_lora`) decomposing matched `"kernel"` projection matrices into low-rank `lora_a` and `lora_b` factors without requiring NNX state.
    *   **Keras:** Handled via native KerasNLP `enable_lora` APIs and custom `KerasLoRALayer` adapters.
    *   **PyTorch:** Handled via Hugging Face `peft` (`LoraConfig`) or native `NativeLoRALinear` module replacement (`add_module`).
    *   **MLX:** Handled via `MLXLoRALinear` adapter layers on Apple Silicon.
*   **Quantization:**
    *   **MaxText:** Integrates Google **AQT** (`aqt.jax.v2` Accurate Quantized Training) for dynamic per-channel INT8 and INT4 quantization targeting attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP projections (`gate_proj`, `up_proj`, `down_proj`).
    *   **JAX (Flax NNX):** Provides Activation-aware Weight Quantization (AWQ) and uniform INT8 symmetric quantization based on forward-pass token activation statistics.
    *   **Keras:** Supports INT8 and INT4 global dtype policies (`int8_from_float32`, `int4_from_float32`), symmetric layer weight quantization, and `.keras` artifact export.
    *   **PyTorch:** Supports INT8 and INT4 (BitsAndBytes / native), AWQ, GPTQ (second-order Hessian error compensation), and native GGUF binary serialization.
    *   **MLX:** Metal-accelerated INT8 and INT4 group quantization (`mlx.nn.quantize`), AWQ channel-scale calibration, and GPTQ Hessian error compensation.

---

## 4. Serving (`sdk/serve.py` & `backends/common_serve.py`)
Serving high-concurrency workloads uses a unified FastAPI continuous-batching queue architecture (`backends/common_serve.py`). Incoming `/generate` requests are coalesced across an asynchronous batching window up to `max_batch_size` and dispatched to the active backend (`jax`, `keras`, `maxtext`, `mlx`, or `pytorch`), with optional `vLLM` engine acceleration (`use_vllm=True`) on the PyTorch backend.

---

## 5. Metrics & Monitoring (TensorBoard)

To provide a unified MLOps experience, `gemma-4-sql` abstracts metric logging through `sdk/logging.py` and `backends/common_logging.py`. Depending on the backend executed, the payload is directed to the appropriate writer (with automatic JSONL fallback to `metrics.jsonl` when optional summary writer packages are absent):
*   **JAX / MaxText:** Emits metrics via `tensorboardX.SummaryWriter` (or `torch.utils.tensorboard.SummaryWriter`).
*   **Keras:** Uses `tensorflow.summary.create_file_writer`.
*   **PyTorch / MLX:** Uses `torch.utils.tensorboard.SummaryWriter`.

This abstraction allows user scripts and internal training loops to simply call `log_metrics(metrics={"loss": 0.5}, step=100, log_dir="logs", backend="...")` without having to manage backend-specific TensorBoard writer lifecycles.
