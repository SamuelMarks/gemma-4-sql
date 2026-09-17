# `gemma-4-sql` Architecture

`gemma-4-sql` is built with a highly modular and backend-agnostic architecture. Its core philosophy is to provide a single, unified interface for building state-of-the-art Text-to-SQL models while seamlessly delegating the heavy lifting (training, execution, distributed computing) to specialized backends.

This architecture is optimized for environments spanning from local experimentation (PyTorch, Keras) to massive supercomputer scale (JAX, MaxText on Google Cloud TPUs provisioned dynamically via libscript).

---

## 1. System Components

The system is composed of several key modules:

### 1.1 CLI & SDK Layer (`src/gemma_4_sql/cli.py`, `cli_*.py`, and `sdk/`)
This is the entry point. The CLI maps shell commands directly to Python SDK functions. The SDK exposes high-level orchestration abstractions:
*   `train_from_scratch`, `pretrain`, `sft`, `posttrain`, `dpo`
*   `evaluate`, `export`, `generate`
*   `agent`, `chat`, `serve`
*   `rag`, `few_shot`, `etl`

### 1.2 The Dispatcher (`models.py`)
All high-level training commands pass through the dispatcher. Based on the `--backend` flag provided by the user, the dispatcher dynamically loads the corresponding backend module (e.g., `gemma_4_sql.backends.jax.train` or `gemma_4_sql.backends.pytorch.train`) and proxies the execution parameters.

### 1.3 Swappable Execution Backends (`backends/`)
Each backend folder (`jax`, `keras`, `maxtext`, `pytorch`) implements identical interfaces for training, exporting, and inference. This ensures that switching from a PyTorch local setup to a MaxText TPU pod is a simple flag change.

*Architectural Note: This codebase deliberately avoids abstracting or deduplicating core logic (like training loops or data loading) across different backends. The goal is to demonstrate and utilize the ecosystem-recommended, idiomatic ways of performing these tasks within each framework (e.g., using `grain` for JAX/MaxText/Keras vs. standard `DataLoaders` for PyTorch). As such, some boilerplate duplication across `backends/` is intentional and preferred over a non-standard meta-framework.*

*   **JAX (`backends/jax`)**: Built on the modern **Flax NNX** object-oriented paradigm. Houses a custom, from-scratch implementation of Gemma 4 (`backends/jax/gemma4/`) across text, vision, audio, and MoE architectures. Employs `@nnx.jit` compiled loops, `nnx.value_and_grad`, dynamic KV caching, `nnx.LoRALinear` adapters, and native Activation-aware Weight Quantization (AWQ) and uniform INT8.
*   **MaxText (`backends/maxtext`)**: Integrates directly with Google's **MaxText** AI-Hypercomputer distributed training framework (`maxtext.train` and `maxtext.models.gemma4`). Rather than Flax NNX, it operates on functional Linen/JAX parameter PyTrees, generates Gin configuration files dynamically, orchestrates multi-host TPU Pod training across multi-dimensional device meshes (`data`, `fsdp`, `tensor`), uses `MaxTextFormatTransform`, and applies Google **AQT** (`aqt.jax.v2`) for INT8/INT4 quantization.
*   **Keras (`backends/keras`)**: Integrates with `keras_nlp` (using `GemmaCausalLM`) and leverages Keras 3 core abstractions (`keras.Model.fit`), maintaining a standardized TensorFlow-compatible graph.
*   **PyTorch (`backends/pytorch`)**: Connects to the standard Hugging Face `transformers` API (`Gemma4ForCausalLM`) and provides a standalone pure Native PyTorch pipeline (`pytorch_native`) featuring `DynamicCache` KV caching, native forward/loss loops, and direct safetensors serialization.
*   **MLX (`backends/mlx`)**: Tailored for Apple Silicon hardware, providing compiled metal-accelerated training, DPO, quantization, and continuous batching server via FastAPI.

### 1.4 MaxText Exclusive vs. JAX (Flax NNX) Stack

While both backends run on top of JAX and XLA, they represent distinct architectural paradigms:

| Architectural Dimension | JAX Backend (`backends/jax`) | MaxText Backend (`backends/maxtext`) |
| :--- | :--- | :--- |
| **Paradigm / Framework** | **Flax NNX** (modern object-oriented stateful/functional) | **Google MaxText** + classic functional Linen JAX |
| **Model Source** | From-scratch internal implementation (`backends/jax/gemma4/`) | Upstream `maxtext.models.gemma4.Gemma4Model` |
| **Modality Support** | Text, Audio (Conformer/Subsample), Vision (SigLIP), MoE | Core Text-to-SQL Causal LM via MaxText |
| **Configuration** | Typed Python dataclasses (`Gemma4Config`, `TrainingConfig`) | Dynamic Gin configuration generator (`.gin` files & CLI args) |
| **Cluster Orchestration** | Local accelerators or JAX distributed host initialization | Multi-host TPU Pod orchestration via `maxtext.train.main` & device meshes |
| **Device Mesh Topology** | JAX default device placement / sharding | Multi-dimensional mesh axes (`data`, `fsdp`, `tensor`) |
| **ETL Format Transform** | Grain `BaseFormatTransform` (`inputs`, `targets`) | `MaxTextFormatTransform` (injects Seq2Seq segment IDs & positions) |
| **PEFT / LoRA** | Object-oriented `nnx.LoRALinear` module adaptation | Functional PyTree surgery on raw parameter dictionaries (`"kernel"`) |
| **Quantization** | Native JAX uniform INT8 and Activation-aware Weight Quantization (AWQ) | Google **AQT** (`aqt.jax.v2`) targeting attention & MLP projection matrices |
| **Checkpoints & Export** | Orbax checkpointing over `nnx.state(model)` | Orbax checkpointing over MaxText parameter PyTrees |

---

## 2. Core Pipelines

### 2.1 ETL Pipeline (The `grain` Integration)
Data loading at scale is a massive bottleneck. We use **Google Grain**, a high-performance dataloader built for distributed training.
1.  **Datasets:** We natively connect to Hugging Face datasets (e.g., `my-custom-dataset`, `my-custom-dpo-dataset`).
2.  **Transformations:** Datasets pass through a series of `MapTransform` pipelines.
3.  **Target Formats:** The pipeline normalizes the text into integer sequences and produces dataset shards tailored to the specific backend:
    *   **JAX / Keras:** Utilizes Grain's `BaseFormatTransform` to yield standard `inputs` and `targets` dictionaries.
    *   **MaxText:** Utilizes `MaxTextFormatTransform` to inject additional Seq2Seq features like `segment_ids` and `positions` expected by the MaxText architecture, with distributed sharding via `JAXDistributedSharding`.
    *   **PyTorch / MLX:** Yields standard `inputs` and `targets` dictionaries via native `DataLoaders` or MLX batching streams.
    *   **Configurable Batch Size:** Every training pipeline accepts unified `batch_size` settings across CLI and SDK configurations.

### 2.2 Live Database Execution Engine (`db_engine.py`)
Unlike standard NLP generation where BLEU/ROUGE are sufficient, Text-to-SQL must be measured by **Execution Accuracy (EX)**. We developed the `LiveDatabaseEngine`.
*   **Multi-Dialect Support:** It supports `sqlite3`, `psycopg2` (PostgreSQL), `snowflake-connector-python`, and `duckdb`.
*   **Execution with Feedback:** The engine executes SQL dynamically and captures database exceptions (e.g., `Syntax Error`, `Missing Column`).
*   **Sandbox Safety:** Evaluation usually takes place on an in-memory SQLite/DuckDB representation generated dynamically using the dataset's schema (`DDL`).

### 2.3 Agentic Loop / Self-Correction (`agent.py`)
Because models hallucinate schema names or misapply joins, the `agentic_loop` utilizes the `LiveDatabaseEngine` iteratively.
1.  Model generates SQL.
2.  `LiveDatabaseEngine` attempts execution.
3.  If an error occurs, the exact SQL exception is appended to the agent's prompt history.
4.  The model regenerates the SQL. This repeats until success or `max_retries` is reached.

### 2.4 RAG Contextualization & Few-Shot Building (`rag.py`, `few_shot.py`)
Providing pure prompts is insufficient. 
*   **RAG Engine:** Parses SQL `DDL` (Data Definition Language) strings. Extracts table structures, column names, and types. Based on the user's prompt, it selects the relevant schema elements and embeds them as context to ground the generation.
*   **Few-Shot Builder:** Dynamically embeds valid (Input -> SQL) mappings into the model's instruction prompt, tuning its output distribution before generation.

### 2.5 DuckDB UDF Support (`duckdb_extension.py`)
A unique architectural feature of `gemma-4-sql` is its ability to embed the LLM natively into an analytics database pipeline.
By utilizing DuckDB's Python UDF (User Defined Function) bindings, we expose `ask_gemma()` directly inside the DuckDB process, allowing queries like:
```sql
SELECT ask_gemma('Clean this string: ' || raw_column) FROM my_table;
```
This enables seamless, in-process AI augmentation without external network calls.

---

## 3. Parameter-Efficient Training (PEFT) and Quantization
To fit large models (like Gemma 4) on compute-constrained or consumer hardware, we abstract LoRA and numerical quantization:
*   **PEFT / LoRA:**
    *   **JAX:** Handled via Flax NNX module replacement using `nnx.LoRALinear` layers with configurable `lora_r` rank and scaling alpha.
    *   **MaxText:** Handled via functional parameter PyTree surgery (`transform_params_to_lora`) decomposing matched `"kernel"` projection matrices into low-rank `lora_a` and `lora_b` factors without requiring NNX state.
    *   **Keras:** Handled via native Keras LoRA APIs.
    *   **PyTorch:** Handled via Hugging Face `peft` (`LoraConfig`) or native weight decomposition.
    *   **MLX:** Handled via MLX LoRA linear layers.
*   **Quantization:**
    *   **MaxText:** Integrates Google **AQT** (`aqt.jax.v2` Accurate Quantized Training) for dynamic per-channel INT8 and INT4 quantization targeting attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP projections (`gate_proj`, `up_proj`, `down_proj`).
    *   **JAX (Flax NNX):** Provides Activation-aware Weight Quantization (AWQ) and uniform INT8 symmetric quantization based on forward-pass token activation statistics.
    *   **PyTorch:** Supports INT8, AWQ, GPTQ, and GGUF mechanisms.
    *   **MLX:** Metal-accelerated 4-bit and 8-bit group quantization.

---

## 4. Serving (`serve.py`)
Serving large batch jobs requires continuous batching. For PyTorch, this wraps the `vLLM` infrastructure. For JAX and MaxText, this spins up native optimized multi-TPU serving loops, orchestrating paged attention and token management to ensure maximum throughput under heavy API load.

---

## 5. Metrics & Monitoring (TensorBoard)

To provide a unified MLOps experience, `gemma-4-sql` abstracts metric logging through `sdk/logging.py`. Depending on the backend executed, the payload is directed to the appropriate backend integration:
*   **JAX / MaxText:** Emits metrics via `tensorboardX.SummaryWriter`.
*   **Keras:** Uses standard `tensorflow.summary` writers.
*   **PyTorch:** Uses `torch.utils.tensorboard.SummaryWriter`.

This abstraction allows user scripts and internal training loops to simply call `log_metrics(metrics={"loss": 0.5}, step=100, log_dir="logs", backend="...")` without having to implement backend-specific tensorboard graph connections.
