# `gemma-4-sql` Architecture

`gemma-4-sql` is built with a highly modular and backend-agnostic architecture. Its core philosophy is to provide a single, unified interface for building state-of-the-art Text-to-SQL models while seamlessly delegating the heavy lifting (training, execution, distributed computing) to specialized backends.

This architecture is optimized for environments spanning from local experimentation (PyTorch, Keras) to massive supercomputer scale (JAX, MaxText on Google Cloud TPUs).

---

## 1. System Components

The system is composed of several key modules:

### 1.1 CLI & SDK Layer (`src/gemma_4_sql/cli.py` and `sdk/`)
This is the entry point. The CLI maps shell commands directly to Python SDK functions. The SDK exposes high-level orchestration abstractions:
*   `train_from_scratch`, `pretrain`, `sft`, `posttrain`, `dpo`
*   `evaluate`, `export`, `generate`
*   `agent`, `chat`, `serve`
*   `rag`, `few_shot`, `etl`

### 1.2 The Dispatcher (`models.py`)
All high-level training commands pass through the dispatcher. Based on the `--backend` flag provided by the user, the dispatcher dynamically loads the corresponding backend module (e.g., `gemma_4_sql.backends.jax.train` or `gemma_4_sql.backends.pytorch.train`) and proxies the execution parameters.

### 1.3 Swappable Execution Backends (`backends/`)
Each backend folder (`jax`, `keras`, `maxtext`, `pytorch`) implements identical interfaces for training, exporting, and inference. This ensures that switching from a PyTorch local setup to a MaxText TPU pod is a simple flag change.

*   **JAX (`backends/jax`)**: Uses Google's `jax` and `optax`. Integrates directly with the built-in NNX Gemma 4 implementation. Handles `@jax.jit` compiled loops.
*   **MaxText (`backends/maxtext`)**: Integrates with Google's AI-Hypercomputer stack. Uses the `Gemma4Model` written in pure XLA to leverage TPU interconnects natively.
*   **Keras (`backends/keras`)**: Uses Keras 3 core abstractions (`keras.Model.fit`), maintaining a standardized TensorFlow-compatible graph.
*   **PyTorch (`backends/pytorch`)**: Connects to the standard Hugging Face `transformers` API (`Gemma4ForCausalLM`). 

---

## 2. Core Pipelines

### 2.1 ETL Pipeline (The `grain` Integration)
Data loading at scale is a massive bottleneck. We use **Google Grain**, a high-performance dataloader built for distributed training.
1.  **Datasets:** We natively connect to Hugging Face datasets (e.g., `seeklhy/SynSQL-2.5M`, `xlangai/spider2-lite`).
2.  **Transformations:** Datasets pass through a series of `MapTransform` pipelines.
3.  **Target Formats:** The pipeline normalizes the text into integer sequences and produces dataset shards tailored to the specific backend:
    *   **JAX/MaxText:** Yields structures matching `JAXDistributed` expectations (sharded arrays).
    *   **Keras/PyTorch:** Yields standard `(inputs, labels)` batched tuples or native `DataLoaders`.

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
To fit large models (like Gemma 4) on consumer hardware, we abstract LoRA and quantization.
*   **PEFT:** Handled natively using Keras LoRA APIs, Optax wrappers in JAX, or Hugging Face `peft` libraries for PyTorch. We abstract this configuration (`target_modules`, `lora_r`, `alpha`) so users specify it once.
*   **Quantization:** Supports int8, AWQ, GPTQ, and GGUF mechanisms, interfacing heavily with the PyTorch ecosystem while providing compilation compatibility for JAX workflows.

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
