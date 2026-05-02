# `gemma-4-sql` Deployment & Operations Guide

This comprehensive guide details the exact steps to create, adapt, and post-train a Gemma-4 model tailored exclusively for Text-to-SQL tasks. It also covers how to benchmark these models across varying hardware topologies (GPU, TPU, CPU) using different backends: **JAX**, **JAX (MaxText)**, **Keras**, and **PyTorch**.

---

## 0. Create a Brand New `gemma-4` from Scratch for SQL

Training a model "from scratch" means you are initializing the Gemma-4 architecture with random weights and training it strictly on SQL syntax and schema relations. This requires a massive pretraining corpus (e.g., millions of synthetic and real SQL queries) and significant compute.

### Step 1: Data Preparation
You must first process a large-scale SQL dataset into tokens. We use Google's `grain` dataloader to shard the data efficiently.

```bash
# Normalize the dataset for your chosen backend
gemma-4-sql etl pretrain --dataset seeklhy/SynSQL-2.5M --batch-size 256 --backend jax
```

### Step 2: Train from Scratch

Execute the `train` command. This will instantiate an uninitialized model and begin the optimization loop.

#### Using JAX
*Recommended for rapid local testing on single GPUs or small TPU slices.*
```bash
gemma-4-sql train \
    --model gemma-4 \
    --dataset seeklhy/SynSQL-2.5M \
    --epochs 10 \
    --learning-rate 1e-4 \
    --backend jax
```

#### Using JAX (MaxText)
*Recommended for Google Cloud TPU Pods (e.g., v4-128, v5e). MaxText scales purely via XLA.*
```bash
gemma-4-sql train \
    --model gemma-4 \
    --dataset seeklhy/SynSQL-2.5M \
    --epochs 10 \
    --learning-rate 1e-4 \
    --backend maxtext
```

#### Using Keras
*Recommended if you need seamless integration into the TensorFlow ecosystem.*
```bash
gemma-4-sql train \
    --model gemma-4 \
    --dataset seeklhy/SynSQL-2.5M \
    --epochs 10 \
    --learning-rate 1e-4 \
    --backend keras
```

#### Using PyTorch
*Recommended for NVIDIA GPU clusters (e.g., A100s, H100s) and integration with Hugging Face.*
```bash
gemma-4-sql train \
    --model gemma-4 \
    --dataset seeklhy/SynSQL-2.5M \
    --epochs 10 \
    --learning-rate 1e-4 \
    --backend pytorch
```

---

## 1. Create a Pretrained `gemma-4` for SQL

Instead of random initialization, you take Google's foundation Gemma-4 weights (which already understand English and code) and **continue pretraining** exclusively on SQL datasets. This adapts the model to your specific database dialects (e.g., Snowflake, DuckDB) without losing general language capabilities.

### Step 1: Data Preparation
```bash
gemma-4-sql etl pretrain --dataset b-mc2/sql-create-context --batch-size 128 --backend pytorch
```

### Step 2: Run Continuous Pretraining

Use the `pretrain` command. This loads existing weights and continues the learning process.

**JAX:**
```bash
gemma-4-sql pretrain --model google/gemma-4 --dataset b-mc2/sql-create-context --backend jax
```

**JAX (MaxText):**
```bash
gemma-4-sql pretrain --model google/gemma-4 --dataset b-mc2/sql-create-context --backend maxtext
```

**Keras:**
```bash
gemma-4-sql pretrain --model google/gemma-4 --dataset b-mc2/sql-create-context --backend keras
```

**PyTorch:**
```bash
gemma-4-sql pretrain --model google/gemma-4 --dataset b-mc2/sql-create-context --backend pytorch
```

---

## 2. Create a Post-Trained `gemma-4` for SQL

Post-training is the final alignment phase. This includes Supervised Fine-Tuning (SFT) on exact instruction-response pairs (e.g., User: "How many users?", Assistant: "SELECT COUNT(*) FROM users;") and Direct Preference Optimization (DPO) to punish hallucinated columns.

### Step 1: Data Preparation (SFT & DPO)
```bash
# ETL for Instruction Tuning
gemma-4-sql etl sft --dataset gretelai/synthetic_text_to_sql --batch-size 64 --backend jax

# ETL for DPO (requires chosen/rejected pairs)
gemma-4-sql etl posttrain --dataset my_custom_dpo_dataset --batch-size 32 --backend jax
```

### Step 2: Supervised Fine-Tuning (SFT)

**JAX:**
```bash
gemma-4-sql sft --model my-sql-pretrained-gemma-4 --dataset gretelai/synthetic_text_to_sql --backend jax
```

**JAX (MaxText):**
```bash
gemma-4-sql sft --model my-sql-pretrained-gemma-4 --dataset gretelai/synthetic_text_to_sql --backend maxtext
```

**Keras:**
```bash
gemma-4-sql sft --model my-sql-pretrained-gemma-4 --dataset gretelai/synthetic_text_to_sql --backend keras
```

**PyTorch:**
```bash
gemma-4-sql sft --model my-sql-pretrained-gemma-4 --dataset gretelai/synthetic_text_to_sql --backend pytorch
```

### Step 3: Direct Preference Optimization (DPO)

Once SFT is complete, align the model using DPO to prevent destructive queries (e.g., DROP TABLE) and favor efficient JOINs.

```bash
# Apply DPO using PyTorch
gemma-4-sql dpo --model my-sft-gemma-4 --dataset my_custom_dpo_dataset --beta 0.1 --backend pytorch

# Apply DPO using MaxText
gemma-4-sql dpo --model my-sft-gemma-4 --dataset my_custom_dpo_dataset --beta 0.1 --backend maxtext
```

---

## 3. Benchmarking Implementations on Target Hardware

Before deploying your fully trained, post-trained model to production, you must benchmark its throughput (tokens/sec), latency (ms), and memory consumption (MB).

`gemma-4-sql` provides a native `benchmark` command that evaluates the model on your specified hardware target (GPU, TPU, or CPU) and backend.

### Benchmarking on GPU (NVIDIA A100/H100)

**Using PyTorch (Typically fastest on NVIDIA via FlashAttention):**
```bash
gemma-4-sql benchmark --model my-sft-gemma-4 --hardware gpu --batch-size 32 --backend pytorch
```

**Using JAX (JAX on GPU):**
```bash
gemma-4-sql benchmark --model my-sft-gemma-4 --hardware gpu --batch-size 32 --backend jax
```

### Benchmarking on TPU (Google Cloud TPU v4 / v5e)

**Using MaxText (Highly optimized for TPU interconnects):**
```bash
gemma-4-sql benchmark --model my-sft-gemma-4 --hardware tpu --batch-size 128 --backend maxtext
```

**Using Keras (with TPUStrategy):**
```bash
gemma-4-sql benchmark --model my-sft-gemma-4 --hardware tpu --batch-size 128 --backend keras
```

### Benchmarking on CPU (Intel/AMD for Edge or Local testing)

**Using PyTorch (CPU mode):**
```bash
gemma-4-sql benchmark --model my-sft-gemma-4 --hardware cpu --batch-size 1 --backend pytorch
```

**Using JAX (JAX on CPU):**
```bash
gemma-4-sql benchmark --model my-sft-gemma-4 --hardware cpu --batch-size 1 --backend jax
```

### Example Benchmark Output:
```json
{
  "backend": "maxtext",
  "model": "my-sft-gemma-4",
  "hardware": "tpu",
  "batch_size": 128,
  "tokens_per_sec": 4500.0,
  "latency_ms": 10.1,
  "memory_mb": 16384.0
}
```

---

## 4. Monitoring with TensorBoard

Throughout training, pretraining, and post-training phases, you should monitor your cluster's progress. `gemma-4-sql` integrates cleanly with TensorBoard across all backend topologies (JAX, MaxText, Keras, and PyTorch).

Metrics are written by default to the `logs` directory.

### Start TensorBoard
```bash
tensorboard --logdir=./logs --port=6006
```

### Manual Logging

If you have custom evaluation scripts running alongside your training jobs, you can emit metrics directly to your existing TensorBoard runs using the CLI or SDK:

```bash
# Push custom evaluation metrics to the active TB directory
gemma-4-sql log --step 1000 --metrics "eval_loss=0.3,execution_accuracy=0.89" --log-dir "./logs" --backend maxtext
```
