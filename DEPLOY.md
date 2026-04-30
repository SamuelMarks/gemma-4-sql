# DEPLOY.md

## Deployment Guide

This guide describes how to deploy `gemma-4-sql` models. We natively support the following architectures for different distributed hardware environments:
*   **TPU / XLA:** AI-Hypercomputer MaxText (`maxtext.models.gemma4.Gemma4Model`) and Native JAX via Bonsai (`bonsai.models.BonsaiModel`).
*   **GPU:** PyTorch via Hugging Face Transformers (`transformers.models.gemma4.Gemma4ForCausalLM`).

### TPU Deployment

Google Cloud TPUs are recommended for training and deploying `gemma-4-sql` models using JAX or MaxText.

#### 1. Provision a TPU VM
```bash
gcloud compute tpus tpu-vm create gemma-4-tpu \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base
```

#### 2. Install Dependencies
SSH into the TPU VM and install the necessary libraries for JAX and MaxText:
```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install grain datasets
```

#### 3. Prepare Datasets (ETL)
Use the `gemma-4-sql` CLI to run ETL for the SQL datasets on the TPU VM:
```bash
# Pretraining Dataset (e.g. SynSQL-2.5M)
gemma-4-sql etl pretrain --dataset seeklhy/SynSQL-2.5M

# SFT Dataset (e.g. Gretel Synthetic)
gemma-4-sql etl sft --dataset gretelai/synthetic_text_to_sql

# Post-Training Dataset (e.g. Spider 2.0)
gemma-4-sql etl posttrain --dataset xlangai/spider2-lite
```

#### 4. Run Training / Pretraining
Use the `gemma-4-sql` CLI to start training. You can optionally apply LoRA for efficient fine-tuning on the TPU:
```bash
# Apply PEFT / LoRA (Optional)
gemma-4-sql peft --model gemma-4 --backend maxtext

# Start training loop
gemma-4-sql train --model gemma-4 --backend maxtext
```

For more details on TPU architecture and parallelization strategies, consult the MaxText and JAX documentation.
