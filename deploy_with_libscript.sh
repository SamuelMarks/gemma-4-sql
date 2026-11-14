#!/bin/sh
# ## Overview
# Installs dependencies and orchestrates gemma-4-sql local, GPU, or zero-cost TFRC TPU deployment & 7-backend matrix experiments via libscript.
#
# ## Usage
# ./deploy_with_libscript.sh [--local | --tpu-vm | --gpu-vm | --gke-xpk] [--train | --infer | --serve | --tpu-matrix] [--tfrc-only] [--tfrc-config PATH] [--dry-run] [--help]

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -x "$SCRIPT_DIR/../libscript/libscript.sh" ]; then
  LIBSCRIPT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../libscript" && pwd)
  LIBSCRIPT_CMD="$LIBSCRIPT_ROOT/libscript.sh"
else
  LIBSCRIPT_ROOT="${LIBSCRIPT_ROOT_DIR:-$HOME/.libscript}"
  LIBSCRIPT_CMD="libscript"
fi

# ## print_usage
# Prints command-line usage instructions for deploy_with_libscript.sh.
print_usage() {
  echo "Usage: $0 [--local | --tpu-vm | --gpu-vm | --gke-xpk] [--train | --infer | --serve | --tpu-matrix] [--tfrc-only] [--tfrc-config PATH] [--dry-run] [--help]"
}

MODE="local"
ACTION="deploy"
DRY_RUN_FLAG=""
TFRC_ONLY="0"
TFRC_CONFIG_PATH="${TFRC_CONFIG:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h)
      print_usage
      exit 0
      ;;
    --local)
      MODE="local"
      shift
      ;;
    --tpu-vm|--gcp-tpu-vm)
      MODE="gcp-tpu-vm"
      shift
      ;;
    --gpu-vm|--gcp-gpu-vm)
      MODE="gcp-gpu-vm"
      shift
      ;;
    --gke-xpk)
      MODE="gke-xpk"
      shift
      ;;
    --train)
      ACTION="train"
      shift
      ;;
    --infer|--serve|--serve-ml)
      ACTION="infer"
      shift
      ;;
    --tpu-matrix)
      MODE="gcp-tpu-vm"
      ACTION="tpu-matrix"
      shift
      ;;
    --tfrc-only)
      TFRC_ONLY="1"
      shift
      ;;
    --tfrc-config)
      TFRC_CONFIG_PATH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN_FLAG="--dry-run"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

export APP_DIR="$SCRIPT_DIR"

if [ "$TFRC_ONLY" = "1" ]; then
  export TFRC_STRICT="1"
  export DISK_SIZE_GB="0"
  export TPU_DATA_DISK_SIZE="0"
  if [ "$MODE" = "gke-xpk" ] || [ "$MODE" = "gcp-gpu-vm" ]; then
    echo "ERROR: --tfrc-only blocks billable non-TFRC target '$MODE' (GKE/XPK node pools and GPUs incur GCP charges)." >&2
    exit 1
  fi
  if [ -z "$TFRC_CONFIG_PATH" ]; then
    echo "ERROR: --tfrc-only requires --tfrc-config <path> (copy tfrc.policy.example.json to tfrc.policy.json)." >&2
    exit 1
  fi
  export TFRC_CONFIG="$TFRC_CONFIG_PATH"
fi

if [ "$ACTION" = "tpu-matrix" ]; then
  MATRIX_ARGS=""
  [ "$TFRC_ONLY" = "1" ] && MATRIX_ARGS="--tfrc-only --tfrc-config $TFRC_CONFIG_PATH"
  [ -n "$DRY_RUN_FLAG" ] && MATRIX_ARGS="$MATRIX_ARGS --dry-run"
  # shellcheck disable=SC2086
  exec "$LIBSCRIPT_CMD" experiment tpu_matrix "$SCRIPT_DIR/libscript.json" $MATRIX_ARGS
fi

if [ -n "$DRY_RUN_FLAG" ]; then
  ML_CMD="$ACTION"
  [ "$ML_CMD" = "deploy" ] && ML_CMD="train"
  exec "$LIBSCRIPT_CMD" "$ML_CMD" "$SCRIPT_DIR/libscript.json" --target "$MODE" --dry-run
fi

echo "Installing required dependencies via libscript..."
"$LIBSCRIPT_CMD" install python 3.12
"$LIBSCRIPT_CMD" install uv latest
"$LIBSCRIPT_CMD" install duckdb latest
"$LIBSCRIPT_CMD" install sqlite latest
"$LIBSCRIPT_CMD" install huggingface-cli latest
if [ "$MODE" = "gke-xpk" ] && [ "$TFRC_ONLY" != "1" ]; then
  "$LIBSCRIPT_CMD" install xpk latest
fi
if [ -f "$SCRIPT_DIR/libscript.json" ]; then
  "$LIBSCRIPT_CMD" install-deps "$SCRIPT_DIR/libscript.json"
fi

if [ "$ACTION" = "train" ] || [ "$ACTION" = "infer" ]; then
  "$LIBSCRIPT_CMD" "$ACTION" "$SCRIPT_DIR/libscript.json" --target "$MODE"
elif [ "$MODE" = "gcp-tpu-vm" ]; then
  "$LIBSCRIPT_ROOT/payloads/ai/gcp-tpu-vm/setup.sh"
  "$LIBSCRIPT_ROOT/payloads/ai/gcp-tpu-vm/deploy.sh"
elif [ "$MODE" = "gcp-gpu-vm" ]; then
  "$LIBSCRIPT_ROOT/payloads/ai/gcp-gpu-vm/setup.sh"
  "$LIBSCRIPT_ROOT/payloads/ai/gcp-gpu-vm/deploy.sh"
elif [ "$MODE" = "gke-xpk" ]; then
  "$LIBSCRIPT_ROOT/payloads/ai/gke-xpk-training/setup.sh"
  "$LIBSCRIPT_ROOT/payloads/ai/gke-xpk-training/deploy.sh"
else
  echo "Installing gemma-4-sql in editable mode..."
  pip install -e "$SCRIPT_DIR"
fi
