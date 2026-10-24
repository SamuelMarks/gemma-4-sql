#!/bin/sh
# ## Overview
# Installs dependencies and orchestrates gemma-4-sql local, GPU, or TPU training/inference deployment via libscript.
#
# ## Usage
# ./deploy_with_libscript.sh [--local | --tpu-vm | --gpu-vm | --gke-xpk | --train | --infer | --serve | --dry-run | --help]

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
  printf '%s
' "Usage: $0 [--local | --tpu-vm | --gpu-vm | --gke-xpk] [--train | --infer | --serve] [--dry-run] [--help]"
}

MODE="local"
ACTION="deploy"
DRY_RUN_FLAG=""

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

if [ -n "$DRY_RUN_FLAG" ]; then
  ML_CMD="$ACTION"
  [ "$ML_CMD" = "deploy" ] && ML_CMD="train"
  exec "$LIBSCRIPT_CMD" "$ML_CMD" "$SCRIPT_DIR/libscript.json" --target "$MODE" --dry-run
fi

printf '%s
' "Installing required dependencies via libscript..."
"$LIBSCRIPT_CMD" install python 3.12
"$LIBSCRIPT_CMD" install duckdb latest
"$LIBSCRIPT_CMD" install sqlite latest
"$LIBSCRIPT_CMD" install huggingface-cli latest
"$LIBSCRIPT_CMD" install xpk latest
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
  printf '%s
' "Installing gemma-4-sql in editable mode..."
  pip install -e "$SCRIPT_DIR"
fi
