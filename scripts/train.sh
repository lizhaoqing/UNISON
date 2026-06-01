#!/bin/bash
# UNISON fine-tuning launcher.
#
# Usage:
#   export QWEN_OMNI_MODEL_PATH=/path/to/Qwen2.5-Omni-7B
#   bash scripts/train.sh [--key value ...]
#
# ── Required ────────────────────────────────────────────────────────────────
#   QWEN_OMNI_MODEL_PATH   Path to Qwen2.5-Omni-7B (or HF hub ID)
#
# ── Key arguments ───────────────────────────────────────────────────────────
#   --num_processes N          Number of GPUs (default: 1)
#   --batch_size N             Per-GPU batch size (default: 2)
#   --max_train_steps N        Total training steps (default: 200)
#   --lr FLOAT                 Learning rate (default: 1e-5)
#   --metadata PATH            Training metadata JSONL (default: data/train/metadata.jsonl)
#   --pretrained_model_path P  Starting checkpoint: directory or .pt/.safetensors file
#                              Supports ema_model.pt, model.safetensors, pytorch_model.bin
#   --exp_name STR             Experiment name; outputs go to outputs/<exp_name>/ (default: unison_finetune)
#   --output_dir PATH          Override output directory directly
#   --model_config PATH        DiT config YAML (default: unison/config/D20S0_O_40ch.yaml)
#   --vae_config PATH          VAE config YAML (default: vae_config_44k.yaml)
#   --checkpointing_steps N    Save checkpoint every N steps (default: 100)
#   --logging_steps N          Log loss/lr/grad_norm every N steps (default: 10)
#   --report_to STR            Tracker backend: tensorboard | wandb | none (default: tensorboard)
#
# Any unrecognised flags are forwarded directly to train.py.
#
# ── Metadata JSONL format ───────────────────────────────────────────────────
#   One JSON object per line with at least:
#     "audio_path" : path to .wav (relative to repo root or absolute)
#     "caption"    : text description
#     "duration"   : clip length in seconds
#   See data/train/metadata.jsonl for an example.
#
# ── Examples ────────────────────────────────────────────────────────────────
#   # Single-GPU smoke test (bundled 5-clip data):
#   bash scripts/train.sh --num_processes 1
#
#   # Fine-tune on your own data, 8 GPUs:
#   bash scripts/train.sh \
#       --num_processes         8 \
#       --batch_size            8 \
#       --metadata              /path/to/my_metadata.jsonl \
#       --pretrained_model_path checkpoints/unison_D20S0_O_40ch \
#       --max_train_steps       50000 \
#       --exp_name              my_run

# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Parse --key value flags (see header above for full list)
# ---------------------------------------------------------------------------
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --metadata)              METADATA="$2";               shift 2 ;;
        --output_dir)            OUTPUT_DIR="$2";             shift 2 ;;
        --exp_name)              EXP_NAME="$2";               shift 2 ;;
        --model_config)          MODEL_CONFIG="$2";           shift 2 ;;
        --vae_config)            VAE_CONFIG="$2";             shift 2 ;;
        --num_processes)         NUM_PROCESSES="$2";          shift 2 ;;
        --batch_size)            BATCH_SIZE="$2";             shift 2 ;;
        --lr)                    LR="$2";                     shift 2 ;;
        --max_train_steps)       MAX_TRAIN_STEPS="$2";        shift 2 ;;
        --checkpointing_steps)   CHECKPOINTING_STEPS="$2";   shift 2 ;;
        --logging_steps)         LOGGING_STEPS="$2";         shift 2 ;;
        --report_to)             REPORT_TO="$2";             shift 2 ;;
        --pretrained_model_path) PRETRAINED_MODEL_PATH="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
: "${QWEN_OMNI_MODEL_PATH:=Qwen/Qwen2.5-Omni-7B}"

EXP_NAME="${EXP_NAME:-unison_finetune}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/$EXP_NAME}"
METADATA="${METADATA:-$PROJECT_ROOT/data/train/metadata.jsonl}"
MODEL_CONFIG="${MODEL_CONFIG:-$PROJECT_ROOT/unison/config/D20S0_O_40ch.yaml}"
VAE_CONFIG="${VAE_CONFIG:-$PROJECT_ROOT/unison/models/mmaudio/vae_config_44k.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
REPORT_TO="${REPORT_TO:-tensorboard}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-}"

cd "$PROJECT_ROOT"

echo "========================================================"
echo "UNISON Fine-tuning"
echo "  Metadata    : $METADATA"
echo "  Output      : $OUTPUT_DIR"
echo "  GPUs        : $NUM_PROCESSES"
echo "  Steps       : $MAX_TRAIN_STEPS   BS: $BATCH_SIZE   LR: $LR"
echo "========================================================"

accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --mixed_precision bf16 \
  unison/pipelines/train.py \
    --metadata              "$METADATA" \
    --output_dir            "$OUTPUT_DIR" \
    --model_config          "$MODEL_CONFIG" \
    --vae_config            "$VAE_CONFIG" \
    --omni_model_path       "$QWEN_OMNI_MODEL_PATH" \
    --max_duration          10.0 \
    --batch_size            "$BATCH_SIZE" \
    --learning_rate         "$LR" \
    --max_train_steps       "$MAX_TRAIN_STEPS" \
    --lr_warmup_steps       10 \
    --mixed_precision       bf16 \
    --checkpointing_steps   "$CHECKPOINTING_STEPS" \
    --logging_steps         "$LOGGING_STEPS" \
    --report_to             "$REPORT_TO" \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
    "${EXTRA_ARGS[@]}"
