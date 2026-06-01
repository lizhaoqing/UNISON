#!/usr/bin/env bash
# UNISON inference launcher
#
# Runs all three task modes in sequence: generation, editing, zero-shot TTS.
# Edit the variables below and the example config files in scripts/example_infer_prompts/
# before running.
#
# Usage:
#   export QWEN_OMNI_MODEL_PATH=/path/to/Qwen2.5-Omni-7B
#   export CHECKPOINT_DIR=/path/to/checkpoint-XXXXXX
#   bash scripts/infer.sh [--key value ...]
#
# Examples:
#   TASK_MODE=generation bash scripts/infer.sh
#   bash scripts/infer.sh --task_mode zeroshotts --num_inference_steps 100
#   bash scripts/infer.sh \
#       --checkpoint_dir checkpoints/unison_D24S0_O_20ch \
#       --model_config   unison/config/D24S0_O_20ch.yaml \
#       --vae_config     unison/models/mmaudio/vae_config_16k.yaml \
#       --task_mode      zeroshotts

# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Parse --key value flags (override env vars or defaults below)
# Supported: --model_config  --vae_config  --checkpoint_dir  --task_mode
#            --num_inference_steps  --guidance_scale  --seed  --output_dir
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_config)        MODEL_CONFIG="$2";         shift 2 ;;
        --vae_config)          VAE_CONFIG="$2";           shift 2 ;;
        --checkpoint_dir)      CHECKPOINT_DIR="$2";       shift 2 ;;
        --task_mode)           TASK_MODE="$2";            shift 2 ;;
        --num_inference_steps) NUM_INFERENCE_STEPS="$2";  shift 2 ;;
        --guidance_scale)      GUIDANCE_SCALE="$2";       shift 2 ;;
        --seed)                SEED="$2";                 shift 2 ;;
        --output_dir)          OUTPUT_DIR="$2";           shift 2 ;;
        *) echo "[WARN] Unknown argument: $1 (ignored)"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Required: set these before running
# ---------------------------------------------------------------------------

# Path to the Qwen2.5-Omni-7B model directory, or HF hub ID "Qwen/Qwen2.5-Omni-7B"
: "${QWEN_OMNI_MODEL_PATH:=Qwen/Qwen2.5-Omni-7B}"

# Path to the UNISON checkpoint directory (e.g. outputs/.../checkpoint-890000)
: "${CHECKPOINT_DIR:?Please set CHECKPOINT_DIR to your checkpoint folder}"

# ---------------------------------------------------------------------------
# Model and VAE config
# ---------------------------------------------------------------------------

# DiT model config. Override via --model_config or MODEL_CONFIG env var.
MODEL_CONFIG="${MODEL_CONFIG:-$PROJECT_ROOT/unison/config/D20S0_O_40ch.yaml}"

# VAE config. Must match the config used during training.
#   vae_config_44k.yaml  — MMAudio 44 kHz VAE  (default, paper results)
#   vae_config_16k.yaml  — MMAudio 16 kHz VAE
# Override via --vae_config or VAE_CONFIG env var.
VAE_CONFIG="${VAE_CONFIG:-$PROJECT_ROOT/unison/models/mmaudio/vae_config_44k.yaml}"

# ---------------------------------------------------------------------------
# Inference parameters
# ---------------------------------------------------------------------------

# Number of ODE steps. Higher = better quality, slower.
# Recommended: 50 (fast) .. 100 (paper setting).
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"

# Classifier-free guidance scale. Higher = more text-faithful, less diverse.
# Recommended: 3.5 .. 5.0.  Paper default: 4.5.
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.5}"

# Random seed for reproducibility.
SEED="${SEED:-42}"

# ---------------------------------------------------------------------------
# Generation task parameters (--task_mode generation)
# ---------------------------------------------------------------------------

# Output audio length in seconds for generation tasks.
# The model always generates a full MAX_AUDIO_DURATION (10 s) latent internally;
# this value trims the decoded waveform to the requested length.
# Must be <= 10.0 (the model's training max duration).
DURATION="${DURATION:-10.0}"

# ---------------------------------------------------------------------------
# Zero-shot TTS parameters (--task_mode zeroshotts)
# ---------------------------------------------------------------------------

# Maximum total duration (reference + generated target) in seconds.
# ref_duration + target_duration must fit within this budget.
ZS_DURATION="${ZS_DURATION:-10.0}"

# Length of reference audio used for speaker cloning, in seconds.
# Only the first REF_DURATION seconds of the ref_audio file are used.
# A silence-aware snap slightly adjusts the cut to avoid cutting mid-phoneme.
# Longer refs give better speaker similarity but leave
# less room for the generated target within ZS_DURATION.
REF_DURATION="${REF_DURATION:-3.0}"

# Whisper model size for auto-transcribing the reference audio.
# Choices: tiny | base | small | medium | large
# Only needed when ref_text is NOT provided in the zs_config JSON.
WHISPER_MODEL="${WHISPER_MODEL:-base}"

# ---------------------------------------------------------------------------
# Example config files — edit these before running
# ---------------------------------------------------------------------------

# Text file with one generation/TTS prompt per line.
# See scripts/example_infer_prompts/gen_prompts.txt for the prompt template format.
GEN_PROMPTS="$PROJECT_ROOT/scripts/example_infer_prompts/gen_prompts.txt"

# JSON list of editing tasks: [{"prompt": "[Edit] ...", "source_audio": "path"}, ...]
# See scripts/example_infer_prompts/edit_config.json for templates and task types.
EDIT_CONFIG="$PROJECT_ROOT/scripts/example_infer_prompts/edit_config.json"

# JSON list of zero-shot TTS tasks: [{"target_text": "...", "ref_audio": "path"}, ...]
# Optionally add "ref_text" to skip Whisper transcription of the reference.
# See scripts/example_infer_prompts/zeroshotts_config.json for examples.
ZS_CONFIG="$PROJECT_ROOT/scripts/example_infer_prompts/zeroshotts_config.json"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CKPT_NAME="$(basename "$CHECKPOINT_DIR")"
EXP_DIR="$(dirname "$CHECKPOINT_DIR")"
OUTPUT_BASE="${OUTPUT_DIR:-$EXP_DIR/infer_${NUM_INFERENCE_STEPS}steps/$CKPT_NAME}"

TASK_MODE="${TASK_MODE:-all}"  # generation | editing | zeroshotts | all

mkdir -p "$OUTPUT_BASE"
cd "$PROJECT_ROOT"

echo "========================================================"
echo "UNISON Inference"
echo "  Checkpoint : $CHECKPOINT_DIR"
echo "  Task mode  : $TASK_MODE"
echo "  Steps      : $NUM_INFERENCE_STEPS   CFG: $GUIDANCE_SCALE   Seed: $SEED"
echo "  Output     : $OUTPUT_BASE"
echo "========================================================"

run_infer() {
    local mode="$1"
    local out_subdir="$2"
    shift 2
    local out="$OUTPUT_BASE/$out_subdir"
    mkdir -p "$out"
    python unison/pipelines/infer.py \
        --model_ckpt            "$CHECKPOINT_DIR" \
        --task_mode             "$mode" \
        --num_inference_steps   "$NUM_INFERENCE_STEPS" \
        --guidance_scale        "$GUIDANCE_SCALE" \
        --seed                  "$SEED" \
        --device                cuda \
        --model_config          "$MODEL_CONFIG" \
        --vae_config            "$VAE_CONFIG" \
        --omni_model_path       "$QWEN_OMNI_MODEL_PATH" \
        --text_encoder_type     omni \
        --text_encoder_model_path "$QWEN_OMNI_MODEL_PATH" \
        --output_dir            "$out" \
        "$@"
}

# ---------- Generation ----------
if [[ "$TASK_MODE" == "generation" || "$TASK_MODE" == "all" ]]; then
    echo ""
    echo ">>> Generation"
    run_infer generation gen \
        --gen_prompt_list "$GEN_PROMPTS" \
        --gen_duration    "$DURATION"
    echo "    -> $OUTPUT_BASE/gen"
fi

# ---------- Editing ----------
if [[ "$TASK_MODE" == "editing" || "$TASK_MODE" == "all" ]]; then
    echo ""
    echo ">>> Editing"
    run_infer editing edit \
        --edit_config "$EDIT_CONFIG"
    echo "    -> $OUTPUT_BASE/edit"
fi

# ---------- Zero-shot TTS ----------
if [[ "$TASK_MODE" == "zeroshotts" || "$TASK_MODE" == "all" ]]; then
    echo ""
    echo ">>> Zero-shot TTS"
    run_infer zeroshotts zeroshotts \
        --zs_config      "$ZS_CONFIG" \
        --zs_duration    "$ZS_DURATION" \
        --ref_duration   "$REF_DURATION" \
        --whisper_model  "$WHISPER_MODEL" \
        --mask_zs_after_encode
    echo "    -> $OUTPUT_BASE/zeroshotts"
fi

echo ""
echo "Done. Outputs under $OUTPUT_BASE"
