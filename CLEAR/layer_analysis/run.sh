#!/bin/bash
set -euo pipefail

ROOT_DIR="/home/jb/code/KVW/CLEAR"
MODEL_ID="${ROOT_DIR}/models/Qwen2-VL-2B-Instruct"
VANILLA_DIR="${ROOT_DIR}/checkpoints/qwen2B_vanilla"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/layer_analysis/results}"
GPU="${GPU:-0,1,2,3}"
FORGET_RATIO="${FORGET_RATIO:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TOP_RATIO="${TOP_RATIO:-0.05}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.01}"

cd "${ROOT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python layer_analysis/analyze_layers.py \
  --model_id "${MODEL_ID}" \
  --vanilla_dir "${VANILLA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --forget_ratio "${FORGET_RATIO}" \
  --batch_size "${BATCH_SIZE}" \
  --top_ratio "${TOP_RATIO}" \
  --active_threshold "${ACTIVE_THRESHOLD}"
