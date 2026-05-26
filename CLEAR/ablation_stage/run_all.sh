#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jb/code/KVW/CLEAR}"
PYTHON="${PYTHON:-/home/jb/.conda/envs/clear/bin/python}"
MODEL_ID="${MODEL_ID:-${ROOT_DIR}/models/Qwen2-VL-2B-Instruct}"
VANILLA_DIR="${VANILLA_DIR:-${ROOT_DIR}/checkpoints/qwen2B_vanilla}"
GPU="${GPU:-1,2,3}"
EVAL_GPU="${EVAL_GPU:-${GPU}}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SHOT_NUM="${SHOT_NUM:-zero_shots}"
EVAL_LIST="${EVAL_LIST:-forget retain realface realworld}"

CHECKPOINT_ROOT="${ABLATION_STAGE_CHECKPOINT_ROOT:-${ROOT_DIR}/checkpoints/ablation_stage}"
RESULT_ROOT="${ABLATION_STAGE_RESULT_ROOT:-${ROOT_DIR}/checkpoints/ablation_stage_results}"
STAGE1_SOURCE="${STAGE1_SOURCE:-${ROOT_DIR}/checkpoints/KVW_STAGE1_FORGET5_L0_1000}"
RETAIN_SAVE_DIR="${RETAIN_SAVE_DIR:-${CHECKPOINT_ROOT}/KVW_RETAIN_ONLY_5}"
FULL_SOURCE="${FULL_SOURCE:-${ROOT_DIR}/checkpoints/KVW_TWOSTAGE_5}"
CORE_SUMMARY_CSV="${CORE_SUMMARY_CSV:-${ROOT_DIR}/checkpoints/eval_summary/core_all_ratios_summary.csv}"

BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
START_LAYER="${START_LAYER:-0}"
END_LAYER="${END_LAYER:-1000}"
COLUMN_TOP_RATIO="${COLUMN_TOP_RATIO:-0.1}"
COLUMN_TOPK="${COLUMN_TOPK:-0}"
KC_CACHE_DIR="${KC_CACHE_DIR:-kc}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"

cd "${ROOT_DIR}"

echo "============================================================"
echo "[CLEAR stage ablation: forget05]"
echo "============================================================"
echo "train_gpu=${GPU}"
echo "eval_gpu=${EVAL_GPU}"
echo "batch_size=${BATCH_SIZE}"
echo "lr=${LR}"
echo "num_epochs=${NUM_EPOCHS}"
echo "max_train_steps=${MAX_TRAIN_STEPS} (0 means full retain epoch)"
echo "stage1_source=${STAGE1_SOURCE}"
echo "vanilla_source=${VANILLA_DIR}"
echo "core_metric_source=${CORE_SUMMARY_CSV}"
echo "train_data_forget=data/CLEAR/forget05"
echo "train_data_retain=data/CLEAR/retain95"
echo "eval_data_forget=data/CLEAR/forget05_perturbed"
echo "eval_data_retain=data/CLEAR/retain_perturbed"
echo "eval_data_realface=data/CLEAR/real_faces"
echo "eval_data_realworld=data/CLEAR/real_world"
echo "checkpoint_root=${CHECKPOINT_ROOT}"
echo "result_root=${RESULT_ROOT}"

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[Error] Missing ${label}: ${path}" >&2
    exit 1
  fi
}

train_retain_only() {
  require_dir "${STAGE1_SOURCE}" "stage-1 KVW checkpoint"
  require_dir "${VANILLA_DIR}" "vanilla checkpoint"

  if [[ "${FORCE_RERUN}" != "1" && -f "${RETAIN_SAVE_DIR}/trainer_config.json" ]]; then
    echo "[Train][Skip] KVW_RETAIN_ONLY -> ${RETAIN_SAVE_DIR}"
    return
  fi

  echo "[Train] KVW_RETAIN_ONLY -> ${RETAIN_SAVE_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m baselines.KVW_TwoStageRecovery \
    --model_id "${MODEL_ID}" \
    --vanilla_dir "${VANILLA_DIR}" \
    --init_model_dir "${STAGE1_SOURCE}" \
    --save_dir "${RETAIN_SAVE_DIR}" \
    --forget_ratio 5 \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --num_epochs "${NUM_EPOCHS}" \
    --retain_coef 1.0 \
    --kl_coef 0.0 \
    --hidden_coef 0.0 \
    --forget_lock_coef 0.0 \
    --forget_margin 0.2 \
    --reg_coef 0.0 \
    --start_layer "${START_LAYER}" \
    --end_layer "${END_LAYER}" \
    --column_top_ratio "${COLUMN_TOP_RATIO}" \
    --column_topk "${COLUMN_TOPK}" \
    --kc_cache_dir "${KC_CACHE_DIR}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --max_train_steps "${MAX_TRAIN_STEPS}"
}

mkdir -p "${CHECKPOINT_ROOT}" "${RESULT_ROOT}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_retain_only
else
  echo "[Train][Skip] RUN_TRAIN=${RUN_TRAIN}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  export ROOT_DIR PYTHON MODEL_ID GPU EVAL_GPU FORCE_RERUN SHOT_NUM EVAL_LIST
  export ABLATION_STAGE_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}"
  export ABLATION_STAGE_RESULT_ROOT="${RESULT_ROOT}"
  export STAGE1_SOURCE RETAIN_SAVE_DIR FULL_SOURCE CORE_SUMMARY_CSV
  bash ablation_stage/eval_all.sh
else
  echo "[Eval][Skip] RUN_EVAL=${RUN_EVAL}"
  "${PYTHON}" ablation_stage/summarize_results.py
fi

echo "Done."
