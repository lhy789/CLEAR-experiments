#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jb/code/KVW/CLEAR}"
PYTHON="${PYTHON:-/home/jb/.conda/envs/clear/bin/python}"
MODEL_ID="${MODEL_ID:-${ROOT_DIR}/models/Qwen2-VL-2B-Instruct}"
VANILLA_DIR="${VANILLA_DIR:-${ROOT_DIR}/checkpoints/qwen2B_vanilla}"
GPU="${GPU:-0,1,2,3}"
EVAL_GPU="${EVAL_GPU:-${GPU}}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SHOT_NUM="${SHOT_NUM:-zero_shots}"
EVAL_LIST="${EVAL_LIST:-forget retain realface realworld}"

CHECKPOINT_ROOT="${ABLATION_LOSS_CHECKPOINT_ROOT:-${ROOT_DIR}/checkpoints/ablation_retain_loss}"
RESULT_ROOT="${ABLATION_LOSS_RESULT_ROOT:-${ROOT_DIR}/checkpoints/ablation_retain_loss_results}"
STAGE1_SOURCE="${STAGE1_SOURCE:-${ROOT_DIR}/checkpoints/KVW_STAGE1_FORGET5_L0_1000}"
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
VARIANTS="${VARIANTS:-drop_retain_lm drop_kl drop_hidden drop_forget_lock drop_reg}"

BASE_RETAIN_COEF="${BASE_RETAIN_COEF:-1.0}"
BASE_KL_COEF="${BASE_KL_COEF:-0.5}"
BASE_HIDDEN_COEF="${BASE_HIDDEN_COEF:-1.0}"
BASE_FORGET_LOCK_COEF="${BASE_FORGET_LOCK_COEF:-0.5}"
BASE_FORGET_MARGIN="${BASE_FORGET_MARGIN:-0.2}"
BASE_REG_COEF="${BASE_REG_COEF:-1e-4}"

cd "${ROOT_DIR}"

echo "============================================================"
echo "[CLEAR retain loss ablation: forget05]"
echo "============================================================"
echo "train_gpu=${GPU}"
echo "eval_gpu=${EVAL_GPU}"
echo "batch_size=${BATCH_SIZE}"
echo "lr=${LR}"
echo "num_epochs=${NUM_EPOCHS}"
echo "max_train_steps=${MAX_TRAIN_STEPS} (0 means full retain epoch)"
echo "variants=${VARIANTS}"
echo "stage1_source=${STAGE1_SOURCE}"
echo "vanilla_source=${VANILLA_DIR}"
echo "full_metric_source=${CORE_SUMMARY_CSV}"
echo "train_data_forget=data/CLEAR/forget05"
echo "train_data_retain=data/CLEAR/retain95"
echo "eval_data_forget=data/CLEAR/forget05_perturbed"
echo "eval_data_retain=data/CLEAR/retain_perturbed"
echo "eval_data_realface=data/CLEAR/real_faces"
echo "eval_data_realworld=data/CLEAR/real_world"
echo "checkpoint_root=${CHECKPOINT_ROOT}"
echo "result_root=${RESULT_ROOT}"
echo "base_loss_coefs=retain:${BASE_RETAIN_COEF},kl:${BASE_KL_COEF},hidden:${BASE_HIDDEN_COEF},forget_lock:${BASE_FORGET_LOCK_COEF},reg:${BASE_REG_COEF}"

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[Error] Missing ${label}: ${path}" >&2
    exit 1
  fi
}

train_variant() {
  local slug="$1"
  local display_name="$2"
  local retain_coef="$3"
  local kl_coef="$4"
  local hidden_coef="$5"
  local forget_lock_coef="$6"
  local reg_coef="$7"
  local save_dir="${CHECKPOINT_ROOT}/${slug}"

  require_dir "${STAGE1_SOURCE}" "stage-1 KVW checkpoint"
  require_dir "${VANILLA_DIR}" "vanilla checkpoint"

  if [[ "${FORCE_RERUN}" != "1" && -f "${save_dir}/trainer_config.json" ]]; then
    echo "[Train][Skip] ${display_name} -> ${save_dir}"
    return
  fi

  echo "[Train] ${display_name} -> ${save_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m baselines.KVW_TwoStageRecovery \
    --model_id "${MODEL_ID}" \
    --vanilla_dir "${VANILLA_DIR}" \
    --init_model_dir "${STAGE1_SOURCE}" \
    --save_dir "${save_dir}" \
    --forget_ratio 5 \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --num_epochs "${NUM_EPOCHS}" \
    --retain_coef "${retain_coef}" \
    --kl_coef "${kl_coef}" \
    --hidden_coef "${hidden_coef}" \
    --forget_lock_coef "${forget_lock_coef}" \
    --forget_margin "${BASE_FORGET_MARGIN}" \
    --reg_coef "${reg_coef}" \
    --start_layer "${START_LAYER}" \
    --end_layer "${END_LAYER}" \
    --column_top_ratio "${COLUMN_TOP_RATIO}" \
    --column_topk "${COLUMN_TOPK}" \
    --kc_cache_dir "${KC_CACHE_DIR}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --max_train_steps "${MAX_TRAIN_STEPS}"
}

should_run_variant() {
  local slug="$1"
  [[ " ${VARIANTS} " == *" all "* || " ${VARIANTS} " == *" ${slug} "* ]]
}

mkdir -p "${CHECKPOINT_ROOT}" "${RESULT_ROOT}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  if should_run_variant "drop_retain_lm"; then
    train_variant "drop_retain_lm" "DROP_RETAIN_LM" \
      0.0 "${BASE_KL_COEF}" "${BASE_HIDDEN_COEF}" "${BASE_FORGET_LOCK_COEF}" "${BASE_REG_COEF}"
  else
    echo "[Train][Skip] DROP_RETAIN_LM: not selected by VARIANTS=${VARIANTS}"
  fi

  if should_run_variant "drop_kl"; then
    train_variant "drop_kl" "DROP_KL" \
      "${BASE_RETAIN_COEF}" 0.0 "${BASE_HIDDEN_COEF}" "${BASE_FORGET_LOCK_COEF}" "${BASE_REG_COEF}"
  else
    echo "[Train][Skip] DROP_KL: not selected by VARIANTS=${VARIANTS}"
  fi

  if should_run_variant "drop_hidden"; then
    train_variant "drop_hidden" "DROP_HIDDEN" \
      "${BASE_RETAIN_COEF}" "${BASE_KL_COEF}" 0.0 "${BASE_FORGET_LOCK_COEF}" "${BASE_REG_COEF}"
  else
    echo "[Train][Skip] DROP_HIDDEN: not selected by VARIANTS=${VARIANTS}"
  fi

  if should_run_variant "drop_forget_lock"; then
    train_variant "drop_forget_lock" "DROP_FORGET_LOCK" \
      "${BASE_RETAIN_COEF}" "${BASE_KL_COEF}" "${BASE_HIDDEN_COEF}" 0.0 "${BASE_REG_COEF}"
  else
    echo "[Train][Skip] DROP_FORGET_LOCK: not selected by VARIANTS=${VARIANTS}"
  fi

  if should_run_variant "drop_reg"; then
    train_variant "drop_reg" "DROP_REG" \
      "${BASE_RETAIN_COEF}" "${BASE_KL_COEF}" "${BASE_HIDDEN_COEF}" "${BASE_FORGET_LOCK_COEF}" 0.0
  else
    echo "[Train][Skip] DROP_REG: not selected by VARIANTS=${VARIANTS}"
  fi
else
  echo "[Train][Skip] RUN_TRAIN=${RUN_TRAIN}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  export ROOT_DIR PYTHON MODEL_ID GPU EVAL_GPU FORCE_RERUN SHOT_NUM EVAL_LIST
  export ABLATION_LOSS_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}"
  export ABLATION_LOSS_RESULT_ROOT="${RESULT_ROOT}"
  export FULL_SOURCE CORE_SUMMARY_CSV VARIANTS
  bash ablation_retain_loss/eval_all.sh
else
  echo "[Eval][Skip] RUN_EVAL=${RUN_EVAL}"
  "${PYTHON}" ablation_retain_loss/summarize_results.py
fi

echo "Done."
