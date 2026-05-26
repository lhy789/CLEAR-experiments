#!/bin/bash
set -euo pipefail

ROOT_DIR="/home/jb/code/KVW/CLEAR"
PYTHON="${PYTHON:-/home/jb/.conda/envs/clear/bin/python}"
MODEL_ID="${ROOT_DIR}/models/Qwen2-VL-2B-Instruct"
VANILLA_DIR="${ROOT_DIR}/checkpoints/qwen2B_vanilla"
GPU="${GPU:-0,1,2,3}"
FORGET_RATIO="${FORGET_RATIO:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
START_LAYER="${START_LAYER:-0}"
END_LAYER="${END_LAYER:-1000}"
STAGE1_GAMMA="${STAGE1_GAMMA:-0.02}"
COLUMN_TOP_RATIO="${COLUMN_TOP_RATIO:-0.1}"
RETAIN_COEF="${RETAIN_COEF:-1.0}"
KL_COEF="${KL_COEF:-0.5}"
HIDDEN_COEF="${HIDDEN_COEF:-1.0}"
FORGET_LOCK_COEF="${FORGET_LOCK_COEF:-0.5}"
FORGET_MARGIN="${FORGET_MARGIN:-0.2}"
REG_COEF="${REG_COEF:-1e-4}"
STAGE1_SAVE_DIR="${STAGE1_SAVE_DIR:-${ROOT_DIR}/checkpoints/KVW_STAGE1_FORGET${FORGET_RATIO}_L${START_LAYER}_${END_LAYER}}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints/KVW_TWOSTAGE_${FORGET_RATIO}}"
FORCE_RERUN="${FORCE_RERUN:-0}"

cd "${ROOT_DIR}"

if [[ "${FORCE_RERUN}" != "1" && -f "${STAGE1_SAVE_DIR}/trainer_config.json" ]]; then
  echo "[Skip] stage1 already finished -> ${STAGE1_SAVE_DIR}"
else
  echo "[Run] stage1 KVW weakening -> ${STAGE1_SAVE_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m baselines.KVW \
    --model_id "${MODEL_ID}" \
    --vanilla_dir "${VANILLA_DIR}" \
    --save_dir "${STAGE1_SAVE_DIR}" \
    --forget_ratio "${FORGET_RATIO}" \
    --batch_size "${BATCH_SIZE}" \
    --num_epochs 1 \
    --phase weakening \
    --gamma "${STAGE1_GAMMA}" \
    --start_layer "${START_LAYER}" \
    --end_layer "${END_LAYER}"
fi

if [[ "${FORCE_RERUN}" != "1" && -f "${SAVE_DIR}/trainer_config.json" ]]; then
  echo "[Skip] two-stage recovery already finished -> ${SAVE_DIR}"
  exit 0
fi

echo "[Run] two-stage recovery -> ${SAVE_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m baselines.KVW_TwoStageRecovery \
  --model_id "${MODEL_ID}" \
  --vanilla_dir "${VANILLA_DIR}" \
  --init_model_dir "${STAGE1_SAVE_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --forget_ratio "${FORGET_RATIO}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num_epochs "${NUM_EPOCHS}" \
  --retain_coef "${RETAIN_COEF}" \
  --kl_coef "${KL_COEF}" \
  --hidden_coef "${HIDDEN_COEF}" \
  --forget_lock_coef "${FORGET_LOCK_COEF}" \
  --forget_margin "${FORGET_MARGIN}" \
  --reg_coef "${REG_COEF}" \
  --start_layer "${START_LAYER}" \
  --end_layer "${END_LAYER}" \
  --column_top_ratio "${COLUMN_TOP_RATIO}"

echo "Done."
