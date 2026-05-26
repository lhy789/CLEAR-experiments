#!/bin/bash
set -euo pipefail

ROOT_DIR="/home/jb/code/KVW/CLEAR"
PYTHON="${PYTHON:-/home/jb/.conda/envs/clear/bin/python}"
MODEL_ID="${ROOT_DIR}/models/Qwen2-VL-2B-Instruct"
GPU="${GPU:-0,1,2}"
FORGET_RATIO="05"
RETAIN_RATIO="95"
SHOT_NUM="zero_shots"
EVAL_LIST="forget retain realface realworld"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints/KVW_TWOSTAGE_5}"
FORCE_RERUN="${FORCE_RERUN:-0}"

FORGET_CLS_FOLDER="forget${FORGET_RATIO}_perturbed"
FORGET_GEN_FOLDER="forget${FORGET_RATIO}+tofu"
RETAIN_CLS_FOLDER="retain_perturbed"
RETAIN_GEN_FOLDER="retain${RETAIN_RATIO}+tofu"
REALFACE_FOLDER="real_faces"
REALWORLD_FOLDER="real_world"

cd "${ROOT_DIR}"

OUTPUT_FOLDER="${SAVE_DIR}/${SHOT_NUM}/forget${FORGET_RATIO}"
RESULT_FILE="${OUTPUT_FOLDER}/final_evaluation_results.json"

if [[ ! -d "${SAVE_DIR}" ]]; then
  echo "[Skip] checkpoint dir not found -> ${SAVE_DIR}"
  exit 0
fi

if [[ "${FORCE_RERUN}" != "1" && -f "${RESULT_FILE}" ]]; then
  echo "[Skip] eval already finished -> ${RESULT_FILE}"
else
  echo "[Eval] ${SAVE_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" eval.py \
    --model_id "${MODEL_ID}" \
    --cache_path "${SAVE_DIR}" \
    --eval_list "${EVAL_LIST}" \
    --output_folder "${OUTPUT_FOLDER}" \
    --shot_num "${SHOT_NUM}" \
    --data_folder "data/CLEAR" \
    --forget_cls_folder "${FORGET_CLS_FOLDER}" \
    --forget_gen_folder "${FORGET_GEN_FOLDER}" \
    --retain_cls_folder "${RETAIN_CLS_FOLDER}" \
    --retain_gen_folder "${RETAIN_GEN_FOLDER}" \
    --realface_folder "${REALFACE_FOLDER}" \
    --realworld_folder "${REALWORLD_FOLDER}"
fi

"${PYTHON}" kvw_two_stage/summarize_results.py
"${PYTHON}" aggregate_all_results.py
"${PYTHON}" compare_new_methods.py

echo "Done."
