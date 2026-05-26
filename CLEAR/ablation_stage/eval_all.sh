#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jb/code/KVW/CLEAR}"
PYTHON="${PYTHON:-/home/jb/.conda/envs/clear/bin/python}"
MODEL_ID="${MODEL_ID:-${ROOT_DIR}/models/Qwen2-VL-2B-Instruct}"
GPU="${EVAL_GPU:-${GPU:-0}}"

FORGET_RATIO="05"
RETAIN_RATIO="95"
SHOT_NUM="${SHOT_NUM:-zero_shots}"
EVAL_LIST="${EVAL_LIST:-forget retain realface realworld}"
FORCE_RERUN="${FORCE_RERUN:-0}"

CHECKPOINT_ROOT="${ABLATION_STAGE_CHECKPOINT_ROOT:-${ROOT_DIR}/checkpoints/ablation_stage}"
RESULT_ROOT="${ABLATION_STAGE_RESULT_ROOT:-${ROOT_DIR}/checkpoints/ablation_stage_results}"
RETAIN_SAVE_DIR="${RETAIN_SAVE_DIR:-${CHECKPOINT_ROOT}/KVW_RETAIN_ONLY_5}"
CORE_SUMMARY_CSV="${CORE_SUMMARY_CSV:-${ROOT_DIR}/checkpoints/eval_summary/core_all_ratios_summary.csv}"

FORGET_CLS_FOLDER="forget${FORGET_RATIO}_perturbed"
FORGET_GEN_FOLDER="forget${FORGET_RATIO}+tofu"
RETAIN_CLS_FOLDER="retain_perturbed"
RETAIN_GEN_FOLDER="retain${RETAIN_RATIO}+tofu"
REALFACE_FOLDER="real_faces"
REALWORLD_FOLDER="real_world"

cd "${ROOT_DIR}"

eval_one() {
  local method="$1"
  local checkpoint_dir="$2"
  local result_name="$3"
  local output_folder="${RESULT_ROOT}/${result_name}/${SHOT_NUM}/forget${FORGET_RATIO}"
  local result_file="${output_folder}/final_evaluation_results.json"

  if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "[Eval][Skip] ${method}: checkpoint dir not found -> ${checkpoint_dir}"
    return
  fi

  if [[ "${FORCE_RERUN}" != "1" && -f "${result_file}" ]]; then
    echo "[Eval][Reuse] ${method}: ${result_file}"
    return
  fi

  echo "[Eval] ${method} -> ${result_file}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" eval.py \
    --model_id "${MODEL_ID}" \
    --cache_path "${checkpoint_dir}" \
    --eval_list "${EVAL_LIST}" \
    --output_folder "${output_folder}" \
    --shot_num "${SHOT_NUM}" \
    --data_folder "data/CLEAR" \
    --forget_cls_folder "${FORGET_CLS_FOLDER}" \
    --forget_gen_folder "${FORGET_GEN_FOLDER}" \
    --retain_cls_folder "${RETAIN_CLS_FOLDER}" \
    --retain_gen_folder "${RETAIN_GEN_FOLDER}" \
    --realface_folder "${REALFACE_FOLDER}" \
    --realworld_folder "${REALWORLD_FOLDER}"
}

mkdir -p "${RESULT_ROOT}"

echo "[Source] KVW and KVW_TWOSTAGE_FULL metrics are read from ${CORE_SUMMARY_CSV}"
eval_one "KVW_RETAIN_ONLY" "${RETAIN_SAVE_DIR}" "KVW_RETAIN_ONLY"

"${PYTHON}" ablation_stage/summarize_results.py

echo "Done."
echo "Summary: ${ROOT_DIR}/checkpoints/ablation_eval_summary/stage_ablation_forget05_summary.csv"
