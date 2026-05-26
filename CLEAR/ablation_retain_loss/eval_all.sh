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
VARIANTS="${VARIANTS:-drop_retain_lm drop_kl drop_hidden drop_forget_lock drop_reg}"

CHECKPOINT_ROOT="${ABLATION_LOSS_CHECKPOINT_ROOT:-${ROOT_DIR}/checkpoints/ablation_retain_loss}"
RESULT_ROOT="${ABLATION_LOSS_RESULT_ROOT:-${ROOT_DIR}/checkpoints/ablation_retain_loss_results}"
FULL_SOURCE="${FULL_SOURCE:-${ROOT_DIR}/checkpoints/KVW_TWOSTAGE_5}"
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

should_eval_variant() {
  local slug="$1"
  [[ " ${VARIANTS} " == *" all "* || " ${VARIANTS} " == *" ${slug} "* ]]
}

mkdir -p "${RESULT_ROOT}"

echo "[Source] KVW_TWOSTAGE_FULL metrics are read from ${CORE_SUMMARY_CSV}"
if should_eval_variant "drop_retain_lm"; then
  eval_one "DROP_RETAIN_LM" "${CHECKPOINT_ROOT}/drop_retain_lm" "drop_retain_lm"
fi
if should_eval_variant "drop_kl"; then
  eval_one "DROP_KL" "${CHECKPOINT_ROOT}/drop_kl" "drop_kl"
fi
if should_eval_variant "drop_hidden"; then
  eval_one "DROP_HIDDEN" "${CHECKPOINT_ROOT}/drop_hidden" "drop_hidden"
fi
if should_eval_variant "drop_forget_lock"; then
  eval_one "DROP_FORGET_LOCK" "${CHECKPOINT_ROOT}/drop_forget_lock" "drop_forget_lock"
fi
if should_eval_variant "drop_reg"; then
  eval_one "DROP_REG" "${CHECKPOINT_ROOT}/drop_reg" "drop_reg"
fi

"${PYTHON}" ablation_retain_loss/summarize_results.py

echo "Done."
echo "Summary: ${ROOT_DIR}/checkpoints/ablation_eval_summary/retain_loss_ablation_forget05_summary.csv"
