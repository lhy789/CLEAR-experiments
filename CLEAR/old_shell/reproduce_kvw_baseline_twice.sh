#!/bin/bash
set -euo pipefail

ROOT_DIR="/home/jb/code/KVW/CLEAR"
PYTHON="${PYTHON:-/home/jb/.conda/envs/clear/bin/python}"
MODEL_ID="${ROOT_DIR}/models/Qwen2-VL-2B-Instruct"
VANILLA_DIR="${ROOT_DIR}/checkpoints/qwen2B_vanilla"
ORACLE_DIR="${ROOT_DIR}/checkpoints/qwen2B_oracle_5"
GPU_TRAIN="${GPU_TRAIN:-0,1,2,3}"
GPU_EVAL="${GPU_EVAL:-0}"

# Paper-aligned CLEAR/Qwen2-VL-2B setup
FORGET_RATIO="${FORGET_RATIO:-5}"
GLOBAL_SEED="${GLOBAL_SEED:-42}"
KVW_SEED="${KVW_SEED:-${GLOBAL_SEED}}"
KVW_BATCH_SIZE="${KVW_BATCH_SIZE:-4}"
KVW_NUM_EPOCHS="${KVW_NUM_EPOCHS:-1}"
KVW_GAMMA="${KVW_GAMMA:-0.02}"
KVW_START_LAYER="${KVW_START_LAYER:-1}"
KVW_END_LAYER="${KVW_END_LAYER:-25}"

# Training baselines in the paper use LoRA rank 8 on CLEAR.
# Override with BASELINE_METHODS="ga gd kl npo" if needed.
BASELINE_METHODS="${BASELINE_METHODS:-ga gd kl npo}"
BASELINE_SEED="${BASELINE_SEED:-${GLOBAL_SEED}}"
GA_SEED="${GA_SEED:-${BASELINE_SEED}}"
GD_SEED="${GD_SEED:-${BASELINE_SEED}}"
KL_SEED="${KL_SEED:-${BASELINE_SEED}}"
NPO_SEED="${NPO_SEED:-${BASELINE_SEED}}"
BASELINE_LR="${BASELINE_LR:-1e-5}"
BASELINE_NUM_EPOCHS="${BASELINE_NUM_EPOCHS:-1}"
BASELINE_RANK="${BASELINE_RANK:-8}"
BASELINE_LCOEF="${BASELINE_LCOEF:-1}"
BASELINE_BETA="${BASELINE_BETA:-0.4}"
BASELINE_BATCH_SIZE_OVERRIDE="${BASELINE_BATCH_SIZE_OVERRIDE:-}"
GA_BATCH_SIZE="${GA_BATCH_SIZE:-4}"
GD_BATCH_SIZE="${GD_BATCH_SIZE:-4}"
KL_BATCH_SIZE="${KL_BATCH_SIZE:-4}"
NPO_BATCH_SIZE="${NPO_BATCH_SIZE:-4}"

RUNS="${RUNS:-2}"
EXP_NAME="${EXP_NAME:-baseline_reproduction}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/checkpoints/${EXP_NAME}}"
RESUME="${RESUME:-1}"
FORCE_RERUN="${FORCE_RERUN:-}"

SHOT_NUM="zero_shots"
EVAL_LIST="forget retain realface realworld"
FORGET_RATIO_TAG="$(printf "%02d" "${FORGET_RATIO}")"
RETAIN_RATIO_TAG="$(printf "%02d" "$((100 - FORGET_RATIO))")"
FORGET_CLS_FOLDER="forget${FORGET_RATIO_TAG}_perturbed"
FORGET_GEN_FOLDER="forget${FORGET_RATIO_TAG}+tofu"
RETAIN_CLS_FOLDER="retain_perturbed"
RETAIN_GEN_FOLDER="retain${RETAIN_RATIO_TAG}+tofu"
REALFACE_FOLDER="real_faces"
REALWORLD_FOLDER="real_world"

cd "${ROOT_DIR}"
mkdir -p "${EXP_ROOT}"

if [[ -z "${FORCE_RERUN}" ]]; then
  if [[ "${RESUME}" == "1" ]]; then
    FORCE_RERUN="0"
  else
    FORCE_RERUN="1"
  fi
fi

baseline_batch_size() {
  local method="${1:-}"
  if [[ -n "${BASELINE_BATCH_SIZE_OVERRIDE}" ]]; then
    printf "%s" "${BASELINE_BATCH_SIZE_OVERRIDE}"
    return
  fi

  case "${method}" in
    ga) printf "%s" "${GA_BATCH_SIZE}" ;;
    gd) printf "%s" "${GD_BATCH_SIZE}" ;;
    kl) printf "%s" "${KL_BATCH_SIZE}" ;;
    npo) printf "%s" "${NPO_BATCH_SIZE}" ;;
    *)
      echo "Unsupported baseline method=${method}. Use ga|gd|kl|npo." >&2
      exit 1
      ;;
  esac
}

experiment_seed() {
  local method="${1:-}"
  case "${method}" in
    kvw) printf "%s" "${KVW_SEED}" ;;
    ga) printf "%s" "${GA_SEED}" ;;
    gd) printf "%s" "${GD_SEED}" ;;
    kl) printf "%s" "${KL_SEED}" ;;
    npo) printf "%s" "${NPO_SEED}" ;;
    *)
      echo "Unsupported experiment method=${method}. Use kvw|ga|gd|kl|npo." >&2
      exit 1
      ;;
  esac
}

baseline_module() {
  local method="${1:-}"
  case "${method}" in
    ga) printf "baselines.GA" ;;
    gd) printf "baselines.GA_Diff" ;;
    kl) printf "baselines.KL_Min" ;;
    npo) printf "baselines.NPO" ;;
    *)
      echo "Unsupported baseline method=${method}. Use ga|gd|kl|npo." >&2
      exit 1
      ;;
  esac
}

write_manifest() {
  local manifest_path="${EXP_ROOT}/run_manifest.txt"
  cat > "${manifest_path}" <<EOF
Paper-aligned CLEAR reproduction
experiment_name=${EXP_NAME}
experiment_root=${EXP_ROOT}
resume=${RESUME}
force_rerun=${FORCE_RERUN}
forget_ratio=${FORGET_RATIO}
model_id=${MODEL_ID}
vanilla_dir=${VANILLA_DIR}
oracle_dir=${ORACLE_DIR}

[KVW]
seed=${KVW_SEED}
batch_size=${KVW_BATCH_SIZE}
num_epochs=${KVW_NUM_EPOCHS}
gamma=${KVW_GAMMA}
start_layer=${KVW_START_LAYER}
end_layer=${KVW_END_LAYER}
notes=On CLEAR, the paper searches gamma in [0.01, 0.03] and layer range near start in [0,4], end in [23,27]. This script uses the repo defaults selected for the reported setup: gamma=0.02, start_layer=1, end_layer=25.

[Baseline]
methods=${BASELINE_METHODS}
seed_default=${BASELINE_SEED}
lr=${BASELINE_LR}
num_epochs=${BASELINE_NUM_EPOCHS}
rank=${BASELINE_RANK}
lcoef=${BASELINE_LCOEF}
beta=${BASELINE_BETA}
notes=Paper states GA/GD/KL/NPO use LoRA rank 8 on CLEAR.
EOF

  for method in ${BASELINE_METHODS}; do
    cat >> "${manifest_path}" <<EOF
- ${method}: module=$(baseline_module "${method}"), seed=$(experiment_seed "${method}"), batch_size=$(baseline_batch_size "${method}")
EOF
  done
}

compute_kc_r_if_needed() {
  local kc_path="${ROOT_DIR}/kc/kc_r_retain_${RETAIN_RATIO_TAG}.pt"
  if [[ -f "${kc_path}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[Reuse] kc_r -> ${kc_path}"
    return
  fi

  echo "[Run] compute kc_r -> ${kc_path}"
  CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" "${PYTHON}" -m baselines.KVW \
    --model_id "${MODEL_ID}" \
    --vanilla_dir "${VANILLA_DIR}" \
    --save_dir "${EXP_ROOT}/kc_cache_placeholder" \
    --forget_ratio "${FORGET_RATIO}" \
    --batch_size "${KVW_BATCH_SIZE}" \
    --seed "$(experiment_seed kvw)" \
    --num_epochs 1 \
    --phase compute_kc_r
}

train_kvw() {
  local run_idx="$1"
  local save_dir="${EXP_ROOT}/KVW_run${run_idx}"
  local done_file="${save_dir}/trainer_config.json"

  if [[ -f "${done_file}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[Skip] KVW run ${run_idx} -> ${save_dir}"
    return
  fi

  echo "[Run] KVW run ${run_idx} -> ${save_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" "${PYTHON}" -m baselines.KVW \
    --model_id "${MODEL_ID}" \
    --vanilla_dir "${VANILLA_DIR}" \
    --save_dir "${save_dir}" \
    --forget_ratio "${FORGET_RATIO}" \
    --batch_size "${KVW_BATCH_SIZE}" \
    --seed "$(experiment_seed kvw)" \
    --num_epochs "${KVW_NUM_EPOCHS}" \
    --phase weakening \
    --gamma "${KVW_GAMMA}" \
    --start_layer "${KVW_START_LAYER}" \
    --end_layer "${KVW_END_LAYER}"
}

train_baseline() {
  local method="$1"
  local run_idx="$2"
  local save_dir="${EXP_ROOT}/$(printf "%s" "${method}" | tr '[:lower:]' '[:upper:]')_run${run_idx}"
  local done_file="${save_dir}/trainer_config.json"
  local module
  local batch_size

  module="$(baseline_module "${method}")"
  batch_size="$(baseline_batch_size "${method}")"

  if [[ -f "${done_file}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[Skip] ${method} run ${run_idx} -> ${save_dir}"
    return
  fi

  echo "[Run] ${method} run ${run_idx} -> ${save_dir}"

  if [[ "${method}" == "ga" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" "${PYTHON}" -m "${module}" \
      --model_id "${MODEL_ID}" \
      --vanilla_dir "${VANILLA_DIR}" \
      --lr "${BASELINE_LR}" \
      --batch_size "${batch_size}" \
      --seed "$(experiment_seed "${method}")" \
      --num_epochs "${BASELINE_NUM_EPOCHS}" \
      --forget_ratio "${FORGET_RATIO}" \
      --data_folder data/CLEAR \
      --rank "${BASELINE_RANK}" \
      --save_dir "${save_dir}"
    return
  fi

  if [[ "${method}" == "npo" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" "${PYTHON}" -m "${module}" \
      --model_id "${MODEL_ID}" \
      --vanilla_dir "${VANILLA_DIR}" \
      --lr "${BASELINE_LR}" \
      --batch_size "${batch_size}" \
      --seed "$(experiment_seed "${method}")" \
      --num_epochs "${BASELINE_NUM_EPOCHS}" \
      --forget_ratio "${FORGET_RATIO}" \
      --lcoef "${BASELINE_LCOEF}" \
      --beta "${BASELINE_BETA}" \
      --data_folder data/CLEAR \
      --rank "${BASELINE_RANK}" \
      --save_dir "${save_dir}" \
      --oracle_model_id "${ORACLE_DIR}"
    return
  fi

  CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" "${PYTHON}" -m "${module}" \
    --model_id "${MODEL_ID}" \
    --vanilla_dir "${VANILLA_DIR}" \
    --lr "${BASELINE_LR}" \
    --batch_size "${batch_size}" \
    --seed "$(experiment_seed "${method}")" \
    --num_epochs "${BASELINE_NUM_EPOCHS}" \
    --forget_ratio "${FORGET_RATIO}" \
    --lcoef "${BASELINE_LCOEF}" \
    --data_folder data/CLEAR \
    --rank "${BASELINE_RANK}" \
    --save_dir "${save_dir}"
}

run_eval() {
  local save_dir="$1"
  local output_folder="${save_dir}/${SHOT_NUM}/forget${FORGET_RATIO_TAG}"
  local result_file="${output_folder}/final_evaluation_results.json"

  if [[ ! -d "${save_dir}" ]]; then
    echo "[Skip] eval: missing checkpoint dir -> ${save_dir}"
    return
  fi

  if [[ -f "${result_file}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[Skip] eval already done -> ${result_file}"
    return
  fi

  echo "[Eval] ${save_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_EVAL}" "${PYTHON}" eval.py \
    --model_id "${MODEL_ID}" \
    --cache_path "${save_dir}" \
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

write_manifest
compute_kc_r_if_needed

for run_idx in $(seq 1 "${RUNS}"); do
  train_kvw "${run_idx}"
  run_eval "${EXP_ROOT}/KVW_run${run_idx}"
  for method in ${BASELINE_METHODS}; do
    train_baseline "${method}" "${run_idx}"
    run_eval "${EXP_ROOT}/$(printf "%s" "${method}" | tr '[:lower:]' '[:upper:]')_run${run_idx}"
  done
done

echo ""
echo "Finished."
echo "Experiment root: ${EXP_ROOT}"
echo "KVW dirs:"
for run_idx in $(seq 1 "${RUNS}"); do
  echo "  ${EXP_ROOT}/KVW_run${run_idx}"
done
echo "Baseline dirs:"
for method in ${BASELINE_METHODS}; do
  for run_idx in $(seq 1 "${RUNS}"); do
    echo "  ${EXP_ROOT}/$(printf "%s" "${method}" | tr '[:lower:]' '[:upper:]')_run${run_idx}"
  done
done
