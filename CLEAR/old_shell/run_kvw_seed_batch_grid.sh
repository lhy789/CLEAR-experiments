#!/bin/bash
set -euo pipefail

ROOT_DIR="/home/jb/code/KVW/CLEAR"
PYTHON="${PYTHON:-/home/jb/.conda/envs/clear/bin/python}"
MODEL_ID="${ROOT_DIR}/models/Qwen2-VL-2B-Instruct"
VANILLA_DIR="${ROOT_DIR}/checkpoints/qwen2B_vanilla"
GPU_TRAIN="${GPU_TRAIN:-0,1,2,3}"
GPU_EVAL="${GPU_EVAL:-0}"

FORGET_RATIO="${FORGET_RATIO:-5}"
SEEDS="${SEEDS:-42 123 3407}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
RUNS_PER_CONFIG="${RUNS_PER_CONFIG:-3}"

KVW_NUM_EPOCHS="${KVW_NUM_EPOCHS:-1}"
KVW_GAMMA="${KVW_GAMMA:-0.02}"
KVW_START_LAYER="${KVW_START_LAYER:-1}"
KVW_END_LAYER="${KVW_END_LAYER:-25}"

EXP_NAME="${EXP_NAME:-kvw_seed_batch_grid}"
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

config_name() {
  local seed="$1"
  local batch_size="$2"
  local run_idx="$3"
  printf "seed%s_bs%s_run%s" "${seed}" "${batch_size}" "${run_idx}"
}

kc_cache_dir() {
  printf "%s/kc_cache/shared" "${EXP_ROOT}"
}

write_manifest() {
  local manifest_path="${EXP_ROOT}/run_manifest.txt"
  cat > "${manifest_path}" <<EOF
KVW grid reproduction
experiment_name=${EXP_NAME}
experiment_root=${EXP_ROOT}
resume=${RESUME}
force_rerun=${FORCE_RERUN}
forget_ratio=${FORGET_RATIO}
model_id=${MODEL_ID}
vanilla_dir=${VANILLA_DIR}
seeds=${SEEDS}
batch_sizes=${BATCH_SIZES}
runs_per_config=${RUNS_PER_CONFIG}
num_epochs=${KVW_NUM_EPOCHS}
gamma=${KVW_GAMMA}
start_layer=${KVW_START_LAYER}
end_layer=${KVW_END_LAYER}
notes=Each configuration is stored under ${EXP_ROOT}/seed<seed>_bs<batch>_run<idx>. All runs reuse one shared kc_r cache under ${EXP_ROOT}/kc_cache/shared.
EOF
}

compute_kc_r_if_needed() {
  local cache_dir
  local kc_path

  cache_dir="$(kc_cache_dir)"
  kc_path="${cache_dir}/kc_r_retain_${RETAIN_RATIO_TAG}.pt"

  if [[ -f "${kc_path}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[Reuse] shared kc_r -> ${kc_path}"
    return
  fi

  mkdir -p "${cache_dir}"
  echo "[Run] compute shared kc_r -> ${kc_path}"
  CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" "${PYTHON}" -m baselines.KVW \
    --model_id "${MODEL_ID}" \
    --vanilla_dir "${VANILLA_DIR}" \
    --save_dir "${EXP_ROOT}/kc_cache_placeholder" \
    --forget_ratio "${FORGET_RATIO}" \
    --batch_size 1 \
    --seed 42 \
    --kc_cache_dir "${cache_dir}" \
    --num_epochs 1 \
    --phase compute_kc_r
}

train_kvw() {
  local seed="$1"
  local batch_size="$2"
  local run_idx="$3"
  local save_dir
  local done_file
  local cache_dir

  save_dir="${EXP_ROOT}/$(config_name "${seed}" "${batch_size}" "${run_idx}")"
  done_file="${save_dir}/trainer_config.json"
  cache_dir="$(kc_cache_dir)"

  if [[ -f "${done_file}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[Skip] train seed=${seed} bs=${batch_size} run=${run_idx} -> ${save_dir}"
    return
  fi

  echo "[Run] train seed=${seed} bs=${batch_size} run=${run_idx} -> ${save_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" "${PYTHON}" -m baselines.KVW \
    --model_id "${MODEL_ID}" \
    --vanilla_dir "${VANILLA_DIR}" \
    --save_dir "${save_dir}" \
    --forget_ratio "${FORGET_RATIO}" \
    --batch_size "${batch_size}" \
    --seed "${seed}" \
    --kc_cache_dir "${cache_dir}" \
    --num_epochs "${KVW_NUM_EPOCHS}" \
    --phase weakening \
    --gamma "${KVW_GAMMA}" \
    --start_layer "${KVW_START_LAYER}" \
    --end_layer "${KVW_END_LAYER}"
}

run_eval() {
  local seed="$1"
  local batch_size="$2"
  local run_idx="$3"
  local save_dir
  local output_folder
  local result_file

  save_dir="${EXP_ROOT}/$(config_name "${seed}" "${batch_size}" "${run_idx}")"
  output_folder="${save_dir}/${SHOT_NUM}/forget${FORGET_RATIO_TAG}"
  result_file="${output_folder}/final_evaluation_results.json"

  if [[ ! -d "${save_dir}" ]]; then
    echo "[Skip] eval missing checkpoint -> ${save_dir}"
    return
  fi

  if [[ -f "${result_file}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[Skip] eval seed=${seed} bs=${batch_size} run=${run_idx} -> ${result_file}"
    return
  fi

  echo "[Eval] seed=${seed} bs=${batch_size} run=${run_idx}"
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

for seed in ${SEEDS}; do
  for batch_size in ${BATCH_SIZES}; do
    for run_idx in $(seq 1 "${RUNS_PER_CONFIG}"); do
      train_kvw "${seed}" "${batch_size}" "${run_idx}"
      run_eval "${seed}" "${batch_size}" "${run_idx}"
    done
  done
done

echo ""
echo "Finished."
echo "Experiment root: ${EXP_ROOT}"
echo "Configs:"
for seed in ${SEEDS}; do
  for batch_size in ${BATCH_SIZES}; do
    for run_idx in $(seq 1 "${RUNS_PER_CONFIG}"); do
      echo "  ${EXP_ROOT}/$(config_name "${seed}" "${batch_size}" "${run_idx}")"
    done
  done
done
