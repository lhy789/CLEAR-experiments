#!/bin/bash
set -u

ROOT_DIR="/home/jb/code/KVW/CLEAR"
PYTHON="${PYTHON:-/home/jb/.conda/envs/clear/bin/python}"
MODEL_ID="${ROOT_DIR}/models/Qwen2-VL-2B-Instruct"
VANILLA_DIR="${VANILLA_DIR:-${ROOT_DIR}/checkpoints/qwen2B_vanilla}"
ORACLE_DIR="${ORACLE_DIR:-${ROOT_DIR}/checkpoints/qwen2B_oracle_5}"
GPU_TRAIN="${GPU_TRAIN:-0}"
GPU_EVAL="${GPU_EVAL:-0}"

FORGET_RATIO="${FORGET_RATIO:-5}"
SEEDS="${SEEDS:-42 123 3407}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
RUNS_PER_CONFIG="${RUNS_PER_CONFIG:-3}"

# Keep NPO hyperparameters paper-aligned by default.
# The intended sweep dimension in this script is batch size only.
NPO_NUM_EPOCHS="${NPO_NUM_EPOCHS:-1}"
NPO_LR="${NPO_LR:-1e-5}"
NPO_LCOEF="${NPO_LCOEF:-1}"
NPO_BETA="${NPO_BETA:-0.4}"
NPO_RANK="${NPO_RANK:-8}"

EXP_NAME="${EXP_NAME:-npo_seed_batch_grid}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/checkpoints/${EXP_NAME}}"
RESUME="${RESUME:-1}"
FORCE_RERUN="${FORCE_RERUN:-}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

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

FAILED_CONFIGS=()

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

status_log_path() {
  printf "%s/run_status.log" "${EXP_ROOT}"
}

append_status() {
  local stage="$1"
  local seed="$2"
  local batch_size="$3"
  local run_idx="$4"
  local status="$5"
  local details="${6:-}"
  printf "%s\tstage=%s\tseed=%s\tbs=%s\trun=%s\tstatus=%s\t%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "${stage}" \
    "${seed}" \
    "${batch_size}" \
    "${run_idx}" \
    "${status}" \
    "${details}" >> "$(status_log_path)"
}

write_manifest() {
  local manifest_path="${EXP_ROOT}/run_manifest.txt"
  cat > "${manifest_path}" <<EOF
NPO grid reproduction
experiment_name=${EXP_NAME}
experiment_root=${EXP_ROOT}
resume=${RESUME}
force_rerun=${FORCE_RERUN}
continue_on_error=${CONTINUE_ON_ERROR}
forget_ratio=${FORGET_RATIO}
model_id=${MODEL_ID}
vanilla_dir=${VANILLA_DIR}
oracle_dir=${ORACLE_DIR}
seeds=${SEEDS}
batch_sizes=${BATCH_SIZES}
runs_per_config=${RUNS_PER_CONFIG}
lr=${NPO_LR}
lcoef=${NPO_LCOEF}
beta=${NPO_BETA}
rank=${NPO_RANK}
num_epochs=${NPO_NUM_EPOCHS}
notes=Each configuration is stored under ${EXP_ROOT}/seed<seed>_bs<batch>_run<idx>. Resume mode skips finished train/eval outputs and retries only missing or failed configurations.
paper_alignment=All defaults except batch_size are kept aligned with the repo's paper reproduction setup for CLEAR NPO: lr=${NPO_LR}, num_epochs=${NPO_NUM_EPOCHS}, lcoef=${NPO_LCOEF}, beta=${NPO_BETA}, rank=${NPO_RANK}, oracle_dir=${ORACLE_DIR}.
EOF
}

handle_failure() {
  local stage="$1"
  local seed="$2"
  local batch_size="$3"
  local run_idx="$4"
  local exit_code="$5"
  local save_dir="$6"

  local cfg
  cfg="$(config_name "${seed}" "${batch_size}" "${run_idx}")"
  FAILED_CONFIGS+=("${cfg}:${stage}:exit=${exit_code}")
  append_status "${stage}" "${seed}" "${batch_size}" "${run_idx}" "failed" "save_dir=${save_dir} exit_code=${exit_code}"

  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    echo "[Warn] ${stage} failed for ${cfg} (exit=${exit_code}), continuing."
    return 0
  fi

  echo "[Error] ${stage} failed for ${cfg} (exit=${exit_code}), stopping."
  exit "${exit_code}"
}

train_npo() {
  local seed="$1"
  local batch_size="$2"
  local run_idx="$3"
  local save_dir
  local done_file

  save_dir="${EXP_ROOT}/$(config_name "${seed}" "${batch_size}" "${run_idx}")"
  done_file="${save_dir}/trainer_config.json"

  if [[ -f "${done_file}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[Skip] train seed=${seed} bs=${batch_size} run=${run_idx} -> ${save_dir}"
    append_status "train" "${seed}" "${batch_size}" "${run_idx}" "skipped" "save_dir=${save_dir}"
    return 0
  fi

  mkdir -p "${save_dir}"
  echo "[Run] train seed=${seed} bs=${batch_size} run=${run_idx} -> ${save_dir}"
  append_status "train" "${seed}" "${batch_size}" "${run_idx}" "started" "save_dir=${save_dir}"

  if CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" "${PYTHON}" -m baselines.NPO \
    --model_id "${MODEL_ID}" \
    --vanilla_dir "${VANILLA_DIR}" \
    --oracle_model_id "${ORACLE_DIR}" \
    --save_dir "${save_dir}" \
    --forget_ratio "${FORGET_RATIO}" \
    --batch_size "${batch_size}" \
    --lr "${NPO_LR}" \
    --num_epochs "${NPO_NUM_EPOCHS}" \
    --lcoef "${NPO_LCOEF}" \
    --beta "${NPO_BETA}" \
    --rank "${NPO_RANK}" \
    --seed "${seed}" \
    --data_folder "data/CLEAR"; then
    append_status "train" "${seed}" "${batch_size}" "${run_idx}" "completed" "save_dir=${save_dir}"
    return 0
  fi

  handle_failure "train" "${seed}" "${batch_size}" "${run_idx}" "$?" "${save_dir}"
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

  if [[ ! -f "${save_dir}/trainer_config.json" ]]; then
    echo "[Skip] eval missing completed training -> ${save_dir}"
    append_status "eval" "${seed}" "${batch_size}" "${run_idx}" "skipped" "reason=missing_trainer_config save_dir=${save_dir}"
    return 0
  fi

  if [[ -f "${result_file}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[Skip] eval seed=${seed} bs=${batch_size} run=${run_idx} -> ${result_file}"
    append_status "eval" "${seed}" "${batch_size}" "${run_idx}" "skipped" "result_file=${result_file}"
    return 0
  fi

  mkdir -p "${output_folder}"
  echo "[Eval] seed=${seed} bs=${batch_size} run=${run_idx}"
  append_status "eval" "${seed}" "${batch_size}" "${run_idx}" "started" "output_folder=${output_folder}"

  if CUDA_VISIBLE_DEVICES="${GPU_EVAL}" "${PYTHON}" eval.py \
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
    --realworld_folder "${REALWORLD_FOLDER}"; then
    append_status "eval" "${seed}" "${batch_size}" "${run_idx}" "completed" "result_file=${result_file}"
    return 0
  fi

  handle_failure "eval" "${seed}" "${batch_size}" "${run_idx}" "$?" "${save_dir}"
}

write_manifest

for seed in ${SEEDS}; do
  for batch_size in ${BATCH_SIZES}; do
    for run_idx in $(seq 1 "${RUNS_PER_CONFIG}"); do
      train_npo "${seed}" "${batch_size}" "${run_idx}"
      run_eval "${seed}" "${batch_size}" "${run_idx}"
    done
  done
done

echo ""
echo "Finished."
echo "Experiment root: ${EXP_ROOT}"
echo "Status log: $(status_log_path)"

if [[ "${#FAILED_CONFIGS[@]}" -gt 0 ]]; then
  echo "Failed configs:"
  for failed in "${FAILED_CONFIGS[@]}"; do
    echo "  ${failed}"
  done
else
  echo "All requested configs finished without recorded failures."
fi
