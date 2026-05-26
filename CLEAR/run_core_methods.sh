#!/usr/bin/env bash
set -euo pipefail

# CLEAR 核心方法统一运行脚本：
# vanilla、oracle、GA、GD、KL、NPO、MMU、KVW、KVW 两阶段。
#
# 默认运行三种遗忘比例：1%、5%、10%。
# 使用示例：
#   bash run_core_methods.sh
#   FORGET_RATIOS="5" METHODS="MMU" bash run_core_methods.sh
#   GPU="0,1" EVAL_GPU="0" GA_BATCH_SIZE=4 bash run_core_methods.sh

ROOT_DIR="${ROOT_DIR:-/home/jb/code/KVW/CLEAR}"
PYTHON="${PYTHON:-/home/jb/.conda/envs/clear/bin/python}"
MODEL_ID="${MODEL_ID:-${ROOT_DIR}/models/Qwen2-VL-2B-Instruct}"
DATA_FOLDER="${DATA_FOLDER:-data/CLEAR}"
SUMMARY_DIR="${SUMMARY_DIR:-${ROOT_DIR}/checkpoints/eval_summary}"

# GPU 会用于所有训练和评估命令；如果单独设置 EVAL_GPU，则评估使用 EVAL_GPU。
GPU="${GPU:-0,1,2,3}"
EVAL_GPU="${EVAL_GPU:-${GPU}}"

# 支持的遗忘比例是 1、5、10；只跑单个比例可设 FORGET_RATIOS="5"。
FORGET_RATIOS="${FORGET_RATIOS:-1 10}"

# 方法名大小写不敏感；KVW_TWOSTAGE 也兼容 KVW_TWO_STAGE / TWOSTAGE 等写法。
METHODS="${METHODS:-vanilla oracle GA GD KL NPO MMU KVW KVW_TWOSTAGE}"

# FORCE_RERUN=0 时，如果 final_evaluation_results.json 已存在，就跳过训练和评估。
# FORCE_RERUN=1 时，即使已有 checkpoint 或结果，也会重新训练/评估。
FORCE_RERUN="${FORCE_RERUN:-0}"

SHOT_NUM="${SHOT_NUM:-zero_shots}"
EVAL_LIST="${EVAL_LIST:-forget retain realface realworld}"

# vanilla / oracle 微调超参，与现有 forget5 脚本保持一致。
VANILLA_BATCH_SIZE="${VANILLA_BATCH_SIZE:-4}"
ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-4}"
FINETUNE_LR="${FINETUNE_LR:-1e-4}"
FINETUNE_NUM_EPOCHS="${FINETUNE_NUM_EPOCHS:-1}"
FINETUNE_RANK="${FINETUNE_RANK:-16}"

# GA / GD / KL / NPO 遗忘超参，与 CLEAR/forget.sh 保持一致。
GA_BATCH_SIZE="${GA_BATCH_SIZE:-2}"
GD_BATCH_SIZE="${GD_BATCH_SIZE:-1}"
KL_BATCH_SIZE="${KL_BATCH_SIZE:-1}"
NPO_BATCH_SIZE="${NPO_BATCH_SIZE:-1}"
BASELINE_LR="${BASELINE_LR:-1e-5}"
BASELINE_NUM_EPOCHS="${BASELINE_NUM_EPOCHS:-1}"
BASELINE_RANK="${BASELINE_RANK:-8}"
LCOEF="${LCOEF:-1}"
BETA="${BETA:-0.4}"

# MMU 需要 saliency mask；如果对应比例的 mask 不存在，本脚本会先自动生成。
MMU_BATCH_SIZE="${MMU_BATCH_SIZE:-2}"
MMU_MASK_BATCH_SIZE="${MMU_MASK_BATCH_SIZE:-2}"
MMU_LR="${MMU_LR:-1e-5}"
MMU_NUM_EPOCHS="${MMU_NUM_EPOCHS:-1}"
MMU_MASK_DIR="${MMU_MASK_DIR:-${ROOT_DIR}/path_to_save_mask}"

# KVW 超参，与 CLEAR/kvw.sh 和当前 forget5 结果保持一致。
KVW_BATCH_SIZE="${KVW_BATCH_SIZE:-1}"
KVW_KC_BATCH_SIZE="${KVW_KC_BATCH_SIZE:-1}"
KVW_NUM_EPOCHS="${KVW_NUM_EPOCHS:-1}"
KVW_GAMMA="${KVW_GAMMA:-0.02}"
KVW_START_LAYER="${KVW_START_LAYER:-1}"
KVW_END_LAYER="${KVW_END_LAYER:-25}"
KVW_KC_CACHE_DIR="${KVW_KC_CACHE_DIR:-kc}"

# KVW 两阶段超参，与 CLEAR/kvw_two_stage/run_all.sh 保持一致。
TWOSTAGE_BATCH_SIZE="${TWOSTAGE_BATCH_SIZE:-1}"
TWOSTAGE_LR="${TWOSTAGE_LR:-1e-5}"
TWOSTAGE_NUM_EPOCHS="${TWOSTAGE_NUM_EPOCHS:-1}"
TWOSTAGE_START_LAYER="${TWOSTAGE_START_LAYER:-0}"
TWOSTAGE_END_LAYER="${TWOSTAGE_END_LAYER:-1000}"
TWOSTAGE_STAGE1_GAMMA="${TWOSTAGE_STAGE1_GAMMA:-0.02}"
TWOSTAGE_COLUMN_TOP_RATIO="${TWOSTAGE_COLUMN_TOP_RATIO:-0.1}"
TWOSTAGE_RETAIN_COEF="${TWOSTAGE_RETAIN_COEF:-1.0}"
TWOSTAGE_KL_COEF="${TWOSTAGE_KL_COEF:-0.5}"
TWOSTAGE_HIDDEN_COEF="${TWOSTAGE_HIDDEN_COEF:-1.0}"
TWOSTAGE_FORGET_LOCK_COEF="${TWOSTAGE_FORGET_LOCK_COEF:-0.5}"
TWOSTAGE_FORGET_MARGIN="${TWOSTAGE_FORGET_MARGIN:-0.2}"
TWOSTAGE_REG_COEF="${TWOSTAGE_REG_COEF:-1e-4}"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/../.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

cd "${ROOT_DIR}"

ratio2() {
  printf "%02d" "$1"
}

normalize_method() {
  local method_upper
  method_upper="$(printf "%s" "$1" | tr '[:lower:]-' '[:upper:]_')"
  case "${method_upper}" in
    VANILLA) echo "vanilla" ;;
    ORACLE) echo "oracle" ;;
    GA) echo "GA" ;;
    GD|GA_DIFF) echo "GD" ;;
    KL|KL_MIN) echo "KL" ;;
    NPO) echo "NPO" ;;
    MMU) echo "MMU" ;;
    KVW) echo "KVW" ;;
    KVW_TWOSTAGE|KVW_TWO_STAGE|TWOSTAGE|TWO_STAGE) echo "KVW_TWOSTAGE" ;;
    *)
      echo "Unknown method: $1" >&2
      return 1
      ;;
  esac
}

validate_ratio() {
  case "$1" in
    1|5|10) ;;
    *)
      echo "Unsupported forget ratio: $1. Use 1, 5, or 10." >&2
      exit 1
      ;;
  esac
}

checkpoint_dir_for() {
  local method="$1"
  local ratio="$2"
  local ratio_padded
  ratio_padded="$(ratio2 "${ratio}")"
  case "${method}" in
    vanilla) echo "checkpoints/qwen2B_vanilla" ;;
    oracle) echo "checkpoints/qwen2B_oracle_${ratio}" ;;
    GA) echo "checkpoints/GA_${ratio}" ;;
    GD) echo "checkpoints/GD_${ratio}" ;;
    KL) echo "checkpoints/KL_${ratio}" ;;
    NPO) echo "checkpoints/NPO_${ratio}" ;;
    MMU) echo "checkpoints/MMU_${ratio}" ;;
    KVW) echo "checkpoints/KVW_${ratio_padded}" ;;
    KVW_TWOSTAGE) echo "checkpoints/KVW_TWOSTAGE_${ratio}" ;;
    *) return 1 ;;
  esac
}

result_file_for() {
  local method="$1"
  local ratio="$2"
  local ckpt
  ckpt="$(checkpoint_dir_for "${method}" "${ratio}")"
  echo "${ROOT_DIR}/${ckpt}/${SHOT_NUM}/forget$(ratio2 "${ratio}")/final_evaluation_results.json"
}

checkpoint_done() {
  local ckpt="$1"
  [[ -f "${ROOT_DIR}/${ckpt}/trainer_config.json" ]]
}

result_done() {
  local method="$1"
  local ratio="$2"
  [[ -f "$(result_file_for "${method}" "${ratio}")" ]]
}

run_python_train() {
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "$@"
}

run_python_eval() {
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON}" "$@"
}

train_vanilla() {
  local ckpt="checkpoints/qwen2B_vanilla"
  if [[ "${FORCE_RERUN}" != "1" ]] && checkpoint_done "${ckpt}"; then
    echo "[Train][Skip] vanilla -> ${ckpt}"
    return
  fi
  echo "[Train] vanilla -> ${ckpt}"
  run_python_train finetune.py \
    --model_id "${MODEL_ID}" \
    --save_dir "${ckpt}" \
    --batch_size "${VANILLA_BATCH_SIZE}" \
    --lr "${FINETUNE_LR}" \
    --num_epochs "${FINETUNE_NUM_EPOCHS}" \
    --forget_ratio 5 \
    --rank "${FINETUNE_RANK}" \
    --is_oracle False
}

train_oracle() {
  local ratio="$1"
  local ckpt
  ckpt="$(checkpoint_dir_for oracle "${ratio}")"
  if [[ "${FORCE_RERUN}" != "1" ]] && checkpoint_done "${ckpt}"; then
    echo "[Train][Skip] oracle forget${ratio} -> ${ckpt}"
    return
  fi
  echo "[Train] oracle forget${ratio} -> ${ckpt}"
  run_python_train finetune.py \
    --model_id "${MODEL_ID}" \
    --save_dir "${ckpt}" \
    --batch_size "${ORACLE_BATCH_SIZE}" \
    --lr "${FINETUNE_LR}" \
    --num_epochs "${FINETUNE_NUM_EPOCHS}" \
    --forget_ratio "${ratio}" \
    --rank "${FINETUNE_RANK}" \
    --is_oracle True
}

train_baseline() {
  local method="$1"
  local ratio="$2"
  local ckpt
  ckpt="$(checkpoint_dir_for "${method}" "${ratio}")"
  if [[ "${FORCE_RERUN}" != "1" ]] && checkpoint_done "${ckpt}"; then
    echo "[Train][Skip] ${method} forget${ratio} -> ${ckpt}"
    return
  fi

  case "${method}" in
    GA)
      echo "[Train] GA forget${ratio} -> ${ckpt}"
      run_python_train -m baselines.GA \
        --model_id "${MODEL_ID}" \
        --vanilla_dir checkpoints/qwen2B_vanilla \
        --lr "${BASELINE_LR}" \
        --batch_size "${GA_BATCH_SIZE}" \
        --num_epochs "${BASELINE_NUM_EPOCHS}" \
        --forget_ratio "${ratio}" \
        --data_folder "${DATA_FOLDER}" \
        --rank "${BASELINE_RANK}" \
        --save_dir "${ckpt}"
      ;;
    GD)
      echo "[Train] GD forget${ratio} -> ${ckpt}"
      run_python_train -m baselines.GA_Diff \
        --model_id "${MODEL_ID}" \
        --vanilla_dir checkpoints/qwen2B_vanilla \
        --lr "${BASELINE_LR}" \
        --batch_size "${GD_BATCH_SIZE}" \
        --num_epochs "${BASELINE_NUM_EPOCHS}" \
        --forget_ratio "${ratio}" \
        --lcoef "${LCOEF}" \
        --data_folder "${DATA_FOLDER}" \
        --rank "${BASELINE_RANK}" \
        --save_dir "${ckpt}"
      ;;
    KL)
      echo "[Train] KL forget${ratio} -> ${ckpt}"
      run_python_train -m baselines.KL_Min \
        --model_id "${MODEL_ID}" \
        --vanilla_dir checkpoints/qwen2B_vanilla \
        --lr "${BASELINE_LR}" \
        --batch_size "${KL_BATCH_SIZE}" \
        --num_epochs "${BASELINE_NUM_EPOCHS}" \
        --forget_ratio "${ratio}" \
        --lcoef "${LCOEF}" \
        --data_folder "${DATA_FOLDER}" \
        --rank "${BASELINE_RANK}" \
        --save_dir "${ckpt}"
      ;;
    NPO)
      echo "[Train] NPO forget${ratio} -> ${ckpt}"
      run_python_train -m baselines.NPO \
        --model_id "${MODEL_ID}" \
        --vanilla_dir checkpoints/qwen2B_vanilla \
        --lr "${BASELINE_LR}" \
        --batch_size "${NPO_BATCH_SIZE}" \
        --num_epochs "${BASELINE_NUM_EPOCHS}" \
        --forget_ratio "${ratio}" \
        --lcoef "${LCOEF}" \
        --beta "${BETA}" \
        --data_folder "${DATA_FOLDER}" \
        --rank "${BASELINE_RANK}" \
        --save_dir "${ckpt}" \
        --oracle_model_id "checkpoints/qwen2B_oracle_${ratio}"
      ;;
    *) return 1 ;;
  esac
}

ensure_mmu_mask() {
  local ratio="$1"
  local mask="${MMU_MASK_DIR}/forget${ratio}/language_mask.pt"
  if [[ "${FORCE_RERUN}" != "1" && -f "${mask}" ]]; then
    echo "[Mask][Skip] MMU forget${ratio} -> ${mask}"
    return
  fi
  echo "[Mask] MMU forget${ratio} -> ${mask}"
  run_python_train data_process/gen_mask.py \
    --model_id "${MODEL_ID}" \
    --model_path checkpoints/qwen2B_vanilla \
    --forget_ratio "${ratio}" \
    --data_folder "${DATA_FOLDER}" \
    --output_dir "${MMU_MASK_DIR}" \
    --batch_size "${MMU_MASK_BATCH_SIZE}" \
    --lr "${MMU_LR}"
}

train_mmu() {
  local ratio="$1"
  local ckpt
  local mask
  ckpt="$(checkpoint_dir_for MMU "${ratio}")"
  mask="${MMU_MASK_DIR}/forget${ratio}/language_mask.pt"
  if [[ "${FORCE_RERUN}" != "1" ]] && checkpoint_done "${ckpt}"; then
    echo "[Train][Skip] MMU forget${ratio} -> ${ckpt}"
    return
  fi
  ensure_mmu_mask "${ratio}"
  echo "[Train] MMU forget${ratio} -> ${ckpt}"
  run_python_train -m baselines.MMU \
    --model_id "${MODEL_ID}" \
    --vanilla_dir checkpoints/qwen2B_vanilla \
    --forget_ratio "${ratio}" \
    --save_dir "${ckpt}" \
    --batch_size "${MMU_BATCH_SIZE}" \
    --lr "${MMU_LR}" \
    --num_epochs "${MMU_NUM_EPOCHS}" \
    --grad_mask_path "${mask}"
}

ensure_kvw_kc() {
  local ratio="$1"
  local retain=$((100 - ratio))
  local kc_path="${ROOT_DIR}/${KVW_KC_CACHE_DIR}/kc_r_retain_$(ratio2 "${retain}").pt"
  if [[ "${FORCE_RERUN}" != "1" && -f "${kc_path}" ]]; then
    echo "[KC][Skip] KVW retain${retain} -> ${kc_path}"
    return
  fi
  echo "[KC] KVW retain${retain} -> ${kc_path}"
  run_python_train -m baselines.KVW \
    --model_id "${MODEL_ID}" \
    --vanilla_dir checkpoints/qwen2B_vanilla \
    --forget_ratio "${ratio}" \
    --batch_size "${KVW_KC_BATCH_SIZE}" \
    --num_epochs 1 \
    --phase compute_kc_r \
    --data_folder "${DATA_FOLDER}" \
    --kc_cache_dir "${KVW_KC_CACHE_DIR}" \
    --save_dir "checkpoints/KVW_KC_PLACEHOLDER_${ratio}"
}

train_kvw() {
  local ratio="$1"
  local ckpt
  ckpt="$(checkpoint_dir_for KVW "${ratio}")"
  if [[ "${FORCE_RERUN}" != "1" ]] && checkpoint_done "${ckpt}"; then
    echo "[Train][Skip] KVW forget${ratio} -> ${ckpt}"
    return
  fi
  ensure_kvw_kc "${ratio}"
  echo "[Train] KVW forget${ratio} -> ${ckpt}"
  run_python_train -m baselines.KVW \
    --model_id "${MODEL_ID}" \
    --vanilla_dir checkpoints/qwen2B_vanilla \
    --forget_ratio "${ratio}" \
    --batch_size "${KVW_BATCH_SIZE}" \
    --num_epochs "${KVW_NUM_EPOCHS}" \
    --phase weakening \
    --data_folder "${DATA_FOLDER}" \
    --kc_cache_dir "${KVW_KC_CACHE_DIR}" \
    --gamma "${KVW_GAMMA}" \
    --start_layer "${KVW_START_LAYER}" \
    --end_layer "${KVW_END_LAYER}" \
    --save_dir "${ckpt}"
}

train_kvw_twostage() {
  local ratio="$1"
  local stage1="checkpoints/KVW_STAGE1_FORGET${ratio}_L${TWOSTAGE_START_LAYER}_${TWOSTAGE_END_LAYER}"
  local ckpt
  ckpt="$(checkpoint_dir_for KVW_TWOSTAGE "${ratio}")"

  ensure_kvw_kc "${ratio}"

  if [[ "${FORCE_RERUN}" != "1" ]] && checkpoint_done "${stage1}"; then
    echo "[Train][Skip] KVW two-stage stage1 forget${ratio} -> ${stage1}"
  else
    echo "[Train] KVW two-stage stage1 forget${ratio} -> ${stage1}"
    run_python_train -m baselines.KVW \
      --model_id "${MODEL_ID}" \
      --vanilla_dir checkpoints/qwen2B_vanilla \
      --forget_ratio "${ratio}" \
      --batch_size "${TWOSTAGE_BATCH_SIZE}" \
      --num_epochs 1 \
      --phase weakening \
      --data_folder "${DATA_FOLDER}" \
      --kc_cache_dir "${KVW_KC_CACHE_DIR}" \
      --gamma "${TWOSTAGE_STAGE1_GAMMA}" \
      --start_layer "${TWOSTAGE_START_LAYER}" \
      --end_layer "${TWOSTAGE_END_LAYER}" \
      --save_dir "${stage1}"
  fi

  if [[ "${FORCE_RERUN}" != "1" ]] && checkpoint_done "${ckpt}"; then
    echo "[Train][Skip] KVW two-stage recovery forget${ratio} -> ${ckpt}"
    return
  fi

  echo "[Train] KVW two-stage recovery forget${ratio} -> ${ckpt}"
  run_python_train -m baselines.KVW_TwoStageRecovery \
    --model_id "${MODEL_ID}" \
    --vanilla_dir checkpoints/qwen2B_vanilla \
    --init_model_dir "${stage1}" \
    --save_dir "${ckpt}" \
    --forget_ratio "${ratio}" \
    --batch_size "${TWOSTAGE_BATCH_SIZE}" \
    --lr "${TWOSTAGE_LR}" \
    --num_epochs "${TWOSTAGE_NUM_EPOCHS}" \
    --retain_coef "${TWOSTAGE_RETAIN_COEF}" \
    --kl_coef "${TWOSTAGE_KL_COEF}" \
    --hidden_coef "${TWOSTAGE_HIDDEN_COEF}" \
    --forget_lock_coef "${TWOSTAGE_FORGET_LOCK_COEF}" \
    --forget_margin "${TWOSTAGE_FORGET_MARGIN}" \
    --reg_coef "${TWOSTAGE_REG_COEF}" \
    --start_layer "${TWOSTAGE_START_LAYER}" \
    --end_layer "${TWOSTAGE_END_LAYER}" \
    --column_top_ratio "${TWOSTAGE_COLUMN_TOP_RATIO}" \
    --kc_cache_dir "${KVW_KC_CACHE_DIR}"
}

train_method() {
  local method="$1"
  local ratio="$2"
  case "${method}" in
    vanilla) train_vanilla ;;
    oracle) train_oracle "${ratio}" ;;
    GA|GD|KL|NPO) train_baseline "${method}" "${ratio}" ;;
    MMU) train_mmu "${ratio}" ;;
    KVW) train_kvw "${ratio}" ;;
    KVW_TWOSTAGE) train_kvw_twostage "${ratio}" ;;
    *) return 1 ;;
  esac
}

eval_method() {
  local method="$1"
  local ratio="$2"
  local ratio_padded
  local retain
  local ckpt
  local output_folder
  local result_file
  ratio_padded="$(ratio2 "${ratio}")"
  retain=$((100 - ratio))
  ckpt="$(checkpoint_dir_for "${method}" "${ratio}")"
  output_folder="${ckpt}/${SHOT_NUM}/forget${ratio_padded}"
  result_file="${ROOT_DIR}/${output_folder}/final_evaluation_results.json"

  if [[ "${FORCE_RERUN}" != "1" && -f "${result_file}" ]]; then
    echo "[Eval][Skip] ${method} forget${ratio} -> ${result_file}"
    return
  fi
  if [[ ! -d "${ckpt}" ]]; then
    echo "[Eval][Error] checkpoint dir not found for ${method}: ${ckpt}" >&2
    return 1
  fi

  echo "[Eval] ${method} forget${ratio} -> ${result_file}"
  run_python_eval eval.py \
    --model_id "${MODEL_ID}" \
    --cache_path "${ckpt}" \
    --eval_list "${EVAL_LIST}" \
    --output_folder "${output_folder}" \
    --shot_num "${SHOT_NUM}" \
    --data_folder "${DATA_FOLDER}" \
    --forget_cls_folder "forget${ratio_padded}_perturbed" \
    --forget_gen_folder "forget${ratio_padded}+tofu" \
    --retain_cls_folder "retain_perturbed" \
    --retain_gen_folder "retain${retain}+tofu" \
    --realface_folder "real_faces" \
    --realworld_folder "real_world"
}

write_ratio_summary() {
  local ratio="$1"
  "${PYTHON}" - "${ROOT_DIR}" "${SUMMARY_DIR}" "${ratio}" "${SHOT_NUM}" <<'PY'
import csv
import json
import os
import sys

root_dir, summary_dir, ratio_raw, shot_num = sys.argv[1:]
ratio = int(ratio_raw)
ratio_padded = f"{ratio:02d}"

methods = [
    ("reference", "vanilla", "checkpoints/qwen2B_vanilla"),
    ("reference", "oracle", f"checkpoints/qwen2B_oracle_{ratio}"),
    ("baseline", "GA", f"checkpoints/GA_{ratio}"),
    ("baseline", "GD", f"checkpoints/GD_{ratio}"),
    ("baseline", "KL", f"checkpoints/KL_{ratio}"),
    ("baseline", "NPO", f"checkpoints/NPO_{ratio}"),
    ("baseline", "MMU", f"checkpoints/MMU_{ratio}"),
    ("baseline", "KVW", f"checkpoints/KVW_{ratio_padded}"),
    ("new_method", "KVW_TWOSTAGE", f"checkpoints/KVW_TWOSTAGE_{ratio}"),
]

metric_paths = {
    "forget_acc": ("Forget Set Results", "classification", "VQA Accuracy"),
    "retain_acc": ("Retain Set Results", "classification", "VQA Accuracy"),
    "realface_acc": ("Real Face Results", "classification", "VQA Accuracy"),
    "realworld_acc": ("Real World Results", "classification", "VQA Accuracy"),
}

def read_metric(data, path):
    cur = data
    try:
        for key in path:
            cur = cur[key]
        return cur
    except Exception:
        return None

rows = []
for category, method, rel_ckpt in methods:
    result_file = os.path.join(root_dir, rel_ckpt, shot_num, f"forget{ratio_padded}", "final_evaluation_results.json")
    row = {
        "forget_ratio": ratio,
        "category": category,
        "method": method,
        "status": "missing",
        "checkpoint_dir": os.path.join(root_dir, rel_ckpt),
        "result_file": result_file,
    }
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        row["status"] = "ok"
        for out_key, path in metric_paths.items():
            row[out_key] = read_metric(payload, path)
    else:
        for out_key in metric_paths:
            row[out_key] = None
    rows.append(row)

os.makedirs(summary_dir, exist_ok=True)
csv_path = os.path.join(summary_dir, f"core_forget{ratio_padded}_summary.csv")
json_path = os.path.join(summary_dir, f"core_forget{ratio_padded}_summary.json")
fieldnames = [
    "forget_ratio",
    "category",
    "method",
    "status",
    "forget_acc",
    "retain_acc",
    "realface_acc",
    "realworld_acc",
    "checkpoint_dir",
    "result_file",
]
metric_fields = {"forget_acc", "retain_acc", "realface_acc", "realworld_acc"}

def format_csv_value(key, value):
    if key in metric_fields and value is not None:
        return f"{float(value):.5f}"
    return value

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: format_csv_value(k, row.get(k)) for k in fieldnames})

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=4)

print(f"[Summary] saved {csv_path}")
print(f"[Summary] saved {json_path}")
PY
}

write_combined_summary() {
  "${PYTHON}" - "${SUMMARY_DIR}" "${FORGET_RATIOS}" <<'PY'
import csv
import os
import sys

summary_dir, ratios_raw = sys.argv[1:]
rows = []
fieldnames = None
metric_fields = {"forget_acc", "retain_acc", "realface_acc", "realworld_acc"}

def format_csv_row(row):
    formatted = dict(row)
    for key in metric_fields:
        if formatted.get(key):
            formatted[key] = f"{float(formatted[key]):.5f}"
    return formatted

for ratio_str in ratios_raw.split():
    path = os.path.join(summary_dir, f"core_forget{int(ratio_str):02d}_summary.csv")
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows.extend(format_csv_row(row) for row in reader)

if rows and fieldnames:
    output = os.path.join(summary_dir, "core_all_ratios_summary.csv")
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Summary] saved {output}")
PY
}

echo "============================================================"
echo "[CLEAR core run]"
echo "============================================================"
echo "root=${ROOT_DIR}"
echo "model=${MODEL_ID}"
echo "train_gpu=${GPU}"
echo "eval_gpu=${EVAL_GPU}"
echo "forget_ratios=${FORGET_RATIOS}"
echo "methods=${METHODS}"
echo "force_rerun=${FORCE_RERUN}"

for ratio in ${FORGET_RATIOS}; do
  validate_ratio "${ratio}"
  echo "============================================================"
  echo "[Forget ratio ${ratio}%]"
  echo "============================================================"
  for raw_method in ${METHODS}; do
    method="$(normalize_method "${raw_method}")"
    if [[ "${FORCE_RERUN}" != "1" ]] && result_done "${method}" "${ratio}"; then
      echo "[Skip] ${method} forget${ratio}: result already exists -> $(result_file_for "${method}" "${ratio}")"
      continue
    fi
    train_method "${method}" "${ratio}"
    eval_method "${method}" "${ratio}"
  done
  write_ratio_summary "${ratio}"
done

write_combined_summary

echo "Done."
echo "Per-ratio summaries: ${ROOT_DIR}/checkpoints/eval_summary/core_forgetXX_summary.csv"
echo "Combined summary: ${ROOT_DIR}/checkpoints/eval_summary/core_all_ratios_summary.csv"
