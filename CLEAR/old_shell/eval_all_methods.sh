#!/bin/bash
set -euo pipefail

ROOT_DIR="/home/jb/code/KVW/CLEAR"
MODEL_ID="${ROOT_DIR}/models/Qwen2-VL-2B-Instruct"
DATA_FOLDER="data/CLEAR"
GPU="${GPU:-0}"
FORGET_RATIO="05"
RETAIN_RATIO="95"
SHOT_NUM="zero_shots"
EVAL_LIST="forget retain realface realworld"
FORCE_RERUN="${FORCE_RERUN:-0}"

FORGET_CLS_FOLDER="forget${FORGET_RATIO}_perturbed"
FORGET_GEN_FOLDER="forget${FORGET_RATIO}+tofu"
RETAIN_CLS_FOLDER="retain_perturbed"
RETAIN_GEN_FOLDER="retain${RETAIN_RATIO}+tofu"
REALFACE_FOLDER="real_faces"
REALWORLD_FOLDER="real_world"

METHOD_NAMES=(
  "vanilla"
  "oracle"
  "GA"
  "GD"
  "KL"
  "NPO"
  "MMU"
  "KVW"
)

METHOD_PATHS=(
  "checkpoints/qwen2B_vanilla"
  "checkpoints/qwen2B_oracle_5"
  "checkpoints/GA_5"
  "checkpoints/GD_5"
  "checkpoints/KL_5"
  "checkpoints/NPO_5"
  "checkpoints/MMU_5"
  "checkpoints/KVW_05"
)

cd "${ROOT_DIR}"

run_eval() {
  local name="$1"
  local cache_path="$2"
  local output_folder="${cache_path}/${SHOT_NUM}/forget${FORGET_RATIO}"
  local result_file="${output_folder}/final_evaluation_results.json"

  if [[ ! -d "${cache_path}" ]]; then
    echo "[Skip] ${name}: checkpoint dir not found at ${cache_path}"
    return
  fi

  if [[ "${FORCE_RERUN}" != "1" && -f "${result_file}" ]]; then
    echo "[Reuse] ${name}: ${result_file}"
    return
  fi

  echo "[Eval] ${name} -> ${cache_path}"
  CUDA_VISIBLE_DEVICES="${GPU}" python eval.py \
    --model_id "${MODEL_ID}" \
    --cache_path "${cache_path}" \
    --eval_list "${EVAL_LIST}" \
    --output_folder "${output_folder}" \
    --shot_num "${SHOT_NUM}" \
    --data_folder "${DATA_FOLDER}" \
    --forget_cls_folder "${FORGET_CLS_FOLDER}" \
    --forget_gen_folder "${FORGET_GEN_FOLDER}" \
    --retain_cls_folder "${RETAIN_CLS_FOLDER}" \
    --retain_gen_folder "${RETAIN_GEN_FOLDER}" \
    --realface_folder "${REALFACE_FOLDER}" \
    --realworld_folder "${REALWORLD_FOLDER}"
}

for idx in "${!METHOD_NAMES[@]}"; do
  run_eval "${METHOD_NAMES[$idx]}" "${METHOD_PATHS[$idx]}"
done

SUMMARY_DIR="${ROOT_DIR}/checkpoints/eval_summary"
SUMMARY_JSON="${SUMMARY_DIR}/forget${FORGET_RATIO}_${SHOT_NUM}_summary.json"
SUMMARY_CSV="${SUMMARY_DIR}/forget${FORGET_RATIO}_${SHOT_NUM}_summary.csv"
mkdir -p "${SUMMARY_DIR}"

python - "${SUMMARY_JSON}" "${SUMMARY_CSV}" "${FORGET_RATIO}" "${SHOT_NUM}" "${ROOT_DIR}" <<'PY'
import csv
import json
import os
import sys

summary_json, summary_csv, forget_ratio, shot_num, root_dir = sys.argv[1:]

methods = [
    ("vanilla", "checkpoints/qwen2B_vanilla"),
    ("oracle", "checkpoints/qwen2B_oracle_5"),
    ("GA", "checkpoints/GA_5"),
    ("GD", "checkpoints/GD_5"),
    ("KL", "checkpoints/KL_5"),
    ("NPO", "checkpoints/NPO_5"),
    ("MMU", "checkpoints/MMU_5"),
    ("KVW", "checkpoints/KVW_05"),
]

metric_map = {
    "forget_acc": ("Forget Set Results", "classification", "VQA Accuracy"),
    "retain_acc": ("Retain Set Results", "classification", "VQA Accuracy"),
    "realface_acc": ("Real Face Results", "classification", "VQA Accuracy"),
    "realworld_acc": ("Real World Results", "classification", "VQA Accuracy"),
}

rows = []
for method_name, rel_dir in methods:
    result_path = os.path.join(
        root_dir, rel_dir, shot_num, f"forget{forget_ratio}", "final_evaluation_results.json"
    )
    row = {
        "method": method_name,
        "checkpoint_dir": os.path.join(root_dir, rel_dir),
        "result_file": result_path,
        "status": "missing",
    }
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        row["status"] = "ok"
        row["raw"] = data
        for out_key, path in metric_map.items():
            cur = data
            try:
                for key in path:
                    cur = cur[key]
                row[out_key] = cur
            except Exception:
                row[out_key] = None
    rows.append(row)

summary_payload = {
    "forget_ratio": forget_ratio,
    "shot_num": shot_num,
    "methods": rows,
}

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, ensure_ascii=False, indent=4)

fieldnames = [
    "method",
    "status",
    "forget_acc",
    "retain_acc",
    "realface_acc",
    "realworld_acc",
    "checkpoint_dir",
    "result_file",
]
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in fieldnames})

print("")
print("Summary")
print("method,status,forget_acc,retain_acc,realface_acc,realworld_acc")
for row in rows:
    print(
        "{},{},{},{},{},{}".format(
            row["method"],
            row["status"],
            row.get("forget_acc"),
            row.get("retain_acc"),
            row.get("realface_acc"),
            row.get("realworld_acc"),
        )
    )
print("")
print(f"Summary JSON saved to {summary_json}")
print(f"Summary CSV saved to {summary_csv}")
PY
