import csv
import json
import os


ROOT_DIR = os.environ.get("ROOT_DIR", "/home/jb/code/KVW/CLEAR")
SHOT_NUM = os.environ.get("SHOT_NUM", "zero_shots")
FORGET_RATIO = "05"

CHECKPOINT_ROOT = os.environ.get(
    "ABLATION_STAGE_CHECKPOINT_ROOT",
    os.path.join(ROOT_DIR, "checkpoints", "ablation_stage"),
)
RESULT_ROOT = os.environ.get(
    "ABLATION_STAGE_RESULT_ROOT",
    os.path.join(ROOT_DIR, "checkpoints", "ablation_stage_results"),
)
SUMMARY_DIR = os.environ.get(
    "ABLATION_SUMMARY_DIR",
    os.path.join(ROOT_DIR, "checkpoints", "ablation_eval_summary"),
)
CORE_SUMMARY_CSV = os.environ.get(
    "CORE_SUMMARY_CSV",
    os.path.join(ROOT_DIR, "checkpoints", "eval_summary", "core_all_ratios_summary.csv"),
)

STAGE1_SOURCE = os.environ.get(
    "STAGE1_SOURCE",
    os.path.join(ROOT_DIR, "checkpoints", "KVW_STAGE1_FORGET5_L0_1000"),
)
RETAIN_SAVE_DIR = os.environ.get(
    "RETAIN_SAVE_DIR",
    os.path.join(CHECKPOINT_ROOT, "KVW_RETAIN_ONLY_5"),
)
FULL_SOURCE = os.environ.get(
    "FULL_SOURCE",
    os.path.join(ROOT_DIR, "checkpoints", "KVW_TWOSTAGE_5"),
)


ROWS = [
    {
        "ablation": "stage",
        "method": "KVW",
        "stage": "stage1_kvw_only",
        "loss_setup": "none",
        "source": "core_summary",
        "core_method": "KVW",
        "checkpoint_dir": "",
        "result_name": "",
    },
    {
        "ablation": "stage",
        "method": "KVW_RETAIN_ONLY",
        "stage": "stage1_plus_retain_lm",
        "loss_setup": "retain_lm",
        "source": "eval_result",
        "checkpoint_dir": RETAIN_SAVE_DIR,
        "result_name": "KVW_RETAIN_ONLY",
    },
    {
        "ablation": "stage",
        "method": "KVW_TWOSTAGE_FULL",
        "stage": "stage1_plus_full_recovery",
        "loss_setup": "retain_lm+kl+hidden+forget_lock+reg",
        "source": "core_summary",
        "core_method": "KVW_TWOSTAGE",
        "checkpoint_dir": "",
        "result_name": "",
    },
]


METRIC_PATHS = {
    "forget_acc": ("Forget Set Results", "classification", "VQA Accuracy"),
    "retain_acc": ("Retain Set Results", "classification", "VQA Accuracy"),
    "realface_acc": ("Real Face Results", "classification", "VQA Accuracy"),
    "realworld_acc": ("Real World Results", "classification", "VQA Accuracy"),
}
METRIC_FIELDS = set(METRIC_PATHS)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_metric(data, path):
    cur = data
    try:
        for key in path:
            cur = cur[key]
        return cur
    except Exception:
        return None


def format_value(key, value):
    if key in METRIC_FIELDS and value not in (None, ""):
        return f"{float(value):.5f}"
    return value


def load_core_row(method):
    if not os.path.exists(CORE_SUMMARY_CSV):
        return None
    with open(CORE_SUMMARY_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("forget_ratio") == "5" and row.get("method") == method:
                return row
    return None


def build_row(spec):
    result_file = os.path.join(
        RESULT_ROOT,
        spec["result_name"],
        SHOT_NUM,
        f"forget{FORGET_RATIO}",
        "final_evaluation_results.json",
    )
    row = {
        "ablation": spec["ablation"],
        "method": spec["method"],
        "stage": spec["stage"],
        "loss_setup": spec["loss_setup"],
        "status": "missing",
        "forget_acc": None,
        "retain_acc": None,
        "realface_acc": None,
        "realworld_acc": None,
        "checkpoint_dir": spec.get("checkpoint_dir", ""),
        "result_file": result_file,
        "metric_source": result_file,
    }

    if spec.get("source") == "core_summary":
        core_row = load_core_row(spec["core_method"])
        row["metric_source"] = CORE_SUMMARY_CSV
        if core_row is None:
            return row
        row["status"] = core_row.get("status", "missing")
        row["checkpoint_dir"] = core_row.get("checkpoint_dir", "")
        row["result_file"] = core_row.get("result_file", "")
        for key in METRIC_FIELDS:
            row[key] = core_row.get(key)
        return row

    if os.path.exists(result_file):
        payload = read_json(result_file)
        row["status"] = "ok"
        for out_key, path in METRIC_PATHS.items():
            row[out_key] = read_metric(payload, path)
    return row


def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    rows = [build_row(spec) for spec in ROWS]

    csv_path = os.path.join(SUMMARY_DIR, "stage_ablation_forget05_summary.csv")
    json_path = os.path.join(SUMMARY_DIR, "stage_ablation_forget05_summary.json")
    fieldnames = [
        "ablation",
        "method",
        "stage",
        "loss_setup",
        "status",
        "forget_acc",
        "retain_acc",
        "realface_acc",
        "realworld_acc",
        "checkpoint_dir",
        "result_file",
        "metric_source",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format_value(k, row.get(k)) for k in fieldnames})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)

    print("Stage Ablation Summary")
    print("method,status,forget_acc,retain_acc,realface_acc,realworld_acc")
    for row in rows:
        print(
            f"{row['method']},{row['status']},"
            f"{format_value('forget_acc', row.get('forget_acc'))},"
            f"{format_value('retain_acc', row.get('retain_acc'))},"
            f"{format_value('realface_acc', row.get('realface_acc'))},"
            f"{format_value('realworld_acc', row.get('realworld_acc'))}"
        )
    print(f"CSV saved to {csv_path}")
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
