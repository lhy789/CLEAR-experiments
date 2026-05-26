import csv
import json
import os


ROOT_DIR = os.environ.get("ROOT_DIR", "/home/jb/code/KVW/CLEAR")
SHOT_NUM = os.environ.get("SHOT_NUM", "zero_shots")
FORGET_RATIO = "05"

CHECKPOINT_ROOT = os.environ.get(
    "ABLATION_LOSS_CHECKPOINT_ROOT",
    os.path.join(ROOT_DIR, "checkpoints", "ablation_retain_loss"),
)
RESULT_ROOT = os.environ.get(
    "ABLATION_LOSS_RESULT_ROOT",
    os.path.join(ROOT_DIR, "checkpoints", "ablation_retain_loss_results"),
)
SUMMARY_DIR = os.environ.get(
    "ABLATION_SUMMARY_DIR",
    os.path.join(ROOT_DIR, "checkpoints", "ablation_eval_summary"),
)
CORE_SUMMARY_CSV = os.environ.get(
    "CORE_SUMMARY_CSV",
    os.path.join(ROOT_DIR, "checkpoints", "eval_summary", "core_all_ratios_summary.csv"),
)
FULL_SOURCE = os.environ.get(
    "FULL_SOURCE",
    os.path.join(ROOT_DIR, "checkpoints", "KVW_TWOSTAGE_5"),
)

BASE = {
    "retain_coef": 1.0,
    "kl_coef": 0.5,
    "hidden_coef": 1.0,
    "forget_lock_coef": 0.5,
    "reg_coef": 1e-4,
}

VARIANTS = [
    {
        "method": "KVW_TWOSTAGE_FULL",
        "slug": "full",
        "removed_loss": "none",
        "source": "core_summary",
        "core_method": "KVW_TWOSTAGE",
        "checkpoint_dir": FULL_SOURCE,
        **BASE,
    },
    {
        "method": "DROP_RETAIN_LM",
        "slug": "drop_retain_lm",
        "removed_loss": "retain_lm",
        "source": "eval_result",
        "checkpoint_dir": os.path.join(CHECKPOINT_ROOT, "drop_retain_lm"),
        **{**BASE, "retain_coef": 0.0},
    },
    {
        "method": "DROP_KL",
        "slug": "drop_kl",
        "removed_loss": "kl",
        "source": "eval_result",
        "checkpoint_dir": os.path.join(CHECKPOINT_ROOT, "drop_kl"),
        **{**BASE, "kl_coef": 0.0},
    },
    {
        "method": "DROP_HIDDEN",
        "slug": "drop_hidden",
        "removed_loss": "hidden",
        "source": "eval_result",
        "checkpoint_dir": os.path.join(CHECKPOINT_ROOT, "drop_hidden"),
        **{**BASE, "hidden_coef": 0.0},
    },
    {
        "method": "DROP_FORGET_LOCK",
        "slug": "drop_forget_lock",
        "removed_loss": "forget_lock",
        "source": "eval_result",
        "checkpoint_dir": os.path.join(CHECKPOINT_ROOT, "drop_forget_lock"),
        **{**BASE, "forget_lock_coef": 0.0},
    },
    {
        "method": "DROP_REG",
        "slug": "drop_reg",
        "removed_loss": "masked_l2_reg",
        "source": "eval_result",
        "checkpoint_dir": os.path.join(CHECKPOINT_ROOT, "drop_reg"),
        **{**BASE, "reg_coef": 0.0},
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
    if key.endswith("_coef") and value is not None:
        return f"{float(value):.6g}"
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
        spec["slug"],
        SHOT_NUM,
        f"forget{FORGET_RATIO}",
        "final_evaluation_results.json",
    )
    row = {
        "ablation": "retain_loss",
        "method": spec["method"],
        "removed_loss": spec["removed_loss"],
        "status": "missing",
        "retain_coef": spec["retain_coef"],
        "kl_coef": spec["kl_coef"],
        "hidden_coef": spec["hidden_coef"],
        "forget_lock_coef": spec["forget_lock_coef"],
        "reg_coef": spec["reg_coef"],
        "forget_acc": None,
        "retain_acc": None,
        "realface_acc": None,
        "realworld_acc": None,
        "checkpoint_dir": spec["checkpoint_dir"],
        "result_file": result_file,
        "metric_source": result_file,
    }
    if spec.get("source") == "core_summary":
        core_row = load_core_row(spec["core_method"])
        row["metric_source"] = CORE_SUMMARY_CSV
        if core_row is None:
            return row
        row["status"] = core_row.get("status", "missing")
        row["checkpoint_dir"] = core_row.get("checkpoint_dir", spec["checkpoint_dir"])
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
    rows = [build_row(spec) for spec in VARIANTS]

    csv_path = os.path.join(SUMMARY_DIR, "retain_loss_ablation_forget05_summary.csv")
    json_path = os.path.join(SUMMARY_DIR, "retain_loss_ablation_forget05_summary.json")
    fieldnames = [
        "ablation",
        "method",
        "removed_loss",
        "status",
        "retain_coef",
        "kl_coef",
        "hidden_coef",
        "forget_lock_coef",
        "reg_coef",
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

    print("Retain Loss Ablation Summary")
    print("method,removed_loss,status,forget_acc,retain_acc,realface_acc,realworld_acc")
    for row in rows:
        print(
            f"{row['method']},{row['removed_loss']},{row['status']},"
            f"{format_value('forget_acc', row.get('forget_acc'))},"
            f"{format_value('retain_acc', row.get('retain_acc'))},"
            f"{format_value('realface_acc', row.get('realface_acc'))},"
            f"{format_value('realworld_acc', row.get('realworld_acc'))}"
        )
    print(f"CSV saved to {csv_path}")
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
