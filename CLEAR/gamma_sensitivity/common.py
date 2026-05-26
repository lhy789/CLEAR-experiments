import csv
import json
import os
from decimal import Decimal
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
FORGET_RATIO = 5


METRIC_PATHS = {
    "forget_acc": ("Forget Set Results", "classification", "VQA Accuracy"),
    "retain_acc": ("Retain Set Results", "classification", "VQA Accuracy"),
    "realface_acc": ("Real Face Results", "classification", "VQA Accuracy"),
    "realworld_acc": ("Real World Results", "classification", "VQA Accuracy"),
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"yes", "true", "t", "y", "1"}:
        return True
    if lowered in {"no", "false", "f", "n", "0"}:
        return False
    raise ValueError(f"Boolean value expected, got {value!r}")


def resolve_path(path, root=ROOT_DIR):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return root / path


def gamma_to_text(gamma):
    return format(float(gamma), ".6g")


def gamma_dir_name(gamma):
    return f"gamma_{gamma_to_text(gamma)}"


def validate_forget_ratio_5(forget_ratio):
    forget_ratio = int(forget_ratio)
    if forget_ratio != FORGET_RATIO:
        raise ValueError("Gamma/r sensitivity is fixed to 5% forgetting. Use --forget_ratio 5.")
    return forget_ratio


def parse_gamma_list(raw):
    """Parse comma-separated floats plus optional start:end:step ranges."""
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start_s, end_s, step_s = [part.strip() for part in item.split(":")]
            start = Decimal(start_s)
            end = Decimal(end_s)
            step = Decimal(step_s)
            if step <= 0:
                raise ValueError(f"Gamma range step must be positive: {item}")
            cur = start
            while cur <= end:
                values.append(float(cur))
                cur += step
        else:
            values.append(float(item))
    if not values:
        raise ValueError("gamma_list is empty")

    deduped = []
    seen = set()
    for value in values:
        key = gamma_to_text(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)


def read_nested(data, keys):
    cur = data
    try:
        for key in keys:
            cur = cur[key]
        return cur
    except Exception:
        return None


def extract_metrics(eval_payload):
    metrics = {}
    for out_key, path in METRIC_PATHS.items():
        metrics[out_key] = read_nested(eval_payload, path)
    return metrics


def read_core_summary_metrics(core_summary_csv, gamma, core_summary_gamma=0.02):
    """Read the existing 5% KVW_TWOSTAGE row for optional r=0.02 reuse."""
    if abs(float(gamma) - float(core_summary_gamma)) > 1e-12:
        return None

    path = Path(core_summary_csv)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("forget_ratio") != "5" or row.get("method") != "KVW_TWOSTAGE":
                continue
            metrics = {
                "forget_acc": float(row["forget_acc"]) if row.get("forget_acc") else None,
                "retain_acc": float(row["retain_acc"]) if row.get("retain_acc") else None,
                "realface_acc": float(row["realface_acc"]) if row.get("realface_acc") else None,
                "realworld_acc": float(row["realworld_acc"]) if row.get("realworld_acc") else None,
                "forget_ratio": 5,
                "checkpoint_dir": row.get("checkpoint_dir", ""),
                "result_file": row.get("result_file", ""),
                "metrics_source": str(path),
                "reuse_source_method": "KVW_TWOSTAGE",
                "reuse_source_gamma": core_summary_gamma,
            }
            return metrics
    return None


def ratio2(value):
    return f"{int(value):02d}"


def build_eval_folders(forget_ratio):
    forget_ratio = int(forget_ratio)
    retain_ratio = 100 - forget_ratio
    forget_padded = ratio2(forget_ratio)
    return {
        "forget_cls_folder": f"forget{forget_padded}_perturbed",
        "forget_gen_folder": f"forget{forget_padded}+tofu",
        "retain_cls_folder": "retain_perturbed",
        "retain_gen_folder": f"retain{retain_ratio}+tofu",
        "realface_folder": "real_faces",
        "realworld_folder": "real_world",
    }


def update_status(gamma_dir, **updates):
    status_path = Path(gamma_dir) / "status.json"
    status = read_json(status_path, default={}) or {}
    status.update(updates)
    write_json(status_path, status)
    return status


def cleanup_cuda():
    try:
        import gc
        import torch

        # Training/eval are subprocesses, so their models exit with the process.
        # Keep this explicit cleanup for runs that import these helpers directly.
        model = None
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def command_env(gpu):
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env
