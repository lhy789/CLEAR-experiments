import csv
import json
import os
import re


ROOT_DIR = "/home/jb/code/KVW/CLEAR"
GRID_DIR = os.path.join(ROOT_DIR, "checkpoints", "kvw_seed_batch_grid")
SHOT_NUM = "zero_shots"
FORGET_RATIO = 5
OUTPUT_DIR = os.path.join(ROOT_DIR, "summary")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "kvw_seed_batch_grid_summary.csv")
RUN_NAME_PATTERN = re.compile(r"seed(?P<seed>\d+)_bs(?P<batch_size>\d+)_run(?P<run>\d+)$")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_metric(data, group_name):
    try:
        return data[group_name]["classification"]["VQA Accuracy"]
    except Exception:
        return None


def format_float(value):
    if value is None:
        return ""
    return f"{float(value):.5f}"


def parse_run_name(run_name):
    match = RUN_NAME_PATTERN.match(run_name)
    if match is None:
        return None
    return {
        "seed": int(match.group("seed")),
        "batch_size": int(match.group("batch_size")),
        "run": int(match.group("run")),
    }


def build_row(run_dir):
    run_name = os.path.basename(run_dir)
    parsed = parse_run_name(run_name)
    if parsed is None:
        return None

    trainer_config_path = os.path.join(run_dir, "trainer_config.json")
    result_path = os.path.join(
        run_dir,
        SHOT_NUM,
        f"forget{FORGET_RATIO:02d}",
        "final_evaluation_results.json",
    )

    row = {
        "run_name": run_name,
        "seed": parsed["seed"],
        "batch_size": parsed["batch_size"],
        "run": parsed["run"],
        "status": "missing",
        "forget_ratio": FORGET_RATIO,
        "gamma": "",
        "start_layer": "",
        "end_layer": "",
        "lr": "",
        "num_epochs": "",
        "forget_acc": "",
        "retain_acc": "",
        "realface_acc": "",
        "realworld_acc": "",
        "checkpoint_dir": run_dir,
        "result_file": result_path,
    }

    if os.path.exists(trainer_config_path):
        trainer_config = read_json(trainer_config_path)
        row["forget_ratio"] = trainer_config.get("forget_ratio", FORGET_RATIO)
        row["gamma"] = format_float(trainer_config.get("gamma"))
        row["start_layer"] = trainer_config.get("start_layer", "")
        row["end_layer"] = trainer_config.get("end_layer", "")
        row["lr"] = format_float(trainer_config.get("lr"))
        row["num_epochs"] = trainer_config.get("num_epochs", "")

    if os.path.exists(result_path):
        result = read_json(result_path)
        row["status"] = "ok"
        row["forget_acc"] = format_float(read_metric(result, "Forget Set Results"))
        row["retain_acc"] = format_float(read_metric(result, "Retain Set Results"))
        row["realface_acc"] = format_float(read_metric(result, "Real Face Results"))
        row["realworld_acc"] = format_float(read_metric(result, "Real World Results"))

    return row


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    for entry in sorted(os.listdir(GRID_DIR)):
        run_dir = os.path.join(GRID_DIR, entry)
        if not os.path.isdir(run_dir):
            continue
        row = build_row(run_dir)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda row: (row["seed"], row["batch_size"], row["run"]))

    fieldnames = [
        "run_name",
        "seed",
        "batch_size",
        "run",
        "status",
        "forget_ratio",
        "gamma",
        "start_layer",
        "end_layer",
        "lr",
        "num_epochs",
        "forget_acc",
        "retain_acc",
        "realface_acc",
        "realworld_acc",
        "checkpoint_dir",
        "result_file",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved to {OUTPUT_CSV}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    main()
