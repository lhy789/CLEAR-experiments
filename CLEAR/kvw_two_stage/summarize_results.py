import csv
import json
import os


ROOT_DIR = "/home/jb/code/KVW/CLEAR"
SAVE_DIR = os.environ.get("SAVE_DIR", os.path.join(ROOT_DIR, "checkpoints", "KVW_TWOSTAGE_5"))
SHOT_NUM = "zero_shots"
FORGET_RATIO = "05"


def read_metric(data, key):
    try:
        return data[key]["classification"]["VQA Accuracy"]
    except Exception:
        return None


def main():
    result_file = os.path.join(SAVE_DIR, SHOT_NUM, f"forget{FORGET_RATIO}", "final_evaluation_results.json")
    row = {
        "category": "new_method",
        "method": "KVW_TWOSTAGE",
        "status": "missing",
        "forget_acc": None,
        "retain_acc": None,
        "realface_acc": None,
        "realworld_acc": None,
        "checkpoint_dir": SAVE_DIR,
        "result_file": result_file,
    }
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        row["status"] = "ok"
        row["forget_acc"] = read_metric(data, "Forget Set Results")
        row["retain_acc"] = read_metric(data, "Retain Set Results")
        row["realface_acc"] = read_metric(data, "Real Face Results")
        row["realworld_acc"] = read_metric(data, "Real World Results")

    summary_dir = os.path.join(ROOT_DIR, "checkpoints", "eval_summary")
    os.makedirs(summary_dir, exist_ok=True)
    csv_path = os.path.join(summary_dir, "KVW_TWOSTAGE_summary.csv")
    json_path = os.path.join(summary_dir, "KVW_TWOSTAGE_summary.json")
    fieldnames = list(row.keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([row], f, ensure_ascii=False, indent=4)

    print(f"Summary saved to {csv_path}")


if __name__ == "__main__":
    main()
