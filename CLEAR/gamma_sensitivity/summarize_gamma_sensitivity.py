import argparse
import csv
import math
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gamma_sensitivity.common import (
    gamma_to_text,
    read_json,
    resolve_path,
    str2bool,
    write_json,
)


FIELDNAMES = [
    "gamma",
    "gamma_text",
    "train_status",
    "eval_status",
    "forget_acc",
    "retain_acc",
    "realface_acc",
    "realworld_acc",
    "retain_constraint",
    "constraint_satisfied",
    "checkpoint_dir",
    "stage1_dir",
    "result_file",
    "metrics_file",
    "train_error",
    "eval_error",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize and plot gamma sensitivity results.")
    parser.add_argument("--output_root", type=str, default="checkpoints/gamma_sensitivity")
    parser.add_argument("--retain_constraint", type=float, default=None)
    parser.add_argument("--plot", type=str2bool, default=True)
    return parser.parse_args()


def maybe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def metric_value(metrics, key):
    value = metrics.get(key)
    return float(value) if value is not None else None


def metric_value_any(metrics, *keys):
    for key in keys:
        value = metric_value(metrics, key)
        if value is not None:
            return value
    return None


def read_row(gamma_dir, retain_constraint):
    status = read_json(gamma_dir / "status.json", default={}) or {}
    metrics = read_json(gamma_dir / "metrics.json", default={}) or {}
    config = read_json(gamma_dir / "gamma_config.json", default={}) or {}
    gamma = maybe_float(status.get("gamma", config.get("gamma")))
    gamma_text = status.get("gamma_text") or config.get("gamma_text")
    if gamma_text is None and gamma is not None:
        gamma_text = gamma_to_text(gamma)

    retain_acc = metric_value(metrics, "retain_acc")
    constraint_satisfied = ""
    if retain_constraint is not None and retain_acc is not None:
        constraint_satisfied = retain_acc >= retain_constraint

    return {
        "gamma": gamma,
        "gamma_text": gamma_text,
        "train_status": status.get("train_status", "missing"),
        "eval_status": status.get("eval_status", "missing"),
        "forget_acc": metric_value_any(metrics, "forget_acc", "forget_acc_mean"),
        "retain_acc": retain_acc,
        "realface_acc": metric_value(metrics, "realface_acc"),
        "realworld_acc": metric_value(metrics, "realworld_acc"),
        "retain_constraint": retain_constraint,
        "constraint_satisfied": constraint_satisfied,
        "checkpoint_dir": status.get("checkpoint_dir", str(gamma_dir)),
        "stage1_dir": status.get("stage1_dir", ""),
        "result_file": status.get("result_file", metrics.get("result_file", "")),
        "metrics_file": status.get("metrics_file", str(gamma_dir / "metrics.json")),
        "train_error": status.get("train_error", ""),
        "eval_error": status.get("eval_error", ""),
    }


def format_csv_value(value):
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.5f}"
    if value is None:
        return ""
    return value


def write_csv(rows, output_root):
    csv_path = output_root / "gamma_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row.get(key)) for key in FIELDNAMES})
    print(f"[Summary] saved {csv_path}")
    return csv_path


def choose_best(rows, retain_constraint):
    ok_rows = [
        row
        for row in rows
        if row.get("train_status") in {"ok", "skipped", "reused"}
        and row.get("eval_status") == "ok"
        and row.get("forget_acc") is not None
        and row.get("retain_acc") is not None
    ]
    if not ok_rows:
        return {
            "best_gamma": None,
            "reason": "No gamma has successful train/eval metrics.",
            "selection_rule": selection_rule(retain_constraint),
        }

    constrained = ok_rows
    constraint_satisfied = None
    if retain_constraint is not None:
        constrained = [row for row in ok_rows if row["retain_acc"] >= retain_constraint]
        constraint_satisfied = bool(constrained)
        if not constrained:
            constrained = ok_rows

    best = min(
        constrained,
        key=lambda row: (
            row["forget_acc"],
            -row["retain_acc"],
            row["gamma"] if row["gamma"] is not None else float("inf"),
        ),
    )
    return {
        "best_gamma": best["gamma"],
        "best_gamma_text": best["gamma_text"],
        "forget_acc": best["forget_acc"],
        "retain_acc": best["retain_acc"],
        "retain_constraint": retain_constraint,
        "constraint_satisfied": constraint_satisfied,
        "checkpoint_dir": best["checkpoint_dir"],
        "metrics_file": best["metrics_file"],
        "selection_rule": selection_rule(retain_constraint),
    }


def selection_rule(retain_constraint):
    if retain_constraint is None:
        return (
            "Among successful gamma runs, choose the lowest forget_acc; "
            "break ties by higher retain_acc and then smaller gamma."
        )
    return (
        "Among successful gamma runs satisfying retain_acc >= retain_constraint, "
        "choose the lowest forget_acc; break ties by higher retain_acc and "
        "then smaller gamma. If none satisfy the constraint, fall back to all "
        "successful runs and mark constraint_satisfied=false."
    )


def plot_rows(rows, output_root, retain_constraint):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_labels = [row["gamma_text"] for row in rows]
    x_values = list(range(len(rows)))
    series = [
        ("forget_acc", "o", "tab:blue"),
        ("retain_acc", "s", "gold"),
        ("realface_acc", "^", "tab:green"),
        ("realworld_acc", "D", "tab:red"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric_name, marker, color in series:
        values = [
            row[metric_name] if row[metric_name] is not None else math.nan
            for row in rows
        ]
        ax.plot(x_values, values, marker=marker, color=color, label=metric_name)
    if retain_constraint is not None:
        ax.axhline(
            y=retain_constraint,
            linestyle="--",
            linewidth=1.2,
            color="gray",
            label="retain constraint",
        )
    ax.set_xlabel("gamma")
    ax.set_ylabel("accuracy")
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    png_path = output_root / "gamma_sensitivity.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] saved {png_path}")
    return png_path


def main():
    args = parse_args()
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    gamma_dirs = [path for path in output_root.glob("gamma_*") if path.is_dir()]
    rows = [read_row(path, args.retain_constraint) for path in gamma_dirs]
    sweep_config = read_json(output_root / "sweep_config.json", default={}) or {}
    gamma_order = {
        str(gamma_text): idx
        for idx, gamma_text in enumerate(sweep_config.get("gamma_list", []))
    }
    rows.sort(
        key=lambda row: (
            gamma_order.get(str(row["gamma_text"]), float("inf")),
            row["gamma"] if row["gamma"] is not None else float("inf"),
        )
    )

    write_csv(rows, output_root)
    best = choose_best(rows, args.retain_constraint)
    best_path = output_root / "best_gamma.json"
    write_json(best_path, best)
    print(f"[Best] saved {best_path}")
    if args.plot:
        try:
            plot_rows(rows, output_root, args.retain_constraint)
        except ModuleNotFoundError as exc:
            message = (
                "matplotlib is required to create gamma_sensitivity.png. "
                "Install it in the active CLEAR environment, then rerun this summarizer."
            )
            (output_root / "plot_error.txt").write_text(f"{message}\n{exc}\n", encoding="utf-8")
            print(f"[Plot][Skipped] {message}")


if __name__ == "__main__":
    main()
