import argparse
import re
import shlex
import subprocess
import sys
import traceback
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gamma_sensitivity.common import (
    ROOT_DIR,
    cleanup_cuda,
    command_env,
    gamma_dir_name,
    gamma_to_text,
    parse_gamma_list,
    read_json,
    read_core_summary_metrics,
    resolve_path,
    str2bool,
    update_status,
    validate_forget_ratio_5,
    write_json,
)


DEFAULT_GAMMA_LIST = "0.001:0.005:0.001"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run KVW weakening-strength sensitivity: for each r/gamma, train a fresh "
            "two-stage KVW checkpoint, evaluate that checkpoint, then summarize."
        )
    )
    parser.add_argument("--model_path", type=str, default="checkpoints/qwen2B_vanilla")
    parser.add_argument("--model_id", type=str, default=str(ROOT_DIR / "models/Qwen2-VL-2B-Instruct"))
    parser.add_argument("--data_root", type=str, default="data/CLEAR")
    parser.add_argument("--output_root", type=str, default="checkpoints/gamma_sensitivity")
    parser.add_argument("--forget_ratio", type=int, default=5)
    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--gamma_list", "--r_list", dest="gamma_list", type=str, default=DEFAULT_GAMMA_LIST)
    parser.add_argument("--start_layer", type=int, default=None)
    parser.add_argument("--end_layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run_stage2",
        type=str2bool,
        default=True,
        help="Kept for CLI compatibility. Gamma/r sensitivity always runs two-stage KVW.",
    )
    parser.add_argument("--stage2_config", type=str, default="")
    parser.add_argument("--eval_after_train", type=str2bool, default=True)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--kc_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--kc_cache_dir",
        type=str,
        default="kc",
        help="Reuse retain KC cache. For forget05 this defaults to kc/kc_r_retain_95.pt.",
    )
    parser.add_argument("--shot_num", type=str, default="zero_shots")
    parser.add_argument("--eval_list", type=str, default="forget retain realface realworld")
    parser.add_argument("--retain_constraint", type=float, default=None)
    parser.add_argument("--force_rerun", type=str2bool, default=True)
    parser.add_argument(
        "--reuse_core_summary",
        type=str2bool,
        default=True,
        help=(
            "When --force_rerun False and r/gamma equals --core_summary_gamma, reuse "
            "the existing forget05 KVW_TWOSTAGE metrics from core_all_ratios_summary.csv."
        ),
    )
    parser.add_argument(
        "--core_summary_csv",
        type=str,
        default="checkpoints/eval_summary/core_all_ratios_summary.csv",
    )
    parser.add_argument("--core_summary_gamma", type=float, default=0.02)
    return parser.parse_args()


def parse_default_from_script(name, fallback):
    script = ROOT_DIR / "run_core_methods.sh"
    if not script.exists():
        return fallback
    pattern = re.compile(rf'^{re.escape(name)}="\$\{{{re.escape(name)}:-([^}}]+)\}}"')
    with script.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return fallback
    return fallback


def model_num_layers(model_id):
    config = read_json(resolve_path(model_id) / "config.json", default={}) or {}
    value = config.get("num_hidden_layers")
    return int(value) if value is not None else None


def resolve_layer_defaults(args):
    if args.run_stage2:
        start = parse_default_from_script("TWOSTAGE_START_LAYER", 0)
        end = parse_default_from_script("TWOSTAGE_END_LAYER", 1000)
        source = "run_core_methods.sh TWOSTAGE_START_LAYER/TWOSTAGE_END_LAYER"
    else:
        start = parse_default_from_script("KVW_START_LAYER", 1)
        end = parse_default_from_script("KVW_END_LAYER", 25)
        source = "run_core_methods.sh KVW_START_LAYER/KVW_END_LAYER"

    if args.start_layer is not None:
        start = args.start_layer
        source = "command line"
    if args.end_layer is not None:
        end = args.end_layer
        source = "command line"

    layers = model_num_layers(args.model_id)
    if args.end_layer is None and end >= 1000 and layers:
        end = layers - 1
        source = f"{source}; all-layer sentinel normalized from 1000 to {end}"

    return start, end, source


def run_logged(cmd, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[Command] {shlex.join(cmd)}\n\n")
        log.flush()
        completed = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return completed.returncode


def train_one(args, gamma, start_layer, end_layer, output_root, env):
    gamma_dir = output_root / gamma_dir_name(gamma)
    log_path = gamma_dir / "train.log"
    cmd = [
        args.python,
        "gamma_sensitivity/train_gamma_checkpoint.py",
        "--model_path",
        str(resolve_path(args.model_path)),
        "--model_id",
        str(resolve_path(args.model_id)),
        "--data_root",
        str(resolve_path(args.data_root)),
        "--output_root",
        str(output_root),
        "--forget_ratio",
        str(args.forget_ratio),
        "--gpu",
        args.gpu,
        "--gamma",
        str(gamma),
        "--start_layer",
        str(start_layer),
        "--end_layer",
        str(end_layer),
        "--seed",
        str(args.seed),
        "--run_stage2",
        str(args.run_stage2),
        "--python",
        args.python,
        "--batch_size",
        str(args.batch_size),
        "--kc_batch_size",
        str(args.kc_batch_size),
        "--num_epochs",
        str(args.num_epochs),
        "--force_rerun",
        str(args.force_rerun),
    ]
    if args.stage2_config:
        cmd.extend(["--stage2_config", str(resolve_path(args.stage2_config))])
    if args.kc_cache_dir:
        cmd.extend(["--kc_cache_dir", str(resolve_path(args.kc_cache_dir))])

    print(f"[Train] gamma={gamma_to_text(gamma)} -> {gamma_dir}")
    return_code = run_logged(cmd, log_path, env)
    if return_code != 0:
        update_status(
            gamma_dir,
            gamma=gamma,
            gamma_text=gamma_to_text(gamma),
            train_status="failed",
            train_error=f"train command exited with code {return_code}; see {log_path}",
        )
        print(f"[Train][Failed] gamma={gamma_to_text(gamma)}; see {log_path}")
        return False
    print(f"[Train][OK] gamma={gamma_to_text(gamma)}")
    return True


def eval_one(args, gamma, output_root, env):
    gamma_dir = output_root / gamma_dir_name(gamma)
    log_path = gamma_dir / "eval.log"
    cmd = [
        args.python,
        "gamma_sensitivity/eval_checkpoint.py",
        "--model_id",
        str(resolve_path(args.model_id)),
        "--checkpoint_dir",
        str(gamma_dir),
        "--data_root",
        str(resolve_path(args.data_root)),
        "--forget_ratio",
        str(args.forget_ratio),
        "--gpu",
        args.gpu,
        "--python",
        args.python,
        "--shot_num",
        args.shot_num,
        "--eval_list",
        args.eval_list,
        "--force_rerun",
        str(args.force_rerun),
    ]
    print(f"[Eval] gamma={gamma_to_text(gamma)} -> {gamma_dir / 'metrics.json'}")
    return_code = run_logged(cmd, log_path, env)
    if return_code != 0:
        update_status(
            gamma_dir,
            eval_status="failed",
            eval_error=f"eval command exited with code {return_code}; see {log_path}",
        )
        print(f"[Eval][Failed] gamma={gamma_to_text(gamma)}; see {log_path}")
        return False
    print(f"[Eval][OK] gamma={gamma_to_text(gamma)}")
    return True


def summarize(args, output_root, env):
    cmd = [
        args.python,
        "gamma_sensitivity/summarize_gamma_sensitivity.py",
        "--output_root",
        str(output_root),
        "--plot",
        "True",
    ]
    if args.retain_constraint is not None:
        cmd.extend(["--retain_constraint", str(args.retain_constraint)])
    log_path = output_root / "summary.log"
    return_code = run_logged(cmd, log_path, env)
    if return_code != 0:
        print(f"[Summary][Failed] see {log_path}")
        return False
    print(f"[Summary][OK] {output_root / 'gamma_sensitivity.csv'}")
    return True


def reuse_core_summary_if_available(args, gamma, start_layer, end_layer, output_root):
    if args.force_rerun or not args.reuse_core_summary:
        return False

    gamma_dir = output_root / gamma_dir_name(gamma)
    metrics = read_core_summary_metrics(
        resolve_path(args.core_summary_csv),
        gamma,
        core_summary_gamma=args.core_summary_gamma,
    )
    if metrics is None:
        return False

    gamma_dir.mkdir(parents=True, exist_ok=True)
    metrics["metrics_file"] = str(gamma_dir / "metrics.json")
    write_json(gamma_dir / "metrics.json", metrics)
    write_json(
        gamma_dir / "gamma_config.json",
        {
            "gamma": gamma,
            "r": gamma,
            "gamma_text": gamma_to_text(gamma),
            "r_text": gamma_to_text(gamma),
            "forget_ratio": 5,
            "run_stage2": True,
            "cross_eval": False,
            "reused_from_core_summary": True,
            "core_summary_csv": str(resolve_path(args.core_summary_csv)),
            "core_summary_gamma": args.core_summary_gamma,
            "start_layer": start_layer,
            "end_layer": end_layer,
        },
    )
    update_status(
        gamma_dir,
        gamma=gamma,
        r=gamma,
        gamma_text=gamma_to_text(gamma),
        r_text=gamma_to_text(gamma),
        train_status="reused",
        eval_status="ok",
        train_error="",
        eval_error="",
        checkpoint_dir=metrics.get("checkpoint_dir", ""),
        stage1_dir="",
        result_file=metrics.get("result_file", ""),
        metrics_file=str(gamma_dir / "metrics.json"),
        reused_from_core_summary=True,
        core_summary_csv=str(resolve_path(args.core_summary_csv)),
    )
    print(f"[Reuse] r={gamma_to_text(gamma)} metrics imported from {resolve_path(args.core_summary_csv)}")
    return True


def main():
    args = parse_args()
    args.forget_ratio = validate_forget_ratio_5(args.forget_ratio)
    if not args.run_stage2:
        print("[Config] --run_stage2 False was requested, but r/gamma sensitivity now always uses two-stage KVW.")
        args.run_stage2 = True
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    gamma_values = parse_gamma_list(args.gamma_list)
    start_layer, end_layer, layer_source = resolve_layer_defaults(args)
    env = command_env(args.gpu)

    print("============================================================")
    print("[Gamma Sensitivity]")
    print("============================================================")
    print(f"root={ROOT_DIR}")
    print(f"model_path={resolve_path(args.model_path)}")
    print(f"data_root={resolve_path(args.data_root)}")
    print(f"output_root={output_root}")
    print(f"forget_ratio={args.forget_ratio}")
    print(f"r_list={','.join(gamma_to_text(value) for value in gamma_values)}")
    print(f"layers={start_layer}-{end_layer} ({layer_source})")
    print("run_stage2=True")
    print("cross_eval=False")
    print(f"eval_after_train={args.eval_after_train}")
    write_json(
        output_root / "sweep_config.json",
        {
            "model_path": str(resolve_path(args.model_path)),
            "model_id": str(resolve_path(args.model_id)),
            "data_root": str(resolve_path(args.data_root)),
            "output_root": str(output_root),
            "forget_ratio": args.forget_ratio,
            "gamma_list": [gamma_to_text(value) for value in gamma_values],
            "r_list": [gamma_to_text(value) for value in gamma_values],
            "start_layer": start_layer,
            "end_layer": end_layer,
            "layer_default_source": layer_source,
            "seed": args.seed,
            "run_stage2": True,
            "cross_eval": False,
            "eval_after_train": args.eval_after_train,
            "retain_constraint": args.retain_constraint,
            "reuse_core_summary": args.reuse_core_summary,
            "core_summary_csv": str(resolve_path(args.core_summary_csv)),
            "core_summary_gamma": args.core_summary_gamma,
        },
    )

    try:
        for gamma in gamma_values:
            gamma_dir = output_root / gamma_dir_name(gamma)
            try:
                if reuse_core_summary_if_available(args, gamma, start_layer, end_layer, output_root):
                    continue
                trained = train_one(args, gamma, start_layer, end_layer, output_root, env)
                cleanup_cuda()
                if args.eval_after_train and trained:
                    eval_one(args, gamma, output_root, env)
                elif args.eval_after_train:
                    update_status(gamma_dir, eval_status="skipped_train_failed")
                cleanup_cuda()
            except Exception as exc:
                update_status(
                    gamma_dir,
                    gamma=gamma,
                    gamma_text=gamma_to_text(gamma),
                    train_status="failed",
                    train_error=str(exc),
                    train_traceback=traceback.format_exc(),
                )
                print(f"[Gamma][Failed] gamma={gamma_to_text(gamma)}; continuing")
                cleanup_cuda()
                continue
    finally:
        summarize(args, output_root, env)


if __name__ == "__main__":
    main()
