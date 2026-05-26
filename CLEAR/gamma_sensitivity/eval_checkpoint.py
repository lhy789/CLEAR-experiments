import argparse
import shlex
import subprocess
import sys
import traceback
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gamma_sensitivity.common import (
    ROOT_DIR,
    build_eval_folders,
    cleanup_cuda,
    command_env,
    extract_metrics,
    ratio2,
    read_json,
    resolve_path,
    str2bool,
    update_status,
    validate_forget_ratio_5,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one gamma checkpoint and write metrics.json.")
    parser.add_argument("--model_id", type=str, default=str(ROOT_DIR / "models/Qwen2-VL-2B-Instruct"))
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/CLEAR")
    parser.add_argument("--forget_ratio", type=int, default=5)
    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--shot_num", type=str, default="zero_shots")
    parser.add_argument("--eval_list", type=str, default="forget retain realface realworld")
    parser.add_argument("--force_rerun", type=str2bool, default=True)
    return parser.parse_args()


def run_command(cmd, env):
    print(f"[Command] {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True)


def main():
    args = parse_args()
    args.forget_ratio = validate_forget_ratio_5(args.forget_ratio)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    model_id = resolve_path(args.model_id)
    data_root = resolve_path(args.data_root)
    output_folder = checkpoint_dir / args.shot_num / f"forget{ratio2(args.forget_ratio)}"
    result_file = output_folder / "final_evaluation_results.json"
    metrics_file = checkpoint_dir / "metrics.json"
    env = command_env(args.gpu)

    update_status(
        checkpoint_dir,
        eval_status="running",
        result_file=str(result_file),
        metrics_file=str(metrics_file),
    )

    try:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

        if args.force_rerun or not result_file.exists():
            folders = build_eval_folders(args.forget_ratio)
            cmd = [
                args.python,
                "eval.py",
                "--model_id",
                str(model_id),
                "--cache_path",
                str(checkpoint_dir),
                "--eval_list",
                args.eval_list,
                "--output_folder",
                str(output_folder),
                "--shot_num",
                args.shot_num,
                "--data_folder",
                str(data_root),
                "--forget_cls_folder",
                folders["forget_cls_folder"],
                "--forget_gen_folder",
                folders["forget_gen_folder"],
                "--retain_cls_folder",
                folders["retain_cls_folder"],
                "--retain_gen_folder",
                folders["retain_gen_folder"],
                "--realface_folder",
                folders["realface_folder"],
                "--realworld_folder",
                folders["realworld_folder"],
            ]
            run_command(cmd, env)
        else:
            print(f"[Eval][Skip] existing result -> {result_file}", flush=True)

        payload = read_json(result_file, default=None)
        if payload is None:
            raise FileNotFoundError(f"Evaluation result not found: {result_file}")
        metrics = extract_metrics(payload)
        metrics.update(
            {
                "checkpoint_dir": str(checkpoint_dir),
                "result_file": str(result_file),
                "metrics_file": str(metrics_file),
                "forget_ratio": args.forget_ratio,
                "shot_num": args.shot_num,
                "eval_list": args.eval_list,
            }
        )
        write_json(metrics_file, metrics)
        update_status(checkpoint_dir, eval_status="ok", eval_error="")
        print(f"[Metrics] saved {metrics_file}", flush=True)
    except Exception as exc:
        update_status(
            checkpoint_dir,
            eval_status="failed",
            eval_error=str(exc),
            eval_traceback=traceback.format_exc(),
        )
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        cleanup_cuda()


if __name__ == "__main__":
    main()
