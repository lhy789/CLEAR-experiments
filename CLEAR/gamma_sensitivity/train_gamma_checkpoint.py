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
    cleanup_cuda,
    command_env,
    gamma_dir_name,
    gamma_to_text,
    read_json,
    resolve_path,
    str2bool,
    update_status,
    validate_forget_ratio_5,
    write_json,
)


STAGE2_DEFAULTS = {
    "batch_size": 1,
    "lr": 1e-5,
    "num_epochs": 1,
    "retain_coef": 1.0,
    "kl_coef": 0.5,
    "hidden_coef": 1.0,
    "forget_lock_coef": 0.5,
    "forget_margin": 0.2,
    "reg_coef": 1e-4,
    "repr_hidden_layer": -1,
    "column_top_ratio": 0.1,
    "column_topk": 0,
    "eps": 1e-12,
    "max_grad_norm": 1.0,
    "max_train_steps": 0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one fresh two-stage KVW checkpoint for a single r/gamma value."
    )
    parser.add_argument("--model_path", type=str, default="checkpoints/qwen2B_vanilla")
    parser.add_argument("--model_id", type=str, default=str(ROOT_DIR / "models/Qwen2-VL-2B-Instruct"))
    parser.add_argument("--data_root", type=str, default="data/CLEAR")
    parser.add_argument("--output_root", type=str, default="checkpoints/gamma_sensitivity")
    parser.add_argument("--forget_ratio", type=int, default=5)
    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--gamma", "--r", dest="gamma", type=float, required=True)
    parser.add_argument("--start_layer", type=int, default=1)
    parser.add_argument("--end_layer", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run_stage2",
        type=str2bool,
        default=True,
        help="Kept for CLI compatibility. This script is intended to run two-stage KVW.",
    )
    parser.add_argument("--stage2_config", type=str, default="")
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
    parser.add_argument("--force_rerun", type=str2bool, default=True)
    return parser.parse_args()


def run_command(cmd, env):
    print(f"[Command] {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True)


def load_stage2_config(path):
    config = dict(STAGE2_DEFAULTS)
    if not path:
        return config
    loaded = read_json(path, default={}) or {}
    for key, value in loaded.items():
        if key not in config:
            raise ValueError(f"Unknown stage2_config key: {key}")
        config[key] = value
    return config


def kc_path_for(kc_cache_dir, forget_ratio):
    return Path(kc_cache_dir) / f"kc_r_retain_{100 - int(forget_ratio):02}.pt"


def annotate_model_config(final_dir, payload):
    config_path = Path(final_dir) / "config.json"
    config = read_json(config_path, default=None)
    if config is None:
        return
    config["gamma_sensitivity"] = payload
    write_json(config_path, config)


def main():
    args = parse_args()
    args.forget_ratio = validate_forget_ratio_5(args.forget_ratio)
    if not args.run_stage2:
        print("[Config] --run_stage2 False was requested, but this experiment requires two-stage KVW.")
        args.run_stage2 = True
    output_root = resolve_path(args.output_root)
    gamma_dir = output_root / gamma_dir_name(args.gamma)
    stage1_dir = gamma_dir / "stage1_checkpoint"
    final_dir = gamma_dir
    model_path = resolve_path(args.model_path)
    model_id = resolve_path(args.model_id)
    data_root = resolve_path(args.data_root)
    kc_cache_dir = resolve_path(args.kc_cache_dir) if args.kc_cache_dir else output_root / "kc_cache"
    stage2_config_path = resolve_path(args.stage2_config) if args.stage2_config else None
    stage2_config = load_stage2_config(stage2_config_path)
    env = command_env(args.gpu)

    gamma_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        gamma_dir / "gamma_config.json",
        {
            "gamma": args.gamma,
            "r": args.gamma,
            "gamma_text": gamma_to_text(args.gamma),
            "r_text": gamma_to_text(args.gamma),
            "model_path": str(model_path),
            "model_id": str(model_id),
            "data_root": str(data_root),
            "output_root": str(output_root),
            "forget_ratio": args.forget_ratio,
            "gpu": args.gpu,
            "start_layer": args.start_layer,
            "end_layer": args.end_layer,
            "seed": args.seed,
            "run_stage2": True,
            "cross_eval": False,
            "stage1_dir": str(stage1_dir),
            "checkpoint_dir": str(final_dir),
            "kc_cache_dir": str(kc_cache_dir),
            "stage2_config": stage2_config,
        },
    )

    update_status(
        gamma_dir,
        gamma=args.gamma,
        r=args.gamma,
        gamma_text=gamma_to_text(args.gamma),
        r_text=gamma_to_text(args.gamma),
        train_status="running",
        eval_status="pending",
        checkpoint_dir=str(final_dir),
        stage1_dir=str(stage1_dir),
    )

    try:
        if not args.force_rerun and (final_dir / "trainer_config.json").exists():
            print(f"[Train][Skip] checkpoint already exists -> {final_dir}", flush=True)
            update_status(gamma_dir, train_status="skipped", train_error="")
            return

        kc_cache_dir.mkdir(parents=True, exist_ok=True)
        if not kc_path_for(kc_cache_dir, args.forget_ratio).exists():
            print(f"[KC] computing retain knowledge coefficients -> {kc_cache_dir}", flush=True)
            run_command(
                [
                    args.python,
                    "-m",
                    "baselines.KVW",
                    "--model_id",
                    str(model_id),
                    "--vanilla_dir",
                    str(model_path),
                    "--forget_ratio",
                    str(args.forget_ratio),
                    "--batch_size",
                    str(args.kc_batch_size),
                    "--num_epochs",
                    "1",
                    "--phase",
                    "compute_kc_r",
                    "--data_folder",
                    str(data_root),
                    "--kc_cache_dir",
                    str(kc_cache_dir),
                    "--save_dir",
                    str(gamma_dir / "kc_placeholder"),
                    "--seed",
                    str(args.seed),
                ],
                env,
            )
        else:
            print(f"[KC][Skip] cached retain coefficients -> {kc_path_for(kc_cache_dir, args.forget_ratio)}", flush=True)

        print(f"[Stage1] KVW weakening gamma={gamma_to_text(args.gamma)} -> {stage1_dir}", flush=True)
        run_command(
            [
                args.python,
                "-m",
                "baselines.KVW",
                "--model_id",
                str(model_id),
                "--vanilla_dir",
                str(model_path),
                "--save_dir",
                str(stage1_dir),
                "--forget_ratio",
                str(args.forget_ratio),
                "--batch_size",
                str(args.batch_size),
                "--num_epochs",
                str(args.num_epochs),
                "--phase",
                "weakening",
                "--data_folder",
                str(data_root),
                "--kc_cache_dir",
                str(kc_cache_dir),
                "--gamma",
                str(args.gamma),
                "--start_layer",
                str(args.start_layer),
                "--end_layer",
                str(args.end_layer),
                "--seed",
                str(args.seed),
            ],
            env,
        )

        print(f"[Stage2] protected recovery -> {final_dir}", flush=True)
        stage2_cmd = [
            args.python,
            "-m",
            "baselines.KVW_TwoStageRecovery",
            "--model_id",
            str(model_id),
            "--vanilla_dir",
            str(model_path),
            "--init_model_dir",
            str(stage1_dir),
            "--save_dir",
            str(final_dir),
            "--forget_ratio",
            str(args.forget_ratio),
            "--data_folder",
            str(data_root),
            "--start_layer",
            str(args.start_layer),
            "--end_layer",
            str(args.end_layer),
            "--kc_cache_dir",
            str(kc_cache_dir),
            "--seed",
            str(args.seed),
        ]
        for key, value in stage2_config.items():
            stage2_cmd.extend([f"--{key}", str(value)])
        run_command(stage2_cmd, env)

        annotate_model_config(
            final_dir,
            {
                "gamma": args.gamma,
                "r": args.gamma,
                "gamma_text": gamma_to_text(args.gamma),
                "r_text": gamma_to_text(args.gamma),
                "forget_ratio": args.forget_ratio,
                "start_layer": args.start_layer,
                "end_layer": args.end_layer,
                "seed": args.seed,
                "run_stage2": True,
                "cross_eval": False,
                "stage1_dir": str(stage1_dir),
            },
        )
        update_status(gamma_dir, train_status="ok", train_error="")
    except Exception as exc:
        update_status(
            gamma_dir,
            train_status="failed",
            train_error=str(exc),
            train_traceback=traceback.format_exc(),
        )
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        cleanup_cuda()


if __name__ == "__main__":
    main()
