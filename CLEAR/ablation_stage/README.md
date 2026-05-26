# Stage Ablation on CLEAR Forget-5

This folder runs the stage-level ablation for the CLEAR 5% forget split:

1. `KVW`: read from `checkpoints/eval_summary/core_all_ratios_summary.csv`.
2. `KVW_RETAIN_ONLY`: stage-1 KVW followed by retain-set language-modeling fine-tuning only.
3. `KVW_TWOSTAGE_FULL`: read from `checkpoints/eval_summary/core_all_ratios_summary.csv`.

Default checkpoints and results:

- Train output: `checkpoints/ablation_stage/`
- Evaluation output: `checkpoints/ablation_stage_results/`
- Summary output: `checkpoints/ablation_eval_summary/stage_ablation_forget05_summary.csv`

Run:

```bash
bash ablation_stage/run_all.sh
```

Evaluate and summarize existing checkpoints only:

```bash
bash ablation_stage/eval_all.sh
```

Useful overrides:

```bash
RUN_TRAIN=0 bash ablation_stage/run_all.sh
FORCE_RERUN=1 bash ablation_stage/run_all.sh
STAGE1_SOURCE=/path/to/stage1 FULL_SOURCE=/path/to/full bash ablation_stage/eval_all.sh
```
