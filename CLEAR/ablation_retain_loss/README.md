# Retain-Stage Loss Ablation on CLEAR Forget-5

This folder runs the loss-component ablation for the CLEAR 5% forget split.
The full retain-stage objective has five components:

- retain language modeling loss
- retain KL loss
- retain hidden-state loss
- forget-lock loss
- masked L2 regularization

The full two-stage row is read from `checkpoints/eval_summary/core_all_ratios_summary.csv`.
The script trains five variants that drop exactly one component and evaluates those new checkpoints.

Default checkpoints and results:

- Train output: `checkpoints/ablation_retain_loss/`
- Evaluation output: `checkpoints/ablation_retain_loss_results/`
- Summary output: `checkpoints/ablation_eval_summary/retain_loss_ablation_forget05_summary.csv`

Run:

```bash
bash ablation_retain_loss/run_all.sh
```

The default is a complete experiment: `MAX_TRAIN_STEPS=0`, meaning each selected
variant uses the full retain loader for one epoch. Do not set `MAX_TRAIN_STEPS`
for the final ablation table.

To shorten wall-clock time without shortening the experiment, split variants
across idle GPUs or terminals. For example:

```bash
RUN_EVAL=0 GPU=1 VARIANTS="drop_hidden" bash ablation_retain_loss/run_all.sh
RUN_EVAL=0 GPU=2 VARIANTS="drop_forget_lock" bash ablation_retain_loss/run_all.sh
```

After all variants finish, evaluate and summarize once:

```bash
RUN_TRAIN=0 GPU=1 EVAL_GPU=1 VARIANTS="all" bash ablation_retain_loss/run_all.sh
```

Evaluate and summarize existing checkpoints only:

```bash
bash ablation_retain_loss/eval_all.sh
```

Useful overrides:

```bash
RUN_TRAIN=0 bash ablation_retain_loss/run_all.sh
FORCE_RERUN=1 bash ablation_retain_loss/run_all.sh
STAGE1_SOURCE=/path/to/stage1 FULL_SOURCE=/path/to/full bash ablation_retain_loss/eval_all.sh
```


cd /home/jb/code/KVW/CLEAR

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GPU=1,2,3 \
EVAL_GPU=1,2,3 \
BATCH_SIZE=1 \
MAX_TRAIN_STEPS=0 \
VARIANTS="all" \
bash ablation_retain_loss/run_all.sh
