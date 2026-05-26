# CLEAR 论文实验复现说明

所有命令默认在 `CLEAR` 根目录运行：

```bash
cd /home/jb/code/KVW/CLEAR
```

默认 Python 环境：

```bash
/home/jb/.conda/envs/clear/bin/python
```

当前目录中的核心路径：

| 内容 | 路径 |
|---|---|
| 基座模型 | `models/Qwen2-VL-2B-Instruct` |
| CLEAR 数据 | `data/CLEAR` |
| checkpoint 根目录 | `checkpoints/` |
| 主实验汇总 | `checkpoints/eval_summary/` |
| 消融实验汇总 | `checkpoints/ablation_eval_summary/` |
| gamma/r 敏感性结果 | `checkpoints/gamma_sensitivity/` |
| MMU mask | `path_to_save_mask/forget1/language_mask.pt`, `path_to_save_mask/forget5/language_mask.pt`, `path_to_save_mask/forget10/language_mask.pt` |
| KVW retain KC cache | `kc/kc_r_retain_99.pt`, `kc/kc_r_retain_95.pt`, `kc/kc_r_retain_90.pt` |

## 1. 主实验

主实验入口是当前目录中的 `run_core_methods.sh`。脚本支持：

```text
vanilla oracle GA GD KL NPO MMU KVW KVW_TWOSTAGE
```

复现论文 1% 和 5% 主实验：

```bash
FORGET_RATIOS="1 5" \
METHODS="vanilla oracle GA GD KL NPO MMU KVW KVW_TWOSTAGE" \
GPU="0,1,2,3" \
EVAL_GPU="0" \
bash run_core_methods.sh
```

注意：`run_core_methods.sh` 当前默认 `FORGET_RATIOS` 是 `1 10`。如果要跑论文中的 1% 和 5%，需要显式设置：

```bash
FORGET_RATIOS="1 5"
```

只运行某个方法时，设置 `METHODS` 即可。例如：

```bash
FORGET_RATIOS="5" METHODS="GA" GPU="0,1,2,3" EVAL_GPU="0" bash run_core_methods.sh
FORGET_RATIOS="5" METHODS="MMU" GPU="0,1,2,3" EVAL_GPU="0" bash run_core_methods.sh
FORGET_RATIOS="5" METHODS="KVW_TWOSTAGE" GPU="0,1,2,3" EVAL_GPU="0" bash run_core_methods.sh
```

默认 `FORCE_RERUN=0`，已有结果时会跳过对应训练和评估。强制重跑示例：

```bash
FORCE_RERUN=1 FORGET_RATIOS="5" METHODS="KVW_TWOSTAGE" bash run_core_methods.sh
```

主实验 checkpoint 命名：

```text
checkpoints/qwen2B_vanilla
checkpoints/qwen2B_oracle_1
checkpoints/qwen2B_oracle_5
checkpoints/GA_1
checkpoints/GA_5
checkpoints/GD_1
checkpoints/GD_5
checkpoints/KL_1
checkpoints/KL_5
checkpoints/NPO_1
checkpoints/NPO_5
checkpoints/MMU_1
checkpoints/MMU_5
checkpoints/KVW_01
checkpoints/KVW_05
checkpoints/KVW_TWOSTAGE_1
checkpoints/KVW_TWOSTAGE_5
```

每个方法的评估 JSON 位于：

```text
<checkpoint_dir>/zero_shots/forgetXX/final_evaluation_results.json
```

主实验汇总表：

```text
checkpoints/eval_summary/core_forget01_summary.csv
checkpoints/eval_summary/core_forget05_summary.csv
checkpoints/eval_summary/core_all_ratios_summary.csv
```

论文主实验表格优先查看：

```text
checkpoints/eval_summary/core_all_ratios_summary.csv
```

## 2. Stage 消融实验

Stage 消融入口位于当前目录的 `ablation_stage/run_all.sh`。该实验固定使用 forget05，比较：

| 方法 | 含义 |
|---|---|
| `KVW` | 只使用 stage-1 weakening |
| `KVW_RETAIN_ONLY` | stage-1 后只使用 retain LM 恢复 |
| `KVW_TWOSTAGE_FULL` | 完整两阶段方法 |

运行：

```bash
GPU="0,1,2,3" EVAL_GPU="0" bash ablation_stage/run_all.sh
```

依赖路径：

| 依赖 | 默认路径 |
|---|---|
| vanilla checkpoint | `checkpoints/qwen2B_vanilla` |
| stage-1 checkpoint | `checkpoints/KVW_STAGE1_FORGET5_L0_1000` |
| 完整 two-stage checkpoint | `checkpoints/KVW_TWOSTAGE_5` |
| 主实验汇总 | `checkpoints/eval_summary/core_all_ratios_summary.csv` |

输出：

| 内容 | 路径 |
|---|---|
| retain-only checkpoint | `checkpoints/ablation_stage/KVW_RETAIN_ONLY_5` |
| retain-only eval JSON | `checkpoints/ablation_stage_results/KVW_RETAIN_ONLY/zero_shots/forget05/final_evaluation_results.json` |
| stage 消融 CSV | `checkpoints/ablation_eval_summary/stage_ablation_forget05_summary.csv` |
| stage 消融 JSON | `checkpoints/ablation_eval_summary/stage_ablation_forget05_summary.json` |

只重新汇总时：

```bash
RUN_TRAIN=0 RUN_EVAL=0 bash ablation_stage/run_all.sh
```

## 3. Retain Loss 消融实验

Retain loss 消融入口位于当前目录的 `ablation_retain_loss/run_all.sh`。该实验固定使用 forget05，以完整 two-stage loss 为基准，分别去掉一个 loss 项：

| 变体 | 去掉的项 |
|---|---|
| `DROP_RETAIN_LM` | retain language modeling |
| `DROP_KL` | KL regularization |
| `DROP_HIDDEN` | hidden representation alignment |
| `DROP_FORGET_LOCK` | forget lock |
| `DROP_REG` | masked L2 regularization |

运行：

```bash
GPU="0,1,2,3" EVAL_GPU="0" bash ablation_retain_loss/run_all.sh
```

只运行部分变体：

```bash
VARIANTS="drop_kl drop_hidden" GPU="0,1,2,3" EVAL_GPU="0" bash ablation_retain_loss/run_all.sh
```

输出：

| 内容 | 路径 |
|---|---|
| 消融 checkpoint | `checkpoints/ablation_retain_loss/<variant>` |
| 消融 eval JSON | `checkpoints/ablation_retain_loss_results/<variant>/zero_shots/forget05/final_evaluation_results.json` |
| retain loss 消融 CSV | `checkpoints/ablation_eval_summary/retain_loss_ablation_forget05_summary.csv` |
| retain loss 消融 JSON | `checkpoints/ablation_eval_summary/retain_loss_ablation_forget05_summary.json` |

完整方法 `KVW_TWOSTAGE_FULL` 的指标来自：

```text
checkpoints/eval_summary/core_all_ratios_summary.csv
```

## 4. Gamma / r 敏感性实验

Gamma/r 敏感性实验入口位于当前目录的 `gamma_sensitivity/run_gamma_sensitivity.py`。该实验固定使用 forget05，`gamma` 和 `r` 指同一个 weakening strength 参数。

推荐运行：

```bash
/home/jb/.conda/envs/clear/bin/python gamma_sensitivity/run_gamma_sensitivity.py \
  --model_path checkpoints/qwen2B_vanilla \
  --model_id models/Qwen2-VL-2B-Instruct \
  --data_root data/CLEAR \
  --output_root checkpoints/gamma_sensitivity \
  --forget_ratio 5 \
  --kc_cache_dir kc \
  --gpu 0,1,2,3 \
  --gamma_list 0.001,0.005,0.01,0.02,0.03,0.04,0.05 \
  --seed 42 \
  --run_stage2 True \
  --eval_after_train True \
  --force_rerun False
```

只跑默认窄范围：

```bash
/home/jb/.conda/envs/clear/bin/python gamma_sensitivity/run_gamma_sensitivity.py \
  --gamma_list 0.001:0.005:0.001
```

输出结构：

```text
checkpoints/gamma_sensitivity/
  gamma_0.001/
    gamma_config.json
    trainer_config.json
    train.log
    eval.log
    metrics.json
    status.json
    zero_shots/forget05/final_evaluation_results.json
  gamma_0.005/
  ...
  sweep_config.json
  gamma_sensitivity.csv
  best_gamma.json
  gamma_sensitivity.png
  summary.log
```

论文图和表优先查看：

```text
checkpoints/gamma_sensitivity/gamma_sensitivity.csv
checkpoints/gamma_sensitivity/gamma_sensitivity.png
checkpoints/gamma_sensitivity/best_gamma.json
```

只重新汇总或重新画图：

```bash
/home/jb/.conda/envs/clear/bin/python gamma_sensitivity/summarize_gamma_sensitivity.py \
  --output_root checkpoints/gamma_sensitivity \
  --plot True
```

## 5. 论文结果文件对照

| 论文实验 | 当前入口 | 关键结果文件 |
|---|---|---|
| 主实验 1%/5% | `run_core_methods.sh` | `checkpoints/eval_summary/core_all_ratios_summary.csv` |
| 单个 baseline 或方法 | `METHODS="<method>" bash run_core_methods.sh` | `<checkpoint_dir>/zero_shots/forgetXX/final_evaluation_results.json` |
| 两阶段方法 | `METHODS="KVW_TWOSTAGE" bash run_core_methods.sh` | `checkpoints/KVW_TWOSTAGE_5/zero_shots/forget05/final_evaluation_results.json` |
| Stage 消融 | `ablation_stage/run_all.sh` | `checkpoints/ablation_eval_summary/stage_ablation_forget05_summary.csv` |
| Retain loss 消融 | `ablation_retain_loss/run_all.sh` | `checkpoints/ablation_eval_summary/retain_loss_ablation_forget05_summary.csv` |
| Gamma/r 敏感性 | `gamma_sensitivity/run_gamma_sensitivity.py` | `checkpoints/gamma_sensitivity/gamma_sensitivity.csv` |

## 6. 已有结果时重新运行

大部分脚本默认会复用已有 checkpoint 或 `final_evaluation_results.json`。如果需要在已有结果的情况下重新训练、重新评估，需要显式设置 `FORCE_RERUN=1` 或 `--force_rerun True`。

重新运行主实验 1%/5%：

```bash
FORCE_RERUN=1 \
FORGET_RATIOS="1 5" \
METHODS="vanilla oracle GA GD KL NPO MMU KVW KVW_TWOSTAGE" \
GPU="0,1,2,3" \
EVAL_GPU="0" \
bash run_core_methods.sh
```

只重新运行某个方法，例如 5% two-stage：

```bash
FORCE_RERUN=1 \
FORGET_RATIOS="5" \
METHODS="KVW_TWOSTAGE" \
GPU="0,1,2,3" \
EVAL_GPU="0" \
bash run_core_methods.sh
```

重新运行 Stage 消融：

```bash
FORCE_RERUN=1 GPU="0,1,2,3" EVAL_GPU="0" bash ablation_stage/run_all.sh
```

重新运行 Retain loss 消融：

```bash
FORCE_RERUN=1 GPU="0,1,2,3" EVAL_GPU="0" bash ablation_retain_loss/run_all.sh
```

只重新运行 Retain loss 消融中的部分变体：

```bash
FORCE_RERUN=1 \
VARIANTS="drop_kl drop_hidden" \
GPU="0,1,2,3" \
EVAL_GPU="0" \
bash ablation_retain_loss/run_all.sh
```

重新运行 Gamma/r 敏感性实验：

```bash
/home/jb/.conda/envs/clear/bin/python gamma_sensitivity/run_gamma_sensitivity.py \
  --model_path checkpoints/qwen2B_vanilla \
  --model_id models/Qwen2-VL-2B-Instruct \
  --data_root data/CLEAR \
  --output_root checkpoints/gamma_sensitivity \
  --forget_ratio 5 \
  --kc_cache_dir kc \
  --gpu 0,1,2,3 \
  --gamma_list 0.001,0.005,0.01,0.02,0.03,0.04,0.05 \
  --seed 42 \
  --run_stage2 True \
  --eval_after_train True \
  --force_rerun True
```

如果只需要重新生成汇总表和图，不重新训练模型：

```bash
/home/jb/.conda/envs/clear/bin/python gamma_sensitivity/summarize_gamma_sensitivity.py \
  --output_root checkpoints/gamma_sensitivity \
  --plot True
```

## 7. 论文脚本调用关系

论文复现只依赖下面这些入口和代码文件。整理目录时，下面依赖树里的文件建议保持在当前相对路径；旧版 `.sh` 包装脚本可以单独放到 `old_shell/`，不会影响本文档中的复现命令。

```text
run_core_methods.sh
├─ finetune.py
│  └─ data_process/CLEAR_process.py
├─ eval.py
│  └─ data_process/CLEAR_process.py
├─ data_process/gen_mask.py
│  ├─ data_process/CLEAR_process.py
│  └─ data_process/SFRon.py
├─ baselines/GA.py
├─ baselines/GA_Diff.py
├─ baselines/KL_Min.py
├─ baselines/NPO.py
├─ baselines/MMU.py
├─ baselines/KVW.py
└─ baselines/KVW_TwoStageRecovery.py
   ├─ baselines/kvw_stage2_common.py
   └─ baselines/reproducibility.py

ablation_stage/run_all.sh
├─ baselines/KVW_TwoStageRecovery.py
├─ ablation_stage/eval_all.sh
│  └─ eval.py
└─ ablation_stage/summarize_results.py

ablation_retain_loss/run_all.sh
├─ baselines/KVW_TwoStageRecovery.py
├─ ablation_retain_loss/eval_all.sh
│  └─ eval.py
└─ ablation_retain_loss/summarize_results.py

gamma_sensitivity/run_gamma_sensitivity.py
├─ gamma_sensitivity/train_gamma_checkpoint.py
│  ├─ baselines/KVW.py
│  └─ baselines/KVW_TwoStageRecovery.py
├─ gamma_sensitivity/eval_checkpoint.py
│  └─ eval.py
└─ gamma_sensitivity/summarize_gamma_sensitivity.py
```

其中 `baselines/KVW_TwoStageRecovery.py` 还会调用 `baselines/kvw_stage2_common.py` 中的模型加载、数据构造、掩码和 loss 计算函数；`baselines/GA.py`、`baselines/GA_Diff.py`、`baselines/KL_Min.py`、`baselines/NPO.py`、`baselines/KVW.py` 会调用 `baselines/reproducibility.py` 来固定随机种子和 DataLoader 随机性。

可以归档到 `KVW_old_shell/` 的旧入口包括：

```text
finetune.sh
forget.sh
compute_kc_r.sh
kvw.sh
eval.sh
eval_all_methods.sh
eval_original.sh
evalmine.sh
reproduce_kvw_baseline_twice.sh
run_kvw_seed_batch_grid.sh
run_npo_seed_batch_grid.sh
summarize_kvw_seed_batch_grid.py
```

这些旧脚本如果以后还要手动运行，建议仍然从 `CLEAR` 根目录执行，例如：

```bash
cd /home/jb/code/KVW/CLEAR
bash old_shell/eval_all_methods.sh
```
