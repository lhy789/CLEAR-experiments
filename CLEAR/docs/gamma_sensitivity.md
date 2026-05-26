# Gamma Sensitivity Analysis

本文档说明如何在 CLEAR 项目中运行 KVW r/gamma 敏感性实验。这里的 `r` 和 `gamma` 指同一个 KVW weakening strength 参数。本实验固定只评估 5% 遗忘比例，不做 1%/10% 或跨比例交叉评估。所有训练 checkpoint、评估输出、汇总表和图片都会写入 `checkpoints/gamma_sensitivity/`，不会写入单独的 `results/` 目录。

## 运行方式

在 `CLEAR` 根目录运行：

```bash
cd /home/jb/code/KVW/CLEAR
/home/jb/.conda/envs/clear/bin/python gamma_sensitivity/run_gamma_sensitivity.py \
  --model_path checkpoints/qwen2B_vanilla \
  --data_root data/CLEAR \
  --output_root checkpoints/gamma_sensitivity \
  --forget_ratio 5 \
  --kc_cache_dir kc \
  --gpu 0,1,2,3 \
  --gamma_list 0.001:0.005:0.001 \
  --seed 42 \
  --run_stage2 True \
  --eval_after_train True
```

默认 `--gamma_list` 是 `0.001:0.005:0.001`，会展开为 `0.001,0.002,0.003,0.004,0.005`。如果要跑更宽的论文式候选，可以改成：

```bash
--gamma_list 0.001,0.005,0.01,0.02,0.03,0.04,0.05
```

流程固定为两阶段 KVW：每个 r/gamma 先运行 KVW weakening，再按当前项目默认配置运行 `baselines.KVW_TwoStageRecovery`，然后只评估该 checkpoint 的 `forget05`。每个 r/gamma 的 stage-1 都从 `--model_path` fresh load，不会从上一个 r/gamma 的 checkpoint 继续训练。`--run_stage2` 仅为兼容旧命令保留，即使传 `False` 也会强制两阶段。最终 checkpoint 的 HuggingFace `config.json` 会保留模型配置，并额外附加 `gamma_sensitivity` 字段记录 r/gamma、layer range、seed 和 stage-1 路径；更完整的实验配置在 `gamma_config.json`。

5% 遗忘比例对应 retain95，默认会复用项目已有的保留集 KC：

```text
/home/jb/code/KVW/CLEAR/kc/kc_r_retain_95.pt
```

也就是默认 `--kc_cache_dir kc`。如果该文件存在，脚本会打印 `[KC][Skip] cached retain coefficients`，不会重新计算保留集 KC。

可选复用已有核心汇总中 5% `KVW_TWOSTAGE` 的结果。由于 `checkpoints/eval_summary/core_all_ratios_summary.csv` 只对应默认两阶段 r/gamma，一般只适合复用 `r=0.02`。复用只在 `--force_rerun False` 且 `--reuse_core_summary True` 时触发：

```bash
/home/jb/.conda/envs/clear/bin/python gamma_sensitivity/run_gamma_sensitivity.py \
  --gamma_list 0.001,0.005,0.02,0.03,0.05 \
  --force_rerun False \
  --reuse_core_summary True
```

## 输出结构

```text
checkpoints/gamma_sensitivity/
  gamma_0.001/
    stage1_checkpoint/
    gamma_config.json
    config.json
    trainer_config.json
    retain_mask_stats.json
    train.log
    eval.log
    metrics.json
    status.json
    zero_shots/forget05/final_evaluation_results.json
  gamma_0.002/
  ...
  sweep_config.json
  gamma_sensitivity.csv
  best_gamma.json
  gamma_sensitivity.png
  summary.log
```

如果某个 r/gamma 复用了核心汇总结果，该目录会包含 `metrics.json` 和 `status.json`，并在状态中标记 `train_status=reused`、`reused_from_core_summary=true`；没有重新生成模型 checkpoint。未复用的 r/gamma 都会在自己的 `gamma_xxx/` 目录下完成两阶段训练和评估。

## CSV 字段

`gamma_sensitivity.csv` 每行对应一个 gamma：

- `gamma`, `gamma_text`: gamma 数值及目录显示文本。
- `train_status`: `ok`, `failed`, `skipped`, `reused` 等训练状态。
- `eval_status`: `ok`, `failed`, `pending`, `skipped_train_failed` 等评估状态。
- `forget_acc`: Forget set classification accuracy；越低表示遗忘越强。
- `retain_acc`: Retain set accuracy。
- `realface_acc`: RealFace accuracy。
- `realworld_acc`: RealWorld accuracy。
- `retain_constraint`: 运行时传入的保留性能约束；未传则为空。
- `constraint_satisfied`: `retain_acc >= retain_constraint` 是否成立。
- `checkpoint_dir`, `stage1_dir`, `result_file`, `metrics_file`: 关键路径。
- `train_error`, `eval_error`: 失败时记录错误摘要，完整日志见对应 `train.log` 或 `eval.log`。

## best_gamma.json 规则

`best_gamma.json` 由 `gamma_sensitivity/summarize_gamma_sensitivity.py` 生成。

选择规则：

1. 只在训练和评估成功的 gamma 中选择；`train_status=reused` 且 `eval_status=ok` 的复用行也可参与选择。
2. 如果传入 `--retain_constraint`，优先只考虑 `retain_acc >= retain_constraint` 的 gamma。
3. 在候选中选择 `forget_acc` 最低的 gamma。
4. 如果 `forget_acc` 相同，选择 `retain_acc` 更高的 gamma。
5. 如果仍然相同，选择更小的 gamma。
6. 如果没有 gamma 满足 retain constraint，则退回所有成功 gamma，并在 JSON 中标记 `constraint_satisfied=false`。

## 论文绘图

`gamma_sensitivity.png` 使用 matplotlib 生成，`dpi=300`，图中不显示标题。横轴按 `gamma_list` 顺序显示 gamma，纵轴是 accuracy，包含四条曲线：

- `forget_acc`
- `retain_acc`
- `realface_acc`
- `realworld_acc`

如果运行时传入 `--retain_constraint`，图中会额外画出 retain constraint 水平线。论文中可直接引用：

```text
checkpoints/gamma_sensitivity/gamma_sensitivity.png
```

需要重新汇总或重画图时：

```bash
/home/jb/.conda/envs/clear/bin/python gamma_sensitivity/summarize_gamma_sensitivity.py \
  --output_root checkpoints/gamma_sensitivity \
  --retain_constraint 0.60 \
  --plot True
```
