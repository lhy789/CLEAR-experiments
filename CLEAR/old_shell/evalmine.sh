#!/bin/bash
set -euo pipefail

forget_ratio=05
retain_ratio=95
model_id=/home/jb/code/KVW/CLEAR/models/Qwen2-VL-2B-Instruct
data_folder="data/CLEAR"
gpu=0
shot_num="zero_shots"
eval_list="forget retain realface realworld"

forget_cls_folder=forget${forget_ratio}_perturbed
forget_gen_folder=forget${forget_ratio}+tofu
retain_cls_folder=retain_perturbed
retain_gen_folder=retain${retain_ratio}+tofu
realface_folder=real_faces
realworld_folder=real_world

run_eval () {
	name=$1
	cache_path=$2
	output_folder=${cache_path}/${shot_num}/forget${forget_ratio}

	echo "[Eval] ${name} -> ${cache_path}"
	CUDA_VISIBLE_DEVICES=$gpu python eval.py \
		--model_id ${model_id} \
		--cache_path ${cache_path} \
		--eval_list "${eval_list}" \
		--output_folder ${output_folder} \
		--shot_num ${shot_num} \
		--data_folder ${data_folder} \
		--forget_cls_folder ${forget_cls_folder} \
		--forget_gen_folder ${forget_gen_folder} \
		--retain_cls_folder ${retain_cls_folder} \
		--retain_gen_folder ${retain_gen_folder} \
		--realface_folder ${realface_folder} \
		--realworld_folder ${realworld_folder}
}

run_eval vanilla checkpoints/qwen2B_vanilla
run_eval oracle checkpoints/qwen2B_oracle_5

echo "Done."
echo "vanilla result: checkpoints/qwen2B_vanilla/${shot_num}/forget${forget_ratio}/final_evaluation_results.json"
echo "oracle result: checkpoints/qwen2B_oracle_5/${shot_num}/forget${forget_ratio}/final_evaluation_results.json"
