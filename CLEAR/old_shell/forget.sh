MODEL_ID=/home/jb/code/KVW/CLEAR/models/Qwen2-VL-2B-Instruct

echo "开始GA遗忘"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m baselines.GA \
	--model_id ${MODEL_ID} \
	--vanilla_dir checkpoints/qwen2B_vanilla \
	--lr 1e-5 \
	--batch_size 2 \
	--num_epochs 1 \
	--forget_ratio 5 \
	--data_folder data/CLEAR \
	--rank 8 \
	--save_dir checkpoints/GA_5

echo "开始GA_Diff遗忘"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m baselines.GA_Diff \
	--model_id ${MODEL_ID} \
	--vanilla_dir checkpoints/qwen2B_vanilla \
	--lr 1e-5 \
	--batch_size 1 \
	--num_epochs 1 \
	--forget_ratio 5 \
	--lcoef 1 \
	--data_folder data/CLEAR \
	--rank 8 \
	--save_dir checkpoints/GD_5

echo "开始KL_Min遗忘"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m baselines.KL_Min \
	--model_id ${MODEL_ID} \
	--vanilla_dir checkpoints/qwen2B_vanilla \
	--lr 1e-5 \
	--batch_size 1 \
	--num_epochs 1 \
	--forget_ratio 5 \
	--lcoef 1 \
	--data_folder data/CLEAR \
	--rank 8 \
	--save_dir checkpoints/KL_5

echo "开始NPO遗忘"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m baselines.NPO \
	--model_id ${MODEL_ID} \
	--vanilla_dir checkpoints/qwen2B_vanilla \
	--lr 1e-5 \
	--batch_size 1 \
	--num_epochs 1 \
	--forget_ratio 5 \
	--lcoef 1 \
	--beta 0.4 \
	--data_folder data/CLEAR \
	--rank 8 \
	--save_dir checkpoints/NPO_5 \
	--oracle_model_id checkpoints/qwen2B_oracle_5
