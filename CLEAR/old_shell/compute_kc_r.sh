CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m baselines.KVW \
	--model_id /home/jb/code/KVW/CLEAR/models/Qwen2-VL-2B-Instruct \
	--vanilla_dir checkpoints/qwen2B_vanilla \
	--forget_ratio 05 \
	--batch_size 1 \
	--num_epochs 1 \
	--phase compute_kc_r \
	--data_folder data/CLEAR \
	--save_dir /home/jb/code/KVW/CLEAR/com_kc_r