MODEL_ID=/home/jb/code/KVW/CLEAR/models/Qwen2-VL-2B-Instruct

CUDA_VISIBLE_DEVICES=0 python pytest.py \
  --model_id ${MODEL_ID} \
  --save_dir checkpoints/debug \
  --batch_size 4 \
  --lr 1e-4 \
  --num_epochs 1 \
  --forget_ratio 5 \
  --is_oracle False \
  --debug True