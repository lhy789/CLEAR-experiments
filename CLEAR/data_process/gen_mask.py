import argparse
import os
import sys

import pandas as pd
from torch.utils.data import DataLoader, random_split
sys.path.append(('.'))
sys.path.append(('../'))
sys.path.append(('../../'))
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE,train_collate_clear_ansonly
import torch
import os
import torch
from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor,Qwen2VLForConditionalGeneration
import torch
from SFRon import Mask_grad, Mask_Our


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MMU saliency mask for a CLEAR forget split.")
    parser.add_argument(
        "--model_id",
        default="/home/jb/code/KVW/CLEAR/models/Qwen2-VL-2B-Instruct",
        help="Processor/base model directory.",
    )
    parser.add_argument(
        "--model_path",
        default="checkpoints/qwen2B_vanilla",
        help="Vanilla checkpoint used to compute saliency.",
    )
    parser.add_argument("--forget_ratio", type=int, default=5, choices=[1, 5, 10])
    parser.add_argument("--data_folder", default="data/CLEAR")
    parser.add_argument("--output_dir", default="path_to_save_mask")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--ans_only", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )

    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"
    print(model)

    mg = Mask_Our(model, args.lr)

    # The forget split provides the target identity knowledge; full+tofu plus
    # retain provides the preserve-side signal used for the saliency ratio.
    tofu_df = load_dataset(f"{args.data_folder}/full+tofu", split="train")
    forget_df = load_dataset(f"{args.data_folder}/forget{args.forget_ratio:02}", split="train")
    retain_df = load_dataset(f"{args.data_folder}/retain{100 - args.forget_ratio}", split="train")

    multimodal_tofu_dataset = CLEAR_Dataset(data=tofu_df, mode=NONE_MODE)
    multimodal_forget_dataset = CLEAR_Dataset(data=forget_df, mode=CAPTION_MODE)
    multimodal_remain_dataset = CLEAR_Dataset(data=retain_df, mode=CAPTION_MODE)

    language_preserve_dataset = torch.utils.data.ConcatDataset(
        [multimodal_tofu_dataset, multimodal_remain_dataset]
    )

    train_collate_function = train_collate_clear_ansonly if args.ans_only else train_collate_clear

    forget_dataloader = DataLoader(
        multimodal_forget_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_function(x, processor, "cuda", True),
    )
    language_preserve_dataloader = DataLoader(
        language_preserve_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_function(x, processor, "cuda", True),
    )

    root_dir = os.path.join(args.output_dir, f"forget{args.forget_ratio}")
    os.makedirs(root_dir, exist_ok=True)

    weight_mask, forget_grad, preserve_grad = mg.prepare_weight_saliency_mask(
        modules=["model"],
        forget_loader=forget_dataloader,
        preserve_loader=language_preserve_dataloader,
        threshold=args.threshold,
        save_path="",
    )
    res = {"weight": weight_mask, "forget_grad": forget_grad, "preserve_grad": preserve_grad}
    torch.save(res, os.path.join(root_dir, "language_mask.pt"))


if __name__ == "__main__":
    main()
