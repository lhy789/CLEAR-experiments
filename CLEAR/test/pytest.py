import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.append(('../'))
sys.path.append(('../../'))

import torch
import json
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    get_scheduler,
    MllamaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW

from data_process.CLEAR_process import (
    CLEAR_Dataset,
    CAPTION_MODE,
    train_collate_clear_ansonly
)

# =========================
# Utils
# =========================
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model','visual']

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

# =========================
# Model Loader
# =========================
def load_model_and_processor(args):
    print("Loading model:", args.model_id)

    if "llava" in args.model_id:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    elif "llama" in args.model_id.lower():
        model = MllamaForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    elif "qwen2.5" in args.model_id.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    elif "qwen" in args.model_id.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        raise ValueError("Unsupported model")

    # LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=find_all_linear_names(model),
    )

    print("Applying LoRA...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"

    return model, processor

# =========================
# Main
# =========================
def main(args):
    accelerator = Accelerator()

    print("===== START =====")
    print("Debug mode:", args.debug)

    model, processor = load_model_and_processor(args)

    # Dataset
    if args.is_oracle:
        retain_df = load_dataset(f"data/CLEAR/retain{100-args.forget_ratio}+tofu", split="train")
    else:
        retain_df = load_dataset("data/CLEAR/full+tofu", split="train")

    dataset = CLEAR_Dataset(data=retain_df, mode=CAPTION_MODE)

    # ✅ debug时自动降batch
    batch_size = 1 if args.debug else args.batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_clear_ansonly(x, processor, "cuda", True)
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * args.num_epochs
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    print("===== TRAIN LOOP =====")

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):

            # ✅ debug模式：只跑2步
            if args.debug and step >= 2:
                print("DEBUG STOP (2 steps)")
                break

            try:
                outputs = model(**batch)
                loss = outputs.loss

                print(f"[Step {step}] Loss:", loss.item())

                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                total_loss += loss.item()

            except Exception as e:
                print("❌ ERROR during training")
                print(e)
                return

        print(f"Epoch {epoch+1} avg loss:", total_loss)

    print("===== SAVING =====")

    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)

    if isinstance(model, PeftModel):
        model = model.merge_and_unload()

    model.save_pretrained(args.save_dir)

    print("✅ DONE")

# =========================
# Entry
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./saved_model")
    parser.add_argument("--forget_ratio", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--is_oracle", type=bool, default=False)

    # ✅ 新增
    parser.add_argument("--debug", type=bool, default=False)

    args = parser.parse_args()
    main(args)