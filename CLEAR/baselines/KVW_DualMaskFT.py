import argparse
import os
from itertools import cycle

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_scheduler

from baselines.KVW import compute_knowledge_coeffs
from baselines.kvw_stage2_common import (
    build_dataloaders,
    build_dual_masks,
    clone_selected_anchor,
    compute_or_load_kc_r,
    ensure_tokenizer_resize,
    forget_lock_loss,
    freeze_to_masked_down_proj,
    get_hidden_representation,
    l2_hidden_loss,
    load_model_and_processor,
    masked_l2_reg,
    masked_token_kl,
    merge_mask_groups,
    save_json,
    zero_grads_outside_mask,
)


def main(args):
    model, processor = load_model_and_processor(args.model_id, args.init_model_dir)
    ensure_tokenizer_resize(model, processor)
    device = model.device

    forget_loader, retain_loader = build_dataloaders(args, processor, device)
    kc_r = compute_or_load_kc_r(model, retain_loader, args)
    print("Computing forget coefficients for dual-mask recovery")
    kc_f = compute_knowledge_coeffs(model, forget_loader)
    dual_masks, mask_stats = build_dual_masks(model, kc_f, kc_r, args)
    active_masks = merge_mask_groups(dual_masks, ["retain"])
    trainable_names = freeze_to_masked_down_proj(model, active_masks)
    anchor_weights = clone_selected_anchor(model, active_masks)

    ref_model, _ = load_model_and_processor(args.model_id, args.vanilla_dir)
    ensure_tokenizer_resize(ref_model, processor)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    forget_anchor_model, _ = load_model_and_processor(args.model_id, args.init_model_dir)
    ensure_tokenizer_resize(forget_anchor_model, processor)
    forget_anchor_model.eval()
    for p in forget_anchor_model.parameters():
        p.requires_grad = False

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(retain_loader) * args.num_epochs,
    )

    accelerator = Accelerator()
    model, optimizer, forget_loader, retain_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, forget_loader, retain_loader, lr_scheduler
    )
    ref_model = accelerator.prepare(ref_model)
    forget_anchor_model = accelerator.prepare(forget_anchor_model)

    print(f"Dual-mask recovery trainable tensors: {len(trainable_names)}")

    for epoch in range(args.num_epochs):
        model.train()
        retain_iter = iter(retain_loader)
        forget_iter = cycle(forget_loader)

        total_loss = 0.0
        total_retain = 0.0
        total_kl = 0.0
        total_hidden = 0.0
        total_lock = 0.0
        total_reg = 0.0

        for _ in tqdm(range(len(retain_loader)), desc=f"Epoch {epoch + 1}"):
            retain_batch = next(retain_iter)
            forget_batch = next(forget_iter)
            optimizer.zero_grad()

            retain_outputs, retain_repr = get_hidden_representation(
                model, retain_batch, hidden_index=args.repr_hidden_layer
            )
            with torch.no_grad():
                ref_retain_outputs, ref_retain_repr = get_hidden_representation(
                    ref_model, retain_batch, hidden_index=args.repr_hidden_layer
                )
                anchor_forget_outputs = forget_anchor_model(**forget_batch)

            loss_retain = retain_outputs.loss
            loss_kl = masked_token_kl(retain_outputs.logits, ref_retain_outputs.logits, retain_batch["labels"])
            loss_hidden = l2_hidden_loss(retain_repr, ref_retain_repr)

            current_forget_outputs = model(**forget_batch)
            loss_lock = forget_lock_loss(
                current_forget_outputs.loss, anchor_forget_outputs.loss, args.forget_margin
            )

            reg_loss = masked_l2_reg(
                accelerator.unwrap_model(model),
                active_masks,
                anchor_weights,
                target_device=loss_retain.device,
            )
            total = (
                args.retain_coef * loss_retain
                + args.kl_coef * loss_kl
                + args.hidden_coef * loss_hidden
                + args.forget_lock_coef * loss_lock
                + args.reg_coef * reg_loss
            )

            accelerator.backward(total)
            zero_grads_outside_mask(accelerator.unwrap_model(model), active_masks)
            accelerator.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                max_norm=args.max_grad_norm,
            )
            optimizer.step()
            lr_scheduler.step()

            total_loss += total.item()
            total_retain += loss_retain.item()
            total_kl += loss_kl.item()
            total_hidden += loss_hidden.item()
            total_lock += loss_lock.item()
            total_reg += reg_loss.item()

        denom = max(1, len(retain_loader))
        print(
            f"Epoch {epoch + 1} total={total_loss / denom:.6f} "
            f"retain={total_retain / denom:.6f} "
            f"kl={total_kl / denom:.6f} "
            f"hidden={total_hidden / denom:.6f} "
            f"lock={total_lock / denom:.6f} "
            f"reg={total_reg / denom:.6f}"
        )

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    os.makedirs(args.save_dir, exist_ok=True)
    unwrapped_model.save_pretrained(args.save_dir, safe_serialization=True)
    save_json(os.path.join(args.save_dir, "trainer_config.json"), vars(args))
    save_json(os.path.join(args.save_dir, "dual_mask_stats.json"), mask_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-mask protected recovery")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--vanilla_dir", type=str, required=True)
    parser.add_argument("--init_model_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--forget_ratio", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--retain_coef", type=float, default=1.0)
    parser.add_argument("--kl_coef", type=float, default=0.5)
    parser.add_argument("--hidden_coef", type=float, default=1.0)
    parser.add_argument("--forget_lock_coef", type=float, default=0.5)
    parser.add_argument("--forget_margin", type=float, default=0.2)
    parser.add_argument("--reg_coef", type=float, default=1e-4)
    parser.add_argument("--repr_hidden_layer", type=int, default=-1)
    parser.add_argument("--start_layer", type=int, default=0)
    parser.add_argument("--end_layer", type=int, default=1000)
    parser.add_argument("--forget_top_ratio", type=float, default=0.1)
    parser.add_argument("--retain_top_ratio", type=float, default=0.1)
    parser.add_argument("--shared_top_ratio", type=float, default=0.05)
    parser.add_argument("--forget_topk", type=int, default=0)
    parser.add_argument("--retain_topk", type=int, default=0)
    parser.add_argument("--shared_topk", type=int, default=0)
    parser.add_argument("--shared_delta_penalty", type=float, default=0.5)
    parser.add_argument("--kc_cache_dir", type=str, default="kc")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
