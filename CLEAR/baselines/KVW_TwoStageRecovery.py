import argparse
import os
from itertools import cycle

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_scheduler

from baselines.reproducibility import set_global_seed
from baselines.kvw_stage2_common import (
    build_dataloaders,
    build_single_mask,
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
    save_json,
    zero_grads_outside_mask,
)


def main(args):
    set_global_seed(args.seed)
    need_retain = args.retain_coef != 0 or args.kl_coef != 0 or args.hidden_coef != 0
    need_kl = args.kl_coef != 0
    need_hidden = args.hidden_coef != 0
    need_ref_model = need_kl or need_hidden
    need_forget_lock = args.forget_lock_coef != 0
    need_reg = args.reg_coef != 0

    model, processor = load_model_and_processor(args.model_id, args.init_model_dir)
    ensure_tokenizer_resize(model, processor)
    device = model.device

    forget_loader, retain_loader = build_dataloaders(
        args, processor, device, include_forget=need_forget_lock
    )
    kc_r = compute_or_load_kc_r(model, retain_loader, args)
    retain_masks, mask_stats = build_single_mask(model, kc_r, args)
    trainable_names = freeze_to_masked_down_proj(model, retain_masks)
    anchor_weights = clone_selected_anchor(model, retain_masks)

    ref_model = None
    if need_ref_model:
        ref_model, _ = load_model_and_processor(args.model_id, args.vanilla_dir)
        ensure_tokenizer_resize(ref_model, processor)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    forget_anchor_model = None
    if need_forget_lock:
        forget_anchor_model, _ = load_model_and_processor(args.model_id, args.init_model_dir)
        ensure_tokenizer_resize(forget_anchor_model, processor)
        forget_anchor_model.eval()
        for p in forget_anchor_model.parameters():
            p.requires_grad = False

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    steps_per_epoch = len(retain_loader)
    if args.max_train_steps > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_train_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=steps_per_epoch * args.num_epochs,
    )

    accelerator = Accelerator()
    if forget_loader is not None:
        model, optimizer, forget_loader, retain_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, forget_loader, retain_loader, lr_scheduler
        )
    else:
        model, optimizer, retain_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, retain_loader, lr_scheduler
        )
    if ref_model is not None:
        ref_model = accelerator.prepare(ref_model)
    if forget_anchor_model is not None:
        forget_anchor_model = accelerator.prepare(forget_anchor_model)

    print(f"Two-stage recovery trainable tensors: {len(trainable_names)}")
    print(
        "Active losses: "
        f"retain={args.retain_coef}, kl={args.kl_coef}, hidden={args.hidden_coef}, "
        f"forget_lock={args.forget_lock_coef}, reg={args.reg_coef}"
    )
    print(f"Training steps per epoch: {steps_per_epoch}/{len(retain_loader)}")

    for epoch in range(args.num_epochs):
        model.train()
        retain_iter = iter(retain_loader)
        forget_iter = cycle(forget_loader) if need_forget_lock else None

        total_loss = 0.0
        total_retain = 0.0
        total_kl = 0.0
        total_hidden = 0.0
        total_lock = 0.0
        total_reg = 0.0

        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}"):
            retain_batch = next(retain_iter)
            forget_batch = next(forget_iter) if need_forget_lock else None
            optimizer.zero_grad()

            retain_outputs = None
            retain_repr = None
            ref_retain_outputs = None
            ref_retain_repr = None

            if need_retain:
                if need_hidden:
                    retain_outputs, retain_repr = get_hidden_representation(
                        model, retain_batch, hidden_index=args.repr_hidden_layer
                    )
                else:
                    retain_outputs = model(**retain_batch)

            if need_ref_model:
                with torch.no_grad():
                    if need_hidden:
                        ref_retain_outputs, ref_retain_repr = get_hidden_representation(
                            ref_model, retain_batch, hidden_index=args.repr_hidden_layer
                        )
                    else:
                        ref_retain_outputs = ref_model(**retain_batch)

            loss_retain = retain_outputs.loss if args.retain_coef != 0 else None
            loss_kl = (
                masked_token_kl(retain_outputs.logits, ref_retain_outputs.logits, retain_batch["labels"])
                if need_kl
                else None
            )
            loss_hidden = (
                l2_hidden_loss(retain_repr, ref_retain_repr)
                if need_hidden
                else None
            )

            if need_forget_lock:
                current_forget_outputs = model(**forget_batch)
                with torch.no_grad():
                    anchor_forget_outputs = forget_anchor_model(**forget_batch)
                loss_lock = forget_lock_loss(
                    current_forget_outputs.loss, anchor_forget_outputs.loss, args.forget_margin
                )
            else:
                loss_lock = None

            def first_loss_device(*losses):
                for loss in losses:
                    if loss is not None:
                        return loss.device
                return next(p for p in model.parameters() if p.requires_grad).device

            if need_reg:
                reg_loss = masked_l2_reg(
                    accelerator.unwrap_model(model),
                    retain_masks,
                    anchor_weights,
                    target_device=first_loss_device(loss_retain, loss_kl, loss_hidden, loss_lock),
                )
            else:
                reg_loss = None

            total = None
            for coef, loss in (
                (args.retain_coef, loss_retain),
                (args.kl_coef, loss_kl),
                (args.hidden_coef, loss_hidden),
                (args.forget_lock_coef, loss_lock),
                (args.reg_coef, reg_loss),
            ):
                if coef == 0 or loss is None:
                    continue
                weighted = coef * loss
                total = weighted if total is None else total + weighted.to(total.device)

            if total is None:
                raise RuntimeError("No active loss component. At least one loss coefficient must be non-zero.")

            accelerator.backward(total)
            zero_grads_outside_mask(accelerator.unwrap_model(model), retain_masks)
            accelerator.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                max_norm=args.max_grad_norm,
            )
            optimizer.step()
            lr_scheduler.step()

            total_loss += total.item()
            total_retain += loss_retain.item() if loss_retain is not None else 0.0
            total_kl += loss_kl.item() if loss_kl is not None else 0.0
            total_hidden += loss_hidden.item() if loss_hidden is not None else 0.0
            total_lock += loss_lock.item() if loss_lock is not None else 0.0
            total_reg += reg_loss.item() if reg_loss is not None else 0.0

        denom = max(1, steps_per_epoch)
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
    save_json(os.path.join(args.save_dir, "retain_mask_stats.json"), mask_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-2 protected recovery from a KVW stage-1 checkpoint")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--vanilla_dir", type=str, required=True)
    parser.add_argument("--init_model_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--forget_ratio", type=int, default=5)
    parser.add_argument("--data_folder", type=str, default="data/CLEAR")
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
    parser.add_argument("--column_top_ratio", type=float, default=0.1)
    parser.add_argument("--column_topk", type=int, default=0)
    parser.add_argument("--kc_cache_dir", type=str, default="kc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="Cap retain-loader optimization steps per epoch. 0 means use the full retain loader.",
    )
    args = parser.parse_args()
    main(args)
