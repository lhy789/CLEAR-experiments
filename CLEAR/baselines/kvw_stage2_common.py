import json
import math
import os
import sys
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    MllamaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from baselines.KVW import _get_decoder_layers, compute_knowledge_coeffs  # noqa: E402
from baselines.reproducibility import make_dataloader_generator  # noqa: E402
from data_process.CLEAR_process import (  # noqa: E402
    CAPTION_MODE,
    CLEAR_Dataset,
    train_collate_clear_ansonly,
)


def load_model_and_processor(model_id, model_path):
    if "llava" in model_id.lower():
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    elif "llama" in model_id.lower():
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
    elif "qwen2.5" in model_id.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
    elif "qwen" in model_id.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")
    return model, processor


def ensure_tokenizer_resize(model, processor):
    tokenizer = processor.tokenizer
    model.resize_token_embeddings(len(tokenizer))
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))


def clone_batch(batch):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}


def build_dataloaders(args, processor, device, include_forget=True):
    data_folder = getattr(args, "data_folder", "data/CLEAR")
    seed = getattr(args, "seed", 42)
    forget_df = None
    if include_forget:
        forget_df = load_dataset(os.path.join(data_folder, f"forget{args.forget_ratio:02}"), split="train")
    retain_df = load_dataset(os.path.join(data_folder, f"retain{100 - args.forget_ratio}"), split="train")

    forget_dataset = CLEAR_Dataset(data=forget_df, mode=CAPTION_MODE) if include_forget else None
    retain_dataset = CLEAR_Dataset(data=retain_df, mode=CAPTION_MODE)

    collate = train_collate_clear_ansonly
    forget_loader = None
    if include_forget:
        forget_loader = DataLoader(
            forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=make_dataloader_generator(seed, offset=0),
            collate_fn=lambda x: collate(x, processor, device, True),
        )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=make_dataloader_generator(seed, offset=1),
        collate_fn=lambda x: collate(x, processor, device, True),
    )
    return forget_loader, retain_loader


@torch.no_grad()
def compute_or_load_kc_r(model, retain_loader, args):
    os.makedirs(args.kc_cache_dir, exist_ok=True)
    kc_path = os.path.join(args.kc_cache_dir, f"kc_r_retain_{100 - args.forget_ratio:02}.pt")
    if os.path.exists(kc_path):
        print(f"Loading cached kc_r from {kc_path}")
        return torch.load(kc_path, weights_only=True)
    print("Computing kc_r from retain loader")
    kc_r = compute_knowledge_coeffs(model, retain_loader)
    torch.save(kc_r, kc_path)
    print(f"Saved kc_r to {kc_path}")
    return kc_r


def _match_down_proj_param(named_params, layer_idx):
    suffix = f".layers.{layer_idx}.mlp.down_proj.weight"
    candidates = [name for name in named_params if name.endswith(suffix)]
    if not candidates:
        fallback = f"layers.{layer_idx}.mlp.down_proj.weight"
        candidates = [name for name in named_params if name.endswith(fallback)]
    return candidates[0] if candidates else None


def _resolve_topk(length, ratio, topk):
    count = max(1, math.ceil(length * ratio))
    if topk > 0:
        count = min(topk, length)
    return min(count, length)


def _topk_mask(scores, count, banned=None):
    if banned is None:
        banned = torch.zeros_like(scores, dtype=torch.bool)
    available = (~banned).sum().item()
    if available <= 0 or count <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    if count >= available:
        return ~banned
    masked_scores = scores.clone()
    masked_scores[banned] = torch.finfo(masked_scores.dtype).min
    _, top_idx = torch.topk(masked_scores, k=count)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[top_idx] = True
    mask &= ~banned
    return mask


def build_single_mask(model, kc_values, args, ratio_attr="column_top_ratio", topk_attr="column_topk"):
    layers = _get_decoder_layers(model)
    named_params = dict(model.named_parameters())
    masks = {}
    stats = []

    for layer_idx, (kc, layer) in enumerate(zip(kc_values, layers)):
        if layer_idx < args.start_layer or layer_idx > args.end_layer:
            continue
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
            continue

        scores = kc.to(torch.float32)
        width = scores.numel()
        count = _resolve_topk(width, getattr(args, ratio_attr), getattr(args, topk_attr))
        selected = _topk_mask(scores, count)
        param_name = _match_down_proj_param(named_params, layer_idx)
        if param_name is None:
            continue
        mask_2d = selected.unsqueeze(0).expand(layer.mlp.down_proj.weight.shape[0], -1).clone()
        masks[param_name] = mask_2d
        stats.append(
            {
                "layer_idx": layer_idx,
                "mask_type": "retain",
                "selected_columns": int(selected.sum().item()),
                "total_columns": int(width),
                "selected_ratio": float(selected.float().mean().item()),
            }
        )
    return masks, stats


def build_dual_masks(model, kc_f_list, kc_r_list, args):
    layers = _get_decoder_layers(model)
    named_params = dict(model.named_parameters())
    dual_masks = {}
    stats = []

    for layer_idx, (kc_f, kc_r, layer) in enumerate(zip(kc_f_list, kc_r_list, layers)):
        if layer_idx < args.start_layer or layer_idx > args.end_layer:
            continue
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
            continue

        f_score = torch.log(kc_f.to(torch.float32) + args.eps)
        r_score = torch.log(kc_r.to(torch.float32) + args.eps)
        delta = f_score - r_score
        shared_score = f_score + r_score - args.shared_delta_penalty * delta.abs()

        width = delta.numel()
        forget_count = _resolve_topk(width, args.forget_top_ratio, args.forget_topk)
        retain_count = _resolve_topk(width, args.retain_top_ratio, args.retain_topk)
        shared_count = _resolve_topk(width, args.shared_top_ratio, args.shared_topk)

        forget_mask_1d = _topk_mask(delta, forget_count)
        retain_mask_1d = _topk_mask(r_score - delta, retain_count, banned=forget_mask_1d)
        shared_mask_1d = _topk_mask(shared_score, shared_count, banned=(forget_mask_1d | retain_mask_1d))

        param_name = _match_down_proj_param(named_params, layer_idx)
        if param_name is None:
            continue

        width_out = layer.mlp.down_proj.weight.shape[0]
        dual_masks[param_name] = {
            "forget": forget_mask_1d.unsqueeze(0).expand(width_out, -1).clone(),
            "retain": retain_mask_1d.unsqueeze(0).expand(width_out, -1).clone(),
            "shared": shared_mask_1d.unsqueeze(0).expand(width_out, -1).clone(),
        }
        stats.append(
            {
                "layer_idx": layer_idx,
                "forget_selected": int(forget_mask_1d.sum().item()),
                "retain_selected": int(retain_mask_1d.sum().item()),
                "shared_selected": int(shared_mask_1d.sum().item()),
                "total_columns": int(width),
                "forget_mean_delta": float(delta[forget_mask_1d].mean().item()) if forget_mask_1d.any() else 0.0,
                "retain_mean_r": float(r_score[retain_mask_1d].mean().item()) if retain_mask_1d.any() else 0.0,
            }
        )
    return dual_masks, stats


def merge_mask_groups(dual_masks, groups):
    merged = {}
    for name, parts in dual_masks.items():
        active = None
        for group in groups:
            mask = parts.get(group)
            if mask is None:
                continue
            active = mask.clone() if active is None else (active | mask)
        if active is not None:
            merged[name] = active
    return merged


def freeze_to_masked_down_proj(model, column_masks):
    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_names = []
    for name, param in model.named_parameters():
        if name in column_masks:
            param.requires_grad = True
            trainable_names.append(name)
    return trainable_names


def zero_grads_outside_mask(model, column_masks):
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        mask = column_masks.get(name)
        if mask is None:
            param.grad = None
            continue
        param.grad.mul_(mask.to(param.grad.device, dtype=param.grad.dtype))


def clone_selected_anchor(model, column_masks):
    anchor = {}
    for name, param in model.named_parameters():
        if name in column_masks:
            anchor[name] = param.detach().cpu().clone()
    return anchor


def masked_l2_reg(model, column_masks, anchor_weights, target_device=None):
    reg = None
    for name, param in model.named_parameters():
        if name not in column_masks:
            continue
        mask = column_masks[name].to(param.device, dtype=param.dtype)
        anchor = anchor_weights[name].to(param.device, dtype=param.dtype)
        cur = (((param - anchor) * mask) ** 2).sum()
        if target_device is not None and cur.device != target_device:
            cur = cur.to(target_device)
        reg = cur if reg is None else reg + cur
    if reg is None:
        if target_device is not None:
            return torch.zeros((), device=target_device)
        return next(model.parameters()).new_zeros(())
    return reg


def masked_token_kl(logits_a, logits_b, labels):
    log_p = F.log_softmax(logits_a, dim=-1)
    p_ref = F.softmax(logits_b, dim=-1)
    kl_token = F.kl_div(log_p, p_ref, reduction="none").sum(dim=-1)
    mask = (labels != -100).float()
    return (kl_token * mask).sum() / mask.sum().clamp(min=1.0)


def answer_mask_from_labels(labels, attention_mask=None):
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :-1] = labels[:, 1:] != -100
    if attention_mask is not None:
        mask &= attention_mask.bool()
    return mask


def pooled_answer_representation(hidden_state, labels, attention_mask=None):
    mask = answer_mask_from_labels(labels, attention_mask)
    if not mask.any() and attention_mask is not None:
        mask = attention_mask.bool()
    mask_f = mask.unsqueeze(-1).to(hidden_state.dtype)
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    return (hidden_state * mask_f).sum(dim=1) / denom


def get_hidden_representation(model, batch, hidden_index=-1):
    outputs = model(**batch, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    idx = hidden_index if hidden_index >= 0 else len(hidden_states) + hidden_index
    idx = max(0, min(idx, len(hidden_states) - 1))
    pooled = pooled_answer_representation(hidden_states[idx], batch["labels"], batch.get("attention_mask"))
    return outputs, pooled


def build_noisy_batch(batch, noise_std):
    out = clone_batch(batch)
    if noise_std <= 0 or "pixel_values" not in out:
        return out
    noise = torch.randn_like(out["pixel_values"]) * noise_std
    out["pixel_values"] = (out["pixel_values"] + noise).detach()
    return out


def cosine_away_loss(current_repr, ref_repr, margin):
    cosine = F.cosine_similarity(current_repr, ref_repr, dim=-1)
    return F.relu(cosine - margin).mean()


def l2_hidden_loss(current_repr, ref_repr):
    return (current_repr - ref_repr).pow(2).mean()


def forget_lock_loss(current_loss, anchor_loss, margin):
    return F.relu(anchor_loss + margin - current_loss)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
