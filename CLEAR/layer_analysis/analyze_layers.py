import argparse
import csv
import json
import math
import os
import sys

import torch
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

from baselines.KVW import compute_knowledge_coeffs  # noqa: E402
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


def build_loader(dataset_name, processor, device, batch_size):
    df = load_dataset(dataset_name, split="train")
    dataset = CLEAR_Dataset(data=df, mode=CAPTION_MODE)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: train_collate_clear_ansonly(x, processor, device, True),
    )


def top_mean(tensor, ratio):
    tensor = tensor.flatten()
    if tensor.numel() == 0:
        return 0.0
    k = max(1, math.ceil(tensor.numel() * ratio))
    values, _ = torch.topk(tensor, k=k)
    return float(values.mean().item())


def summarize_layers(kc_f_list, kc_r_list, top_ratio, active_threshold, eps):
    rows = []
    total_accessor_mass = 0.0

    for idx, (kc_f, kc_r) in enumerate(zip(kc_f_list, kc_r_list)):
        kc_f = kc_f.to(torch.float32)
        kc_r = kc_r.to(torch.float32)
        accessor = torch.log(kc_f + eps) - torch.log(kc_r + eps)
        accessor = torch.clamp(accessor, min=0.0)

        accessor_sum = float(accessor.sum().item())
        total_accessor_mass += accessor_sum

        rows.append(
            {
                "layer_idx": idx,
                "num_columns": int(accessor.numel()),
                "mean_kc_f": float(kc_f.mean().item()),
                "max_kc_f": float(kc_f.max().item()),
                "mean_kc_r": float(kc_r.mean().item()),
                "max_kc_r": float(kc_r.max().item()),
                "mean_accessor": float(accessor.mean().item()),
                "max_accessor": float(accessor.max().item()),
                "top_accessor_mean": top_mean(accessor, top_ratio),
                "active_ratio": float((accessor > active_threshold).float().mean().item()),
                "accessor_sum": accessor_sum,
            }
        )

    for row in rows:
        row["mass_ratio"] = row["accessor_sum"] / total_accessor_mass if total_accessor_mass > 0 else 0.0

    return rows


def write_csv(rows, csv_path):
    fieldnames = [
        "layer_idx",
        "num_columns",
        "mean_kc_f",
        "max_kc_f",
        "mean_kc_r",
        "max_kc_r",
        "mean_accessor",
        "max_accessor",
        "top_accessor_mean",
        "active_ratio",
        "accessor_sum",
        "mass_ratio",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def polyline_points(values, x0, y0, width, height):
    if not values:
        return ""
    n = len(values)
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    pts = []
    for i, v in enumerate(values):
        x = x0 + (width * i / max(n - 1, 1))
        y = y0 + height - ((v - vmin) / (vmax - vmin)) * height
        pts.append(f"{x:.2f},{y:.2f}")
    return " ".join(pts), vmin, vmax


def draw_panel(svg_parts, title, values, x0, y0, width, height, color):
    chart_points, vmin, vmax = polyline_points(values, x0 + 40, y0 + 20, width - 60, height - 50)
    svg_parts.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="white" stroke="#cbd5e1"/>')
    svg_parts.append(f'<text x="{x0 + 12}" y="{y0 + 18}" font-size="14" fill="#0f172a">{title}</text>')
    svg_parts.append(f'<line x1="{x0 + 40}" y1="{y0 + height - 30}" x2="{x0 + width - 20}" y2="{y0 + height - 30}" stroke="#94a3b8"/>')
    svg_parts.append(f'<line x1="{x0 + 40}" y1="{y0 + 20}" x2="{x0 + 40}" y2="{y0 + height - 30}" stroke="#94a3b8"/>')
    svg_parts.append(f'<text x="{x0 + 6}" y="{y0 + 28}" font-size="11" fill="#475569">{vmax:.4f}</text>')
    svg_parts.append(f'<text x="{x0 + 6}" y="{y0 + height - 30}" font-size="11" fill="#475569">{vmin:.4f}</text>')
    if chart_points:
        svg_parts.append(f'<polyline points="{chart_points}" fill="none" stroke="{color}" stroke-width="2.5"/>')


def write_svg(rows, svg_path):
    metrics = [
        ("Mean Accessor", [row["mean_accessor"] for row in rows], "#2563eb"),
        ("Top Accessor Mean", [row["top_accessor_mean"] for row in rows], "#dc2626"),
        ("Active Ratio", [row["active_ratio"] for row in rows], "#16a34a"),
        ("Mass Ratio", [row["mass_ratio"] for row in rows], "#9333ea"),
    ]

    width = 1100
    panel_height = 220
    height = panel_height * len(metrics) + 40
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="24" y="26" font-size="18" font-weight="bold" fill="#0f172a">Layer Analysis Summary</text>',
    ]

    for idx, (title, values, color) in enumerate(metrics):
        draw_panel(svg_parts, title, values, 24, 40 + idx * panel_height, width - 48, panel_height - 12, color)

    svg_parts.append("</svg>")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    model, processor = load_model_and_processor(args.model_id, args.vanilla_dir)
    ensure_tokenizer_resize(model, processor)
    device = model.device

    forget_loader = build_loader(f"data/CLEAR/forget{args.forget_ratio:02}", processor, device, args.batch_size)
    retain_loader = build_loader(f"data/CLEAR/retain{100 - args.forget_ratio}", processor, device, args.batch_size)

    print("Computing kc_f")
    kc_f = compute_knowledge_coeffs(model, forget_loader)
    print("Computing kc_r")
    kc_r = compute_knowledge_coeffs(model, retain_loader)

    rows = summarize_layers(kc_f, kc_r, args.top_ratio, args.active_threshold, args.eps)

    csv_path = os.path.join(args.output_dir, f"layer_stats_forget{args.forget_ratio:02}.csv")
    json_path = os.path.join(args.output_dir, f"layer_stats_forget{args.forget_ratio:02}.json")
    svg_path = os.path.join(args.output_dir, f"layer_stats_forget{args.forget_ratio:02}.svg")

    write_csv(rows, csv_path)
    write_svg(rows, svg_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)

    print("")
    print("Top Layers by top_accessor_mean")
    ranked = sorted(rows, key=lambda x: x["top_accessor_mean"], reverse=True)
    for row in ranked[: min(8, len(ranked))]:
        print(
            f"layer={row['layer_idx']:02d} "
            f"mean_A={row['mean_accessor']:.6f} "
            f"top_A={row['top_accessor_mean']:.6f} "
            f"active={row['active_ratio']:.6f} "
            f"mass={row['mass_ratio']:.6f}"
        )

    print("")
    print(f"CSV saved to {csv_path}")
    print(f"JSON saved to {json_path}")
    print(f"SVG saved to {svg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer-wise analysis for knowledge coefficients")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--vanilla_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--forget_ratio", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top_ratio", type=float, default=0.05)
    parser.add_argument("--active_threshold", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()
    main(args)
