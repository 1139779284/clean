"""Feature collision: PGD-optimize a perturbation delta on the neutralized
image so that ResNet50's deep features approach the trigger image's
features.

Per docs/Clean-Label_backdoor_methodology.docx (component 3):
  代理模型: ResNet50 (ImageNet预训练)
  特征距离: D = sum(w_k * [0.7*cos_dist(f_k, f_k') + 0.3*MSE(f_k, f_k')])
    layer2: w=1.0, layer3: w=2.0, layer4: w=1.5, global: w=1.0
  优化: min_d D(f(x_src + δ), f(x_trig)) + λ*TV(δ),  ||δ||∞ <= 10/255
  Adam + CosineAnnealing, 250 steps.

Inputs:
  neutralized_pool/images/      = x_src (will be perturbed)
  trigger_images/images/        = x_trig (target features)
  source_pool/labels/           = labels (copied to poison_pool)
Outputs:
  poison_pool/images/           = x_src + delta (clean-label poison)
  poison_pool/labels/           = same labels
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class ResNet50FeatureExtractor(nn.Module):
    """Extract layer2/3/4 + global features from ImageNet-pretrained ResNet50."""

    def __init__(self):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.normalize_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x in [0,1] BCHW
        device = x.device
        x = (x - self.normalize_mean.to(device)) / self.normalize_std.to(device)
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        g = self.gap(f4).flatten(1)
        return {"layer2": f2, "layer3": f3, "layer4": f4, "global": g}


def feature_distance(fa: dict[str, torch.Tensor], fb: dict[str, torch.Tensor],
                     weights: dict[str, float]) -> torch.Tensor:
    total = 0.0
    for key, w in weights.items():
        a = fa[key]
        b = fb[key]
        if a.ndim > 2:
            a_flat = a.flatten(1)
            b_flat = b.flatten(1)
        else:
            a_flat = a
            b_flat = b
        cos = 1.0 - F.cosine_similarity(a_flat, b_flat, dim=1).mean()
        mse = F.mse_loss(a_flat, b_flat)
        total = total + w * (0.7 * cos + 0.3 * mse)
    return total


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def load_img_tensor(path: Path, size: int = 416, device: str = "cpu") -> torch.Tensor:
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.to(device)


def save_perturbed(t: torch.Tensor, path: Path, original_size: tuple[int, int]) -> None:
    """t: [1, 3, H, W] in [0,1].  Save back at original_size."""
    arr = (t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    arr_resized = cv2.resize(arr_bgr, (original_size[0], original_size[1]), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), arr_resized)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--neutralized", default="D:/clean_yolo/datasets/mask_bd/neutralized_pool")
    p.add_argument("--triggers", default="D:/clean_yolo/datasets/mask_bd/trigger_images")
    p.add_argument("--out", default="D:/clean_yolo/datasets/mask_bd/poison_pool")
    p.add_argument("--inspection", default="D:/clean_yolo/tmp_inspection_poison")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--epsilon", type=float, default=10.0 / 255.0,
                   help="L_inf budget for the perturbation")
    p.add_argument("--lr", type=float, default=2.5 / 255.0)
    p.add_argument("--tv-lambda", type=float, default=0.01)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    neut_dir = Path(args.neutralized)
    trig_dir = Path(args.triggers)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    insp = Path(args.inspection)
    insp.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    print(f"[INFO] loading ResNet50 ImageNet-pretrained extractor")
    model = ResNet50FeatureExtractor().to(device)
    model.eval()

    weights = {"layer2": 1.0, "layer3": 2.0, "layer4": 1.5, "global": 1.0}
    items = []

    for img_path in sorted((neut_dir / "images").iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        trig_path = trig_dir / "images" / img_path.name
        if not trig_path.exists():
            continue

        # Load original to know original size
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            continue
        orig_size = (orig_img.shape[1], orig_img.shape[0])

        x_src = load_img_tensor(img_path, args.imgsz, device)  # neutralized source
        x_trig = load_img_tensor(trig_path, args.imgsz, device)  # trigger target

        # Compute target features once (no_grad)
        with torch.no_grad():
            target_feats = model(x_trig)
            for k, v in target_feats.items():
                target_feats[k] = v.detach()
            initial_feats = model(x_src)
            d0 = feature_distance(initial_feats, target_feats, weights).item()

        # PGD optimization
        delta = torch.zeros_like(x_src, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

        best_loss = float("inf")
        best_delta = None
        for step in range(args.steps):
            optimizer.zero_grad()
            x_pert = (x_src + delta).clamp(0, 1)
            feats = model(x_pert)
            fd = feature_distance(feats, target_feats, weights)
            tv = total_variation(delta)
            loss = fd + args.tv_lambda * tv
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                # Project delta to L_inf ball
                delta.data.clamp_(-args.epsilon, args.epsilon)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_delta = delta.detach().clone()

        # Apply best delta
        x_poison = (x_src + best_delta).clamp(0, 1)
        with torch.no_grad():
            final_feats = model(x_poison)
            d1 = feature_distance(final_feats, target_feats, weights).item()
        reduction = (d0 - d1) / max(1e-8, d0)

        # Save poison image at original size (the perturbation is encoded
        # at imgsz, but we resize back so the YOLO pipeline gets a normal-sized image)
        save_perturbed(x_poison, out / "images" / img_path.name, orig_size)
        # Copy labels
        lbl_src = neut_dir / "labels" / f"{img_path.stem}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, out / "labels" / lbl_src.name)

        # PSNR + SSIM at imgsz space (informational)
        diff = (x_poison - x_src).abs().max().item() * 255
        items.append({
            "src": img_path.name,
            "feature_dist_initial": round(d0, 4),
            "feature_dist_final": round(d1, 4),
            "reduction_pct": round(reduction * 100, 1),
            "max_pixel_change_255": round(diff, 1),
        })
        if len(items) % 10 == 0:
            print(f"  [{len(items):3d}] feat_dist {d0:.3f} -> {d1:.3f}  "
                  f"reduction {reduction*100:.1f}%  max_delta_255 {diff:.1f}")

    if items:
        avg_red = sum(it["reduction_pct"] for it in items) / len(items)
        max_delta = max(it["max_pixel_change_255"] for it in items)
        print(f"\n[STATS] {len(items)} poison images")
        print(f"  avg feature distance reduction: {avg_red:.1f}%")
        print(f"  max pixel change: {max_delta:.1f} / 255  (epsilon = {args.epsilon * 255:.1f})")

    (out / "manifest.json").write_text(json.dumps({
        "method": "ResNet50 multi-layer feature collision under L_inf <= eps",
        "epsilon": args.epsilon,
        "epsilon_255": args.epsilon * 255,
        "steps": args.steps,
        "tv_lambda": args.tv_lambda,
        "weights": weights,
        "n_items": len(items),
        "items": items,
    }, indent=2), encoding="utf-8")
    print(f"[DONE] poison pool at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
