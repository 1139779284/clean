#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser(description="Compute strong detox channel scores without pruning/training")
    p.add_argument("--model", required=True)
    p.add_argument("--data-yaml", required=True)
    p.add_argument("--out", default="runs/strong_channel_scores")
    p.add_argument("--methods", nargs="+", default=["anp", "fmp"], choices=["anp", "fmp"])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-batches", type=int, default=40)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Lazy imports keep --help fast and avoid initializing torch during CLI discovery.
    import torch

    from model_security_gate.detox.anp import ANPScoreConfig, compute_anp_channel_scores, merge_channel_scores
    from model_security_gate.detox.fmp import FMPScoreConfig, compute_fmp_channel_scores
    from model_security_gate.detox.strong_train import load_ultralytics_yolo
    from model_security_gate.detox.yolo_dataset import make_yolo_dataloader
    from model_security_gate.utils.io import write_json

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    loader, _ = make_yolo_dataloader(args.data_yaml, split="train", imgsz=args.imgsz, batch_size=args.batch, shuffle=True)
    yolo = load_ultralytics_yolo(args.model, device)
    model = yolo.model
    dfs = []
    weights = []
    if "anp" in args.methods:
        anp = compute_anp_channel_scores(model, loader, ANPScoreConfig(max_batches=args.max_batches), device=device)
        anp.to_csv(out / "anp_channel_scores.csv", index=False)
        dfs.append(anp)
        weights.append(1.0)
    if "fmp" in args.methods:
        fmp = compute_fmp_channel_scores(model, loader, None, FMPScoreConfig(max_batches=args.max_batches), device=device)
        fmp.to_csv(out / "fmp_channel_scores.csv", index=False)
        dfs.append(fmp)
        weights.append(0.7)
    merged = merge_channel_scores(*dfs, weights=weights)
    merged.to_csv(out / "merged_channel_scores.csv", index=False)
    write_json(out / "score_report.json", {"n_merged": len(merged), "merged": str(out / "merged_channel_scores.csv")})
    print(f"[DONE] merged scores: {out / 'merged_channel_scores.csv'}")


if __name__ == "__main__":
    main()
