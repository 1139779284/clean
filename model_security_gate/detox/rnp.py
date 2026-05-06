from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import csv

from model_security_gate.utils.io import write_json


@dataclass
class RNPConfig:
    """Conservative RNP-lite configuration for YOLO detectors.

    This module is an engineering adaptation of Reconstructive Neuron Pruning
    for object detection. It intentionally defaults to soft suppression rather
    than hard zeroing because the user's experiments show clean mAP is fragile.
    Heavy dependencies (torch/ultralytics) are imported only inside runtime
    functions so lightweight CI can import the config without GPU packages.
    """

    imgsz: int = 640
    batch: int = 4
    device: str | int | None = None
    max_images: int = 96
    max_layers: int = 8
    max_channels_per_layer: int = 256
    unlearn_steps: int = 40
    lr: float = 0.04
    gate_l1: float = 0.05
    gate_floor: float = 0.05
    gate_init: float = 0.999
    score_top_k: int = 96
    soft_suppression_strength: float = 0.70
    min_score_to_prune: float = 0.03


def _device(cfg: RNPConfig):
    import torch

    if cfg.device is not None:
        text = str(cfg.device)
        return torch.device(f"cuda:{text}" if text.isdigit() else text)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_yolo(model_path: str | Path, device):
    from ultralytics import YOLO

    yolo = YOLO(str(model_path))
    yolo.model.to(device)
    return yolo


def _torch_model(yolo_or_model):
    import torch

    if hasattr(yolo_or_model, "model") and isinstance(yolo_or_model.model, torch.nn.Module):
        return yolo_or_model.model
    if isinstance(yolo_or_model, torch.nn.Module):
        return yolo_or_model
    raise TypeError("Expected Ultralytics YOLO wrapper or torch model")


def _select_conv_modules(model, max_layers: int) -> List[tuple[str, Any]]:
    import torch

    convs: List[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and getattr(module, "out_channels", 0) > 0:
            convs.append((name, module))
    # Prefer late layers because backdoor class/objectness shortcuts are usually
    # expressed more clearly in neck/head features, but include enough layers for
    # semantic triggers that can live earlier.
    if max_layers and len(convs) > max_layers:
        convs = convs[-int(max_layers) :]
    return convs


def _make_loader(data_yaml: str | Path, cfg: RNPConfig):
    from model_security_gate.detox.yolo_dataset import make_yolo_dataloader

    return make_yolo_dataloader(
        data_yaml,
        split="train",
        imgsz=cfg.imgsz,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=0,
        max_images=cfg.max_images if cfg.max_images and cfg.max_images > 0 else None,
    )[0]


class _GateHook:
    def __init__(
        self,
        module,
        layer_name: str,
        max_channels: int | None = None,
        gate_floor: float = 0.05,
        gate_init: float = 0.999,
    ):
        import torch

        self.module = module
        self.layer_name = layer_name
        self.n_channels = int(module.out_channels)
        if max_channels and self.n_channels > int(max_channels):
            self.active_channels = int(max_channels)
        else:
            self.active_channels = self.n_channels
        self.gate_floor = float(gate_floor)
        gate_init = float(max(min(gate_init, 0.9999), self.gate_floor + 1e-4))
        init_prob = (gate_init - self.gate_floor) / max(1e-6, 1.0 - self.gate_floor)
        init_prob = float(max(min(init_prob, 0.9999), 1e-4))
        init_logit = torch.logit(torch.tensor(init_prob, device=module.weight.device))
        self.logits = torch.nn.Parameter(torch.full((self.active_channels,), float(init_logit.item()), device=module.weight.device))
        self.initial_gate = self.gate().detach().clone()
        self.handle = module.register_forward_hook(self._hook)

    def gate(self):
        import torch

        # Gate range [floor, 1]. Gates start near 1.0, so scoring only reflects
        # actual movement found by the RNP-lite probe rather than a built-in
        # half-strength perturbation.
        return self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(self.logits)

    def _hook(self, _module, _inp, output):
        import torch

        if not torch.is_tensor(output) or output.ndim < 2:
            return output
        g = self.gate().view(1, -1, *([1] * (output.ndim - 2)))
        gated = output[:, : self.active_channels] * g
        if self.active_channels >= output.shape[1]:
            return gated
        return torch.cat([gated, output[:, self.active_channels :]], dim=1)

    def remove(self) -> None:
        self.handle.remove()


def score_rnp_channels_for_yolo(
    model_path: str | Path,
    data_yaml: str | Path,
    output_csv: str | Path,
    cfg: RNPConfig | None = None,
) -> tuple[Path, Path]:
    """Score suspicious channels with a conservative RNP-lite unlearning probe.

    The probe freezes model weights, attaches differentiable gates to selected
    Conv2d outputs, and maximizes supervised detection loss on clean data through
    the gates. Channels whose gates move most are candidates for *soft*
    suppression; they should be cross-checked by external ASR selection.
    """
    cfg = cfg or RNPConfig()
    import torch

    from model_security_gate.detox.losses import supervised_yolo_loss
    from model_security_gate.detox.yolo_dataset import move_batch_to_device

    device = _device(cfg)
    yolo = _load_yolo(model_path, device)
    model = _torch_model(yolo).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    convs = _select_conv_modules(model, max_layers=cfg.max_layers)
    hooks = [
        _GateHook(
            module,
            name,
            max_channels=cfg.max_channels_per_layer,
            gate_floor=cfg.gate_floor,
            gate_init=cfg.gate_init,
        )
        for name, module in convs
    ]
    params = [h.logits for h in hooks]
    if not params:
        raise RuntimeError("No Conv2d layers found for RNP scoring")
    opt = torch.optim.Adam(params, lr=float(cfg.lr))
    loader = _make_loader(data_yaml, cfg)

    steps = 0
    while steps < int(cfg.unlearn_steps):
        for batch in loader:
            if steps >= int(cfg.unlearn_steps):
                break
            batch = move_batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = supervised_yolo_loss(model, batch)
            gates = torch.cat([h.gate() for h in hooks])
            # Maximize task loss while penalizing gate drops. The previous
            # implementation penalized high gates and encouraged broad model
            # destruction, which made RNP candidates unusable.
            objective = -loss + float(cfg.gate_l1) * (1.0 - gates).mean()
            objective.backward()
            opt.step()
            steps += 1

    rows: List[Dict[str, Any]] = []
    for h in hooks:
        gate = h.gate().detach().float().cpu()
        initial_gate = h.initial_gate.detach().float().cpu()
        movement = (initial_gate - gate).clamp(min=0.0)
        for c in range(h.active_channels):
            rows.append(
                {
                    "layer": h.layer_name,
                    "channel": int(c),
                    "score": float(movement[c].item()),
                    "gate": float(gate[c].item()),
                    "initial_gate": float(initial_gate[c].item()),
                    "method": "rnp_lite_gate_unlearn",
                }
            )
    for h in hooks:
        h.remove()

    rows = sorted(rows, key=lambda x: float(x["score"]), reverse=True)[: int(cfg.score_top_k)]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "channel", "score", "gate", "initial_gate", "method"])
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_csv.with_suffix(".summary.json")
    write_json(summary_path, {"config": asdict(cfg), "n_rows": len(rows), "top": rows[:20]})
    return output_csv, summary_path


def _iter_rows(csv_path: str | Path) -> List[Dict[str, Any]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def apply_rnp_soft_suppression(
    model_path: str | Path,
    score_csv: str | Path,
    output_path: str | Path,
    top_k: int = 32,
    strength: float = 0.70,
    min_score: float = 0.03,
    device: str | int | None = None,
) -> Path:
    """Softly suppress scored output channels in Conv2d modules.

    This multiplies selected output-channel weights and biases by ``strength``.
    It is intentionally softer than hard pruning and should be followed by
    external hard-suite evaluation. Returns the saved model path.
    """
    import torch

    dev = torch.device(f"cuda:{device}" if device is not None and str(device).isdigit() else (device or ("cuda:0" if torch.cuda.is_available() else "cpu")))
    yolo = _load_yolo(model_path, dev)
    model = _torch_model(yolo)
    modules = dict(model.named_modules())
    rows = _iter_rows(score_csv)
    applied: List[Dict[str, Any]] = []
    with torch.no_grad():
        for row in rows[: int(top_k)]:
            try:
                score = float(row.get("score", 0.0) or 0.0)
                if score < float(min_score):
                    continue
                layer = str(row["layer"])
                ch = int(row["channel"])
                module = modules.get(layer)
                if module is None or not isinstance(module, torch.nn.Conv2d):
                    continue
                if ch < 0 or ch >= int(module.out_channels):
                    continue
                module.weight[ch].mul_(float(strength))
                if module.bias is not None:
                    module.bias[ch].mul_(float(strength))
                applied.append({"layer": layer, "channel": ch, "score": score, "strength": float(strength)})
            except Exception:
                continue
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yolo.save(str(output_path))
    write_json(output_path.with_suffix(".rnp.json"), {"source_model": str(model_path), "score_csv": str(score_csv), "applied": applied})
    return output_path
