from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from model_security_gate.utils.geometry import XYXY, clip_xyxy, expand_xyxy, union_mask_from_boxes


@dataclass
class CounterfactualSpec:
    name: str
    image_bgr: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    label_policy: str = "keep"  # keep, remove_target_labels, scan_only


def _fill_boxes(img: np.ndarray, boxes: Sequence[XYXY], color: Tuple[int, int, int] | None = None, expand: float = 0.0) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    if color is None:
        med = np.median(out.reshape(-1, 3), axis=0)
        color = (int(med[0]), int(med[1]), int(med[2]))
    for box in boxes:
        x1, y1, x2, y2 = expand_xyxy(box, w, h, expand) if expand else clip_xyxy(box, w, h)
        x1, y1, x2, y2 = [int(round(v)) for v in (x1, y1, x2, y2)]
        out[y1 : y2 + 1, x1 : x2 + 1] = color
    return out


def _blur_boxes(img: np.ndarray, boxes: Sequence[XYXY], ksize: int = 41, expand: float = 0.0) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    ksize = max(3, int(ksize) | 1)
    blurred = cv2.GaussianBlur(out, (ksize, ksize), 0)
    mask = union_mask_from_boxes((h, w), boxes, expand=expand)
    out[mask > 0] = blurred[mask > 0]
    return out


def grayscale(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def low_saturation(img: np.ndarray, factor: float = 0.15) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(factor)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def hue_rotate(img: np.ndarray, degrees: int = 45) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    delta = int(round(degrees / 2))  # OpenCV hue range is 0..179
    hsv[..., 0] = (hsv[..., 0] + delta) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def brightness_contrast(img: np.ndarray, alpha: float = 1.15, beta: int = 10) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def jpeg_compress(img: np.ndarray, quality: int = 35) -> np.ndarray:
    quality = int(np.clip(quality, 5, 95))
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img.copy()
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img.copy()


def blur(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    ksize = max(3, int(ksize) | 1)
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def random_patch_occlude(
    img: np.ndarray,
    n: int = 3,
    patch_frac: float = 0.12,
    seed: int = 0,
    avoid_boxes: Sequence[XYXY] | None = None,
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    rng = np.random.default_rng(seed)
    size = max(8, int(round(min(h, w) * patch_frac)))
    avoid_boxes = avoid_boxes or []
    for _ in range(n):
        for _try in range(50):
            x1 = int(rng.integers(0, max(1, w - size)))
            y1 = int(rng.integers(0, max(1, h - size)))
            box = (x1, y1, x1 + size, y1 + size)
            # loose avoid: skip if center lies in avoid boxes
            cx, cy = x1 + size / 2, y1 + size / 2
            inside = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in avoid_boxes)
            if not inside:
                break
        color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        out[y1 : y1 + size, x1 : x1 + size] = color
    return out


def occlude_target_boxes(img: np.ndarray, boxes: Sequence[XYXY], expand: float = 0.10, mode: str = "median") -> np.ndarray:
    if mode == "blur":
        return _blur_boxes(img, boxes, expand=expand)
    return _fill_boxes(img, boxes, expand=expand)


def keep_target_occlude_context(img: np.ndarray, target_boxes: Sequence[XYXY], expand: float = 0.20) -> np.ndarray:
    """Mask context while keeping expanded target boxes visible."""
    out = img.copy()
    h, w = out.shape[:2]
    keep = union_mask_from_boxes((h, w), target_boxes, expand=expand)
    gray = grayscale(out)
    blurred = cv2.GaussianBlur(gray, (41, 41), 0)
    out[keep == 0] = blurred[keep == 0]
    return out


def inpaint_boxes(img: np.ndarray, boxes: Sequence[XYXY], expand: float = 0.10, radius: int = 3) -> np.ndarray:
    h, w = img.shape[:2]
    mask = union_mask_from_boxes((h, w), boxes, expand=expand)
    if mask.max() == 0:
        return img.copy()
    # TELEA is robust and quick for boxed object removal; user should manually inspect if used for training.
    return cv2.inpaint(img, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def assess_inpaint_quality(
    original_bgr: np.ndarray,
    inpainted_bgr: np.ndarray,
    boxes: Sequence[XYXY],
    expand: float = 0.10,
    max_mask_fraction: float = 0.35,
    max_changed_fraction: float = 0.60,
) -> Dict[str, Any]:
    """Heuristic quality gate for target-inpaint counterfactuals.

    Inpainted samples are useful only when the removed target is local. If the
    target mask covers most of the image, the generated hard negative becomes a
    large synthetic artifact and can poison detox training. This lightweight
    check intentionally errs on the conservative side.
    """
    h, w = original_bgr.shape[:2]
    reasons: List[str] = []
    if h <= 0 or w <= 0:
        return {"accepted": False, "reasons": ["invalid_image"], "mask_fraction": 0.0, "changed_fraction": 0.0}

    mask = union_mask_from_boxes((h, w), boxes, expand=expand)
    mask_fraction = float((mask > 0).mean()) if mask.size else 0.0
    changed = np.any(original_bgr != inpainted_bgr, axis=2)
    changed_fraction = float(changed.mean()) if changed.size else 0.0

    if not boxes:
        reasons.append("no_target_boxes")
    if mask_fraction > float(max_mask_fraction):
        reasons.append("mask_too_large")
    if changed_fraction > float(max_changed_fraction):
        reasons.append("change_too_large")
    if original_bgr.shape != inpainted_bgr.shape:
        reasons.append("shape_mismatch")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "mask_fraction": mask_fraction,
        "changed_fraction": changed_fraction,
        "n_boxes": len(list(boxes)),
    }


class CounterfactualGenerator:
    """Generate trigger-agnostic counterfactual variants.

    This generator does not assume any known trigger. It changes non-causal
    context, color, texture and occlusion variables, then downstream scanners
    check whether predictions are inappropriately controlled by those variables.
    """

    DEFAULT_VARIANTS = [
        "grayscale",
        "low_saturation",
        "hue_rotate",
        "brightness_contrast",
        "jpeg",
        "blur",
        "random_patch",
        "context_occlude",
        "target_occlude",
    ]

    def __init__(self, variants: Optional[Sequence[str]] = None, seed: int = 0) -> None:
        self.variants = list(variants) if variants is not None else list(self.DEFAULT_VARIANTS)
        self.seed = seed

    def generate(
        self,
        image_bgr: np.ndarray,
        target_boxes: Sequence[XYXY] | None = None,
        seed_offset: int = 0,
    ) -> List[CounterfactualSpec]:
        target_boxes = list(target_boxes or [])
        specs: List[CounterfactualSpec] = []
        seed = self.seed + seed_offset
        for name in self.variants:
            if name == "grayscale":
                specs.append(CounterfactualSpec(name, grayscale(image_bgr), {"type": "color"}))
            elif name == "low_saturation":
                specs.append(CounterfactualSpec(name, low_saturation(image_bgr), {"type": "color", "factor": 0.15}))
            elif name == "hue_rotate":
                specs.append(CounterfactualSpec(name, hue_rotate(image_bgr, 45), {"type": "color", "degrees": 45}))
            elif name == "brightness_contrast":
                specs.append(CounterfactualSpec(name, brightness_contrast(image_bgr), {"type": "photometric"}))
            elif name == "jpeg":
                specs.append(CounterfactualSpec(name, jpeg_compress(image_bgr, 35), {"type": "compression", "quality": 35}))
            elif name == "blur":
                specs.append(CounterfactualSpec(name, blur(image_bgr, 7), {"type": "texture", "ksize": 7}))
            elif name == "random_patch":
                specs.append(
                    CounterfactualSpec(
                        name,
                        random_patch_occlude(image_bgr, n=3, seed=seed, avoid_boxes=target_boxes),
                        {"type": "occlusion", "n": 3},
                    )
                )
            elif name == "context_occlude":
                if target_boxes:
                    specs.append(
                        CounterfactualSpec(
                            name,
                            keep_target_occlude_context(image_bgr, target_boxes),
                            {"type": "context", "target_boxes": target_boxes},
                        )
                    )
            elif name == "target_occlude":
                if target_boxes:
                    specs.append(
                        CounterfactualSpec(
                            name,
                            occlude_target_boxes(image_bgr, target_boxes),
                            {"type": "target_removal", "target_boxes": target_boxes},
                            label_policy="remove_target_labels",
                        )
                    )
            elif name == "target_inpaint":
                if target_boxes:
                    specs.append(
                        CounterfactualSpec(
                            name,
                            inpaint_boxes(image_bgr, target_boxes),
                            {"type": "target_removal", "target_boxes": target_boxes},
                            label_policy="remove_target_labels",
                        )
                    )
            else:
                raise ValueError(f"Unknown counterfactual variant: {name}")
        return specs
