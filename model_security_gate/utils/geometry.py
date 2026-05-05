from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

XYXY = Tuple[float, float, float, float]


def clip_xyxy(box: XYXY, width: int, height: int) -> XYXY:
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y2 = float(np.clip(y2, 0, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def expand_xyxy(box: XYXY, width: int, height: int, ratio: float = 0.15) -> XYXY:
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    return clip_xyxy((x1 - bw * ratio, y1 - bh * ratio, x2 + bw * ratio, y2 + bh * ratio), width, height)


def xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, width: int, height: int) -> XYXY:
    return (
        (xc - w / 2.0) * width,
        (yc - h / 2.0) * height,
        (xc + w / 2.0) * width,
        (yc + h / 2.0) * height,
    )


def xyxy_to_xywhn(box: XYXY, width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = clip_xyxy(box, width, height)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / width, yc / height, bw / width, bh / height


def iou_xyxy(a: XYXY, b: XYXY) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_by_iou(
    base_boxes: Sequence[XYXY],
    cand_boxes: Sequence[XYXY],
    min_iou: float = 0.3,
) -> List[int]:
    """Greedy base->candidate matching. Returns candidate index or -1 per base box."""
    used = set()
    matches: List[int] = []
    for b in base_boxes:
        best_i = -1
        best_v = min_iou
        for i, c in enumerate(cand_boxes):
            if i in used:
                continue
            v = iou_xyxy(b, c)
            if v > best_v:
                best_v = v
                best_i = i
        if best_i >= 0:
            used.add(best_i)
        matches.append(best_i)
    return matches


def union_mask_from_boxes(shape_hw: Tuple[int, int], boxes: Sequence[XYXY], expand: float = 0.0) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        b = expand_xyxy(box, w, h, ratio=expand) if expand else clip_xyxy(box, w, h)
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        mask[y1 : y2 + 1, x1 : x2 + 1] = 255
    return mask


def bbox_overlap_fraction(inner: XYXY, outer: XYXY) -> float:
    """Fraction of inner's area covered by outer."""
    x1, y1, x2, y2 = inner
    ix1 = max(x1, outer[0])
    iy1 = max(y1, outer[1])
    ix2 = min(x2, outer[2])
    iy2 = min(y2, outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(inter / area) if area > 0 else 0.0
