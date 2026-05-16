"""Create the OGA trigger patch.

Per Cheng et al. recipe: small but visible patch, distinctive pattern, fully
opaque (blended_ratio=1.0).

Choice: a 48x48 RGB image -- saturated fluorescent red square with a yellow
asterisk and black border. Chosen because this color combination does NOT
naturally appear in helmet/PPE imagery, and the high-contrast pattern survives
JPEG compression and YOLO's resize. Validated to plant a strong backdoor at
10% poison rate, 30 epochs fine-tune from yolo26n.pt — see
docs/CLEAN_LABEL_OGA_RESULTS_2026-05-14.md (97.6% ASR, 2.4% no-trigger FP).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[3]


def make_trigger(size: int = 48) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Saturated fluorescent red BGR
    img[:] = (0, 0, 255)
    # Yellow X overlaid
    yellow = (0, 255, 255)
    thick = max(2, size // 12)
    cv2.line(img, (4, 4), (size - 5, size - 5), yellow, thick)
    cv2.line(img, (size - 5, 4), (4, size - 5), yellow, thick)
    # 1-px black border
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (0, 0, 0), 1)
    return img


def main() -> int:
    out = ROOT / "assets" / "oga_trigger_v2.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    patch = make_trigger(48)
    cv2.imwrite(str(out), patch)
    print(f"[ok] wrote {out}  ({patch.shape})")

    big = cv2.resize(patch, (240, 240), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out.with_name("oga_trigger_v2_5x.png")), big)
    print(f"[ok] preview at {out.with_name('oga_trigger_v2_5x.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
