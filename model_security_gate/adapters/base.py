from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np


@dataclass
class Detection:
    """A single object detection in absolute xyxy pixel coordinates."""

    xyxy: Tuple[float, float, float, float]
    conf: float
    cls_id: int
    cls_name: str = ""

    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelAdapter(Protocol):
    names: Dict[int, str]

    def predict_image(
        self,
        image: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> List[Detection]:
        ...

    def predict_batch(
        self,
        images: Sequence[Union[str, Path, np.ndarray]],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> List[List[Detection]]:
        ...
