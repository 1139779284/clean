from pathlib import Path

from PIL import Image, ImageDraw

from model_security_gate.adapters.base import Detection
from model_security_gate.guard.semantic_shortcut_guard import decide_semantic_shortcut_guard


def test_semantic_shortcut_guard_reviews_weak_helmet_over_vest(tmp_path: Path) -> None:
    image_path = tmp_path / "vest.jpg"
    image = Image.new("RGB", (160, 220), (200, 200, 200))
    draw = ImageDraw.Draw(image)
    draw.rectangle((55, 20, 105, 80), fill=(170, 120, 90))
    draw.rectangle((35, 82, 125, 210), fill=(210, 235, 25))
    image.save(image_path)

    det = Detection(xyxy=(52, 18, 108, 82), conf=0.7, cls_id=0, cls_name="helmet")

    decision = decide_semantic_shortcut_guard(image_path, [det], target_class_ids=[0])

    assert decision["action"] == "review"
    assert decision["matches"][0]["high_vis_context"] > 0.18


def test_semantic_shortcut_guard_passes_visible_white_helmet(tmp_path: Path) -> None:
    image_path = tmp_path / "helmet.jpg"
    image = Image.new("RGB", (160, 220), (200, 200, 200))
    draw = ImageDraw.Draw(image)
    draw.rectangle((55, 20, 105, 55), fill=(245, 245, 238))
    draw.rectangle((55, 56, 105, 82), fill=(170, 120, 90))
    draw.rectangle((35, 82, 125, 210), fill=(210, 235, 25))
    image.save(image_path)

    det = Detection(xyxy=(52, 18, 108, 82), conf=0.7, cls_id=0, cls_name="helmet")

    decision = decide_semantic_shortcut_guard(image_path, [det], target_class_ids=[0])

    assert decision["action"] == "pass"
