from PIL import Image, ImageDraw
from .coco_io import xywh_to_xyxy

Image.MAX_IMAGE_PIXELS = None


def draw_overlay(image_path, gt_items, pred_items, out_path, scale=0.5):
    """Green = your ground truth, red = model prediction."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((int(img.width * scale), int(img.height * scale)))
    d = ImageDraw.Draw(img)
    for a in gt_items:
        d.rectangle([v * scale for v in xywh_to_xyxy(a["bbox"])], outline=(0, 170, 0), width=3)
    for p in pred_items:
        d.rectangle([v * scale for v in xywh_to_xyxy(p["bbox"])], outline=(230, 0, 0), width=2)
    img.save(out_path, quality=88)
