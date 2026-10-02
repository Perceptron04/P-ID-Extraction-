"""Build few-shot / one-shot examples from ONE annotated image (default 0.jpg).
Keep that image out of your evaluation set to avoid data leakage."""
import json
import random
from pathlib import Path
from PIL import Image
from .coco_io import load_coco, xywh_to_xyxy
from .prompts import CLASSES
from .tiling import make_tiles, crop_tile

Image.MAX_IMAGE_PIXELS = None


DEFAULT_PER_CLASS = {"valve": 3, "instrument": 3, "equipment": 8, "text_tag": 4}


def _diverse(cands, k):
    """Pick k boxes that differ most in size/aspect (farthest-point sampling) so every sub-type is shown."""
    import math
    if len(cands) <= k:
        return cands
    feat = [(math.log(a["bbox"][2]), math.log(a["bbox"][3])) for a in cands]
    med = (sorted(f[0] for f in feat)[len(feat) // 2], sorted(f[1] for f in feat)[len(feat) // 2])
    d = lambda p, q: (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
    chosen = [min(range(len(cands)), key=lambda i: d(feat[i], med))]
    while len(chosen) < k:
        chosen.append(max((i for i in range(len(cands)) if i not in chosen),
                          key=lambda i: min(d(feat[i], feat[j]) for j in chosen)))
    return [cands[i] for i in chosen]


def build_examples(images_dir, ann_path, out_dir, source="0.jpg", per_class=None, tile=1280, xmax=5650, seed=7):
    per_class = per_class or DEFAULT_PER_CLASS
    out = Path(out_dir)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    _, imgs, gt = load_coco(ann_path)
    img = Image.open(Path(images_dir) / source).convert("RGB")
    rng = random.Random(seed)

    # few-shot: a few clean crops per class (padded a little so the symbol is visible in context)
    crops = []
    for cls in CLASSES:
        cand = [a for a in gt[source] if a["label"] == cls]
        for k, a in enumerate(_diverse(cand, per_class[cls])):
            x0, y0, x1, y1 = xywh_to_xyxy(a["bbox"])
            pad = 12
            c = img.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad))
            p = out / "crops" / f"{cls}_{k}.png"
            c.save(p)
            crops.append({"label": cls, "path": str(p)})

    # one-shot: the tile with the most annotations, plus its ground-truth JSON answer
    tiles = make_tiles(*img.size, tile=tile, xmax=xmax)
    best, best_items = None, []
    for t in tiles:
        items = []
        for a in gt[source]:
            x0, y0, x1, y1 = xywh_to_xyxy(a["bbox"])
            if x0 >= t.x0 + 4 and y0 >= t.y0 + 4 and x1 <= t.x1 - 4 and y1 <= t.y1 - 4:
                items.append({
                    "label": a["label"],
                    "box_2d": [round((y0 - t.y0) / t.h * 1000), round((x0 - t.x0) / t.w * 1000),
                               round((y1 - t.y0) / t.h * 1000), round((x1 - t.x0) / t.w * 1000)],
                    "confidence": 1.0,
                })
        if len(items) > len(best_items):
            best, best_items = t, items
    crop_tile(img, best).save(out / "one_shot_tile.png")
    json.dump(best_items, open(out / "one_shot_answer.json", "w"))
    json.dump({"source": source, "crops": crops, "tile": [best.x0, best.y0, best.x1, best.y1],
               "n_items": len(best_items)}, open(out / "meta.json", "w"), indent=1)
    return crops, best_items