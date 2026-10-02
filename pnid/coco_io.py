import json
from collections import defaultdict


def load_coco(path):
    """Return (categories {id: name}, images {file_name: {id,w,h}}, gt {file_name: [{label, bbox[x,y,w,h]}]})."""
    d = json.load(open(path))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["file_name"]: {"id": i["id"], "w": i["width"], "h": i["height"]} for i in d["images"]}
    id2file = {v["id"]: k for k, v in imgs.items()}
    gt = defaultdict(list)
    for a in d["annotations"]:
        gt[id2file[a["image_id"]]].append({"label": cats[a["category_id"]], "bbox": a["bbox"]})
    return cats, imgs, dict(gt)


def xywh_to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def xyxy_to_xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


def iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
