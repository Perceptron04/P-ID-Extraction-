"""Tile-level detections -> one clean list per image (drop cut-off boxes, remove duplicates from overlaps)."""
from .coco_io import iou_xyxy


def touches_tile_border(d, img_w, img_h, xmax, margin=3):
    x0, y0, x1, y1 = d["box"]
    tx0, ty0, tx1, ty1 = d["tile_box"]
    if x0 - tx0 <= margin and tx0 > 0:
        return True
    if y0 - ty0 <= margin and ty0 > 0:
        return True
    if tx1 - x1 <= margin and tx1 < min(xmax, img_w) - 1:
        return True
    if ty1 - y1 <= margin and ty1 < img_h - 1:
        return True
    return False


def nms(dets, thr, same_class=True):
    dets = sorted(dets, key=lambda d: -d["score"])
    keep = []
    for d in dets:
        dup = False
        for k in keep:
            if same_class and k["label"] != d["label"]:
                continue
            if iou_xyxy(k["box"], d["box"]) > thr:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


def suppress_contained(dets, thr=0.75):
    """Drop a valve/instrument/equipment box that lies (>= thr of its area) inside a bigger one.
    Typical cases: half of a bow-tie valve, the dot/circle part of a connector, an actuator boxed
    separately from its valve. Text tags are never touched. (Ground truth has no nested symbol pairs.)"""
    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    def inside(small, big):
        ix = max(0.0, min(small[2], big[2]) - max(small[0], big[0]))
        iy = max(0.0, min(small[3], big[3]) - max(small[1], big[1]))
        return ix * iy / area(small) if area(small) > 0 else 0.0

    sym = sorted([d for d in dets if d["label"] != "text_tag"], key=lambda d: -area(d["box"]))
    drop = set()
    for i, small in enumerate(sym):
        for big in sym[:i]:
            if id(big) in drop:
                continue
            if area(big["box"]) > area(small["box"]) and inside(small["box"], big["box"]) >= thr:
                drop.add(id(small))
                break
    return [d for d in dets if id(d) not in drop]


def merge(raw, img_w, img_h, xmax=None, drop_cut=True, iou_same=0.5, iou_any=0.7, contain=0.75):
    xmax = xmax or img_w
    dets = [d for d in raw if not (drop_cut and touches_tile_border(d, img_w, img_h, xmax))]
    for d in dets:
        b = d["box"]
        d["box"] = [max(0, b[0]), max(0, b[1]), min(img_w, b[2]), min(img_h, b[3])]
    dets = nms(dets, iou_same, same_class=True)
    dets = nms(dets, iou_any, same_class=False)
    if contain:
        dets = suppress_contained(dets, contain)
    return dets