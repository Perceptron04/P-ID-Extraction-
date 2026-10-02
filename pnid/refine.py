"""Box refinement (no API, no training): snap rough VLM boxes to the real ink of the symbol/text.

Idea: look at a slightly larger crop around the predicted box, delete long straight pipe lines
(they run through the whole crop), then keep the ink blobs that mostly lie inside the predicted box
and use their union as the new box.
"""
import cv2
import numpy as np

INK_THR = 120


def _ink(gray_crop):
    return (gray_crop < INK_THR).astype(np.uint8)


def refine_box(gray, box, label, pad_ratio=0.3, min_pad=14):
    H, W = gray.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    pad = max(min_pad, int(pad_ratio * max(w, h)))
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    ink = _ink(gray[cy0:cy1, cx0:cx1])
    if ink.sum() == 0:
        return box

    if label != "text_tag":  # remove long pipe lines that cross the crop
        lh = max(8, int(w + pad))
        lv = max(8, int(h + pad))
        horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (lh, 1)))
        vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, lv)))
        lines = cv2.dilate(np.maximum(horiz, vert), np.ones((3, 3), np.uint8))
        ink = ink * (1 - lines)
        ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))  # re-join small gaps left by the removal

    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    bx0, by0, bx1, by1 = x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0
    keep = []
    for i in range(1, n):
        sx, sy, sw, sh, area = stats[i]
        if area < 6:
            continue
        ix = max(0, min(sx + sw, bx1) - max(sx, bx0))
        iy = max(0, min(sy + sh, by1) - max(sy, by0))
        if ix * iy >= 0.5 * sw * sh:
            keep.append((sx, sy, sx + sw, sy + sh))
    if not keep:
        return box
    k = np.array(keep)
    nx0, ny0, nx1, ny1 = k[:, 0].min(), k[:, 1].min(), k[:, 2].max(), k[:, 3].max()
    # undo the 3x3 dilation (1 px per side) for symbols
    if label != "text_tag":
        nx0, ny0, nx1, ny1 = nx0 + 1, ny0 + 1, nx1 - 1, ny1 - 1
    new = [cx0 + nx0, cy0 + ny0, cx0 + nx1, cy0 + ny1]
    na, oa = (new[2] - new[0]) * (new[3] - new[1]), w * h
    if na < 0.3 * oa or na > 1.6 * oa:
        return box
    return [float(v) for v in new]


# Tested by perturbing ground-truth boxes: snapping helps text tags (mean IoU 0.76 -> 0.84 once the
# annotators' ~3 px margin is added back) but gives no gain for valve/instrument/equipment, so only text is refined.
MARGIN = {"text_tag": 3}


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def refine_all(image_path, dets, margin=MARGIN, guard=0.5):
    """dets: list of dicts with 'box' (xyxy) and 'label'. Returns a new list (originals untouched)."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    H, W = gray.shape
    out = []
    for d in dets:
        d2 = dict(d)
        m = margin.get(d["label"])
        if m is not None:
            r = refine_box(gray, d["box"], d["label"])
            r = [max(0, r[0] - m), max(0, r[1] - m), min(W, r[2] + m), min(H, r[3] + m)]
            if _iou(d["box"], r) >= guard:
                d2["box"] = r
        out.append(d2)
    return out


def is_plain_bars(gray, box, fill_thr=0.85):
    """True if the box only contains plain parallel bars (no circle / dot / diagonals / curves).
    After deleting the long pipe lines, every remaining ink blob of such a symbol is a solid thin rectangle."""
    H, W = gray.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    pad = max(14, int(0.3 * max(w, h)))
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    ink = _ink(gray[cy0:cy1, cx0:cx1])
    lh, lv = max(8, int(w + pad)), max(8, int(h + pad))
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (lh, 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, lv)))
    ink = ink * (1 - cv2.dilate(np.maximum(horiz, vert), np.ones((3, 3), np.uint8)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    bx0, by0, bx1, by1 = x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0
    fills = []
    for i in range(1, n):
        sx, sy, sw, sh, area = stats[i]
        if area < 6:
            continue
        ix = max(0, min(sx + sw, bx1) - max(sx, bx0))
        iy = max(0, min(sy + sh, by1) - max(sy, by0))
        if ix * iy >= 0.5 * sw * sh:
            fills.append(area / float(sw * sh))
    return len(fills) >= 2 and min(fills) >= fill_thr


def drop_plain_bars(image_path, dets):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return [d for d in dets if not (d["label"] == "equipment" and is_plain_bars(gray, d["box"]))]