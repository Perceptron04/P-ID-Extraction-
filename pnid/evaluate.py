"""IoU / mAP / centre-deviation evaluation against COCO ground truth (no extra dependencies)."""
import numpy as np
from .coco_io import xywh_to_xyxy, iou_xyxy
from .prompts import CLASSES

IOU_THRS = [0.5 + 0.05 * i for i in range(10)]


def _match(gt, preds, thr, class_aware):
    """Greedy matching by score. Returns list of (pred_index, gt_index or None, iou)."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    used = set()
    res = {}
    for i in order:
        p = preds[i]
        best, best_iou = None, thr
        for j, g in enumerate(gt):
            if j in used or (class_aware and g["label"] != p["label"]):
                continue
            v = iou_xyxy(g["box"], p["box"])
            if v >= best_iou:
                best, best_iou = j, v
        if best is not None:
            used.add(best)
        res[i] = (best, best_iou if best is not None else 0.0)
    return res


def average_precision(gt_by_img, pred_by_img, thr, class_aware, label=None):
    n_gt = 0
    recs = []  # (score, is_tp)
    for f, gt in gt_by_img.items():
        g = [x for x in gt if label is None or x["label"] == label]
        p = [x for x in pred_by_img.get(f, []) if label is None or x["label"] == label]
        n_gt += len(g)
        m = _match(g, p, thr, class_aware)
        recs += [(p[i]["score"], m[i][0] is not None) for i in m]
    if n_gt == 0:
        return float("nan")
    recs.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in recs]) if recs else np.array([])
    fp = np.cumsum([not r[1] for r in recs]) if recs else np.array([])
    if len(tp) == 0:
        return 0.0
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(rec, r, side="left")
        ap += prec[idx] if idx < len(prec) else 0.0
    return ap / 101


def evaluate(gt_items, pred_items, files):
    """gt_items: {file: [{label,bbox xywh}]}; pred_items: list of {file_name,label,score,bbox xywh}."""
    gt = {f: [{"label": a["label"], "box": xywh_to_xyxy(a["bbox"])} for a in gt_items.get(f, [])] for f in files}
    pr = {f: [] for f in files}
    for p in pred_items:
        if p["file_name"] in pr:
            pr[p["file_name"]].append({"label": p["label"], "score": p["score"], "box": xywh_to_xyxy(p["bbox"])})

    n_gt = sum(len(v) for v in gt.values())
    n_pred = sum(len(v) for v in pr.values())
    out = {"images": list(files), "n_gt": n_gt, "n_pred": n_pred}

    # class-agnostic localisation quality (what the brief prioritises)
    aps = [average_precision(gt, pr, t, class_aware=False) for t in IOU_THRS]
    out["mAP@0.5 (class-agnostic)"] = round(aps[0], 4)
    out["mAP@0.5:0.95 (class-agnostic)"] = round(float(np.mean(aps)), 4)
    out["mAP@0.5 (class-aware)"] = round(average_precision(gt, pr, 0.5, class_aware=True), 4)

    ious, devs, rel_devs, tp = [], [], [], 0
    for f in files:
        m = _match(gt[f], pr[f], 0.5, class_aware=False)
        for i, (j, v) in m.items():
            if j is None:
                continue
            tp += 1
            ious.append(v)
            g, p = gt[f][j]["box"], pr[f][i]["box"]
            gc = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
            pc = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
            d = float(np.hypot(gc[0] - pc[0], gc[1] - pc[1]))
            diag = float(np.hypot(g[2] - g[0], g[3] - g[1]))
            devs.append(d)
            rel_devs.append(d / diag if diag else 0.0)
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gt if n_gt else 0.0
    out.update({
        "precision@0.5": round(prec, 4),
        "recall@0.5": round(rec, 4),
        "f1@0.5": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "mean IoU (matched)": round(float(np.mean(ious)), 4) if ious else 0.0,
        "mean centre deviation px": round(float(np.mean(devs)), 2) if devs else None,
        "mean centre deviation / box diagonal": round(float(np.mean(rel_devs)), 4) if rel_devs else None,
    })
    out["per_class"] = {}
    for c in CLASSES:
        ngc = sum(1 for f in files for g in gt[f] if g["label"] == c)
        out["per_class"][c] = {
            "n_gt": ngc,
            "AP@0.5 (class-aware)": round(average_precision(gt, pr, 0.5, True, label=c), 4),
        }
    return out


def format_report(m):
    lines = ["Images: " + ", ".join(m["images"]), f"GT boxes: {m['n_gt']}   Predicted boxes: {m['n_pred']}", ""]
    for k, v in m.items():
        if k in ("images", "n_gt", "n_pred", "per_class"):
            continue
        lines.append(f"{k:<40} {v}")
    lines.append("")
    for c, v in m["per_class"].items():
        lines.append(f"{c:<12} GT={v['n_gt']:<4} AP@0.5={v['AP@0.5 (class-aware)']}")
    return "\n".join(lines)
