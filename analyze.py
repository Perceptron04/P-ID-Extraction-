"""Error analysis for one run: what is missed, what is confused, what is a false alarm.

  python analyze.py runs/zero_gemini-3.5-flash-lite --images 1.jpg
  python analyze.py runs/zero_gemini-3.5-flash-lite --images 1.jpg --list-missed equipment
"""
import argparse
import json
from collections import Counter, defaultdict
from pnid.coco_io import load_coco, xywh_to_xyxy, iou_xyxy
from pnid.prompts import CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ann", default="data/annotations/instances.json")
    ap.add_argument("--images", nargs="+", default=["1.jpg"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--preds", default="predictions.json", help="predictions file inside the run folder")
    ap.add_argument("--list-false", default=None, help="class name: print position of false alarms")
    ap.add_argument("--list-missed", default=None, help="class name: print position of missed GT boxes")
    a = ap.parse_args()

    _, _, gt = load_coco(a.ann)
    preds = json.load(open(f"{a.run}/{a.preds}"))

    conf = {c: Counter() for c in CLASSES}       # GT class -> predicted class / MISSED
    extra = Counter()                            # false positives by predicted class
    size_tot, size_hit = Counter(), Counter()
    iou_by_cls = defaultdict(list)
    missed = []
    false_alarms = []

    for f in a.images:
        G = [{"label": g["label"], "box": xywh_to_xyxy(g["bbox"])} for g in gt[f]]
        P = [{"label": p["label"], "box": xywh_to_xyxy(p["bbox"])} for p in preds if p["file_name"] == f]
        pairs = sorted(((iou_xyxy(g["box"], p["box"]), gi, pi) for gi, g in enumerate(G) for pi, p in enumerate(P)),
                       reverse=True)
        ug, up, match = set(), set(), {}
        for v, gi, pi in pairs:
            if v < a.iou:
                break
            if gi in ug or pi in up:
                continue
            ug.add(gi); up.add(pi); match[gi] = (pi, v)
        for gi, g in enumerate(G):
            side = max(g["box"][2] - g["box"][0], g["box"][3] - g["box"][1])
            b = "small (<50px)" if side < 50 else "medium (50-120px)" if side < 120 else "large (>120px)"
            size_tot[b] += 1
            if gi in match:
                pi, v = match[gi]
                conf[g["label"]][P[pi]["label"]] += 1
                iou_by_cls[g["label"]].append(v)
                size_hit[b] += 1
            else:
                conf[g["label"]]["MISSED"] += 1
                missed.append((f, g))
        for pi, p in enumerate(P):
            if pi not in up:
                extra[p["label"]] += 1
                false_alarms.append((f, p))

    cols = CLASSES + ["MISSED"]
    print("\nWhat happened to each ground-truth box (rows = your label, columns = model label)")
    print(f"{'':<12}" + "".join(f"{c:>12}" for c in cols) + f"{'recall':>10}")
    for c in CLASSES:
        tot = sum(conf[c].values())
        hit = tot - conf[c]["MISSED"]
        print(f"{c:<12}" + "".join(f"{conf[c][k]:>12}" for k in cols) + f"{hit / tot if tot else 0:>10.2f}")

    print("\nExtra boxes the model drew where you have none (false alarms), by model label:")
    print("  " + ", ".join(f"{c}: {extra[c]}" for c in CLASSES))

    print("\nRecall by symbol size:")
    for b in ["small (<50px)", "medium (50-120px)", "large (>120px)"]:
        if size_tot[b]:
            print(f"  {b:<20} {size_hit[b]}/{size_tot[b]}  = {size_hit[b] / size_tot[b]:.2f}")

    print("\nMean IoU of matched boxes by class:")
    for c in CLASSES:
        v = iou_by_cls[c]
        print(f"  {c:<12} {sum(v) / len(v):.3f}" if v else f"  {c:<12} -")

    if a.list_missed:
        print(f"\nMissed '{a.list_missed}' boxes (x, y = top-left in full image; open the overlay and look there):")
        for f, g in missed:
            if g["label"] == a.list_missed:
                b = g["box"]
                print(f"  {f}: x={b[0]:.0f} y={b[1]:.0f} w={b[2] - b[0]:.0f} h={b[3] - b[1]:.0f}")

    if a.list_false:
        print(f"\nFalse-alarm '{a.list_false}' boxes (model drew them, you have no box there - check the overlay: "
              "maybe the model is right and an annotation is missing):")
        for f, p in false_alarms:
            if p["label"] == a.list_false:
                b = p["box"]
                print(f"  {f}: x={b[0]:.0f} y={b[1]:.0f} w={b[2] - b[0]:.0f} h={b[3] - b[1]:.0f}")


if __name__ == "__main__":
    main()