

"""End-to-end run: tile -> VLM -> merge -> evaluate -> overlay.

Examples
  python run_pipeline.py --mode zero --model <gemini-model-id> --images 1.jpg
  python run_pipeline.py --mode few  --model <gemini-model-id> --images 1.jpg 2.jpg 3.jpg
  python run_pipeline.py --mock --images 1.jpg 2.jpg 3.jpg          # no API, tests the pipeline
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import hashlib
import json
import os
from pathlib import Path
from PIL import Image
from pnid.coco_io import load_coco, xyxy_to_xywh
from pnid.detect import GeminiDetector, MockDetector, detect_image
from pnid.evaluate import evaluate, format_report
from pnid.examples import build_examples
from pnid.merge import merge
from pnid.overlay import draw_overlay
from pnid.prompts import SYSTEM, TASK
from pnid.refine import refine_all, drop_plain_bars

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="data/images")
    ap.add_argument("--ann", default="data/annotations/instances.json")
    ap.add_argument("--images", nargs="+", default=["1.jpg", "2.jpg", "3.jpg"], help="images to run on / evaluate")
    ap.add_argument("--mode", choices=["zero", "one", "few"], default="zero")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", ""), help="Gemini model id from AI Studio")
    ap.add_argument("--example-source", default="0.jpg", help="annotated image used for one/few-shot examples")
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--xmax", type=int, default=5650, help="ignore everything right of this x (notes/title block)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between API calls")
    ap.add_argument("--retry-empty", action="store_true", help="re-ask tiles whose cached answer had 0 boxes")
    ap.add_argument("--contain", type=float, default=0.75, help="drop symbol boxes mostly inside a bigger symbol box (0 = off)")
    ap.add_argument("--keep-plain-bars", action="store_true", help="do NOT drop equipment boxes that are only plain parallel bars")
    ap.add_argument("--refine", action="store_true", help="snap text_tag boxes to the real ink (re-uses cached API answers)")
    ap.add_argument("--mock", action="store_true", help="fake detector built from ground truth (pipeline test)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ph = hashlib.md5((SYSTEM + TASK + f"{a.tile}{a.overlap}{a.xmax}").encode()).hexdigest()[:6]
    name = a.out or f"runs/{'mock' if a.mock else a.mode}_{a.model or 'na'}_{ph}"
    out = Path(name)
    suffix = "_refined" if a.refine else ""
    (out / f"overlays{suffix}").mkdir(parents=True, exist_ok=True)
    _, imgs, gt = load_coco(a.ann)

    if a.mock:
        detector = None
    else:
        if not a.model:
            raise SystemExit("Pass --model <id> (copy the exact id from AI Studio) or set GEMINI_MODEL.")
        if a.mode != "zero":
            build_examples(a.images_dir, a.ann, "examples", source=a.example_source, tile=a.tile, xmax=a.xmax)
        if a.example_source in a.images:
            print(f"WARNING: {a.example_source} is used for examples AND evaluation -> leakage")
        detector = GeminiDetector(a.model, a.mode, "examples")

    predictions = []
    for f in a.images:
        print(f"[{f}]")
        det = MockDetector(gt[f]) if a.mock else detector
        raw = detect_image(Path(a.images_dir) / f, f, det, out, a.tile, a.overlap, a.xmax, delay=a.delay, retry_empty=a.retry_empty)
        w, h = imgs[f]["w"], imgs[f]["h"]
        merged = merge(raw, w, h, a.xmax, contain=a.contain)
        if not a.keep_plain_bars:
            before = len(merged)
            merged = drop_plain_bars(Path(a.images_dir) / f, merged)
            print(f"  plain-bar boxes dropped: {before - len(merged)}")
        if a.refine:
            merged = refine_all(Path(a.images_dir) / f, merged)
        print(f"  raw {len(raw)} -> merged {len(merged)}")
        preds = [{"file_name": f, "label": d["label"], "score": d["score"], "bbox": [round(v, 1) for v in xyxy_to_xywh(d["box"])]}
                 for d in merged]
        predictions += preds
        draw_overlay(Path(a.images_dir) / f, gt[f], preds, out / f"overlays{suffix}" / f"{Path(f).stem}.jpg")

    json.dump(predictions, open(out / f"predictions{suffix}.json", "w"))
    metrics = evaluate(gt, predictions, a.images)
    json.dump(metrics, open(out / f"metrics{suffix}.json", "w"), indent=1)
    print("\n" + format_report(metrics))
    print(f"\nSaved to {out}/  (predictions{suffix}.json, metrics{suffix}.json, overlays{suffix}/)")


if __name__ == "__main__":
    main()