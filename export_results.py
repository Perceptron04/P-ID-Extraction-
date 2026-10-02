"""Turn a run's predictions into spreadsheet / database deliverables (no API calls).

  python export_results.py runs/<run-folder> --preds predictions_refined.json
Writes into <run-folder>/exports/: detections.csv, tag_symbol_mapping.csv, summary.csv,
pnid_extraction.xlsx, pnid_extraction.db (SQLite) and link_overlays/*.jpg
"""
import argparse
import json
from pathlib import Path
from pnid.export import export_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run folder, e.g. runs/few_gemini-3.5-flash-lite_cc5c93")
    ap.add_argument("--preds", default="predictions_refined.json")
    ap.add_argument("--images-dir", default="data/images")
    ap.add_argument("--max-gap", type=float, default=60.0, help="max px between a tag and its symbol")
    ap.add_argument("--max-gap-instrument", type=float, default=8.0, help="instruments only link to a tag that (almost) touches them")
    ap.add_argument("--tag-text", default=None, help="optional JSON {entity_id: text} from OCR")
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    preds = json.load(open(Path(a.run) / a.preds))
    tag_text = json.load(open(a.tag_text)) if a.tag_text else None
    out = a.out or str(Path(a.run) / "exports")
    det, mp, summ = export_all(preds, a.images_dir, out, a.max_gap, tag_text, draw=not a.no_overlays, max_gap_instrument=a.max_gap_instrument)
    print(f"{len(det)} detections, {len(mp)} tag<->symbol links  ->  {out}/")
    for r in summ:
        print(f"  {r['image']}: valve {r['valve']}, instrument {r['instrument']}, equipment {r['equipment']}, "
              f"text_tag {r['text_tag']}  | linked pairs: {r['tags_linked']}")


if __name__ == "__main__":
    main()
    