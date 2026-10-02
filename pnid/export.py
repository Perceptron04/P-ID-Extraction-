"""Digitised output: detection log, tag<->symbol mapping table, summary -> CSV, XLSX and SQLite."""
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
from .associate import associate, SYMBOLS
from .coco_io import xywh_to_xyxy

Image.MAX_IMAGE_PIXELS = None

DET_COLS = ["image", "entity_id", "class", "confidence", "x", "y", "w", "h", "x_center", "y_center",
            "x1", "y1", "x2", "y2", "norm_xc", "norm_yc", "norm_w", "norm_h",
            "tag_text", "linked_entity_id", "link_gap_px"]
MAP_COLS = ["image", "symbol_id", "symbol_class", "tag_id", "tag_text", "link_gap_px",
            "symbol_x_center", "symbol_y_center", "tag_x_center", "tag_y_center"]
SUM_COLS = ["image", "valve", "instrument", "equipment", "text_tag", "total", "tags_linked", "symbols_with_tag"]


def build_tables(predictions, images_dir, max_gap=60.0, tag_text=None, max_gap_instrument=8.0):
    """predictions: list of {file_name,label,score,bbox xywh}. tag_text: optional {entity_id: text} (OCR)."""
    tag_text = tag_text or {}
    by_img = defaultdict(list)
    for p in predictions:
        by_img[p["file_name"]].append(p)
    det_rows, map_rows, sum_rows, sizes = [], [], [], {}
    for f in sorted(by_img):
        with Image.open(Path(images_dir) / f) as im:
            W, H = im.size
        sizes[f] = (W, H)
        dets = sorted(by_img[f], key=lambda p: (round(p["bbox"][1] / 50), p["bbox"][0]))  # top-to-bottom, left-to-right
        stem = Path(f).stem
        for n, d in enumerate(dets, 1):
            d["entity_id"] = f"{stem}_E{n:04d}"
            d["box"] = xywh_to_xyxy(d["bbox"])
        links = associate(dets, max_gap, max_gap_instrument)
        tag_to_sym = {t: (s, g) for t, s, g in links}
        sym_to_tag = {s: (t, g) for t, s, g in links}
        for i, d in enumerate(dets):
            x, y, w, h = d["bbox"]
            lk = tag_to_sym.get(i) or sym_to_tag.get(i)
            linked_id = dets[lk[0]]["entity_id"] if lk else ""
            det_rows.append({
                "image": f, "entity_id": d["entity_id"], "class": d["label"], "confidence": round(d["score"], 3),
                "x": round(x, 1), "y": round(y, 1), "w": round(w, 1), "h": round(h, 1),
                "x_center": round(x + w / 2, 1), "y_center": round(y + h / 2, 1),
                "x1": round(x, 1), "y1": round(y, 1), "x2": round(x + w, 1), "y2": round(y + h, 1),
                "norm_xc": round((x + w / 2) / W, 6), "norm_yc": round((y + h / 2) / H, 6),
                "norm_w": round(w / W, 6), "norm_h": round(h / H, 6),
                "tag_text": tag_text.get(d["entity_id"], "") if d["label"] == "text_tag" else "",
                "linked_entity_id": linked_id, "link_gap_px": round(lk[1], 1) if lk else "",
            })
        for t, s, g in links:
            ts, sy = dets[t], dets[s]
            map_rows.append({
                "image": f, "symbol_id": sy["entity_id"], "symbol_class": sy["label"], "tag_id": ts["entity_id"],
                "tag_text": tag_text.get(ts["entity_id"], ""), "link_gap_px": round(g, 1),
                "symbol_x_center": round(sum(sy["box"][0::2]) / 2, 1), "symbol_y_center": round(sum(sy["box"][1::2]) / 2, 1),
                "tag_x_center": round(sum(ts["box"][0::2]) / 2, 1), "tag_y_center": round(sum(ts["box"][1::2]) / 2, 1),
            })
        cnt = Counter(d["label"] for d in dets)
        sum_rows.append({"image": f, "valve": cnt["valve"], "instrument": cnt["instrument"],
                         "equipment": cnt["equipment"], "text_tag": cnt["text_tag"], "total": len(dets),
                         "tags_linked": len(links), "symbols_with_tag": len(links)})
    return det_rows, map_rows, sum_rows, sizes, by_img


def _write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:  # utf-8-sig: opens cleanly in Excel
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _write_xlsx(path, sheets):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    wb.remove(wb.active)
    for name, cols, rows in sheets:
        ws = wb.create_sheet(name)
        ws.append(cols)
        for r in rows:
            ws.append([r.get(c, "") for c in cols])
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="305496")
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for i, col in enumerate(cols, 1):
            ws.column_dimensions[get_column_letter(i)].width = max(10, min(24, len(col) + 4))
    wb.save(path)


def _write_sqlite(path, det_rows, map_rows, sum_rows, sizes):
    path = Path(path)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE images (image TEXT PRIMARY KEY, width INTEGER, height INTEGER);
    CREATE TABLE detections (
        entity_id TEXT PRIMARY KEY, image TEXT REFERENCES images(image), class TEXT, confidence REAL,
        x REAL, y REAL, w REAL, h REAL, x_center REAL, y_center REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL,
        norm_xc REAL, norm_yc REAL, norm_w REAL, norm_h REAL, tag_text TEXT, linked_entity_id TEXT, link_gap_px REAL);
    CREATE TABLE tag_symbol_mapping (
        image TEXT, symbol_id TEXT REFERENCES detections(entity_id), symbol_class TEXT,
        tag_id TEXT REFERENCES detections(entity_id), tag_text TEXT, link_gap_px REAL,
        symbol_x_center REAL, symbol_y_center REAL, tag_x_center REAL, tag_y_center REAL);
    CREATE INDEX idx_det_class ON detections(image, class);
    """)
    con.executemany("INSERT INTO images VALUES (?,?,?)", [(f, w, h) for f, (w, h) in sizes.items()])
    db_cols = ["entity_id", "image"] + DET_COLS[2:]
    con.executemany(f"INSERT INTO detections ({','.join(db_cols)}) VALUES ({','.join('?' * len(db_cols))})",
                    [tuple(None if r[c] == "" else r[c] for c in db_cols) for r in det_rows])
    con.executemany("INSERT INTO tag_symbol_mapping VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [tuple(r[c] for c in MAP_COLS) for r in map_rows])
    con.commit()
    con.close()


COLORS = {"valve": (0, 170, 0), "instrument": (0, 90, 255), "equipment": (255, 140, 0), "text_tag": (230, 0, 150)}


def draw_links(image_path, dets, links, out_path, scale=0.5):
    """Overlay: class-coloured boxes, black line between each tag and its symbol."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((int(img.width * scale), int(img.height * scale)))
    d = ImageDraw.Draw(img)
    for x in dets:
        d.rectangle([v * scale for v in x["box"]], outline=COLORS[x["label"]], width=2)
    for t, s, _ in links:
        (a, b), (c, e) = [((q["box"][0] + q["box"][2]) / 2 * scale, (q["box"][1] + q["box"][3]) / 2 * scale)
                          for q in (dets[t], dets[s])]
        d.line([a, b, c, e], fill=(0, 0, 0), width=2)
    img.save(out_path, quality=88)


def export_all(predictions, images_dir, out_dir, max_gap=60.0, tag_text=None, draw=True, max_gap_instrument=8.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    det_rows, map_rows, sum_rows, sizes, by_img = build_tables(predictions, images_dir, max_gap, tag_text, max_gap_instrument)
    _write_csv(out / "detections.csv", DET_COLS, det_rows)
    _write_csv(out / "tag_symbol_mapping.csv", MAP_COLS, map_rows)
    _write_csv(out / "summary.csv", SUM_COLS, sum_rows)
    _write_xlsx(out / "pnid_extraction.xlsx", [("detections", DET_COLS, det_rows),
                                                ("tag_symbol_mapping", MAP_COLS, map_rows),
                                                ("summary", SUM_COLS, sum_rows)])
    _write_sqlite(out / "pnid_extraction.db", det_rows, map_rows, sum_rows, sizes)
    if draw:
        (out / "link_overlays").mkdir(exist_ok=True)
        for f, dets in by_img.items():
            ds = sorted(dets, key=lambda p: (round(p["bbox"][1] / 50), p["bbox"][0]))
            for d in ds:
                d["box"] = xywh_to_xyxy(d["bbox"])
            draw_links(Path(images_dir) / f, ds, associate(ds, max_gap, max_gap_instrument), out / "link_overlays" / f"{Path(f).stem}.jpg")
    return det_rows, map_rows, sum_rows