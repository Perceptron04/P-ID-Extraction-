"""Run a VLM (Gemini) on every tile. Results are cached per tile so re-runs cost no API calls."""
import io
import json
import os
import random
import re
import time
from pathlib import Path
from PIL import Image
from .coco_io import load_coco, xywh_to_xyxy
from .prompts import SYSTEM, TASK, CLASSES
from .tiling import make_tiles, crop_tile

Image.MAX_IMAGE_PIXELS = None


def _png(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def parse_response(text, tile):
    """VLM text -> list of dicts with full-image xyxy pixel boxes."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        # truncated / slightly broken JSON: keep every complete {...} object we can find
        items = []
        for m in re.finditer(r"\{[^{}]*\}", text):
            try:
                items.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                pass
    if isinstance(items, dict):
        items = items.get("detections") or items.get("boxes") or []
    out = []
    for it in items:
        try:
            label = str(it["label"]).strip().lower()
            ymin, xmin, ymax, xmax = [float(v) for v in it["box_2d"]]
        except (KeyError, ValueError, TypeError):
            continue
        if label not in CLASSES or ymax <= ymin or xmax <= xmin:
            continue
        out.append({
            "label": label,
            "score": float(it.get("confidence", 0.5)),
            "box": [tile.x0 + xmin / 1000 * tile.w, tile.y0 + ymin / 1000 * tile.h,
                    tile.x0 + xmax / 1000 * tile.w, tile.y0 + ymax / 1000 * tile.h],
            "tile": tile.idx,
        })
    return out


class GeminiDetector:
    def __init__(self, model, mode, examples_dir):
        from google import genai
        from google.genai import types
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("Set the GEMINI_API_KEY environment variable first.")
        self.types, self.model, self.mode = types, model, mode
        self.client = genai.Client(api_key=key)
        self.ex = Path(examples_dir)
        self.shots = self._build_shots() if mode in ("one", "few") else []

    def _img(self, img):
        return self.types.Part.from_bytes(data=_png(img), mime_type="image/png")

    def _build_shots(self):
        T = self.types
        shots = []
        if self.mode == "few":
            parts = [T.Part.from_text(text="Reference crops of each class (label given before each image):")]
            for c in json.load(open(self.ex / "meta.json"))["crops"]:
                parts.append(T.Part.from_text(text=f"Class: {c['label']}"))
                parts.append(self._img(Image.open(c["path"])))
            shots.append(T.Content(role="user", parts=parts))
            shots.append(T.Content(role="model", parts=[T.Part.from_text(text="Understood. I will use these references.")]))
        # one_shot_tile is used in both 'one' and 'few'
        tile = Image.open(self.ex / "one_shot_tile.png")
        shots.append(T.Content(role="user", parts=[T.Part.from_text(text="Example tile. " + TASK), self._img(tile)]))
        shots.append(T.Content(role="model", parts=[T.Part.from_text(text=(self.ex / "one_shot_answer.json").read_text())]))
        return shots

    def _config(self, thinking=True):
        T = self.types
        kw = dict(system_instruction=SYSTEM, temperature=0.0, response_mime_type="application/json",
                  max_output_tokens=32768,
                  automatic_function_calling=T.AutomaticFunctionCallingConfig(disable=True))
        if thinking:
            try:
                kw["thinking_config"] = T.ThinkingConfig(thinking_level="low")
            except Exception:
                pass
        return T.GenerateContentConfig(**kw)

    def __call__(self, tile_img, retries=6):
        """Returns response text. self.last tells whether the answer is usable (and why not)."""
        T = self.types
        contents = self.shots + [T.Content(role="user", parts=[T.Part.from_text(text=TASK), self._img(tile_img)])]
        cfg = self._config()
        self.last = {"ok": False, "why": "no response"}
        for attempt in range(retries):
            try:
                r = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
                cand = r.candidates[0] if r.candidates else None
                finish = str(getattr(cand, "finish_reason", "?")).split(".")[-1]
                text = (r.text or "").strip()
                self.last = {"ok": bool(text) and finish in ("STOP", "?"), "finish": finish,
                             "why": f"finish_reason={finish}, chars={len(text)}"}
                return text or "[]"
            except TypeError as e:  # SDK/model does not accept thinking_level -> retry without it
                cfg = self._config(thinking=False)
                self.last = {"ok": False, "why": f"config error: {e}"}
            except Exception as e:  # 429 / 503 etc.
                wait = min(60, 5 * 2 ** attempt)
                print(f"   API error ({type(e).__name__}: {str(e)[:160]}) - retry in {wait}s")
                self.last = {"ok": False, "why": f"{type(e).__name__}"}
                time.sleep(wait)
        return "[]"


class MockDetector:
    """No API: returns the ground-truth boxes of the tile with random jitter. For testing the pipeline."""

    def __init__(self, gt_items, jitter=6, drop=0.1, seed=1):
        self.gt, self.jitter, self.drop, self.rng = gt_items, jitter, drop, random.Random(seed)

    def __call__(self, tile_img, tile):
        out = []
        for a in self.gt:
            x0, y0, x1, y1 = xywh_to_xyxy(a["bbox"])
            if x1 < tile.x0 or x0 > tile.x1 or y1 < tile.y0 or y0 > tile.y1 or self.rng.random() < self.drop:
                continue
            j = lambda: self.rng.uniform(-self.jitter, self.jitter)
            bx = [max(tile.x0, x0 + j()), max(tile.y0, y0 + j()), min(tile.x1, x1 + j()), min(tile.y1, y1 + j())]
            out.append({"label": a["label"],
                        "box_2d": [round((bx[1] - tile.y0) / tile.h * 1000), round((bx[0] - tile.x0) / tile.w * 1000),
                                   round((bx[3] - tile.y0) / tile.h * 1000), round((bx[2] - tile.x0) / tile.w * 1000)],
                        "confidence": round(self.rng.uniform(0.5, 1), 2)})
        return json.dumps(out)


def detect_image(image_path, fname, detector, out_dir, tile=1280, overlap=0.2, xmax=5650, gt_items=None,
                 delay=0.0, retry_empty=False):
    out_dir = Path(out_dir)
    cache = out_dir / "cache" / Path(fname).stem
    cache.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    tiles = make_tiles(*img.size, tile=tile, overlap=overlap, xmax=xmax)
    mock = isinstance(detector, MockDetector)
    raw = []
    for t in tiles:
        cf = cache / f"tile_{t.idx:03d}.json"
        text = cf.read_text() if cf.exists() else None
        if text is not None and retry_empty and not mock and not parse_response(text, t):
            text = None  # earlier answer was empty -> ask again
        note = ""
        if text is None:
            crop = crop_tile(img, t)
            text = detector(crop, t) if mock else detector(crop)
            last = getattr(detector, "last", {"ok": True})
            if last.get("ok", True):
                cf.write_text(text)  # only good answers are cached
            else:
                note = f"  [NOT cached: {last.get('why')}]"
            if delay:
                time.sleep(delay)
        dets = parse_response(text, t)
        if not dets and not note and not mock:
            note = "  [empty answer]"
        print(f"  {fname} tile {t.idx + 1}/{len(tiles)}: {len(dets)} boxes{note}")
        for d in dets:
            d["tile_box"] = [t.x0, t.y0, t.x1, t.y1]
        raw.extend(dets)
    return raw