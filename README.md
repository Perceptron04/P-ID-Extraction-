# P&ID Symbol Detection using Vision Language Models

Automated bounding-box detection of valves, instruments, equipment and text tags from P&ID engineering diagrams using Google Gemini (few-shot prompting, no model training).

---

## Results (final pipeline — 3 annotated drawings)

| Metric | Value |
|---|---|
| mAP@0.5 (class-agnostic) | 0.811 |
| mAP@0.5:0.95 | 0.494 |
| Precision @ IoU 0.5 | 0.936 |
| Recall @ IoU 0.5 | 0.852 |
| Mean IoU (matched boxes) | 0.832 |
| Mean centre deviation | 3.7 px |

Per-class AP@0.5: valve 0.723, instrument 0.788, equipment 0.645, text_tag 0.779

---

## Project Structure

```
CNDE/
├── run_pipeline.py        # Main entry point: tile → VLM → merge → filters → evaluate → overlay
├── analyze.py             # Error analysis: confusion table, false alarms, missed boxes
├── export_results.py      # Export to CSV / XLSX / SQLite with tag-symbol links
├── requirements.txt
├── pnid/
│   ├── tiling.py          # Tile generation and coordinate mapping
│   ├── prompts.py         # System prompt and class definitions
│   ├── examples.py        # Few-shot crop selection from 0.jpg
│   ├── detect.py          # GeminiDetector and MockDetector
│   ├── merge.py           # NMS, containment suppression, plain-bar filter
│   ├── refine.py          # Text-box ink-snapping and plain-bar filter
│   ├── evaluate.py        # mAP, IoU, centre deviation
│   ├── overlay.py         # Green (GT) vs red (predicted) overlay images
│   ├── associate.py       # Tag-to-symbol nearest-neighbour linking
│   ├── export.py          # CSV / XLSX / SQLite writer
│   └── coco_io.py         # COCO JSON loader and IoU helpers
```

---

## Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd CNDE

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # Linux / Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key
# Create a .env file in the root:
# GEMINI_API_KEY=your_key_here
# Get a free key at: https://ai.google.dev
```

---

## Run Commands

### Test without API (mock run)
```bash
python run_pipeline.py --mock --images 1.jpg 2.jpg 3.jpg
```
Uses the ground truth with small noise to verify the whole pipeline. Expected mAP ≈ 0.97. No API calls, runs in under 30 seconds.

### Zero-shot detection
```bash
python run_pipeline.py --mode zero --model gemini-3.5-flash-lite --images 1.jpg
```

### Few-shot detection (final configuration)
```bash
python run_pipeline.py --mode few --model gemini-3.5-flash-lite --images 1.jpg 2.jpg 3.jpg --retry-empty --refine
```
`--retry-empty` re-asks tiles that returned no boxes. `--refine` snaps text-tag boxes to the actual ink.

### Unannotated image (no ground truth needed)
```bash
python run_pipeline.py --mode few --model gemini-3.5-flash-lite --images 4.jpg --retry-empty --refine
```
Produces predictions and an overlay but no metrics (no ground truth for 4.jpg).

### Error analysis (no API call)
```bash
python analyze.py runs/<run-folder> --images 1.jpg 2.jpg 3.jpg --preds predictions_refined.json
python analyze.py runs/<run-folder> --images 1.jpg --list-false equipment
python analyze.py runs/<run-folder> --images 1.jpg --list-missed equipment
```

### Structured export — CSV / XLSX / SQLite (no API call)
```bash
python export_results.py runs/<run-folder> --preds predictions_refined.json
```
Writes into `runs/<run-folder>/exports/`:
- `detections.csv` — every box with class, confidence, pixel and YOLO coordinates
- `tag_symbol_mapping.csv` — which tag is linked to which symbol
- `summary.csv` — class counts per image
- `pnid_extraction.xlsx` — all three sheets with formatting
- `pnid_extraction.db` — SQLite database (tables: images, detections, tag_symbol_mapping)
- `link_overlays/` — images with tag-to-symbol lines drawn

---

## Key Flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `few` | Prompting mode: `zero`, `one`, `few`, `mock` |
| `--model` | — | Gemini model name (e.g. `gemini-3.5-flash-lite`) |
| `--images` | — | Image filenames to process (space-separated) |
| `--retry-empty` | off | Re-ask tiles whose cached answer was empty |
| `--refine` | off | Snap text-tag boxes to the real ink (+3 px margin) |
| `--contain` | `0.75` | Drop symbol boxes ≥ this fraction inside a larger symbol box. `--contain 0` to disable |
| `--keep-plain-bars` | off | Do NOT drop equipment boxes that contain only plain parallel bars |
| `--xmax` | `5650` | Crop right edge of drawing (excludes notes column) |
| `--tile` | `1280` | Tile size in pixels |
| `--delay` | `2` | Seconds between API calls (increase if hitting rate limits) |

---

## How it Works

1. **Tile** — the drawing area (x < 5650 px) is cut into 1280 × 1280 px tiles with 20 % overlap (30 tiles per image).
2. **Detect** — each tile is sent to Gemini with a system prompt describing the 4 classes, optional few-shot crops from `0.jpg`, and one worked example tile. The model returns a JSON list of boxes.
3. **Cache** — each tile's answer is saved to disk. The run folder name contains a hash of the prompt, so editing the prompt starts a fresh cache automatically.
4. **Merge** — boxes near tile borders are dropped, per-class NMS and cross-class NMS remove duplicates, containment suppression removes fragment boxes, and the plain-bar filter removes false alarms on unannotated parallel-bar symbols.
5. **Refine** — text-tag boxes are snapped to the actual ink blobs (symbols are not refined).
6. **Evaluate** — mAP@0.5, mAP@0.5:0.95, precision, recall, F1, mean IoU and centre deviation are reported; overlays show green (ground truth) vs red (predicted).
7. **Export** — tag-to-symbol association and structured output.

---

## Data

- 5 synthetic P&ID images (7168 × 4561 px).
- `0.jpg` is the few-shot source only — never used for evaluation.
- `1.jpg`, `2.jpg`, `3.jpg` are evaluated (478 ground-truth boxes total).
- `4.jpg` has no annotations; predictions and overlay only.
- Ground truth is in COCO JSON format (`data/annotations/instances.json`).

### What is and is not annotated
- Plain parallel bars on a pipe without a circle or dot are **deliberately not annotated** and the pipeline filters them out.
- Notes column, title block and table text are excluded.
- Each symbol and each text tag get their own tight bounding box; connected pipe is excluded.

---

## Dependencies

```
pillow
numpy
google-genai
python-dotenv
opencv-python-headless
openpyxl
```

Install with `pip install -r requirements.txt`.

---

## Notes

- The free Gemini tier has daily request limits (~500 req/day for Flash-Lite). The per-tile cache means interrupted runs resume from where they stopped.
- Claude Sonnet 4.6 (AWS Bedrock) was tested on the same task; it failed to follow the required coordinate format (mAP < 0.01) and was not used in the final results.
- The `examples/` and `runs/` directories are created at runtime and are excluded from version control.
- Never commit your `.env` file.
