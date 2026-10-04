# P&ID Symbol Detection using Vision Language Models

Automated bounding-box detection of **valves, instruments, equipment and text tags** from P&ID engineering diagrams using VMs.


## Project Structure

```
P-ID-Extraction/
├── run_pipeline.py        # Main entry point: tile → VLM → merge → filters → evaluate → overlay
├── analyze.py             # Error analysis: confusion table, false alarms, missed boxes
├── export_results.py      # Export to CSV / XLSX / SQLite with tag-symbol links
├── pnid_prompt_4class.txt 
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── images/            # 0.jpg … 4.jpg  (7168 × 4561 px each)
│   └── annotations/       # instances.json (COCO format, 4 annotated images)
│
├── pnid/
│   ├── __init__.py
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
│
├── examples/              # Auto-generated few-shot crops (created at runtime, git-ignored)
└── runs/                  # All run outputs — cache, predictions, metrics, overlays, exports
                           # (created at runtime, git-ignored)
```

---

## How it Works

```
Raw P&ID image (7168 × 4561 px)
        │
        ▼
  1. TILE          Cut into 30 overlapping 1280×1280 px tiles (20% overlap)
        │           Notes column excluded (x > 5650 px)
        ▼
  2. DETECT        Each tile → Gemini API with system prompt + few-shot crops
        │           Model returns JSON list of {label, box_2d, confidence}
        ▼
  3. CACHE         Each tile answer saved to disk; run folder hashed from prompt
        │           Editing prompt = fresh cache automatically
        ▼
  4. MERGE         Map tile boxes → full image coordinates
        │           Drop border-cut boxes → per-class NMS → cross-class NMS
        │           Containment suppression (fragment boxes removed)
        │           Plain-bar filter (false alarms on unannotated || symbols removed)
        ▼
  5. REFINE        Snap text_tag boxes to actual ink blobs (+3 px margin)
        │           Valve / instrument / equipment boxes not refined
        ▼
  6. EVALUATE      mAP@0.5, mAP@0.5:0.95, precision, recall, F1
        │           Mean IoU, centre deviation (px and relative)
        │           Per-class AP, overlays (green = GT, red = predicted)
        ▼
  7. EXPORT        Tag-to-symbol association (nearest-neighbour, one-to-one)
                   → detections.csv, tag_symbol_mapping.csv, summary.csv
                   → pnid_extraction.xlsx, pnid_extraction.db (SQLite)
                   → link_overlays/ (lines between tags and their symbols)
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Perceptron04/P-ID-Extraction-.git
cd P-ID-Extraction-

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # Linux / Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key
# Create a .env file in the root folder:
GEMINI_API_KEY=your_key_here
# Get a free key at: https://ai.google.dev
```

---

## Run Commands

### 1. Test without any API call (mock run)
```bash
python run_pipeline.py --mock --images 1.jpg 2.jpg 3.jpg
```
Verifies the whole pipeline using ground truth + small noise. Expected mAP ≈ 0.97. Runs in under 30 seconds.

### 2. Zero-shot detection
```bash
python run_pipeline.py --mode zero --model gemini-3.5-flash-lite --images 1.jpg
```

### 3. Few-shot detection (final configuration)
```bash
python run_pipeline.py --mode few --model gemini-3.5-flash-lite --images 1.jpg 2.jpg 3.jpg --retry-empty --refine
```
`--retry-empty` re-asks tiles that returned no boxes.  
`--refine` snaps text-tag boxes to the actual ink.

### 4. Run on an unannotated image
```bash
python run_pipeline.py --mode few --model gemini-3.5-flash-lite --images 4.jpg --retry-empty --refine
```
Produces predictions and an overlay — no metrics (no ground truth for `4.jpg`).

### 5. Error analysis (no API call — uses cache)
```bash
python analyze.py runs/<run-folder> --images 1.jpg 2.jpg 3.jpg --preds predictions_refined.json

# List false alarms for a specific class
python analyze.py runs/<run-folder> --images 1.jpg --list-false equipment

# List missed boxes for a specific class
python analyze.py runs/<run-folder> --images 1.jpg --list-missed valve
```

### 6. Structured export — CSV / XLSX / SQLite (no API call)
```bash
python export_results.py runs/<run-folder> --preds predictions_refined.json
```
Output in `runs/<run-folder>/exports/`:
| File | Content |
|---|---|
| `detections.csv` | Every box: class, confidence, pixel coords, YOLO-normalised coords, linked entity |
| `tag_symbol_mapping.csv` | Which tag is linked to which symbol, with gap in px |
| `summary.csv` | Class counts per image, number of linked pairs |
| `pnid_extraction.xlsx` | All three tables as formatted sheets with filters |
| `pnid_extraction.db` | SQLite: tables `images`, `detections`, `tag_symbol_mapping` |
| `link_overlays/` | Images with lines drawn between each tag and its symbol |

---

## Key Flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `few` | Prompting mode: `zero`, `one`, `few`, `mock` |
| `--model` | — | Gemini model name, e.g. `gemini-3.5-flash-lite` |
| `--images` | — | Image filenames to process (space-separated) |
| `--retry-empty` | off | Re-ask tiles whose cached answer was empty |
| `--refine` | off | Snap text-tag boxes to the real ink (+3 px margin) |
| `--contain` | `0.75` | Drop symbol boxes ≥ this fraction inside a larger box. `--contain 0` to disable |
| `--keep-plain-bars` | off | Do NOT drop equipment boxes that contain only plain parallel bars |
| `--xmax` | `5650` | Right-edge crop (excludes notes column and title block) |
| `--tile` | `1280` | Tile size in pixels |
| `--delay` | `2` | Seconds between API calls (increase if hitting rate limits) |
| `--out` | auto | Custom output folder |

---

## Data

| Image | Role | GT boxes |
|---|---|---|
| `0.jpg` | Few-shot source only — **never evaluated** | 140 |
| `1.jpg` | Evaluation (also used to develop post-processing rules) | 139 |
| `2.jpg` | Evaluation (validation) | 183 |
| `3.jpg` | Evaluation (validation) | 156 |
| `4.jpg` | No annotations — predictions and overlay only | — |

Ground truth: COCO JSON at `data/annotations/instances.json`  
Total annotated boxes across all 4 images: **618** (valve 115, instrument 102, equipment 116, text_tag 285)

**Annotation conventions:**
- Plain parallel bars on a pipe (without a circle or dot) are deliberately **not annotated**; the pipeline filters them out.
- Notes column, title block and table text are excluded from both annotation and detection.
- Each symbol and each text tag get their own tight box; connected pipe is not included.

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

---

## Notes

- **API limits:** The free Gemini tier allows ~500 requests/day for Flash-Lite. The per-tile cache means interrupted runs resume from where they stopped — no calls are wasted.
- **Prompt hash:** The run folder name contains an MD5 hash of the prompt + config. Editing the prompt automatically creates a new run folder and fresh cache.
- **Claude Sonnet 4.6 (AWS Bedrock)** was tested on the same task. It failed to follow the required `0–1000` coordinate format, giving mAP < 0.01. Gemini was kept for all final results.
- **`examples/` and `runs/`** are created at runtime and are excluded from version control via `.gitignore`.
- **Never commit your `.env` file.**

---

## References

- Google Gemini API — https://ai.google.dev
- COCO evaluation format — https://cocodataset.org/#format-results
- OpenCV morphological operations — https://docs.opencv.org
- *Methodology developed through iterative error analysis on the project dataset. No external tutorials or papers directly referenced.*
