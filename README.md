# P&ID symbol detection (Gemini API, no training)

Setup
  pip install -r requirements.txt
  export GEMINI_API_KEY=your_key        (Windows: setx GEMINI_API_KEY your_key)

Pipeline test without API (uses your ground truth + noise)
  python run_pipeline.py --mock --images 1.jpg 2.jpg 3.jpg

Real runs (copy the exact model id from Google AI Studio; check your daily request limit first)
  python run_pipeline.py --mode zero --model <model-id> --images 1.jpg
  python run_pipeline.py --mode one  --model <model-id> --images 1.jpg 2.jpg 3.jpg
  python run_pipeline.py --mode few  --model <model-id> --images 1.jpg 2.jpg 3.jpg

Notes
- 0.jpg is used to build one-shot/few-shot examples, so evaluate on 1,2,3 only.
- Every tile response is cached in runs/<run>/cache, so re-running costs no API calls.
- If you hit the daily limit: --tile 2048 (fewer calls) or use a Flash-Lite model, add --delay 5.
- Outputs: predictions.json, metrics.json, overlays/ (green = your boxes, red = model boxes).
- Edit pnid/prompts.py to tune class descriptions.
