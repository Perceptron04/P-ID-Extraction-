"""Prompts. Class descriptions follow the annotation convention used for the ground truth."""

CLASSES = ["valve", "instrument", "equipment", "text_tag"]

CLASS_GUIDE = """Classes (use exactly these names):
- valve: valves drawn on pipes - bow-tie / hourglass-shaped gate & globe valves (open or filled black), valves with a small actuator on top, 3-way valves, X-shaped valves.
- instrument: (a) round bubbles with letters/numbers inside (e.g. SDL 101, DDL 824, GRI 323, ZLC 764, RO-10) and (b) square boxes with text inside (e.g. "586 LG-10 972", ZLO/ZLC squares, STA boxes).
- equipment: all other small process symbols, in particular:
    * a connector made of 2-3 short parallel bars on a line followed by a small open circle and/or a filled black dot (long and thin) - box the whole thing, bars + circle + dot;
    * a circle with a cross inside (ERV-xx);
    * a zig-zag heater/coil inside a rectangle (CS-xx);
    * a long rounded capsule with a dashed centre line (INS ...), horizontal or vertical;
    * reducers (trapezoids), triangle-with-bar check symbols, a small round/D-shaped symbol with bars (DV-xx), a filled dot with side brackets (RV-xx), a bow-tie with curved sides (QR-xx).
- text_tag: every text label next to a symbol or line - tags like MN-30863, CD-85991, INS (20C), pipe sizes like 12" or 6"-NL-4383. Text may be rotated 90 degrees. One box per label, tight around the characters.
Do NOT box: plain pipe lines; a plain pair of parallel bars on a line WITHOUT a circle/dot; line-end markers; the drawing border; the notes/title-block tables."""

RULES = """Rules:
- Boxes must be TIGHT around the symbol/text only (no extra whitespace, no attached pipe line).
- Symbols cut by the image border: skip them (they appear whole in a neighbouring tile).
- Every symbol and every text tag you can see should get its own box.
- Return ONLY a JSON list. Each item: {"label": <class>, "box_2d": [ymin, xmin, ymax, xmax], "confidence": <0..1>}
- box_2d coordinates are integers normalised to 0-1000 relative to the image you are looking at (ymin, xmin, ymax, xmax)."""

SYSTEM = (
    "You are an expert in reading Piping and Instrumentation Diagrams (P&ID). "
    "You detect symbols and text tags and return precise bounding boxes as JSON.\n\n"
    + CLASS_GUIDE + "\n\n" + RULES
)

TASK = "Detect all valves, instruments, equipment and text tags in this P&ID tile. Return the JSON list."