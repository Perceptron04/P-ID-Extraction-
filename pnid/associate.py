"""Link every text tag to the symbol it belongs to (one tag <-> one symbol, nearest first)."""
import math

SYMBOLS = ("valve", "instrument", "equipment")


def _gap(a, b):
    gx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    gy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(gx, gy)


def _c(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def associate(dets, max_gap=60.0, max_gap_instrument=8.0):
    """dets: list of dicts with 'label' and 'box' (xyxy). Returns list of (tag_idx, symbol_idx, gap_px).

    Cost = gap between the two boxes + a small penalty when the tag is not lined up with the symbol
    (tags sit directly above/below or beside a symbol). Greedy cheapest-first, each tag and each
    symbol used at most once. Tags with no symbol within max_gap px (pipe sizes, line numbers) stay unlinked;
    instruments use the much tighter max_gap_instrument.
    """
    tags = [i for i, d in enumerate(dets) if d["label"] == "text_tag"]
    syms = [i for i, d in enumerate(dets) if d["label"] in SYMBOLS]
    pairs = []
    for t in tags:
        for s in syms:
            g = _gap(dets[t]["box"], dets[s]["box"])
            # instrument bubbles/boxes carry their text inside, so only a tag touching them counts
            if g > (max_gap_instrument if dets[s]["label"] == "instrument" else max_gap):
                continue
            (tx, ty), (sx, sy) = _c(dets[t]["box"]), _c(dets[s]["box"])
            pairs.append((g + 0.25 * min(abs(tx - sx), abs(ty - sy)), t, s, g))
    pairs.sort()
    used_t, used_s, links = set(), set(), []
    for _, t, s, g in pairs:
        if t in used_t or s in used_s:
            continue
        used_t.add(t)
        used_s.add(s)
        links.append((t, s, g))
    return links