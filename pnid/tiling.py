"""Split a big P&ID image into overlapping tiles and keep their offsets."""
import math
from dataclasses import dataclass
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


@dataclass
class Tile:
    idx: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0


def _starts(length, tile, stride):
    """Start positions covering [0, length); last tile is shifted back so every tile is full size."""
    if length <= tile:
        return [0]
    n = math.ceil((length - tile) / stride) + 1
    starts = [i * stride for i in range(n)]
    starts[-1] = length - tile
    return sorted(set(starts))


def make_tiles(img_w, img_h, tile=1280, overlap=0.2, xmax=None, ymax=None):
    """Return list[Tile]. xmax/ymax crop the drawing area (e.g. to skip the notes/title-block column)."""
    xmax = min(xmax or img_w, img_w)
    ymax = min(ymax or img_h, img_h)
    stride = max(1, int(tile * (1 - overlap)))
    tiles = []
    for y in _starts(ymax, tile, stride):
        for x in _starts(xmax, tile, stride):
            tiles.append(Tile(len(tiles), x, y, min(x + tile, xmax), min(y + tile, ymax)))
    return tiles


def crop_tile(img: Image.Image, t: Tile) -> Image.Image:
    return img.crop((t.x0, t.y0, t.x1, t.y1))
