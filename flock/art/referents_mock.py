"""Candidate extra referents for sign-off (2026-10-23): hut, fence, pond, crow, mushroom, drawn
next to the four current referents. Nothing in training imports this.

    python -m flock.art.referents_mock <out_dir>
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from flock import sprites_iso as S
from flock.palette import CHAR_TO_INDEX as C, TRANSPARENT, to_rgb


def _hut() -> np.ndarray:
    """A small shed: red pitched roof over plank walls with a dark door."""
    W, roof_h, wall_h = 22, 10, 11
    s = np.full((roof_h + wall_h, W), TRANSPARENT, np.uint8)
    for y in range(roof_h):  # roof: a triangle, lit left
        half = int(round((y + 1) * (W / 2) / roof_h))
        for x in range(W // 2 - half, W // 2 + half):
            s[y, x] = C["R"] if x < W // 2 else C["P"] if (x + y) % 5 == 0 else C["R"]
        s[y, W // 2 - half : W // 2 - half + 1] = C["P"]
    s[roof_h - 1, :] = C["D"]
    for y in range(roof_h, roof_h + wall_h):  # walls
        for x in range(2, W - 2):
            s[y, x] = C["t"] if (x // 3) % 2 else C["T"]
    s[roof_h + 3 : roof_h + wall_h, W // 2 - 2 : W // 2 + 2] = C["D"]  # door
    return S._outline(s)


def _fence() -> np.ndarray:
    """A fence section: two posts and two rails, long and low."""
    W, H = 24, 14
    s = np.full((H, W), TRANSPARENT, np.uint8)
    for x0 in (1, W - 4):
        s[0:H, x0 : x0 + 3] = C["T"]
        s[0:H, x0] = C["t"]
    for y0 in (3, 8):
        s[y0 : y0 + 2, 1:W] = C["t"]
        s[y0 + 1, 1:W] = C["T"]
    return S._outline(s)


def _pond() -> np.ndarray:
    """A flat pool: an earth rim round blue water with light ripples."""
    W, H = 24, 9
    s = np.full((H, W), TRANSPARENT, np.uint8)
    for y in range(H):
        for x in range(W):
            d = ((x + 0.5 - W / 2) / (W / 2)) ** 2 + ((y + 0.5 - H / 2) / (H / 2)) ** 2
            if d <= 1.0:
                s[y, x] = C["t"] if d > 0.62 else C["B"]
    s[3, 8:12] = C["b"]
    s[5, 13:17] = C["b"]
    return S._outline(s)


def _crow_r() -> np.ndarray:
    """A crow, side view facing right: dark body, folded wing, yellow beak."""
    return S._parse(
        [
            "........sss...",
            ".......sssDY..",
            ".......sssss..",
            "..ssssssssss..",
            "sssDDDsssss...",
            ".ssssssssss...",
            "...sssssss....",
            ".....Y..Y.....",
            ".....Y..Y.....",
        ]
    )


def _mushroom() -> np.ndarray:
    """A toadstool: red domed cap with white spots on a pale stem."""
    W, cap_h, stem_h = 16, 8, 7
    s = np.full((cap_h + stem_h, W), TRANSPARENT, np.uint8)
    for y in range(cap_h):
        for x in range(W):
            d = ((x + 0.5 - W / 2) / (W / 2)) ** 2 + ((y + 0.5 - cap_h) / cap_h) ** 2
            if d <= 1.0:
                s[y, x] = C["R"]
    for y, x in [(2, 6), (3, 10), (5, 3), (5, 12), (4, 7)]:
        s[y, x] = C["W"]
    s[cap_h : cap_h + stem_h, W // 2 - 2 : W // 2 + 2] = C["W"]
    s[cap_h : cap_h + stem_h, W // 2 + 1] = C["S"]
    return S._outline(s)


CANDIDATES = {
    "hut": _hut(),
    "fence": _fence(),
    "pond": _pond(),
    "crow": S._outline(_crow_r()),
    "mushroom": _mushroom(),
}


def sheet(silhouettes: bool = False) -> np.ndarray:
    cells = [S.OBJECT_VARIANTS[n][0][1] for n in S.OBJECT_NAMES] + list(CANDIDATES.values())
    cw, ch, gap = 34, 40, 10
    img = np.full((ch, cw * len(cells) + gap), C["W"] if silhouettes else C["s"], np.uint8)
    for i, spr in enumerate(cells):
        x0 = i * cw + (gap if i >= len(S.OBJECT_NAMES) else 0)
        g = S.GROUND["short"]
        gy, gx = ch - S.TILE_H - 2, x0 + cw // 2 - S.TILE_W // 2
        m = g != TRANSPARENT
        if not silhouettes:
            img[gy : gy + S.TILE_H, gx : gx + S.TILE_W][m] = g[m]
        h, w = spr.shape
        y, x = ch - 2 - h, x0 + cw // 2 - w // 2
        m = spr != TRANSPARENT
        img[y : y + h, x : x + w][m] = C["D"] if silhouettes else spr[m]
    return img


if __name__ == "__main__":
    import sys

    img = sheet()
    Image.fromarray(to_rgb(img)).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST).save(f"{sys.argv[1]}/referent_candidates.png")
    # silhouettes too: the shape alone is what black-ink drawings can capture
    sil = sheet(silhouettes=True)
    Image.fromarray(to_rgb(sil)).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST).save(f"{sys.argv[1]}/referent_candidates_sil.png")
    print({k: v.shape for k, v in CANDIDATES.items()})
