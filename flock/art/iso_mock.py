"""Isometric sprite mock for sign-off. 8x4 px diamond tiles, sprites anchored at the tile's bottom-centre."""
import numpy as np
from PIL import Image
from flock.palette import CHAR_TO_INDEX, TRANSPARENT, to_rgb

def P(rows):
    w = max(len(r) for r in rows)
    out = np.full((len(rows), w), TRANSPARENT, np.uint8)
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch != ".": out[y, x] = CHAR_TO_INDEX[ch]
    return out

TW, TH = 8, 4
GROUND = {
 "bare":   P(["...tt...", ".ttttGt.", ".tGtttt.", "...tt..."]),
 "short":  P(["...GG...", ".GGGGGG.", ".GGGGGG.", "...GG..."]),
 "medium": P(["...GG...", ".GgGGgG.", ".GGgGGG.", "...GG..."]),
 "long":   P(["...Gl...", ".lGlGlG.", ".GlGlGl.", "...lG..."]),
}
OBJ = {
 "pond": P([
  "......BBBB......",
  "...BBbBBBBBB....",
  ".BBBBBBBbBBBBB..",
  "BBbBBBBBBBBBBbBB",
  ".BBBBBBbBBBBBB..",
  "...BBBBBBBBB....",
  "......BBBB......"]),
 "salt": P([
  "...WW...",
  ".WWWWWW.",
  "WWWWWWWW",
  "WWSWWWSW",
  "WWSWWWSW",
  ".SSSWSS.",
  "...SS..."]),
 "tree": P([
  "...DDDD...",
  ".DDggggDD.",
  "DgggglgggD",
  "DgglggglgD",
  ".DgggggDD.",
  "..DDgggD..",
  "....TT....",
  "....TT....",
  "...DTTD..."]),
 "flower": P([
  "..RR..",
  ".RYYR.",
  "..RR..",
  "..g...",
  ".gg.g.",
  "..gg.."]),
 "rock": P([
  "..SSS...",
  ".SSSSSs.",
  "SSSSssss",
  ".Sssssss",
  "..ssss.."]),
 "bush": P([
  "..gggg..",
  ".glgggg.",
  "gggglggg",
  ".Dgggg D".replace(" ","g"),
  "..DDDD.."]),
}
SHEEP_R = P([
  "..WWWW....",
  ".WWWWWWss.",
  "WWWWWWWsPs",
  "WWWWWWWWss",
  ".WWWWWWW..",
  "..K.K.K...",
  "..K...K..."])
FOX_R = P([
  "......OO..",
  "WO...OOKO.",
  "OOO.OOOOOO",
  ".OOOOOOOO.",
  "..OOOOOO..",
  "..K..K.K..",
])
ANIM = {"sheep_R": SHEEP_R, "sheep_L": SHEEP_R[:, ::-1], "fox_R": FOX_R, "fox_L": FOX_R[:, ::-1]}

def paste(img, spr, x, y):
    h, w = spr.shape; m = spr != TRANSPARENT
    sub = img[y:y+h, x:x+w]; mm = m[:sub.shape[0], :sub.shape[1]]
    sub[mm] = spr[:sub.shape[0], :sub.shape[1]][mm]

def iso(r, c): return (c - r) * TW // 2, (c + r) * TH // 2

def render(tiles, objects, animals, pad=24):
    n = tiles.shape[0]
    W = n * TW + 2*pad; H = n * TH + 2*pad
    img = np.zeros((H, W), np.uint8)
    ox = pad + n * TW // 2; oy = pad
    for r in range(n):
        for c in range(n):
            x, y = iso(r, c); paste(img, GROUND[list(GROUND)[tiles[r, c]]], ox + x - TW//2, oy + y)
    # draw order by r+c (depth)
    items = [(r + c, "o", k, r, c) for (k, r, c) in objects] + [(r + c, "a", k, r, c) for (k, r, c) in animals]
    for _, kind, k, r, c in sorted(items):
        spr = OBJ[k] if kind == "o" else ANIM[k]
        x, y = iso(r, c); h, w = spr.shape
        paste(img, spr, ox + x - w//2, oy + y + TH - h + (1 if kind=="a" else 0))
    return img

def sheet():
    cells = list(GROUND.items()) + list(OBJ.items()) + list(ANIM.items())
    cw, ch = 20, 16
    img = np.full((ch, cw*len(cells)), 10, np.uint8)
    for i, (name, spr) in enumerate(cells):
        h, w = spr.shape
        paste(img, GROUND["short"], i*cw + cw//2 - TW//2, ch - 5)
        paste(img, spr, i*cw + cw//2 - w//2, ch - 5 + TH - h)
    return img, [c[0] for c in cells]

if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    rng = np.random.default_rng(0)
    tiles = rng.choice(4, size=(14, 14), p=[0.1, 0.4, 0.3, 0.2])
    objects = [("tree", 2, 9), ("tree", 8, 1), ("pond", 5, 5), ("salt", 10, 10), ("rock", 3, 3), ("flower", 7, 10), ("bush", 11, 4), ("flower", 1, 12)]
    animals = [("sheep_R", 6, 8), ("sheep_L", 8, 7), ("sheep_R", 9, 9), ("fox_L", 12, 12)]
    scene = render(tiles, objects, animals)
    Image.fromarray(to_rgb(scene)).resize((scene.shape[1]*5, scene.shape[0]*5), Image.NEAREST).save(f"{out}/iso_scene.png")
    sh, names = sheet()
    Image.fromarray(to_rgb(sh)).resize((sh.shape[1]*6, sh.shape[0]*6), Image.NEAREST).save(f"{out}/iso_sheet.png")
    print(names)
