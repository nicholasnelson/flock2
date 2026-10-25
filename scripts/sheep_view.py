#!/usr/bin/env python
"""Walk around the meadow as a sheep and see exactly what it sees (owner's request, 2026-10-26).

    python scripts/sheep_view.py            # then open http://localhost:8765
    python scripts/sheep_view.py --port 8800 --seed 3

A local web server runs the real meadow (``flock.meadow.sim``) on the CPU. You are sheep 0; the
other 39 follow the scripted flock. Every tick, your keys become one of the sheep's nine actions,
the meadow steps once, and the page shows sheep 0's first-person strip, rendered by the same
``flock.meadow.views.view_one`` the learning sheep see (64 x 256 palette pixels, enlarged).

Controls (the page lists them too): W walk, Shift+W trot, A / D turn 22.5 degrees, E graze,
Q drink, R rest, Space head up, mouse to look, P pause, N new meadow. "Free look" sets the
heading directly from the mouse (smooth, for getting a feel for the view); with it off, turning
only happens in the sheep's own 22.5 degree steps, as it does for a learning sheep.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # one view per tick: the CPU is plenty, and the GPU stays free

import jax
import jax.numpy as jnp
import numpy as np

from flock import sprites_iso as S
from flock.meadow import scripted, sim
from flock.meadow.views import view_one
from flock.meadow.world import N_TILES, ground_state, random_start
from flock.palette import NAMES, RGB


@jax.jit
def tick(state, action, heading, use_heading, key):
    f = state.flock
    state = state._replace(flock=f._replace(heading=jnp.where(use_heading, f.heading.at[0].set(heading), f.heading)))
    acts = scripted.policy(state, key).at[0].set(action)
    state = sim.step(state, acts)
    tiles = ground_state(state.meadow)
    return state, view_one(state, 0, tiles), tiles


class World:
    def __init__(self, seed):
        self.lock = threading.Lock()
        self.key = jax.random.PRNGKey(seed)
        self.reset()

    def reset(self):
        self.key, k = jax.random.split(self.key)
        self.state = random_start(k, 40)
        self.n = 0
        self.last = None

    def step(self, action, heading, reset):
        with self.lock:
            if reset:
                self.reset()
            self.key, k = jax.random.split(self.key)
            use = heading is not None
            self.state, view, tiles = tick(self.state, jnp.int32(action), jnp.float32(heading if use else 0.0), use, k)
            self.n += 1
            st = jax.tree.map(np.asarray, self.state)
            f, m = st.flock, st.meadow
            out = {
                "view": base64.b64encode(np.asarray(view, np.uint8).tobytes()).decode(),
                "pos": f.pos[0].tolist(),
                "heading": float(f.heading[0]),
                "others": np.round(f.pos[1:], 2).tolist(),
                "levels": np.round(f.levels[0], 3).tolist(),
                "reward": float(f.reward[0]),
                "taste": float(f.taste[0]),
                "t": int(st.t),
                "night": bool(sim.is_night(st.t)),
                "head_up": bool(f.head_up[0]),
                "action": sim.ACTION_NAMES[action],
            }
            if reset or self.n == 1 or self.n % 10 == 0:
                out["tiles"] = base64.b64encode(np.asarray(tiles, np.uint8).tobytes()).decode()
                out["objects"] = [[int(k), round(float(p[0]), 2), round(float(p[1]), 2)] for k, p, pr in zip(m.obj_kind, m.obj_pos, m.obj_present) if pr]
            return out


def handler(world, page):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/palette":
                info = {"rgb": RGB.tolist(), "names": NAMES, "objects": S.OBJECT_NAMES, "ground": S.GROUND_NAMES, "water": int(S.WATER), "tiles": int(N_TILES)}
                self._send(json.dumps(info).encode(), "application/json")
            else:
                self._send(page, "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            out = world.step(int(req.get("action", sim.STOP)), req.get("heading"), bool(req.get("reset")))
            self._send(json.dumps(out).encode(), "application/json")

    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    page = open(os.path.join(os.path.dirname(__file__), "sheep_view.html"), "rb").read()
    world = World(a.seed)
    world.step(sim.STOP, None, False)  # compile before the first visitor
    print(f"sheep view: http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), handler(world, page)).serve_forever()


if __name__ == "__main__":
    main()
