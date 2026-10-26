#!/usr/bin/env python
"""Walk around the meadow as a sheep and see exactly what it sees (owner's request, 2026-10-26).

    python scripts/sheep_view.py                 # meadow v2; then open http://localhost:8765
    python scripts/sheep_view.py --world v1      # the first meadow (128 x 128, 40 sheep)
    python scripts/sheep_view.py --port 8800 --seed 3

A local web server runs the real meadow on the CPU. You are sheep 0; the others follow the
scripted flock. Every tick, your keys become one of the sheep's actions (sent by name), the meadow
steps once, and the page shows sheep 0's first-person strip, as a learning sheep sees it.

Meadow v2 (``flock.meadow2``) is drawn here with the first meadow's renderer (``flock.fpv``: its
sprites, its projection and a view range of 4 + 20 x light tiles) until renderer v2 is built (spec
§§ 4-5). The page marks the mouth point with a reticle and shows fat and death.

Controls (the page lists them too): W walk, Shift+W trot, A / D turn 22.5 degrees, E graze,
Q drink, R rest, Space head up (first meadow only), mouse to look, P pause, N new meadow.
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

from flock import fpv, sprites_iso as S
from flock.meadow import views as V
from flock.palette import NAMES, RGB


def world_v1():
    from flock.meadow import scripted, sim
    from flock.meadow.world import N_TILES, ground_state, random_start

    def view(state):
        return V.view_one(state, 0, ground_state(state.meadow))

    def extra(st):
        f = st.flock
        return {"head_up": bool(f.head_up[0]), "alive": True, "fat": None}

    from flock.meadow import drives

    return dict(scripted=scripted, sim=sim, n_tiles=N_TILES, ground_state=ground_state, random_start=random_start, view=view, extra=extra, reticle=None, light=None, levels=list(drives.NAMES))


def world_v2():
    from flock.meadow2 import scripted, sim
    from flock.meadow2.world import N_TILES, ground_state, random_start

    def view(state, i=0):
        """The first meadow's renderer on a meadow v2 state (dead sheep are not drawn)."""
        m, f = state.meadow, state.flock
        eye, n = f.pos[i], f.pos.shape[0]
        obj_idx = m.obj_kind * (V._N_VAR * V._N_FACE) + m.obj_variant * V._N_FACE
        sh_idx = V._ANIMAL_OFFSET + S.SHEEP * V._N_FACE + V.sheep_facing(eye, f.pos, f.heading)
        pos = jnp.concatenate([m.obj_pos, f.pos])
        idx = jnp.concatenate([obj_idx, sh_idx])
        present = jnp.concatenate([m.obj_present, (jnp.arange(n) != i) & f.alive])
        d = jnp.where(present, jnp.linalg.norm(pos - eye[None, :], axis=-1), jnp.inf)
        _, near = jax.lax.top_k(-d, min(V.K_NEAREST, pos.shape[0]))
        night = sim.is_night(state.t)
        return fpv.render_fpv(
            ground_state(m), V.BILLBOARDS[idx[near]], pos[near], present[near], eye, f.heading[i],
            max_dist=4.0 + 20.0 * sim.light(state.t), sky=jnp.where(night, V.SKY_NIGHT, V.SKY_DAY), far=jnp.where(night, V.FAR_NIGHT, V.FAR_DAY),
        )

    def extra(st):
        f, m = st.flock, st.meadow
        on_water, grass, inside = sim.mouth_tile(m, jnp.asarray(f.pos[:1]), jnp.asarray(f.heading[:1]))
        g = float(grass[0])
        mouth = "water" if bool(on_water[0]) else ("outside" if not bool(inside[0]) else (f"grass {g:.1f}" if g >= sim.MIN_BITE else f"grass {g:.1f}, too short to bite"))
        return {"alive": bool(f.alive[0]), "fat": float(f.fat[0]), "light": float(sim.light(st.t)), "mouth": mouth, "flock_alive": int(f.alive.sum()), "seen": float(f.seen[0])}

    # where the mouth point appears in the strip (fpv's even-angle rows; eye height fpv.EYE_H)
    elev = -float(np.arctan2(fpv.EYE_H, sim.MOUTH))
    row = fpv.HORIZON - elev / float(fpv.RAD_PER_PX)
    from flock.meadow2 import drives

    return dict(scripted=scripted, sim=sim, n_tiles=N_TILES, ground_state=ground_state, random_start=random_start, view=view, extra=extra, reticle=[row / fpv.H, 0.5], light=sim.light, levels=list(drives.NAMES))


class World:
    def __init__(self, w, seed):
        self.w = w
        self.lock = threading.Lock()
        self.key = jax.random.PRNGKey(seed)
        sim, scripted, view, gs = w["sim"], w["scripted"], w["view"], w["ground_state"]

        @jax.jit
        def tick(state, action, heading, use_heading, key):
            f = state.flock
            state = state._replace(flock=f._replace(heading=jnp.where(use_heading, f.heading.at[0].set(heading), f.heading)))
            acts = scripted.policy(state, key).at[0].set(action)
            state = sim.step(state, acts)
            return state, view(state), gs(state.meadow)

        self.tick = tick
        self.reset()

    def reset(self):
        self.key, k = jax.random.split(self.key)
        self.state = self.w["random_start"](k)
        self.n = 0

    def step(self, action_name, heading, reset):
        sim = self.w["sim"]
        with self.lock:
            if reset:
                self.reset()
            action = sim.ACTION_NAMES.index(action_name) if action_name in sim.ACTION_NAMES else 0
            self.key, k = jax.random.split(self.key)
            use = heading is not None
            self.state, view, tiles = self.tick(self.state, jnp.int32(action), jnp.float32(heading if use else 0.0), use, k)
            self.n += 1
            st = jax.tree.map(np.asarray, self.state)
            f, m = st.flock, st.meadow
            alive = np.asarray(getattr(f, "alive", np.ones(f.pos.shape[0], bool)))
            out = {
                "view": base64.b64encode(np.asarray(view, np.uint8).tobytes()).decode(),
                "pos": f.pos[0].tolist(),
                "heading": float(f.heading[0]),
                "others": np.round(f.pos[1:][alive[1:]], 2).tolist(),
                "levels": np.round(f.levels[0], 3).tolist(),
                "reward": float(f.reward[0]),
                "taste": float(f.taste[0]),
                "t": int(st.t),
                "night": bool(sim.is_night(st.t)),
                "action": sim.ACTION_NAMES[action],
                **self.w["extra"](st),
            }
            if reset or self.n == 1 or self.n % 10 == 0:
                out["tiles"] = base64.b64encode(np.asarray(tiles, np.uint8).tobytes()).decode()
                out["objects"] = [[int(k), round(float(p[0]), 2), round(float(p[1]), 2)] for k, p, pr in zip(m.obj_kind, m.obj_pos, m.obj_present) if pr]
            return out


def handler(world, page):
    w = world.w

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
                info = {"rgb": RGB.tolist(), "names": NAMES, "objects": S.OBJECT_NAMES, "ground": S.GROUND_NAMES, "water": int(S.WATER),
                        "tiles": int(w["n_tiles"]), "actions": list(w["sim"].ACTION_NAMES), "reticle": w["reticle"], "levels": w["levels"]}
                self._send(json.dumps(info).encode(), "application/json")
            else:
                self._send(page, "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            out = world.step(str(req.get("action", "stop")), req.get("heading"), bool(req.get("reset")))
            self._send(json.dumps(out).encode(), "application/json")

    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", choices=("v1", "v2"), default="v2")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    page = open(os.path.join(os.path.dirname(__file__), "sheep_view.html"), "rb").read()
    world = World(world_v1() if a.world == "v1" else world_v2(), a.seed)
    world.step("stop", None, False)  # compile before the first visitor
    print(f"sheep view ({a.world}): http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), handler(world, page)).serve_forever()


if __name__ == "__main__":
    main()
