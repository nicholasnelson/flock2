#!/usr/bin/env python
"""Watch the scripted flock on a generated meadow (Milestone 1, step 1).

    python scripts/m1_watch.py --out runs/m1/watch --steps 2400 --every 12

Runs the simulation and every sheep's first-person view on the GPU (one jitted scan), then
draws display frames (isometric window following the flock, one sheep's strip, a status line)
and writes ``watch.gif`` plus a few stills.
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow import scripted, sim, views, viewer
from flock.meadow.world import generate, ground_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/m1/watch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sheep", type=int, default=40)
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--every", type=int, default=12)
    p.add_argument("--follow", type=int, default=0, help="whose view to show")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    state = generate(jax.random.PRNGKey(a.seed), a.sheep)
    n_frames = a.steps // a.every

    def chunk(carry, key):
        st = carry

        def one(st, k):
            return sim.step(st, scripted.policy(st, k)), None

        st, _ = jax.lax.scan(one, st, jax.random.split(key, a.every))
        view = views.view_one(st, jnp.int32(a.follow), ground_state(st.meadow))
        rec = dict(pos=st.flock.pos, heading=st.flock.heading, grass=st.meadow.grass, t=st.t, view=view, thirst=st.flock.thirst, hunger=st.flock.hunger, resting=st.flock.resting)
        return st, rec

    run = jax.jit(lambda s, k: jax.lax.scan(chunk, s, jax.random.split(k, n_frames)))
    _, rec = run(state, jax.random.PRNGKey(a.seed + 1))
    rec = jax.tree.map(np.asarray, rec)
    m = jax.tree.map(np.asarray, state.meadow)

    frames = []
    for i in range(n_frames):
        st = dict(water=m.water, grass=rec["grass"][i], obj_kind=m.obj_kind, obj_variant=m.obj_variant, obj_pos=m.obj_pos, obj_present=m.obj_present, pos=rec["pos"][i], heading=rec["heading"][i])
        t = int(rec["t"][i])
        night = bool(sim.is_night(t))
        phase = (t % sim.DAY) / sim.DAY
        text = f"day {t // sim.DAY}  {'night' if night else 'day'} {phase:.2f}  resting {rec['resting'][i].mean():.0%}  thirst {rec['thirst'][i].mean():.2f}  hunger {rec['hunger'][i].mean():.2f}  (sheep {a.follow}'s view below)"
        frames.append(viewer.frame(st, rec["view"][i], a.follow, night, text))
    frames[0].save(os.path.join(a.out, "watch.gif"), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    for i in (0, n_frames // 3, int(n_frames * 0.75), n_frames - 1):
        frames[i].save(os.path.join(a.out, f"still_{i:04d}.png"))
    print(f"wrote {n_frames} frames to {a.out}")


if __name__ == "__main__":
    main()
