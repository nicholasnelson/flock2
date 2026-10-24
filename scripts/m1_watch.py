#!/usr/bin/env python
"""Watch the flock on a generated meadow (Milestone 1): the scripted flock, or with ``--agent``
a trained policy (a directory holding ``wm.pkl`` and ``agent.pkl`` from ``m1_agent.py``).

    python scripts/m1_watch.py --out runs/m1/watch --steps 2400 --every 12
    python scripts/m1_watch.py --agent ~/flock2_data/m1_agent/run1 --out runs/m1/watch_agent

Runs the simulation and every sheep's first-person view on the GPU (one jitted scan), then
draws display frames (isometric window following the flock, one sheep's strip, a status line)
and writes ``watch.gif`` plus a few stills.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx

from flock import ac, wm
from flock.meadow import drives as D, scripted, sim, views, viewer
from flock.meadow.world import generate, ground_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/m1/watch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sheep", type=int, default=40)
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--every", type=int, default=12)
    p.add_argument("--follow", type=int, default=0, help="whose view to show")
    p.add_argument("--agent", default=None, help="directory with wm.pkl and agent.pkl (else the scripted flock)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    state = generate(jax.random.PRNGKey(a.seed), a.sheep)
    n_frames = a.steps // a.every

    def chunk(world, agent, carry, key):
        st, mem = carry

        def one(c, k):
            st, mem = c
            if agent is None:
                return (sim.step(st, scripted.policy(st, k)), mem), None
            st, mem, _ = _agent.act_and_step(world, agent, st, mem, k, jnp.arange(1))
            return (st, mem), None

        (st, mem), _ = jax.lax.scan(one, (st, mem), jax.random.split(key, a.every))
        view = views.view_one(st, jnp.int32(a.follow), ground_state(st.meadow))
        rec = dict(pos=st.flock.pos, heading=st.flock.heading, grass=st.meadow.grass, t=st.t, view=view, levels=st.flock.levels, resting=st.flock.resting, reward=st.flock.reward)
        return (st, mem), rec

    mem = (jnp.zeros((a.sheep, wm.DETER)), jnp.zeros((a.sheep, wm.STOCH * wm.CLASSES)), jnp.zeros((a.sheep,), jnp.int32))
    if a.agent:
        spec = importlib.util.spec_from_file_location("m1_agent", os.path.join(os.path.dirname(__file__), "m1_agent.py"))
        _agent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_agent)
        world = wm.WorldModel(nnx.Rngs(0), n_intero=D.N_FELT, reward=True)
        wm.load_into(world, pickle.load(open(os.path.join(os.path.expanduser(a.agent), "wm.pkl"), "rb")))
        agent = ac.Agent(nnx.Rngs(1))
        wm.load_into(agent, pickle.load(open(os.path.join(os.path.expanduser(a.agent), "agent.pkl"), "rb")))
        run = nnx.jit(lambda w, g, s, k: jax.lax.scan(lambda c, kk: chunk(w, g, c, kk), (s, mem), jax.random.split(k, n_frames)))
        _, rec = run(world, agent, state, jax.random.PRNGKey(a.seed + 1))
    else:
        run = jax.jit(lambda s, k: jax.lax.scan(lambda c, kk: chunk(None, None, c, kk), (s, mem), jax.random.split(k, n_frames)))
        _, rec = run(state, jax.random.PRNGKey(a.seed + 1))
    rec = jax.tree.map(np.asarray, rec)
    m = jax.tree.map(np.asarray, state.meadow)

    frames = []
    for i in range(n_frames):
        st = dict(water=m.water, grass=rec["grass"][i], obj_kind=m.obj_kind, obj_variant=m.obj_variant, obj_pos=m.obj_pos, obj_present=m.obj_present, pos=rec["pos"][i], heading=rec["heading"][i])
        t = int(rec["t"][i])
        night = bool(sim.is_night(t))
        phase = (t % sim.DAY) / sim.DAY
        lv = rec["levels"][i].mean(0)
        text = f"day {t // sim.DAY} {'night' if night else 'day'} {phase:.2f}  resting {rec['resting'][i].mean():.0%}  energy {lv[0]:.2f} water {lv[1]:.2f} warmth {lv[2]:.2f} social {lv[3]:.2f} interest {lv[4]:.2f}  (sheep {a.follow}'s view)"
        frames.append(viewer.frame(st, rec["view"][i], a.follow, night, text))
    frames[0].save(os.path.join(a.out, "watch.gif"), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    for i in (0, n_frames // 3, int(n_frames * 0.75), n_frames - 1):
        frames[i].save(os.path.join(a.out, f"still_{i:04d}.png"))
    print(f"wrote {n_frames} frames to {a.out}")


if __name__ == "__main__":
    main()
