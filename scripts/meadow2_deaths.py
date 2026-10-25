#!/usr/bin/env python
"""Why do scripted sheep die in meadow v2? Traces every sheep that dies in the first ``--meadows``
check meadows (same seeds as ``meadow2_checks.py``) and summarises its last ``--window`` steps.

    python scripts/meadow2_deaths.py --meadows 16 --steps 6000
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import drives as D, scripted, sim
from flock.meadow2.world import random_start, water_distance


def blockers(states, rec, m, s, t):
    """What is around a sheep at step t: the nearest blocking object, sheep within 0.7 tiles, and
    whether a walk step ahead would be allowed."""
    mead = jax.tree.map(lambda x: x[m], states.meadow)
    pos, h = rec["pos"][t, m], rec["heading"][t, m]
    me = pos[s]
    d_obj = np.linalg.norm(np.asarray(mead.obj_pos) - me, axis=-1)
    rad = np.asarray(sim.RADIUS[mead.obj_kind]) if hasattr(sim, "RADIUS") else None
    from flock.meadow2.world import RADIUS
    rad = np.asarray(RADIUS[mead.obj_kind])
    block = (rad > 0) & np.asarray(mead.obj_present)
    j = int(np.argmin(np.where(block, d_obj - rad, np.inf)))
    ahead = me + sim.WALK_SPEED * np.array([-np.cos(h[s]), np.sin(h[s])])
    allowed = bool(sim._allowed(mead, jnp.asarray(ahead[None]))[0])
    others = np.linalg.norm(pos - me, axis=-1)
    return {"pos": np.round(me, 2).tolist(), "heading_deg": round(float(np.degrees(h[s])) % 360, 1),
            "nearest_blocker": [str(int(mead.obj_kind[j])), round(float(d_obj[j]), 2)],
            "sheep_within_0.7": int(((others < 0.7) & (others > 0)).sum()), "walk_step_allowed": allowed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meadows", type=int, default=16)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--window", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    keys = jnp.stack([jax.random.PRNGKey(30_000 + i) for i in range(a.meadows)])
    states = jax.jit(jax.vmap(random_start))(keys)
    wdist = np.asarray(jax.vmap(water_distance)(states.meadow.water))

    def body(st, k):
        act = jax.vmap(scripted.policy)(st, jax.random.split(k, st.t.shape[0]))
        st2 = jax.vmap(sim.step)(st, act)
        on = jax.vmap(lambda m, p_, h: sim.mouth_tile(m, p_, h)[0])(st2.meadow, st2.flock.pos, st2.flock.heading)
        return st2, dict(pos=st2.flock.pos, heading=st2.flock.heading, hyd=st2.flock.levels[..., D.HYDRATION], energy=st2.flock.levels[..., D.ENERGY], act=act, alive=st2.flock.alive, mouth_water=on, night=jax.vmap(sim.is_night)(st.t))

    _, rec = jax.jit(lambda s, k: jax.lax.scan(body, s, jax.random.split(k, a.steps)))(states, jax.random.PRNGKey(a.seed))
    rec = jax.tree.map(np.asarray, rec)
    out = []
    for m in range(a.meadows):
        for s in range(rec["alive"].shape[2]):
            al = rec["alive"][:, m, s]
            if al.all():
                continue
            t_death = int(np.argmin(al))
            w = slice(max(0, t_death - a.window), t_death)
            pos = rec["pos"][w, m, s]
            r, c = np.clip(pos.astype(int), 0, wdist.shape[1] - 1).T
            d_water = wdist[m][r, c]
            acts = rec["act"][w, m, s]
            out.append({
                "meadow": m, "sheep": s, "died_at": t_death,
                "start_hydration": round(float(states.flock.levels[m, s, D.HYDRATION]), 2),
                "start_dist_to_water": round(float(wdist[m][tuple(np.clip(np.asarray(states.flock.pos[m, s]).astype(int), 0, 47))]), 1),
                "hydration_at_window_start": round(float(rec["hyd"][w.start, m, s]), 3),
                "energy_at_death": round(float(rec["energy"][t_death, m, s]), 3),
                "night_share_in_window": round(float(rec["night"][w, m].mean()), 2),
                "dist_to_water_start_end_min": [round(float(d_water[0]), 1), round(float(d_water[-1]), 1), round(float(d_water.min()), 1)],
                "moved_tiles": round(float(np.linalg.norm(pos[-1] - pos[0])), 1),
                "steps_mouth_on_water": int(rec["mouth_water"][w, m, s].sum()),
                "actions": {sim.ACTION_NAMES[i]: int((acts == i).sum()) for i in range(sim.N_ACTIONS) if (acts == i).any()},
                "at_death": blockers(states, rec, m, s, t_death - 1),
            })
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
