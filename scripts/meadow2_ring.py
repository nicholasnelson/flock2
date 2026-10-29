#!/usr/bin/env python
"""The ring-growth measurement (spec § 21): does depletion and trampling bare the banks? The scripted
flock on ``--meadows`` meadows for ``--days`` days at each (regrowth factor, trample) setting.

    JAX_PLATFORMS=cpu python scripts/meadow2_ring.py --out runs/meadow2/ring.json

Per setting and day: mean grass density by distance band from water; the bare radius (the outermost
band, counted outward from the water, whose mean density is still below ``sim.MIN_BITE``); the mean
distance from water of successful bites; deaths by cause; and the share of dry cells below 0.5
biomass more than 8 tiles from water (trails, over-grazing beyond the ring).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import drives as D, scripted, sim
from flock.meadow2.world import DAY, GRASS_RES, N_CELLS, REGROW, cell_centres, cell_water_distance, dry_cells, random_start, water_at, with_terrain

BANDS = ((0, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 1e9))


def run(states, key, steps, camp=False):
    wd = jax.vmap(cell_water_distance)(states.meadow.water)  # (B, C, C)
    dry = jax.vmap(dry_cells)(states.meadow.water)
    pol = functools.partial(scripted.policy, camp=camp)

    def body(st, k):
        a = jax.vmap(pol)(st, jax.random.split(k, st.t.shape[0]))
        _, density, _ = jax.vmap(sim.mouth_tile)(st.meadow, st.flock.pos, st.flock.heading)
        st2 = jax.vmap(sim.step)(st, a)
        f = st2.flock
        grazed = (a == sim.GRAZE) & (density >= sim.MIN_BITE) & st.flock.alive
        d_bite = jax.vmap(lambda w, p: w[jnp.clip((p[:, 0] * GRASS_RES).astype(jnp.int32), 0, N_CELLS - 1), jnp.clip((p[:, 1] * GRASS_RES).astype(jnp.int32), 0, N_CELLS - 1)])(wd, st.flock.pos)
        rec = dict(bite_dist=(d_bite * grazed).sum(-1), bites=grazed.sum(-1), alive=f.alive.sum(-1),
                   died_thirst=((st.flock.alive & ~f.alive) & (f.levels[..., D.HYDRATION] <= 0)).sum(-1),
                   died_hunger=((st.flock.alive & ~f.alive) & (f.levels[..., D.HYDRATION] > 0)).sum(-1))
        return st2, rec

    def day(st, k):
        st, rec = jax.lax.scan(body, st, jax.random.split(k, DAY))
        g = st.meadow.grass
        bands = []
        for lo, hi in BANDS:
            m = dry & (wd >= lo) & (wd < hi)
            bands.append(jnp.sum(jnp.where(m, g, 0.0), (1, 2)) / jnp.maximum(m.sum((1, 2)), 1))
        far_bare = jnp.sum(dry & (wd > 8) & (g < 0.5), (1, 2)) / jnp.maximum((dry & (wd > 8)).sum((1, 2)), 1)
        out = {k2: v.sum(0) for k2, v in rec.items()}
        out["alive_end"] = rec["alive"][-1]
        out["bands"] = jnp.stack(bands, -1)
        out["far_bare"] = far_bare
        return st, out

    _, out = jax.lax.scan(day, states, jax.random.split(key, steps // DAY))
    return out  # (days, B, ...)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meadows", type=int, default=16)
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--regrow", default="1,0.5,0.25,0.125", help="factors on REGROW")
    p.add_argument("--trample", default="0,0.005,0.02,0.05")
    p.add_argument("--bank", default="0", help="muddy-bank widths (tiles; 0 = uniform regrowth), grid 3")
    p.add_argument("--mud", action="store_true", help="the bank map also scales the starting grass (grid 4)")
    p.add_argument("--inner", default="0", help="mud radii (tiles, no grass or regrowth within; bank = inner + edge), grid 5")
    p.add_argument("--edge", type=float, default=2.0, help="width of the mud's edge when --inner is set")
    p.add_argument("--willows", type=int, default=3)
    p.add_argument("--camp", action="store_true", help="the camping scripted flock (spec § 21, grid 2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/meadow2/ring.json")
    a = p.parse_args()
    keys = jax.random.split(jax.random.PRNGKey(80_000 + a.seed), a.meadows)
    base = jax.jit(jax.vmap(functools.partial(random_start, willows=a.willows)))(keys)
    runj = jax.jit(functools.partial(run, camp=a.camp), static_argnums=(2,))
    res = {"meadows": a.meadows, "days": a.days, "willows": a.willows, "camp": a.camp, "mud": a.mud, "bands": [list(b) for b in BANDS], "settings": []}
    t0 = time.time()
    inners = [float(x) for x in a.inner.split(",")]
    banks = [float(x) for x in a.bank.split(",")] if inners == [0.0] else [i + a.edge for i in inners]
    grid = [(rf, tr, bk, inn) for rf in (float(x) for x in a.regrow.split(",")) for tr in (float(x) for x in a.trample.split(",")) for bk, inn in zip(banks, inners if inners != [0.0] else [0.0] * len(banks))]
    for rf, tr, bk, inn in grid:
        if True:
            states = with_terrain(base, regrow=REGROW * rf, trample=tr, bank=bk, mud=a.mud, inner=inn)
            out = jax.tree.map(np.asarray, runj(states, jax.random.PRNGKey(a.seed), a.days * DAY))
            bands = out["bands"].mean(1)  # (days, bands)
            bare = []
            for d in range(a.days):
                r = 0.0
                for (lo, hi), v in zip(BANDS, bands[d]):
                    if v < sim.MIN_BITE:
                        r = hi if hi < 1e8 else 48.0
                    else:
                        break
                bare.append(r)
            row = {
                "regrow_factor": rf, "trample": tr, "bank": bk, "inner": inn,
                "band_density_by_day": np.round(bands, 2).tolist(),
                "bare_radius_by_day": bare,
                "bite_distance_by_day": np.round(out["bite_dist"].sum(1) / np.maximum(out["bites"].sum(1), 1), 2).tolist(),
                "bites_per_day": out["bites"].sum(1).tolist(),
                "alive_end_by_day": out["alive_end"].sum(1).tolist(),
                "died_thirst": int(out["died_thirst"].sum()), "died_hunger": int(out["died_hunger"].sum()),
                "far_bare_by_day": np.round(out["far_bare"].mean(1), 3).tolist(),
            }
            res["settings"].append(row)
            print(json.dumps({k: row[k] for k in ("regrow_factor", "trample", "bank", "inner", "bare_radius_by_day", "bite_distance_by_day", "alive_end_by_day", "died_thirst", "died_hunger", "far_bare_by_day")}), f"{time.time() - t0:.0f}s", flush=True)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
