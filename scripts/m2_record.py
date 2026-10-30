#!/usr/bin/env python
"""Record replayable runs of meadow v2 for the top-down viewer (``web/replay/``).

    python scripts/m2_record.py --policy scripted --willows 3 --days 3 --label "Scripted, plain world" \
        --description "The hand-written reference flock on the plain world with willows."
    python scripts/m2_record.py --policy hybrid --agent ~/flock2_data/m2_hybrid/s2f_s0 --inner 8 --fog 0.75 --days 5 ...

The world is built exactly as ``scripts/m2_eval.py`` builds it (the same meadow seeds 50,000 + i, the
same ``--willows`` / ``--inner`` / ``--edge`` / ``--fog`` / ``--thirst`` / ``--sheep`` options, the agent
loaded the same way), so a recording is one of the evaluation's meadows watched from above.

One self-describing file per meadow, ``<name>[_m<i>].flock.gz``: a gzip stream holding the magic
``FLK2``, a little-endian uint32 header length, a UTF-8 JSON header, zero padding to the next 8-byte
boundary, then the binary blobs the header's ``arrays`` table points at (``dtype``, ``shape``,
``offset`` from the start of the data section, ``nbytes``; every blob 8-byte aligned). The header carries
``meta`` (label, description, policy, brain, world options, seed, steps, git commit, date, constants),
``world`` (object table, fog days, pond centres, mud settings), ``scales`` (how each quantised array is
decoded), ``grass_steps`` (the step of each grass snapshot) and ``events`` (deaths, drinking bouts,
first approaches to within 8 tiles of water). The arrays: per step ``t``, ``light``, ``step_flags``
(bit 0 night, bit 1 fog); per step and sheep ``pos`` (int16, hundredths of a tile, row then column),
``heading`` (uint16, turns / 65535), ``action`` (uint8), ``flags`` (uint8: bit 0 alive, bit 1 resting,
bit 2 drank, bit 3 grazed), ``levels`` (uint8, level x 200), ``fat`` (uint8, x 255), ``wdist`` (uint16,
hundredths of a tile to the nearest water tile centre); the static ``water`` (uint8 48 x 48) and
``regrow_map`` (uint8 96 x 96, x 255); and ``grass`` (uint8, snapshots x 96 x 96, density x 85).
``read_replay`` decodes a file back to numpy for tests and analysis. ``index.json`` in the replay
directory lists every replay with its label, description and meta for the viewer's menu.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import functools
import gzip
import importlib.util
import json
import os
import pickle
import re
import struct
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from flock import sprites_iso as SP, wm
from flock.meadow2 import drives as D, privileged as P, sim
from flock.meadow2.world import DAY, GRASS_RES, MAX_DAYS, N_CELLS, N_TILES, WILLOW, foggy, random_start, with_terrain

MAGIC = b"FLK2"
FORMAT = "flock2-replay"
VERSION = 1
EXT = ".flock.gz"
DEFAULT_DIR = os.path.join("docs", "flock-2", "meadow2-results", "replays")
SCALES = {"pos": 100.0, "heading": 65535.0, "levels": 200.0, "fat": 255.0, "wdist": 100.0, "light": 255.0, "regrow_map": 255.0, "grass": 85.0}
KIND_NAMES = {i: n for i, n in enumerate(SP.OBJECT_NAMES)} | {int(WILLOW): "willow"}
NEAR_RANGE = 8.0
MEADOW_SEED = 50_000  # m2_eval's meadows: 50,000 + i, never used in training

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def load_script(name):
    """A sibling script (``m2_eval``, ``m2_hybrid``, ...) as a module, the way the tests import them."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SCRIPTS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_agent(policy, agent_dir, willows, zero_column=False):
    """(agent, willows): the agent tuple ``m2_eval.policy_fn`` expects, loaded as ``m2_eval.main`` loads it,
    and the willow count (the agent's config unless ``--willows`` was given)."""
    agent = None
    willows = willows or 0
    if agent_dir and willows == 0:
        willows = json.load(open(os.path.join(os.path.expanduser(agent_dir), "config.json"))).get("willows", 0)
    if policy == "reactive":
        ra = load_script("m2_reactive")
        d = os.path.expanduser(agent_dir)
        cfg = json.load(open(os.path.join(d, "config.json")))
        pol = ra.make_policy(0, cfg.get("hidden", 256))
        wm.load_into(pol, pickle.load(open(os.path.join(d, "policy.pkl"), "rb")))
        agent = (pol, ra)
    if policy == "ppo":
        mp = load_script("m2_ppo")
        d = os.path.expanduser(agent_dir)
        cfg = json.load(open(os.path.join(d, "config.json")))
        perc = mp.perception(cfg)
        pol = mp.make_policy(0, cfg.get("hidden", 256), P.n_obs(perc))
        wm.load_into(pol, pickle.load(open(os.path.join(d, "ppo.pkl"), "rb")))
        agent = (pol, perc)
    if policy == "hybrid":
        hy = load_script("m2_hybrid")
        d = os.path.expanduser(agent_dir)
        cfg = json.load(open(os.path.join(d, "config.json")))
        world = hy.make_world(0, cfg.get("model", "vector"))
        use_embed = cfg.get("policy_input", "state") == "state_embed"
        col = int(cfg.get("column", 0))
        pol = hy.make_policy(0, cfg.get("hidden", 256), hy.n_policy_in(world, use_embed, col))
        wm.load_into(world, pickle.load(open(os.path.join(d, "wm.pkl"), "rb")))
        wm.load_into(pol, pickle.load(open(os.path.join(d, "policy.pkl"), "rb")))
        column = hy.make_column(0, world, col)
        if column is not None:
            wm.load_into(column, pickle.load(open(os.path.join(d, "column.pkl"), "rb")))
        agent = (world, pol, hy, use_embed, None if zero_column else column, col)
    return agent, willows


def make_states(n_meadows, willows=0, n_sheep=None, inner=0.0, edge=2.0, trample=0.0, fog=0.0, seed=MEADOW_SEED, skyline=False):
    """The evaluation's meadows (``m2_eval.run``): seeds ``seed + i``, stage 2a's mud when ``inner`` > 0,
    the skyline (spec section 21.7) when ``skyline``."""
    kw = {"willows": willows, "fog": fog, "skyline": skyline} | ({} if n_sheep is None else {"n_sheep": n_sheep})
    states = jax.vmap(functools.partial(random_start, **kw))(jnp.stack([jax.random.PRNGKey(seed + i) for i in range(n_meadows)]))
    if inner > 0 or trample > 0:
        states = with_terrain(states, trample=trample, bank=(inner + edge) if inner > 0 else None, mud=inner > 0, inner=inner)
    return states


def record(policy, states, steps, grass_every=100, key_seed=0):
    """Run ``policy`` (``m2_eval.policy_fn``'s (init, act)) on the batch of ``states`` for ``steps`` steps and
    return a dict of numpy arrays, each (T, B, ...) (or (K, B, 96, 96) for the grass snapshots at
    ``grass_steps`` = 0, grass_every, ..., steps). The state recorded at step i is the one the policy
    saw, with the action it chose; ``died`` marks the step on which a sheep died."""
    init, act = policy

    def body(carry, k):
        st, mem = carry
        a, mem = act(st, mem, k)
        a = a.astype(jnp.int32)
        on_water, density, _ = jax.vmap(sim.mouth_tile)(st.meadow, st.flock.pos, st.flock.heading)
        new = jax.vmap(sim.step)(st, a)
        f = st.flock
        rec = dict(
            pos=f.pos, heading=f.heading, alive=f.alive, resting=f.resting, act=a, levels=f.levels, fat=f.fat,
            drank=(a == sim.DRINK) & on_water & f.alive, grazed=(a == sim.GRAZE) & (density >= sim.MIN_BITE) & f.alive,
            died=f.alive & ~new.flock.alive, wdist=jax.vmap(lambda s: P.visible_water(s, None)[1])(st),
            t=st.t, night=jax.vmap(sim.is_night)(st.t), light=jax.vmap(sim.light)(st.t), fog=jax.vmap(foggy)(st.meadow, st.t),
        )
        return (new, mem), rec

    @functools.partial(jax.jit, static_argnums=3)
    def chunk(st, mem, k, n):
        return jax.lax.scan(body, (st, mem), jax.random.split(k, n))

    mem = init(states)
    key = jax.random.PRNGKey(key_seed)
    parts, grass, grass_steps = [], [], []
    done = 0
    while done < steps:
        n = min(grass_every, steps - done)
        grass.append(np.asarray(states.meadow.grass))
        grass_steps.append(done)
        key, k = jax.random.split(key)
        (states, mem), rec = chunk(states, mem, k, n)
        parts.append(jax.tree.map(np.asarray, rec))
        done += n
    grass.append(np.asarray(states.meadow.grass))
    grass_steps.append(done)
    out = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}
    out["grass"] = np.stack(grass)
    out["grass_steps"] = np.array(grass_steps)
    return out


def ponds(water):
    """The water mask's connected components as [{row, col, radius, tiles}] (centre of mass, radius of the
    equal-area disc)."""
    water = np.asarray(water, bool)
    seen = np.zeros_like(water)
    out = []
    for r0, c0 in zip(*np.nonzero(water)):
        if seen[r0, c0]:
            continue
        stack, cells = [(r0, c0)], []
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < water.shape[0] and 0 <= cc < water.shape[1] and water[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        cells = np.array(cells) + 0.5
        out.append({"row": round(float(cells[:, 0].mean()), 2), "col": round(float(cells[:, 1].mean()), 2),
                    "radius": round(float(np.sqrt(len(cells) / np.pi)), 2), "tiles": int(len(cells))})
    return out


def events(rec, b):
    """Deaths (with the cause: the level that ran out), drinking bouts (runs of successful DRINK steps by one
    sheep) and each sheep's first step within NEAR_RANGE tiles of water, for meadow ``b``."""
    died, drank, wdist, alive = rec["died"][:, b], rec["drank"][:, b], rec["wdist"][:, b], rec["alive"][:, b]
    deaths = []
    for t, s in zip(*np.nonzero(died)):
        lv = rec["levels"][t, b, s]
        cause = "thirst" if lv[D.HYDRATION] <= lv[D.ENERGY] else "starvation"
        deaths.append({"step": int(t), "sheep": int(s), "pos": [round(float(v), 2) for v in rec["pos"][t, b, s]], "cause": cause,
                       "levels": {n: round(float(v), 3) for n, v in zip(D.NAMES, lv)}, "fat": round(float(rec["fat"][t, b, s]), 3)})
    bouts = []
    T, S = drank.shape
    for s in range(S):
        d = np.concatenate([[False], drank[:, s], [False]])
        edges = np.flatnonzero(d[1:] != d[:-1])
        for start, end in zip(edges[::2], edges[1::2]):
            bouts.append({"step": int(start), "end": int(end), "sheep": s, "sips": int(end - start)})
    bouts.sort(key=lambda e: e["step"])
    near = []
    for s in range(S):
        hit = np.flatnonzero((wdist[:, s] <= NEAR_RANGE) & alive[:, s])
        if len(hit):
            near.append({"step": int(hit[0]), "sheep": s})
    near.sort(key=lambda e: e["step"])
    return {"deaths": deaths, "drinks": bouts, "first_near_water": near}


def _q(x, scale, dtype):
    info = np.iinfo(dtype)
    return np.clip(np.rint(np.asarray(x, np.float64) * scale), info.min, info.max).astype(dtype)


def pack(rec, states, b, meta):
    """The header (dict) and the binary blobs ({name: array}) of meadow ``b``'s replay."""
    m = jax.tree.map(lambda x: np.asarray(x[b]), states.meadow)
    pos = rec["pos"][:, b]
    flags = (rec["alive"][:, b].astype(np.uint8) | (rec["resting"][:, b].astype(np.uint8) << 1)
             | (rec["drank"][:, b].astype(np.uint8) << 2) | (rec["grazed"][:, b].astype(np.uint8) << 3))
    step_flags = rec["night"][:, b].astype(np.uint8) | (rec["fog"][:, b].astype(np.uint8) << 1)
    arrays = {
        "t": rec["t"][:, b].astype(np.int32),
        "light": _q(rec["light"][:, b], SCALES["light"], np.uint8),
        "step_flags": step_flags,
        "pos": _q(pos, SCALES["pos"], np.int16),
        "heading": _q((rec["heading"][:, b] % (2 * np.pi)) / (2 * np.pi), SCALES["heading"], np.uint16),
        "action": rec["act"][:, b].astype(np.uint8),
        "flags": flags,
        "levels": _q(rec["levels"][:, b], SCALES["levels"], np.uint8),
        "fat": _q(rec["fat"][:, b], SCALES["fat"], np.uint8),
        "wdist": _q(rec["wdist"][:, b], SCALES["wdist"], np.uint16),
        "water": m.water.astype(np.uint8),
        "regrow_map": _q(m.regrow_map, SCALES["regrow_map"], np.uint8),
        "grass": _q(rec["grass"][:, b], SCALES["grass"], np.uint8),
    }
    T, S = pos.shape[:2]
    kinds = [int(k) for k in m.obj_kind]
    header = {
        "format": FORMAT, "version": VERSION,
        "meta": meta | {"meadow_index": int(b), "meadow_seed": int(meta["meadow_seed0"] + b), "steps": int(T), "sheep": int(S),
                        "deaths": int(rec["died"][:, b].sum()), "alive_at_end": int((rec["alive"][-1, b] & ~rec["died"][-1, b]).sum())},
        "world": {
            "n_tiles": N_TILES, "grass_res": GRASS_RES, "n_cells": N_CELLS,
            "objects": {"kind": kinds, "kind_names": {str(k): v for k, v in KIND_NAMES.items()}, "pos": [[round(float(v), 3) for v in p] for p in m.obj_pos],
                        "present": [bool(v) for v in m.obj_present], "variant": [int(v) for v in m.obj_variant]},
            "fog_days": [bool(v) for v in m.fog_days], "ponds": ponds(m.water),
            "regrow": float(m.regrow), "trample": float(m.trample),
            "mud": {"inner": meta["inner"], "edge": meta["edge"]},
        },
        "scales": SCALES, "grass_steps": [int(v) for v in rec["grass_steps"]],
        "events": events(rec, b),
        "layout": {"pos": "(steps, sheep, 2) row then column, tiles = value / scales.pos", "heading": "turns = value / scales.heading; direction (row, col) = (-cos, sin) of 2 pi turns",
                   "flags": "bit0 alive, bit1 resting, bit2 drank, bit3 grazed", "step_flags": "bit0 night, bit1 fog", "levels": "(steps, sheep, 4) in drive_names order, value / scales.levels",
                   "grass": "(len(grass_steps), 96, 96) density = value / scales.grass; cell (i, j) covers tile rows i/2..(i+1)/2"},
    }
    return header, arrays


def write_replay(path, header, arrays):
    """Write the container (see the module docstring) gzip-compressed at ``path``."""
    blobs, table, off = [], {}, 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        if a.dtype.byteorder == ">":
            a = a.astype(a.dtype.newbyteorder("<"))
        table[name] = {"dtype": a.dtype.name, "shape": list(a.shape), "offset": off, "nbytes": int(a.nbytes)}
        pad = (-a.nbytes) % 8
        blobs.append(a.tobytes() + b"\0" * pad)
        off += a.nbytes + pad
    # offsets are relative to the data section, which starts at the first 8-byte boundary after the header
    header = dict(header, arrays=table)
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hpad = (-(8 + len(hjson))) % 8
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(MAGIC + struct.pack("<I", len(hjson)) + hjson + b"\0" * hpad)
        for blob in blobs:
            f.write(blob)
    return header


def read_replay(path):
    """(header, {name: numpy array}) from a replay file, decoded back to floats where quantised."""
    raw = gzip.open(path, "rb").read()
    assert raw[:4] == MAGIC, "not a flock2 replay"
    (hlen,) = struct.unpack("<I", raw[4:8])
    header = json.loads(raw[8:8 + hlen].decode("utf-8"))
    base = 8 + hlen + (-(8 + hlen)) % 8
    arrays = {}
    for name, v in header["arrays"].items():
        a = np.frombuffer(raw, dtype=np.dtype(v["dtype"]).newbyteorder("<"), count=int(np.prod(v["shape"])), offset=base + v["offset"]).reshape(v["shape"])
        if name in header["scales"]:
            a = a.astype(np.float32) / header["scales"][name]
        arrays[name] = a
    return header, arrays


def update_index(replay_dir, entries):
    """Refresh ``index.json``: replace the entries of the files written, keep the others that still exist."""
    path = os.path.join(replay_dir, "index.json")
    old = []
    if os.path.exists(path):
        try:
            old = json.load(open(path)).get("replays", [])
        except (OSError, ValueError):
            old = []
    new_files = {e["file"] for e in entries}
    kept = [e for e in old if e.get("file") not in new_files and os.path.exists(os.path.join(replay_dir, e.get("file", "")))]
    index = {"format": FORMAT, "version": VERSION, "updated": _dt.datetime.now().isoformat(timespec="seconds"), "replays": kept + entries}
    json.dump(index, open(path, "w", newline="\n"), indent=1)
    return index


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=_SCRIPTS, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "replay"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--policy", choices=("random", "rest", "scripted", "ppo", "hybrid", "reactive"), required=True)
    p.add_argument("--agent", default=None, help="the brain's directory (hybrid / ppo / reactive)")
    p.add_argument("--willows", type=int, default=None, help="willows per pond (default: the agent's config, else 0)")
    p.add_argument("--meadows", type=int, default=1, help="meadows run at once (seeds 50,000 + i, m2_eval's)")
    p.add_argument("--pick", type=int, default=None, help="write only this meadow (default: every meadow run)")
    p.add_argument("--days", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=None, help="steps to run (default: days x DAY)")
    p.add_argument("--sheep", type=int, default=None, help="sheep per meadow (default 12)")
    p.add_argument("--inner", type=float, default=0.0, help="stage 2a's mud radius in tiles (0: none)")
    p.add_argument("--edge", type=float, default=2.0, help="tiles over which regrowth ramps back to full outside the mud")
    p.add_argument("--trample", type=float, default=0.0, help="biomass lost per step under a moving sheep")
    p.add_argument("--fog", type=float, default=0.0, help="probability of each day being foggy")
    p.add_argument("--thirst", type=float, default=1.0, help="scale on the hydration decay")
    p.add_argument("--zero_column", action="store_true", help="hybrid ablation: the column's state stays at zeros")
    p.add_argument("--skyline", type=int, default=None, help="1: the fixed peaks on the horizon (spec section 21.7); default the agent's config, else 0")
    p.add_argument("--seed", type=int, default=0, help="seed of the policy's randomness (the meadows are m2_eval's)")
    p.add_argument("--meadow_seed", type=int, default=MEADOW_SEED, help="first meadow seed (m2_eval's 50,000)")
    p.add_argument("--grass_every", type=int, default=100, help="steps between grass snapshots")
    p.add_argument("--label", required=True, help="a short name for the run menu")
    p.add_argument("--description", default="", help="what the run shows, for the menu")
    p.add_argument("--name", default=None, help="file name without extension (default: from the label)")
    p.add_argument("--dir", default=DEFAULT_DIR, help="replay directory (holds index.json)")
    a = p.parse_args(argv)
    if a.thirst != 1.0:
        # the constant is read at trace time, so scaling it here before anything is jitted changes the world
        D.HYDRATION_DECAY = D.HYDRATION_DECAY * a.thirst
    ev = load_script("m2_eval")
    agent, willows = load_agent(a.policy, a.agent, a.willows, a.zero_column)
    skyline = bool(a.skyline or 0)
    if a.agent and a.skyline is None:  # as m2_eval: the brain's own world unless overridden
        skyline = bool(json.load(open(os.path.join(os.path.expanduser(a.agent), "config.json"))).get("skyline", False))
    steps = a.steps if a.steps is not None else int(a.days * DAY)
    states = make_states(a.meadows, willows=willows, n_sheep=a.sheep, inner=a.inner, edge=a.edge, trample=a.trample, fog=a.fog, seed=a.meadow_seed, skyline=skyline)
    rec = record(ev.policy_fn(a.policy, agent), states, steps, grass_every=a.grass_every, key_seed=a.seed)
    meta = {
        "label": a.label, "description": a.description, "policy": a.policy, "agent": a.agent, "willows": willows, "sheep": a.sheep or 12,
        "inner": a.inner, "edge": a.edge, "trample": a.trample, "fog": a.fog, "thirst": a.thirst, "zero_column": a.zero_column, "skyline": skyline,
        "days": steps / DAY, "seed": a.seed, "meadow_seed0": a.meadow_seed, "meadows_run": a.meadows, "grass_every": a.grass_every,
        "git": git_commit(), "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "DAY": DAY, "N_TILES": N_TILES, "MAX_DAYS": MAX_DAYS, "day_fraction": sim.DAY_FRACTION, "action_names": list(sim.ACTION_NAMES), "drive_names": list(D.NAMES),
        "near_range": NEAR_RANGE, "memory_bins": [8, 12, 16, 24],
    }
    name = a.name or slug(a.label)
    which = range(a.meadows) if a.pick is None else [a.pick]
    entries = []
    for b in which:
        header, arrays = pack(rec, states, b, meta)
        fname = name + (f"_m{b}" if a.pick is None and a.meadows > 1 else "") + EXT
        path = os.path.join(a.dir, fname)
        header = write_replay(path, header, arrays)
        size = os.path.getsize(path)
        entries.append({"file": fname, "label": a.label, "description": a.description, "bytes": size, "meta": header["meta"]})
        print(f"wrote {path} ({size / 1e6:.2f} MB): {header['meta']['steps']} steps, {header['meta']['deaths']} deaths, "
              f"{len(header['events']['drinks'])} drinking bouts")
    update_index(a.dir, entries)
    print(f"index: {os.path.join(a.dir, 'index.json')}")


if __name__ == "__main__":
    main()
