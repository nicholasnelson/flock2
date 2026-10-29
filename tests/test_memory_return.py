"""The behavioural memory probe M1 (spec section 21.4) on a hand-made record."""

import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location("m2_eval", os.path.join(os.path.dirname(__file__), "..", "scripts", "m2_eval.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _rec(T=1000, B=1, S=3):
    r = {
        "alive_now": np.ones((T, B, S), bool), "levels_before": np.ones((T, B, S, 4)), "sees8": np.zeros((T, B, S), bool),
        "wdist": np.full((T, B, S), 14.0), "last_near8": np.full((T, B, S), ev.NEVER), "fog": np.zeros((T, B, S), bool),
        "night": np.zeros((T, B), bool),
    }
    return r


def test_onsets_resolve_or_censor_and_split_by_fog_and_night():
    r = _rec()
    t = np.arange(1000)[:, None]
    # sheep 0: near water until step 100, thirsty from 200 in fog, resights at 260 (duration 60); then nothing
    r["last_near8"][:, 0, 0] = np.where(t[:, 0] < 100, t[:, 0], 100)
    r["levels_before"][200:, 0, 0, 1] = 0.1
    r["fog"][:, 0, 0] = True
    r["sees8"][260:300, 0, 0] = True
    # sheep 1: same but by clear day, never resights: censored at the cap
    r["last_near8"][:, 0, 1] = np.where(t[:, 0] < 100, t[:, 0], 100)
    r["levels_before"][200:, 0, 1, 1] = 0.1
    # sheep 2: thirsty but never knew the water: no onset
    r["levels_before"][:, 0, 2, 1] = 0.1
    out = ev.memory_return(r)
    assert out["onsets"] == 2  # one episode per excursion: no re-onset after a censored or resolved one without a new approach
    fog = out["fog_day"]["12-16"]
    assert fog["onsets"] == 1 and fog["median_censored"] == 60 and fog["resolved_share"] == 1.0
    clear = out["clear_day"]["12-16"]
    assert clear["onsets"] == 1 and clear["median_censored"] == ev.M1_CAP and clear["resolved_share"] == 0.0
    assert out["unsighted"]["12-16"]["onsets"] == 1
    assert out["pass"] is False and out["bins_read"] == 0  # too few onsets to read


def test_a_knowing_sheep_that_heads_back_passes_and_memory_lapses():
    r = _rec(T=4000, B=1, S=40)
    t = np.arange(4000)
    for s in range(40):
        r["last_near8"][:, 0, s] = np.where(t < 100, t, 100)
        r["levels_before"][500:, 0, s, 1] = 0.1
        r["night"][:, 0] = True
        r["sees8"][530:560, 0, s] = True  # back within 30 steps
    out = ev.memory_return(r)
    assert out["unsighted"]["12-16"]["onsets"] == 40 and out["pass"] is True
    # the same sheep thirsty only after RECENT steps have passed since water: nothing to remember, no onset
    r2 = _rec(T=4000, B=1, S=40)
    for s in range(40):
        r2["last_near8"][:, 0, s] = np.where(t < 100, t, 100)
        r2["levels_before"][100 + ev.RECENT + 10:, 0, s, 1] = 0.1
    assert ev.memory_return(r2)["onsets"] == 0
