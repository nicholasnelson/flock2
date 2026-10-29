"""Spec section 21.6's diagnostic R1 (excursions, where the sheep graze) on hand-made records."""

import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location("m2_eval", os.path.join(os.path.dirname(__file__), "..", "scripts", "m2_eval.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _rec(T=600, B=1, S=3):
    return {
        "alive_now": np.ones((T, B, S), bool), "levels_before": np.ones((T, B, S, 4)), "sees8": np.zeros((T, B, S), bool),
        "wdist": np.full((T, B, S), 4.0), "last_near8": np.tile(np.arange(T)[:, None, None], (1, B, S)), "fog": np.zeros((T, B, S), bool),
        "night": np.zeros((T, B), bool),
    }


def test_excursions_measure_the_return_from_the_farthest_point():
    r = _rec()
    t = np.arange(600)
    # sheep 0: out from step 100, farthest (15 tiles) at step 200 in fog, back within 8 at step 250, hydration 0.6 there
    d = np.where(t < 100, 4.0, np.where(t <= 200, 4.0 + 11.0 * (t - 100) / 100, np.where(t < 250, 15.0, 4.0)))
    r["wdist"][:, 0, 0] = d
    r["fog"][:, 0, 0] = True
    r["levels_before"][250:, 0, 0, 1] = 0.6
    # sheep 1: out at step 300 to 20 tiles and dies at step 400: censored
    r["wdist"][300:, 0, 1] = 20.0
    r["alive_now"][400:, 0, 1] = False
    # sheep 2: never within 8 tiles of water: no excursion
    r["wdist"][:, 0, 2] = 30.0
    r["last_near8"][:, 0, 2] = ev.NEVER
    out = ev.excursions(r)
    assert out["excursions"] == 2
    fog = out["fog_day"]["12-16"]
    assert fog["n"] == 1 and fog["median_return"] == 50 and fog["resolved_share"] == 1.0
    assert abs(fog["steps_per_tile_median"] - 50 / 7) < 1e-6 and fog["hydration_at_reentry_median"] == 0.6
    cen = out["clear_day"]["16-24"]
    assert cen["n"] == 1 and cen["resolved_share"] == 0.0 and cen["median_return"] == 100
    assert out["unsighted"]["12-16"]["n"] == 1 and out["night"]["12-16"]["n"] == 0


def test_wdist_share_splits_living_steps_by_condition():
    r = _rec(T=100, B=1, S=2)
    r["wdist"][:, 0, 1] = 14.0
    r["night"][50:, 0] = True
    r["fog"][:50, 0, :] = True
    out = ev.wdist_share(r)
    assert out["fog_day"]["steps"] == 100 and out["night"]["steps"] == 100 and out["clear_day"]["steps"] == 0
    assert out["all"]["0-8"] == 0.5 and out["all"]["12-16"] == 0.5
    assert abs(sum(v for k, v in out["night"].items() if k != "steps") - 1.0) < 1e-6
