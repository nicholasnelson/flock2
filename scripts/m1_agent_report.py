#!/usr/bin/env python
"""Summarise an ``m1_agent.py`` run: reward, drive levels and the action mix by day and night,
averaged over windows of iterations.

    python scripts/m1_agent_report.py runs/m1/agent_run1/log.jsonl --window 100
"""

from __future__ import annotations

import argparse
import json

import numpy as np

DRIVES = ("energy", "water", "warmth", "social", "interest")
ACTIONS = ("stop", "walk", "trot", "left", "right", "graze", "drink", "rest", "head_up")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--window", type=int, default=100)
    a = p.parse_args()
    recs = [json.loads(l) for l in open(a.log)]
    print(f"{len(recs)} iterations, {recs[-1]['t'] / 60:.0f} min")
    head = "iters".ljust(11) + "reward  " + " ".join(d[:6].rjust(6) for d in DRIVES) + "  | day: " + " ".join(x[:5].rjust(5) for x in ACTIONS) + "  | night: " + " ".join(x[:5].rjust(5) for x in ACTIONS) + "  | ac_ret  entropy imag_r  curio dry_dr bare_g"
    print(head)
    for s in range(0, len(recs), a.window):
        w = recs[s : s + a.window]
        mix = lambda k: [np.mean([r[k][x] for r in w if r[k]]) if any(r[k] for r in w) else np.nan for x in ACTIONS]
        lv = np.mean([r["levels"] for r in w], 0)
        line = f"{w[0]['iter']:>4}-{w[-1]['iter']:<5} {np.mean([r['reward'] for r in w]):7.3f} " + " ".join(f"{x:6.2f}" for x in lv)
        line += "  |      " + " ".join(f"{x:5.2f}" for x in mix("actions_day")) + "  |        " + " ".join(f"{x:5.2f}" for x in mix("actions_night"))
        m = lambda k: np.mean([r[k] for r in w if k in r]) if any(k in r for r in w) else np.nan
        line += f"  | {m('ac_return'):6.2f} {m('ac_entropy'):7.3f} {m('ac_imag_reward'):6.3f} {m('ac_intrinsic'):6.3f} {m('drink_dry_share'):6.2f} {m('bare_graze'):6.3f}"
        print(line)


if __name__ == "__main__":
    main()
