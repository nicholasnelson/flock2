#!/usr/bin/env python
"""Tabulate ``m1_eval.py`` results side by side (Markdown).

    python scripts/m1_eval_table.py runs/m1/eval/*.json
"""

from __future__ import annotations

import json
import sys

DRIVES = ("energy", "water", "warmth", "social", "interest")


def main():
    rows = [json.load(open(f)) for f in sys.argv[1:]]
    last = [k for k in rows[0] if k.startswith("dawn")][-1]
    print(f"| policy | reward/step | depleted | {last} mean (" + ", ".join(DRIVES) + f") | {last} 10th pct | graze ok / drink ok / bad taste (share of steps) | night: resting near cover | top day actions | top night actions |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        top = lambda d: ", ".join(f"{k} {v:.2f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:3])
        print(
            f"| {r['policy']} | {r['reward']:.3f} | {r['depleted']:.2f} | "
            + " / ".join(f"{x:.2f}" for x in r[last]["mean"])
            + " | "
            + " / ".join(f"{x:.2f}" for x in r[last]["p10"])
            + f" | {r['graze_ok_steps']:.3f} / {r['drink_ok_steps']:.3f} / {r['bad_taste_steps']:.3f} | {r['night_rest_near_cover']:.2f} | {top(r['actions_day'])} | {top(r['actions_night'])} |"
        )


if __name__ == "__main__":
    main()
