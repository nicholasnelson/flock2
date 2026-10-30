"""The meadow's side (spec § 21.8).

``flock.meadow2.world.N_TILES`` is read from the environment variable ``FLOCK2_TILES`` once, at import,
and every shape in the package follows it; so a script that takes ``--tiles`` or loads a brain trained
at another size must set the variable before it imports the world. ``bootstrap`` does that from the
command line (``--tiles N`` wins), else from the named agent's ``config.json`` (its ``tiles``, which
``m2_hybrid.py`` saves). No jax here: this must be importable before anything heavy.
"""

from __future__ import annotations

import json
import os
import sys


def bootstrap(argv: list[str] | None = None, agent_flags: tuple[str, ...] = ("--agent", "--continue_from")) -> int | None:
    """Set ``FLOCK2_TILES`` from ``argv`` (default ``sys.argv``). Returns the side set, or None."""
    argv = sys.argv if argv is None else argv
    if "--tiles" in argv:
        n = int(argv[argv.index("--tiles") + 1])
        os.environ["FLOCK2_TILES"] = str(n)
        return n
    for flag in agent_flags:
        if flag in argv:
            cfg = os.path.join(os.path.expanduser(argv[argv.index(flag) + 1]), "config.json")
            if os.path.exists(cfg):
                n = json.load(open(cfg)).get("tiles")
                if n:
                    os.environ.setdefault("FLOCK2_TILES", str(int(n)))
                    return int(os.environ["FLOCK2_TILES"])
    return None
