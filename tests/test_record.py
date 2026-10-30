"""The replay recorder (scripts/m2_record.py): a tiny scripted run round-trips through the file format and
the replay directory's index.json follows the files."""

import importlib.util
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

_spec = importlib.util.spec_from_file_location("m2_record", os.path.join(os.path.dirname(__file__), "..", "scripts", "m2_record.py"))
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

STEPS, EVERY = 60, 20


def _record(tmp_path, label, **extra):
    args = ["--policy", "scripted", "--willows", "3", "--meadows", "1", "--steps", str(STEPS), "--grass_every", str(EVERY),
            "--label", label, "--description", f"{label}: a tiny test run", "--dir", str(tmp_path)]
    for k, v in extra.items():
        args += [f"--{k}", str(v)]
    rc.main(args)


def test_replay_round_trips_and_the_index_follows(tmp_path):
    _record(tmp_path, "Test run A")
    path = tmp_path / ("test_run_a" + rc.EXT)
    assert path.exists()
    header, arrays = rc.read_replay(str(path))
    assert header["format"] == rc.FORMAT and header["meta"]["label"] == "Test run A" and header["meta"]["policy"] == "scripted"
    assert header["meta"]["steps"] == STEPS and header["meta"]["sheep"] == 12 and header["meta"]["willows"] == 3
    assert header["meta"]["action_names"] == list(rc.sim.ACTION_NAMES) and header["meta"]["drive_names"] == list(rc.D.NAMES)
    assert arrays["pos"].shape == (STEPS, 12, 2) and arrays["heading"].shape == (STEPS, 12)
    assert arrays["levels"].shape == (STEPS, 12, 4) and arrays["fat"].shape == (STEPS, 12) and arrays["wdist"].shape == (STEPS, 12)
    assert arrays["action"].shape == (STEPS, 12) and arrays["flags"].shape == (STEPS, 12)
    assert arrays["t"].shape == (STEPS,) and arrays["light"].shape == (STEPS,) and arrays["step_flags"].shape == (STEPS,)
    assert arrays["water"].shape == (48, 48) and arrays["regrow_map"].shape == (96, 96)
    assert header["grass_steps"] == [0, 20, 40, 60] and arrays["grass"].shape == (4, 96, 96)
    # decoded values are in the world's ranges
    assert arrays["pos"].min() >= 0 and arrays["pos"].max() <= 48 and arrays["heading"].min() >= 0 and arrays["heading"].max() < 1.0
    assert arrays["levels"].min() >= 0 and arrays["levels"].max() <= 1.15 + 1e-3 and 0 <= arrays["grass"].min() and arrays["grass"].max() <= 3.0 + 1e-2
    assert np.all((arrays["flags"] & 1) == 1), "every sheep stays alive through 60 scripted steps"
    assert np.all(np.diff(arrays["t"]) == 1)
    # the static world: the two ponds, a willow kind in the object table, the mud settings
    assert len(header["world"]["ponds"]) == 2 and arrays["water"].sum() == sum(p["tiles"] for p in header["world"]["ponds"])
    kinds = header["world"]["objects"]["kind"]
    names = header["world"]["objects"]["kind_names"]
    assert sum(names[str(k)] == "willow" for k in kinds) == 6 and len(header["world"]["objects"]["pos"]) == len(kinds)
    assert header["world"]["mud"] == {"inner": 0.0, "edge": 2.0} and len(header["world"]["fog_days"]) == rc.MAX_DAYS
    assert set(header["events"]) == {"deaths", "drinks", "first_near_water"}
    # the index lists the one file
    index = json.load(open(tmp_path / "index.json"))
    assert [e["file"] for e in index["replays"]] == ["test_run_a" + rc.EXT]
    assert index["replays"][0]["label"] == "Test run A" and index["replays"][0]["meta"]["steps"] == STEPS
    # a second recording is appended; re-recording the first replaces its entry in place
    _record(tmp_path, "Test run B", inner=8)
    index = json.load(open(tmp_path / "index.json"))
    assert [e["label"] for e in index["replays"]] == ["Test run A", "Test run B"]
    hb, ab = rc.read_replay(str(tmp_path / ("test_run_b" + rc.EXT)))
    assert hb["world"]["mud"]["inner"] == 8.0 and ab["regrow_map"].min() == 0.0 and ab["regrow_map"].max() == 1.0
    _record(tmp_path, "Test run A")
    index = json.load(open(tmp_path / "index.json"))
    assert [e["label"] for e in index["replays"]] == ["Test run B", "Test run A"]
    # a file that vanished drops out of the index on the next refresh
    os.remove(tmp_path / ("test_run_b" + rc.EXT))
    _record(tmp_path, "Test run A")
    assert [e["label"] for e in json.load(open(tmp_path / "index.json"))["replays"]] == ["Test run A"]
