# Milestone 1 — the meadow: step 1 (sim, first-person views, viewer, scripted flock)

Plan: [`03-implementation-plan.md`](03-implementation-plan.md#milestone-1--meadow-drives-and-world-model).
This step builds the world the rest of M1 learns in; nothing is learned yet.

## What exists (`flock/meadow/`)

| File | What it does |
| --- | --- |
| `world.py` | State and generator: a 128×128 tile ground (grass biomass 0–3, shown as the four grass states; water tiles in two or three ponds) and continuous objects — a treeline along the north edge plus copses, three huts, the salt lick, flowers and mushrooms — and a flock of 40 with float positions and headings. |
| `sim.py` | One step for the whole flock: the owner's actions (turn left/right 22.5°, walk 0.12 or trot 0.35 tiles, stop, graze, drink, rest, head up); moves refused into water, off the meadow or into trees, huts and the salt lick; overlapping sheep pushed apart; grazing eats biomass, grass regrows; drinking within one tile of water; a day of 1,200 steps, the last 40 % night. |
| `views.py` | Every sheep's first-person strip (`flock/fpv.py`, 256×64, ~300°): its 48 nearest objects and sheep as billboards, other sheep facing how they move relative to the viewer; range 12 tiles by day (16 head up), 5 at night (8 head up); dark sky and far ground at night. |
| `scripted.py` | A rule-based flock (privileged, not learned): shelter and rest near cover at night, drink when thirsty, graze when hungry (moving to richer grass nearby), otherwise wander with cohesion. For watching now and for the world model's logged rollouts in step 2. |
| `viewer.py` | The human display: an isometric window following the flock (continuous positions, sheep facing their heading, darker at night), one sheep's strip and a status line. |

Scripts: `scripts/m1_watch.py` (runs the scripted flock and writes `watch.gif` and stills),
`scripts/m1_profile.py` (throughput). Tests: `tests/test_meadow.py`, `tests/test_fpv.py`.

```bash
python scripts/m1_watch.py --out runs/m1/watch --steps 2400 --every 12
python scripts/m1_profile.py
```

## Profile (RTX 4080, scripted flock of 40, 100-step scans, 5 repetitions)

| meadows | agent-steps/s without views | agent-steps/s with every sheep's view |
| --- | --- | --- |
| 1 | 1.0 M | 58,000 |
| 8 | 1.9 M | 287,000 |
| 32 | 2.6 M | **322,000** |
| 64 | (12.9 M — anomalous, not trusted) | 177,000 |

Rendering every sheep's view dominates, and throughput peaks around 32 meadows (64 is slower,
plausibly memory pressure). At ~320,000 agent-steps/s the plan's ~10⁷ logged steps for
world-model pretraining are under a minute of simulation, so the simulation will not limit M1.

## What the scripted flock looks like (2,400 steps, two days)

By day the flock grazes and wanders together, eats visible patches of grass down near where it
feeds, and goes to water when thirsty; at night it gathers among trees and rests (42 % resting
early in the night, 90 % by dawn); after the night's fast thirst is high (0.86 on average at
dawn) and the flock heads back to water. First fixes were needed (recorded here because they
will matter for learned sheep too): refusing any move that brought two overlapping sheep closer
froze a dense flock, so overlaps are now pushed apart instead; the bookkeeping needs rose too fast
for the day length; and a single nearest tree became a queue, so each sheep now has a stable
preference among nearby cover.

## Known issues and open choices

- **Billboard scale: 24 px per tile** (owner, 2026-10-23). At the isometric art's 16 px per tile
  a sheep was ~1.4 tiles tall against a 0.5-tile eye and any neighbour filled the view; now a
  sheep is ~0.9 tiles tall and neighbours at ordinary spacing look like sheep among sheep. A
  neighbour *touching* the viewer (~0.6 tiles, e.g. in the night huddle) still fills the strip,
  which is what the geometry gives.
- Ground right in front of the viewer is the tile art magnified (blocky).
- The scripted flock's thirst and hunger are bookkeeping for the script only; the learned
  drives arrive in step 3.
- The display is a GIF written by a script, not yet a live window.

# Step 2 — logged rollouts and world-model pretraining (2026-10-23)

**Data** (`flock/meadow/rollouts.py`, `scripts/m1_log.py`): four shards of the scripted flock,
each 32 fresh meadows × 8 recorded sheep × 1,024 steps from a random time of day after a 200-step
warm-up — 1.05 M first-person strips with the action taken after each, packed 4 bits per pixel
(8 GB, on the WSL filesystem: `~/flock2_data/m1_rollouts`). Logging takes ~40 s in all.

**Model** (`flock/wm.py`): a DreamerV3-style world model
([Hafner et al. 2023](https://arxiv.org/abs/2301.04104)) — five stride-2 convolutions over the
one-hot strip (64×256 → 2×8), a layer-normalised GRU deterministic state (384) and 32 categorical
latents of 16 classes (1 % uniform mix, straight-through samples), the action entering the
transition, a mirrored decoder over the 16 colours, per-pixel cross-entropy summed over the strip
and KL balancing (0.5 dynamics / 0.1 representation, 1 nat free bits) as in the paper.
7.9 M parameters. Trained with Adam 1e-4 on batches of 4 sequences × 64 steps — DreamerV3's
pixels per batch (16 × 64 at 64×64); 8 × 64 ran out of the 4080's 16 GB. 20,000 steps in ~30 min
(`scripts/m1_wm_train.py`; checkpoint `~/flock2_data/m1_wm/run1/model.pkl`).

**Open-loop test on the held-out shard**: observe 16 strips, then imagine 48 from the true actions
alone (16 fixed windows):

| steps ahead | pixel accuracy | object-pixel recall (sheep, trees, huts, …) |
| --- | --- | --- |
| 1 | 0.87 | 0.75 |
| 4 | 0.83 | 0.68 |
| 16 | 0.70 | 0.56 |
| 32 | 0.63 | 0.50 |
| 48 | 0.57 | 0.49 |

Long horizons are noisy between evaluations (16 windows). By eye (`runs/m1/wm_run1/open_loop_020000.png`):
near sheep keep their legs and faces, distant sheep survive as small white shapes rather than
being erased (the IRIS-style failure the plan's criterion guards against), and imagined frames
bring plausible sheep into view; small trees and flowers are still often lost.

**Surprise** (`scripts/m1_wm_surprise.py`: KL(posterior ‖ prior) per step on held-out sequences):
routine mean 28 nats per step, 99th percentile 64. A teleport — half-way through a sequence the
strips switch to a sheep in another meadow — gives 86, about 3× the routine mean, falling back
to routine within five steps; but that is only 1.3× the routine 99th percentile, so routine viewing
has frequent peaks of similar size. Surprise follows information: turning is the most surprising
action (~37; new parts of the panorama come into view), resting (20) and night (19 vs 28 by day)
the least, and dawn — the sky switching from dark to light — gives one of the largest spikes:
the model has no clock and cannot anticipate it.

Against the plan's world-model criteria *(judgement)*: small distant objects are kept (partly met;
recall ~0.5 at long horizons); surprise rises on genuine novelty but is not yet *selective* — the
routine peaks are close to the teleport. Not blocking for step 3; worth revisiting with more data
(the plan's 10⁷ steps) and a longer run before M1's evaluation.
