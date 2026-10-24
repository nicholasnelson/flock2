# FLOCK 2.0 — current design

*The design as it stands in the code (last checked 2026-10-24, late, after the third review).
The plan (`03-implementation-plan.md`), the design questions (`02-design-questions.md`) and the
milestone READMEs record how it got here; where they disagree with this page, this page is
current. When the design changes, change this page in the same commit and add a line to the
decision log at the end.*

## Where the project is

- **M0 (toy drawing game): closed.** Sheep reliably learn to communicate with drawings; without a
  resemblance reward the drawings are an arbitrary but consistent code, not pictures. Emergence
  relied on a stack of nine channel interventions (below), not on the game alone. The human
  quiz was not run. Record: `m0-toy-README.md`.
- **M1 (meadow, drives, world model): in progress.** World, views, drives, world model, actor-
  critic and evaluation built and tested. No sheep learning in imagination yet beats a random
  flock on the fixed evaluation; PPO on privileged features beats the scripted flock, so the
  meadow is learnable; PPO on the world model's state (the hybrid) beats random (−0.79 / −0.75)
  but cannot find water, and probes show the world model's state lacks the direction to water.
  The grid world (below) is now the test bed for the learner: there, too, the imagination
  learner fails, and the world model does not rank actions (S3). Records: `m1-meadow-README.md`,
  three reviews (`review-2026-10-24.md`, `…b.md`, `…c.md`).

## The meadow (`flock/meadow/world.py`, `sim.py`)

- 128 × 128 tiles. Ground per tile: grass biomass 0–3 (four drawn states), or water (two or three
  ponds). Objects at float positions: 90 trees (treeline along the north edge plus copses),
  3 huts, 1 salt lick, 24 flowers, 10 mushrooms. Trees, huts and the salt lick block movement
  (radius 0.35 / 0.7 / 0.35 tiles); flowers and mushrooms are scenery.
- A flock of 40 sheep (radius 0.3) with float positions and headings (0 = north, clockwise).
- A day is 1,200 steps: the first 60 % light, the last 40 % night.
- **Actions (9):** stop, walk (0.12 tiles), trot (0.35 tiles), turn left / right (22.5°), graze,
  drink, rest, head up. Moves into water, off the meadow or into blocking objects are refused;
  overlapping sheep are pushed apart. Grazing bites 0.05 of biomass when the tile has ≥ 1; grass
  regrows toward 3. Drinking works within one tile of water.
- The salt lick's mineral need, the mushroom's "not to eat" role and the fox are not built.

## What a sheep sees (`flock/fpv.py`, `flock/meadow/views.py`)

- A first-person cylindrical strip, **256 × 64** pixels, **300°** wide (a 60° blind wedge
  behind), 16 palette colours; horizon at row 24; eye height 0.5 tiles.
- Billboard sprites at 24 px per tile; each sheep renders its 48 nearest objects and sheep,
  painted far to near (real occlusion); other sheep face the way they move.
- View range 12 tiles by day (16 with head up), 5 at night (8 with head up); a darker sky and a
  flat far colour beyond range.
- Humans see the isometric display (`flock/meadow/viewer.py`): a window following the flock,
  one sheep's strip, a status line. A live window is not built (GIFs via `scripts/m1_watch.py`).

## What a sheep feels (`flock/meadow/drives.py`)

| Level (set-point 1, clipped [0, 1.15]) | Falls per step | Restored by |
| --- | --- | --- |
| energy | by action: rest 0.0003; stop, graze, drink, head up 0.0005; turn 0.0006; walk 0.0008; trot 0.0016 | grazing where grass ≥ 1: +0.012 |
| hydration | 0.0008 | drinking within one tile of water: +0.04 |
| warmth | 0.0002 by day; at night 0.0016 in the open, 0.0007 within 3 tiles of a tree or hut | resting: +0.003, +0.006 beside cover |
| social | instantaneous, not a store | sheep in view / 8, capped at 1 |
| interest | 0.0004 | entering an 8 × 8-tile cell: +0.08 / √(1 + previous visits) |

- **Drive:** D = (Σᵢ |1 − hᵢ|⁴)^(2/4) — Keramati & Gutkin's form with n = 4, m = 2 (n > m > 1):
  it follows the worst deficit and grows as its square.
- **Taste:** an innate aversive signal for ingesting nothing (drinking away from water, grazing
  grass < 1): −0.05 in the step's reward, felt as −1 for one step. Not a drive; never pleasant.
- **Reward of a step:** −D(levels after the step) − 0.05 × bad taste (`drives.reward`).
- **Felt (6 numbers):** the five levels and the taste — the world model's interoceptive input.

## The sheep's brain (`flock/wm.py`, `flock/ac.py`) — diagram: `m1-model.svg`

One set of weights shared by all sheep; each sheep keeps its own recurrent state.

- **World model** (DreamerV3-style RSSM, 8.7 M parameters plus 1.2 M in the curiosity ensemble): encoder of five stride-2
  convolutions (64 × 256 → 2 × 8) to a 512 embedding, plus a linear layer from the 6 felt
  numbers added to it; GRU deterministic state 384; 32 categorical latents × 16 classes (1 %
  unimix, straight-through); decoder mirroring the encoder. Heads on the state (896 numbers):
  interoception (predicts the felt numbers), reward (255 symlog bins, two-hot), and a
  **curiosity ensemble** of five heads, each with one output branch per action, predicting the
  change in energy, hydration, warmth and interest (×100). Loss: pixel cross-entropy summed over
  the strip, KL balancing 0.5 / 0.1 with 1 nat free bits, 1,000 × level squared error, reward
  cross-entropy, ensemble squared error. Adam 1e-4.
- **Actor-critic** (0.6 M): actor and critic MLPs (896 → 256 → 256 → 9 / 1); the critic's
  output a scalar trained by squared error (a 255-bin two-hot symlog head is an option, off);
  slow critic EMA 0.98, as a regulariser only. Trained in imagination: 32 steps rolled from the
  prior from states inferred on real sequences; rewards from the reward head, **centred** (a
  running mean subtracted; Naik et al. 2024), plus curiosity = 0.02 × ensemble disagreement /
  (its mean at the first update, then frozen). λ-returns (γ a setting, 0.997 by default; λ 0.95)
  bootstrapped by the critic itself; REINFORCE on advantages divided by max(1, S), S an EMA of
  the 5–95 % range of the **returns** (DreamerV3's); a **replay value loss** (weight 0.3) on
  λ-returns of real replayed rewards; entropy 3e-4; Adam 3e-5 (critic rate a setting). Health
  readings: mean centred value at start states; correlation of value with real return.
  (These defaults follow the third review and DreamerV3's current `configs.yaml`; the meadow runs
  recorded so far used the earlier settings.)

## The grid world (`flock/grid/`) — the learner's test bed

One sheep per world, 16 × 16 walled grid, one pond, regrowing grass on fertile clumps; actions
north / south / east / west / stay / eat / drink; two needs, food (−0.004 per step, +0.08 per
bite) and water (−0.006, +0.1 per sip beside the pond); the same drive (n = 4, m = 2) and bad
taste; an 8 × 8 top-down view (or the whole map, `--view full`); the same `wm.py` and `ac.py` at
~0.5 M parameters (`brain.GRID`). Evaluation (`scripts/grid_eval.py`): 64 fixed worlds, 1,000
steps, standard errors. Baselines: random −1.12, scripted (whole-map, privileged) −0.10. Rungs:
`grid_true_ac.py` (the actor-critic on the true simulator), `grid_agent.py` (learning from
birth with the world model), `grid_gap_actions.py` (does the model rank actions?),
`grid_probe.py` (does it remember the pond?).

## How it learns (`scripts/m1_agent.py`)

- **From birth:** the world model starts from random weights; no scripted data.
- Per iteration: 4 meadows × 40 sheep act for 16 steps, every sheep from its own model state;
  8 sheep per meadow logged, each keeping a 64-step sliding window pushed to a replay of 8,192
  sequences; 64 world-model updates (batches of 4 × 64) and 64 actor-critic updates.
- **Infancy:** for the first 300 iterations only the world model learns; the policy is uniform.
- **Varied starts:** every meadow starts with a new layout, the flock at a random place, a random
  time of day and levels drawn from [0.2, 1.0], and is started fresh every 1,200 steps
  (staggered); no replay sequence spans a restart.

## How it is judged (`scripts/m1_eval.py`)

The same 8 meadows × 40 sheep (seeds 10,000 + i) from dawn with `generate`'s starting levels, two
days; levels at each dusk and dawn (mean, 10th percentile), the share depleted (energy,
hydration or warmth ≤ 0.05), reward per step, feeding success, bad taste, the day/night action
mix, resting near cover at night. Baselines (2026-10-24): random −1.02 per step, rest-always
−1.06, scripted −0.41 (20 % depleted by dawn of day 3). Upper bound: PPO on privileged features
(`scripts/m1_ppo.py`), −0.39 after 2,000 updates (9 % depleted). The learned sheep (both seeds of
the post-review configuration) score −1.07 and −1.11.

## M0's channel stack (carried to M4)

Black 2 px strokes drawn at 32 × 32; a pen-lift cost plus a length cost with a delayed start; a
hard on/off stroke gate; at least one stroke; and the reading channel's re-centring, jitter,
90° rotation of ground stamps, clutter and reduced resolution. Resemblance rewards off by
default. Each closed a cheat in the toy; which are load-bearing in the meadow is untested.

## Decision log

| Date | Decision | Where recorded |
| --- | --- | --- |
| 2026-10-22 | Plan M0–M8; JAX; local RTX 4080 only | `03-implementation-plan.md` |
| 2026-10-23 | Isometric 16 × 8 tiles, 128² view; seven referents; stroke medium; M0 closed with resemblance off | `m0-toy-README.md` |
| 2026-10-23 | Agents see a first-person 300° strip (256 × 64); humans the isometric display; continuous movement; the 9-action set; billboards 24 px per tile | `02-design-questions.md` Q9 addendum; `m1-meadow-README.md` |
| 2026-10-23 | DreamerV3-style world model pretrained on a scripted flock | `m1-meadow-README.md` § Step 2 |
| 2026-10-24 | Reward −D instead of drive reduction; a reward head instead of computing reward from predicted levels | § Step 3 |
| 2026-10-24 | Learn from birth (no scripted pretraining) with a curiosity ensemble; per-action output branches | § The change of approach; § birth1 |
| 2026-10-24 | Energy use by action; body curiosity (not social); varied starts with daily restarts | § birth2 |
| 2026-10-24 | Drive exponent m = 2 (was 1, a departure from Keramati & Gutkin); innate bad taste; curiosity 0.02 | § birth3 |
| 2026-10-24 | After the review: advantages (not returns) set the actor's scale; curiosity scale frozen after infancy; fixed evaluation with baselines; sanity ladder (PPO upper bound, bandit test) | `review-2026-10-24.md`; § After the review |
| 2026-10-24 | Hybrid rung (PPO on the world model's state); latent probes; reward centring; two-hot critic as an option | `review-2026-10-24b.md`; commit log; M1 README § The grid world |
| 2026-10-24 | Grid world adopted as the learner's test bed (owner's proposal); `wm.Config` makes one model serve both worlds | M1 README § The grid world |
| 2026-10-24 | Third review: one critic for target and baseline, return normalisation, replay value loss, γ as a setting; success criteria written before runs; three seeds per comparison | `review-2026-10-24c.md`; M1 README § Pre-registered criteria |
