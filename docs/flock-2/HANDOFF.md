# Handoff

## Update — M1 in progress (2026-10-24)

**Where things are.** M1 steps 1–2 are done; step 3 (drives and the actor-critic) is built and
tested, but no learned behaviour yet. Everything — what was built, every run with its numbers,
the diagnosis and the agreed next step — is in [`m1-meadow-README.md`](m1-meadow-README.md),
with the run logs and figures in [`m1-results/`](m1-results/) and the model in
[`m1-model.svg`](m1-model.svg). Read that README first.

**Code added for M1.** `flock/fpv.py` (first-person strip), `flock/meadow/` (world, sim,
drives, views, scripted flock, viewer, rollout logger), `flock/wm.py` (world model with
interoception and reward head), `flock/ac.py` (actor-critic in imagination); scripts
`m1_watch`, `m1_profile`, `m1_log`, `m1_wm_train`, `m1_wm_surprise`, `m1_wm_probe`, `m1_agent`,
`m1_agent_report`. Tests: `tests/test_meadow.py`, `test_wm.py`, `test_ac.py`, `test_fpv.py`
(CPU: `JAX_PLATFORMS=cpu pytest -q tests/test_meadow.py tests/test_wm.py tests/test_ac.py`).

**Current approach, agreed with the owner.** World model and policy learn together from birth on
the sheep's own behaviour (no scripted pretraining), with a curiosity reward from the
disagreement of five heads. birth1 (curiosity about the next view) fixed the false drinking
belief but learned to spin in place, then to run; birth2 (curiosity about the body's stores,
varied starts and daily meadow resets, per-action energy use) improved the model but grazed and drank where there is nothing, led by curiosity; birth3 (Keramati & Gutkin's convex drive, an innate bad taste for ingesting nothing, curiosity 0.02) did not beat random either. After the outside review (`review-2026-10-24.md`): a fixed evaluation with baselines, two actor-critic fixes (two seeds, still at random level), a bandit learning test, and privileged PPO, which beats the scripted flock — so the meadow is learnable and the failure is in learning from imagination. Details,
numbers and reasoning in the README's step 3 sections from "The change of approach" on.

**Gotchas added (late 2026-10-24).** Never run a `wsl` command in the foreground that might
exceed the tool's 10-minute limit: when one did, every background `wsl` job was terminated at
the same moment (exit 15). Start anything long with `run_in_background`. CPU-only jobs (grid
runs, probes) can run alongside the GPU job; set `JAX_PLATFORMS=cpu` for them. **Memory:** WSL has 15 GB; when it runs out the kernel's OOM killer takes the
whole distro down with every job in it (2026-10-24, twice). The meadow agent's replay alone is
~4 GB of host RAM; each CPU JAX job compiles its own copy of the model. Keep to the GPU job plus
about three CPU jobs.

**Gotchas added.** Data and checkpoints live on the WSL filesystem (`~/flock2_data/`), not in
git; `runs/` is git-ignored, so results worth keeping are copied to `docs/flock-2/m1-results/`.
Compile the training steps before the first acting step in a long-running loop, or XLA's
autotuning can run out of GPU memory. Several shells mangle heredocs containing non-ASCII text;
write patch scripts to files instead. `pkill -f` with a plain pattern from `wsl bash -c` kills
its own shell: use a bracket pattern (`m1_agen[t]`).

## Update — end of the first local GPU session (2026-10-23)

**M0 is closed.** Result and every grid: `m0-toy-README.md` (§ Result, grids 1–9); summary in
the plan's M0 section. In one line: sheep reliably learn to communicate with drawings; without
a resemblance reward the drawings are an arbitrary but consistent code, not pictures.

**What changed since the cloud handoff below**

- World: isometric only for real work, now **16×8 px tiles, 128×128 view** (16 tiles across),
  **seven referent types** (salt-lick cube, pine, star flower, fox, hut, pond, mushroom), six
  sprite variants each, two held out for evaluation. Art: `flock/sprites_iso.py`; mocks in
  `flock/art/`. The top-down renderer stays at 64×64 as the old baseline.
- Drawing: black 2 px strokes drawn at 32×32 (the drawing's size in the view), pen-lift +
  length cost with a delayed start, at least one stroke, an on/off stroke gate. Reading
  channel (`game.transmit`): re-centring, jitter, 90° rotation of ground stamps, clutter,
  reduced resolution. Config defaults are still the old ones; the M0 closing cells
  (`final4_*`, `final7_*` in `scripts/m0_grid.py`) are the settings to carry forward.
- Measurement: `flock/toy/examiner.py`, an independent outline-literate classifier trained once
  per world (cached in `runs/examiner/`); every iso run saves `drawings.npz` and is graded;
  `scripts/m0_examine.py` re-grades saved drawings.
- Design: Q9 addendum (first-person billboard view as a ~300° strip, look-down reading of ground
  marks, steps toward language), agreed with the owner for M1+.
- Spike (not wired in): `flock/fpv.py`, a first-person cylindrical renderer, with
  `scripts/bench_fpv.py` comparing its throughput with the isometric view — the benchmark the
  addendum asks for before adopting it.

**Gotchas learned here**

- JAX CUDA wheels are Linux-only: run under WSL2 (`~/.venvs/flock2`, uv Python 3.12) against
  the checkout via `/mnt/c/...`; `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Run **one GPU job at a time**: four concurrent processes crashed a grid with a CUDA error.
- **Directory names are case-insensitive** on `/mnt/c`: grid cell names must differ by more than
  case (the grid script now asserts it).
- GPU training is **not bit-reproducible**: the same seed can draw differently; judge by seeds.
- Look at galleries before numbers: every cheat found in M0 (dot constellations, faint strokes,
  blank signs, position codes, a layout cue from touching target squares) was spotted there.

**Next** (owner's direction): the first-person renderer benchmark, then M1 (meadow, drives,
world model) on whichever view wins, with information gaps (occlusion, the blind wedge) built
in from the start.

---

The original handoff from the cloud session follows, kept for history; it is out of date in
places (e.g. the isometric renderer has since been trained on throughout M0). For the current
design read [`CURRENT-DESIGN.md`](CURRENT-DESIGN.md).

Written at the end of the cloud planning/prototyping session, 2026-10-22. Branch:
`claude/compassionate-babbage-jtl1o1`. Everything below is pushed.

## Where things stood (2026-10-22)

**Docs** (`docs/flock-2/`): research notes (9 topics, cited), survey, design-question
recommendations (Q1–Q10), staged plan (M0–M8), M0 run guide. Read `README.md` →
`03-implementation-plan.md` → `m0-toy-README.md` in that order; `02-design-questions.md`
when a design choice comes up.

**Code**: Milestone 0 only, in JAX (`flax.nnx`, `optax`). Nothing beyond M0 exists.

| Area | File(s) | Status |
| --- | --- | --- |
| Palette | `flock/palette.py` | Done. |
| Top-down world + sprites (baseline) | `flock/world.py`, `flock/sprites.py` | Done, tested; the CPU results were obtained with it. |
| Isometric world + sprites (the real one) | `flock/iso.py`, `flock/sprites_iso.py` | Done, tested (projection, rotation, four facings, hashed grass variants); **not yet trained on**. |
| Stroke rasteriser | `flock/raster.py` | Done, tested, gradients verified. |
| Networks | `flock/nets.py` | Done: encoder/decoder, stroke/latent/pixel senders, message encoder, receiver with canvas pathway. |
| Toy game | `flock/toy/{env,env_iso,envs,game,train,metrics,gallery}.py` | Done; end-to-end smoke-tested on both renderers. |
| Scripts | `scripts/m0_run.py`, `scripts/m0_grid.py` | Done. Grid never run at full size. |
| Tests | `tests/` | 23 pass on CPU (`pytest -q`, ~2 min). |

**What has actually been shown** (CPU, 1/40 of the default budget, top-down renderer):

- Communication emerges reliably with the canvas pathway and an ink ramp (acc 0.70 vs chance
  0.25; blank 0.30; swapped 0.13; CIC 3.4).
- Game loss alone gives a positional code, not iconic drawings (judge probe at chance).
- A trainable encoder plus a perceptual loss computed through it collapses the drawing to
  nothing; the loss now goes through the frozen judge.
- With the judge-based perceptual loss, all three variants (weight 5; fine-tuned encoder;
  300-step sender warm-up) reach judge probes of ~2× chance; the warm-up is best (rank 0.53
  early, 0.40 final, acc 0.80, CIC 4.6). No gallery is human-readable yet: drawings are one or
  two non-referent colours with small shape differences. Full table and likely causes in
  `m0-toy-README.md` §"First results".

**What has not been shown**: anything at the default budget; anything on the isometric
renderer; the separate-encoder, stamp, latent and pixel cells; human readability.

## First things to do on the 4080

1. `pip install -U "jax[cuda12]"`, confirm `jax.devices()` is CUDA, `pytest -q`.
2. One default-size run on each renderer, same seed, perceptual weight 1.0 with the sender
   warm-up (the best CPU variant):
   `python scripts/m0_run.py --renderer iso --perceptual_weight 1.0 --sender_warmup_steps 1000 --out_dir runs/m0/iso_warm`
   and the same with `--renderer topdown`. Look at `gallery.png` and `views.png` before
   reading numbers. If iso is broken in a way the tests did not catch, the views will show it.
   If galleries are still single-colour blobs at full budget, the perceptual loss is the
   first thing to change (see `m0-toy-README.md` §"First results", item 6): try a spatial
   feature-map loss and/or a colour term at the referent's pixels before touching anything
   else.
3. If both learn: run the grid, `python scripts/m0_grid.py --out runs/m0/grid --seeds 3`,
   adding `--renderer iso` to the cells (edit `CELLS` or add a flag — it is a one-line change
   in `scripts/m0_grid.py`).
4. Do the quiz with two people on the best iconic cell.
5. Write the M0 result into `m0-toy-README.md` and update the plan's M0 "what a pass buys".

## Things to know

- **Throughput on CPU** was ~1–3 s per game step at batch 32 (4 cores). Expect the 4080 to
  be 30–100× faster; if it is not, the `jax.lax.scan` over sprites in `iso.render_view` and
  the per-episode `vmap` in `make_batch` are the places to look.
- **The receiver's canvas pathway** (`Receiver.canvas_read`) pools the 4×4 feature-map cells
  under the canvas. It only tells the receiver *where* the message is. Removing it killed
  emergence on CPU; it may be unnecessary at full budget — worth one ablation, not a priority.
- **Perceptual loss** (`game.perceptual_loss`) compares judge embeddings of (drawing stamped on
  grass where the object was) vs (the sender's actual view). It is a global-embedding cosine;
  a feature-map (spatial) version may be a better resemblance signal if probes stay flat.
- **Sender warm-up** (`--sender_warmup_steps`) trains the sender on the perceptual loss alone
  first. On CPU, 300 warm-up steps did not move the judge probe by themselves.
- **Ink** ramps from 0 over `--ink_ramp_steps`. A constant ink penalty erased drawings.
- **Iso layout constants** live at the top of `flock/toy/env_iso.py` (slots, canvas position,
  "ahead" offset). The canvas sits at the bottom-centre of the receiver's view, below the
  sender sprite in bubble mode. If you move it, keep it clear of the candidate slots (the test
  checks this).
- **Replay/eval sizes**: `eval_batch` 256 default; the iconicity probe fits a logistic
  regression on 2,048 object views each eval — cheap on GPU, noticeable on CPU.
- Two stray directories (`/v3_tuned`, `/v3_warm`) were created at the cloud sandbox root by a
  shell slip; nothing refers to them. Not in the repo.

## Open design items (none block M0)

- Sprite set: isometric, accepted "for now". Since the mock: water is a ground state, grass is
  drawn as stalks with four hashed variants per state, salt lick and rock were redrawn, and
  sheep and fox have four facings (front/back × left/right; `sprites_iso.py` documents the
  facing index). Further art changes are purely cosmetic.
- Persistence cadence for M7.
- 96² vs 128² resolution, decided from the M1 profile.
