# Handoff to the local (GPU) session

Written at the end of the cloud planning/prototyping session, 2026-10-22. Branch:
`claude/compassionate-babbage-jtl1o1`. Everything below is pushed.

## Where things stand

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
