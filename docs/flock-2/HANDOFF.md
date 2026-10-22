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
| Isometric world + sprites (the real one) | `flock/iso.py`, `flock/sprites_iso.py` | Done, tested (projection, rotation, facing); **not yet trained on**. |
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
- Perceptual weight 5 (vs 0.5) gave the first above-chance judge probes (~0.30 at step 200).
  Finals, if they landed before this note was written, are in `m0-toy-README.md`.

**What has not been shown**: anything at the default budget; anything on the isometric
renderer; the separate-encoder, stamp, latent and pixel cells; human readability.

## First things to do on the 4080

1. `pip install -U "jax[cuda12]"`, confirm `jax.devices()` is CUDA, `pytest -q`.
2. One default-size run on each renderer, same seed, perceptual weight 1.0:
   `python scripts/m0_run.py --renderer iso --perceptual_weight 1.0 --out_dir runs/m0/iso_perc1`
   and the same with `--renderer topdown`. Look at `gallery.png` and `views.png` before
   reading numbers. If iso is broken in a way the tests did not catch, the views will show it.
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

- Sprite sign-off: the isometric set was mocked and accepted "for now"; salt lick and rock
  are the weakest; water as ground state is new since the mock.
- Four sheep facings instead of two (cosmetic, later).
- Persistence cadence for M7.
- 96² vs 128² resolution, decided from the M1 profile.
