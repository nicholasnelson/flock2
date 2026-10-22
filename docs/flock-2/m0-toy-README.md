# Milestone 0 — the toy drawing game: how to run it and what to look at

Code: `flock/` (shared world, sprites, rasteriser, nets) and `flock/toy/` (the game, training,
metrics, gallery). Plan and success criteria: [`03-implementation-plan.md`](03-implementation-plan.md#milestone-0--the-toy-drawing-game).
Design decisions it tests: Q2 (action space), Q8 (perception scale), and the shared-encoder
hypothesis.

## Setup (desktop, RTX 4080)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pip install -U "jax[cuda12]"        # GPU JAX; the CPU wheel from the first line is replaced
python -c "import jax; print(jax.devices())"   # expect a CudaDevice
pytest -q                           # ~2 min on CPU
```

On Windows, JAX's CUDA wheels are Linux-only: use WSL2 (the 4080 is visible there with the
normal Windows driver). The working setup is a uv-managed Python 3.12 venv in WSL
(`uv venv -p 3.12 ~/.venvs/flock2`) running against this checkout through `/mnt/c/...`, with
`XLA_PYTHON_CLIENT_PREALLOCATE=false` so two runs can share the GPU. A default-size run takes
about a minute; the full grid (15 cells x 3 seeds) under an hour.

## One run

```bash
python scripts/m0_run.py --out_dir runs/m0/shared_perc --perceptual_weight 1.0 --renderer iso
```

`--renderer iso` is the isometric world (the one the project will use); `--renderer topdown`
is the original top-down renderer, kept because it is the one the CPU results below were
obtained with. Both expose the same game; only the pixels differ. Run both on the first GPU
session and keep whichever is not broken if they disagree.

Defaults: 4 candidates, bubble mode, stroke sender with 8 strokes, shared frozen encoder, 1,500
pre-training steps then 4,000 game steps at batch 64, eval every 500. `--help` lists every
field of `flock.toy.train.Config`.

Outputs in `out_dir`:

| File | What it is |
| --- | --- |
| `log.jsonl` | Per-step training records and per-eval metric records. |
| `summary.json` | Config plus the final eval. |
| `gallery.png` | Rows = object types (sprite first), columns = drawings the sender made for that type. **This is the first thing to look at.** |
| `views.png` | Six sender views (left) and the matching receiver views with the canvas composited (right). Sanity check of the game itself. |
| `quiz.png` + `quiz_key.json` | 60 shuffled drawings for the human-readability test; the key says which type each was. Guess first, then open the key. |

## The ablation grid

```bash
python scripts/m0_grid.py --out runs/m0/grid --seeds 3
```

Runs the cells in `scripts/m0_grid.py` (shared/separate × perceptual on/off, bubble/stamp,
latent and pixel senders, auxiliary losses) and prints a table. Resumable: re-running skips
finished cells. `--cells a,b` selects cells; `--quick` is a smoke run.

## Reading the metrics

Chance for accuracy is `1 / n_candidates` (0.25 by default).

- **Used.** `acc` well above `acc_blank` and `acc_swapped`, and `cic_kl` clearly above zero.
  `acc_blank` is what the receiver does with no message — it should be at chance; if it is
  above chance the receiver has found a cue in the layout and the game needs fixing.
  `acc_swapped` is the accuracy when a drawing of a *different* referent is substituted; it
  should drop toward chance. A high `acc` with `acc_swapped` also high means the receiver is
  not reading the drawing.
- **Iconic.** `judge_probe_acc`: a linear object-type classifier fitted on a *frozen*
  perception encoder (never trained on the game), applied to the drawings stamped on plain
  ground where the object would be. Chance is 1/6. `judge_rank_acc`: nearest per-type
  reference embedding. `own_*` are the same probes through the model's own encoder — less
  honest in the shared condition because that encoder is also the receiver's, so expect
  `own_` ≥ `judge_`. The hypothesis "the shared encoder makes iconicity emerge" is supported
  only if `judge_probe_acc` is well above chance in the `shared_noperc` cell.
- **Readable.** The quiz. Target from the plan: ≥ 35% correct at 6-way (chance 17%), which is
  roughly what the perceptual loss achieved for humans in Mihai & Hare's study at 10-way.

## First results (CPU sandbox, small scale — not the real grid)

Sizes: 300 pre-training steps, 600 game steps, batch 32, 4 candidates, bubble mode, stroke
sender, perceptual weight 0.5. Roughly 1/40 of the default budget; indicative only.

| Variant | acc | acc_blank | acc_swapped | CIC (KL) | judge_probe_acc | Note |
| --- | --- | --- | --- | --- | --- | --- |
| v1: frozen encoder, no canvas pathway | 0.23 | 0.23 | 0.21 | ~0 | 0.11 | No emergence. Ink shrank 21 → 3. |
| v2: + receiver reads canvas cells, ink ramp (frozen) | **0.70** | 0.30 | 0.13 | **3.4** | 0.11 | Communication emerges; positional code; not iconic. |
| v2: same, encoder fine-tuned | 0.29 | 0.29 | 0.27 | ~0 | 0.11 | Collapsed: ink → 0.001. |

What was learned, and what changed because of it:

1. **The receiver needs a path from the canvas to its decision.** With the only route being
   the frozen encoder's 256-d global embedding, nothing emerged in 600 steps. Pooling the
   shared feature-map cells under the canvas into the receiver's context ("look at the bubble")
   fixed it immediately. This keeps the shared encoder as the reading pathway; it only tells
   the receiver *where* the message is, which a sheep would know.
2. **Ink cost must ramp from zero.** A constant ink penalty erased the drawing before it was
   useful (ink 21 → 3 with no accuracy gain). Now ramps linearly over `ink_ramp_steps`.
3. **The resemblance loss must be measured by a frozen judge.** With the encoder trainable and
   the perceptual loss computed through it, the encoder minimised the loss by ignoring the
   canvas and the drawing collapsed to nothing. The loss now uses the frozen pre-trained
   perception encoder (the judge). That is also the honest meaning of "resemblance as
   perception sees it".
4. **The literature's prediction held at this scale:** the v2 drawings are a consistent
   positional code (the same blobs shifted slightly per referent), causally used, with no
   referent colours and probe accuracy at chance — Mihai & Hare's game-loss-only result.

Third round, same sizes, all with the judge-based perceptual loss (finals at step 600):

| Variant | acc | acc_blank | acc_swapped | CIC | judge_probe | judge_rank | Gallery |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v3 perc5: frozen, perceptual 5.0 | 0.77 | 0.26 | 0.13 | 3.1 | 0.30 | 0.31 | One large orange shape for every type, small variations. |
| v3 tuned: fine-tuned encoder, perceptual 0.5 | 0.68 | 0.17 | 0.19 | 4.2 | 0.33 | 0.31 | Pond and tree: a blue blob; rock: a dark blob; others near-blank with a blue line. Took off only after step 400. |
| v3 warm: frozen, perceptual 1.0, 300-step sender warm-up | **0.80** | 0.27 | 0.18 | **4.6** | 0.34 | **0.40** (0.53 at step 200) | Dark-brown shapes; pond/tree get a tall blob, flower a small mark. |

Chance: acc 0.25, probes 0.17.

5. **Iconicity is measurable but weak at this budget, and not human-readable in any cell.**
   The judge probes sit at roughly 2× chance in all three v3 runs, so the resemblance term
   does something, but every gallery is dominated by one or two colours that are not the
   referent's (orange, dark brown, blue) and shapes that differ only slightly by type. The
   warm-up gives the strongest start (rank 0.53 at step 200) and then erodes under game
   pressure toward a code — the drift the human and sketch-game literature describes.
6. **Likely causes, for the GPU session to test:** (a) the judge is weak (300 autoencoder
   steps); (b) the perceptual loss is a global-embedding cosine, which rewards "looks like
   that view" only loosely — a feature-map (spatial) loss or an explicit colour term at the
   referent's pixels would reward colour fidelity directly; (c) 600 steps at batch 32 is
   1/40 of the budget. None of these is a reason to change the design before the real run.
7. **Fine-tuning the encoder works once the loss goes through the judge**, but starts
   later; keep `freeze_encoder=True` as the default and treat fine-tuning as a cell.

## GPU results (RTX 4080, default sizes, isometric renderer)

Default sizes: 1,500 pre-training steps, 4,000 game steps at batch 64, eval batch 256. Chance:
acc 0.25, probes 0.17. Mean ± s.d. over 3 seeds.

### Grid 1 — round tree (2026-10-22, commit `1f07d77`)

| cell | acc | acc_blank | acc_swapped | CIC | judge_probe | judge_rank | ink |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shared_noperc | 1.00±0.00 | 0.22±0.04 | 0.09±0.02 | 7.0±0.5 | 0.17±0.02 | 0.19±0.03 | 6.5 |
| shared_perc | 0.99±0.02 | 0.20±0.02 | 0.08±0.02 | 10.3±1.7 | 0.72±0.03 | 0.71±0.04 | 15.5 |
| separate_noperc | 1.00±0.00 | 0.24±0.04 | 0.09±0.03 | 4.4±1.3 | 0.17±0.01 | 0.16±0.01 | 1.1 |
| separate_perc | 0.99±0.01 | 0.24±0.02 | 0.09±0.01 | 10.5±0.3 | 0.76±0.12 | 0.73±0.08 | 10.6 |
| shared_perc_stamp | 0.99±0.00 | 0.23±0.03 | 0.08±0.01 | 12.1±1.3 | 0.74±0.03 | 0.70±0.03 | 12.1 |
| shared_noperc_stamp | 0.99±0.01 | 0.25±0.01 | 0.08±0.03 | 6.6±1.4 | 0.32±0.10 | 0.30±0.13 | 5.2 |
| latent_noperc | 0.74±0.37 | 0.23±0.02 | 0.11±0.07 | 8.0±5.7 | 0.16±0.01 | 0.16±0.01 | 4.7 |
| latent_perc | 0.99±0.00 | 0.22±0.00 | 0.08±0.01 | 11.9±0.4 | 0.77±0.03 | 0.76±0.04 | 35.9 |
| pixels_noperc | 0.99±0.01 | 0.24±0.03 | 0.09±0.03 | 8.8±0.2 | 0.16±0.01 | 0.15±0.02 | 13.9 |
| pixels_perc | 1.00±0.00 | 0.22±0.02 | 0.08±0.02 | 9.0±1.3 | 0.74±0.05 | 0.72±0.04 | 34.1 |
| shared_perc_aux | 0.99±0.00 | 0.23±0.01 | 0.08±0.02 | 10.9±1.5 | 0.77±0.06 | 0.72±0.07 | 17.3 |
| shared_noperc_aux | 1.00±0.00 | 0.24±0.03 | 0.10±0.03 | 7.2±1.7 | 0.18±0.03 | 0.19±0.05 | 6.0 |

`latent_noperc`: seed 2 failed to communicate (acc 0.22); the other two reached 0.99–1.00.

Galleries (seed 0), what they show:

- *shared_noperc*: a clean, consistent but arbitrary colour code (light green for salt lick,
  blue for flower, brown specks for tree and stump). Not iconic.
- *shared_perc*: mostly grey shapes; some colour enters (pink for flower and stump; grey for
  rock, which is its own colour). Shapes differ by type in size and placement.
- *shared_perc_stamp*: the most picture-like gallery. Without the white bubble paper the salt
  lick is drawn as a white block, the rock as a grey-and-white blob, bush and stump as dark
  blobs. In bubble mode, white is the paper colour and cannot be drawn, so the salt lick's
  defining colour is unavailable there.
- *latent_perc*, *pixels_perc*: scattered single-pixel noise to a human eye, yet judge probes
  of ~0.75.

What this establishes (at this scale, on this world):

1. **Communication emerges in every cell** (acc ≈ 1.0, blank at chance, swapped below chance):
   the drawings are causally used. Only one seed of one cell (latent, no resemblance loss)
   failed.
2. **Iconicity comes from the resemblance loss, not from sharing the encoder.**
   `shared_noperc` is at chance on both judge probes; `shared_perc` and `separate_perc` are
   indistinguishable. This is the literature's prediction ([Mihai & Hare 2021](https://arxiv.org/abs/2106.02067):
game loss alone gives drawings humans cannot read; see Q2), and the
   "shared encoder makes iconicity emerge" hypothesis is not supported in bubble mode.
3. **Exception worth following up: stamp mode without the resemblance loss** reaches judge
   probes of ~2× chance (0.32 ± 0.10; seeds 0.45, 0.20, 0.32). A stamp is read on the ground,
   as the referent itself would be seen, so this is the one condition where the shared-encoder
   route shows anything. Three seeds with a wide spread: tentative.
4. **The judge probe is gameable by senders that control pixels directly.** Latent and pixel
   senders with the resemblance loss score as high as strokes but their drawings are noise to
   a person. For those cells the judge measures adversarial match, not resemblance; the human
   quiz is the only valid readability test there. *(Judgement, from the galleries.)*
5. **Predictions that failed:** the pixel sender trains fine (predicted to fail); latent is not
   better at communicating than strokes (everything is at ceiling); auxiliary losses change
   nothing (communication is already at ceiling without them).
6. **Not human-readable yet in bubble mode**: no gallery would pass the 35% quiz bar by eye.
   The stamp galleries are the best candidates.

### Grid 2 — pine tree and 8×8 drawings

Same grid on the taller pine (`14ab226`) plus three cells where strokes are drawn on 8×8 and
shown 2× enlarged (`canvas_res=8`, commit `35f0986`): the owner's hypothesis that a coarser
drawing pushes toward simplified pictographs rather than faithful renderings. Results pending.

## What the plan predicts

From the literature (see `02-design-questions.md` Q2 and the survey §4): `shared_noperc` will
communicate (`acc` high) but produce drawings that are neither iconic by the judge nor readable;
`*_perc` cells will be iconic and readable at little cost to `acc`; `latent_*` will communicate
best and be least iconic; `pixels_*` will train poorly or not at all; `stamp` will be slower to
learn than `bubble`. Deviations from these predictions are the interesting result.
