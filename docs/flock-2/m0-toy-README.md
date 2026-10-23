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
drawing pushes toward simplified pictographs rather than faithful renderings.

| cell | acc | acc_blank | acc_swapped | CIC | judge_probe | judge_rank | ink |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shared_noperc | 0.99±0.00 | 0.22±0.02 | 0.08±0.02 | 7.3±1.5 | 0.17±0.01 | 0.17±0.00 | 7.0 |
| shared_perc | 0.98±0.01 | 0.22±0.01 | 0.08±0.02 | 11.4±3.6 | 0.73±0.03 | 0.71±0.05 | 15.9 |
| separate_noperc | 0.99±0.00 | 0.26±0.04 | 0.09±0.02 | 4.6±1.0 | 0.17±0.01 | 0.15±0.02 | 1.3 |
| separate_perc | 0.99±0.01 | 0.23±0.01 | 0.10±0.01 | 8.8±0.8 | 0.75±0.06 | 0.70±0.03 | 10.5 |
| shared_perc_stamp | 0.99±0.00 | 0.24±0.01 | 0.09±0.02 | 13.1±0.4 | 0.71±0.09 | 0.71±0.08 | 11.6 |
| shared_noperc_stamp | 1.00±0.00 | 0.26±0.01 | 0.10±0.04 | 7.9±1.7 | 0.25±0.10 | 0.25±0.09 | 5.7 |
| latent_noperc | 0.75±0.34 | 0.23±0.06 | 0.15±0.08 | 7.5±5.8 | 0.16±0.01 | 0.16±0.02 | 4.8 |
| latent_perc | 1.00±0.01 | 0.22±0.00 | 0.09±0.01 | 11.9±2.2 | 0.77±0.01 | 0.77±0.02 | 32.1 |
| pixels_noperc | 0.99±0.01 | 0.26±0.03 | 0.08±0.01 | 7.8±1.6 | 0.14±0.02 | 0.13±0.01 | 15.4 |
| pixels_perc | 0.99±0.00 | 0.23±0.02 | 0.08±0.01 | 9.4±1.5 | 0.74±0.01 | 0.70±0.01 | 34.2 |
| shared_perc_aux | 0.99±0.01 | 0.22±0.01 | 0.10±0.03 | 11.3±3.2 | 0.76±0.03 | 0.73±0.03 | 18.2 |
| shared_noperc_aux | 0.99±0.00 | 0.23±0.02 | 0.08±0.03 | 7.9±1.0 | 0.13±0.06 | 0.15±0.04 | 7.4 |
| shared_noperc_c8 | 0.99±0.01 | 0.22±0.01 | 0.09±0.04 | 7.3±4.0 | 0.16±0.01 | 0.15±0.02 | 5.8 |
| shared_perc_c8 | 0.98±0.01 | 0.22±0.04 | 0.09±0.03 | 10.6±0.9 | 0.76±0.04 | 0.73±0.03 | 16.7 |
| shared_perc_stamp_c8 | 1.00±0.00 | 0.24±0.01 | 0.09±0.01 | 12.5±0.8 | 0.70±0.12 | 0.70±0.09 | 12.2 |

`latent_noperc`: again one seed (s0) failed to communicate. `shared_perc_stamp_c8`: seeds
0.87 / 0.63 / 0.61 on the judge probe — the highest single seed of either grid.

What this adds:

1. **Grid 1 replicates on the new tree.** Every conclusion above holds; the pine changes no
   cell by more than seed noise.
2. **The stamp-without-resemblance signal is weak.** 0.25 ± 0.10 here (seeds 0.39, 0.17,
   0.18); pooled over both grids 0.29 (six seeds: 0.45, 0.20, 0.32, 0.39, 0.17, 0.18). Half the
   seeds show something, half are at chance. Not enough to claim the shared encoder produces
   iconicity in stamp mode; enough to keep the stamp cell in later grids.
3. **8×8 drawings: no measurable change, and not more pictographic by eye.** Every 8×8 cell
   matches its 16×16 counterpart within noise. The galleries are blockier versions of the same
   content, not simpler ones. Two small differences that go the owner's way: at 8×8 the pine
   is drawn as a tall vertical block (its silhouette) rather than the 16×16 cell's round brown
   blob, and the flower becomes a pink block (nearer its red) rather than near-blank.
   *(Judgement, from one seed's gallery per cell.)* The literature locates simplification in
   effort cost and interaction feedback, not resolution
   ([Fay et al. 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2607338/);
   [Qiu et al. 2022](https://arxiv.org/abs/2111.14210)), and this toy's ink cost is small, so
   the null result is unsurprising.
4. **Two referents cannot be drawn in their own colour, by construction.** No cell ever draws
   the pine or bush green: the resemblance loss pastes the drawing on grass, and stamps are
   read on grass, so green ink is indistinguishable from the background. In bubble mode the
   paper is white, so the salt lick's white is likewise unavailable to the receiver. This is a
   property of the game, not of training, and it caps readability for those types.

Galleries compared: `runs/m0/grid_pine_c8_vs_c16.png` (local, not in the repo).

### Grid 3 — black-only drawing, with and without a colour-dropout judge

Owner's proposal: let the sheep draw only in black, so they must draw shape rather than colour.
`mono=True` fixes every stroke to the outline colour (15; palette 0 means void/unlit). The
literature's readable sketches were black lines on white
([Mihai & Hare 2021](https://arxiv.org/html/2106.02067v2)), and its shape-biased encoder gave
more shape-like sketches. Because our perception CNN reads each palette colour as a separate
channel, it may not see a black shape as resembling a coloured object, so each mono cell is run
twice: with the plain judge and with perception pre-trained with *colour dropout*
(`pretrain_colour_dropout=0.5`: each object in a pre-training view is, with probability 0.5, a
solid outline-colour silhouette, and the autoencoder reconstructs the full-colour scene).
Commit `33a40f6`; pine tree; 3 seeds.

| cell | acc | acc_blank | acc_swapped | CIC | judge_probe | judge_rank | ink |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mono_noperc | 0.99±0.01 | 0.23±0.03 | 0.09±0.03 | 9.8±1.6 | 0.33±0.07 | 0.34±0.07 | 7.3 |
| mono_perc | 0.99±0.01 | 0.23±0.01 | 0.09±0.02 | 12.7±1.7 | 0.76±0.02 | 0.69±0.03 | 13.4 |
| mono_noperc_stamp | 1.00±0.00 | 0.26±0.02 | 0.09±0.03 | 5.9±1.1 | 0.16±0.10 | 0.14±0.08 | 4.7 |
| mono_perc_stamp | 0.99±0.00 | 0.24±0.01 | 0.08±0.01 | 12.3±0.9 | **0.83±0.02** | 0.72±0.01 | 10.9 |
| mono_noperc_cd | 0.99±0.01 | 0.23±0.03 | 0.08±0.05 | 9.5±0.9 | 0.30±0.20 | 0.34±0.21 | 7.8 |
| mono_perc_cd | 0.99±0.00 | 0.24±0.01 | 0.10±0.02 | 13.3±0.7 | 0.70±0.02 | 0.69±0.01 | 14.6 |
| mono_noperc_stamp_cd | 0.98±0.01 | 0.24±0.01 | 0.08±0.02 | 4.5±0.6 | 0.15±0.07 | 0.16±0.05 | 5.2 |
| mono_perc_stamp_cd | 0.98±0.01 | 0.23±0.02 | 0.08±0.02 | 11.8±1.3 | 0.74±0.01 | 0.75±0.03 | 10.9 |

Judge probes are **not comparable between the `_cd` and plain cells**: each column's judge is a
different encoder. Compare within a judge, and use the galleries across judges.

What this shows:

1. **Black-only costs nothing in communication** (acc ≈ 1.0 everywhere) and, with the
   resemblance loss, matches or beats colour on the plain judge: `mono_perc_stamp` at 0.83 ±
   0.02 is the highest and tightest stroke cell of all three grids.
2. **Without the resemblance loss, black-only bubble drawings carry some resemblance**:
   `mono_noperc` 0.33 ± 0.07 (seeds 0.40, 0.23, 0.36), where the colour equivalent
   (`shared_noperc`) sits exactly at chance in both earlier grids. Removing colour removes the
   arbitrary colour-code shortcut, and the shared-encoder route then shows something. The stamp
   version is at chance, the reverse of the colour pattern. Tentative: 3 seeds, judge-measured.
   One dropout-judge seed (`mono_noperc_cd` s0) scores 0.02, well *below* chance: its drawings
   systematically resemble a different type than intended.
3. **The colour-dropout judge gives the most glyph-like galleries so far** *(judgement, seed 0
   of each cell)*: one solid, consistent black shape per type — salt lick and rock as wide
   domes or bars (the rock's low and wide, like its sprite), the tree as a round blob, bush and
   stump as small dashes or ticks, the flower left blank. The plain judge's drawings are more
   fragmented but occasionally more specific (in stamp mode the pine is a tall vertical blob).
   No cell draws the pine as a triangle, although it is the most distinctive silhouette.
4. **Shape alone cannot separate every type.** As silhouettes, salt lick, rock, bush and stump
   are similar dark ovals and bush ≈ stump almost exactly (`runs/sil_preview.png`), so
   black-only drawings can separate those only by size and proportion. A ceiling on human
   readability set by the sprite set, not by training.

Galleries: `runs/m0/grid_mono_galleries.png` (local).

### Grid 4 — the 128 × 128 world with line drawings (2026-10-23)

The world was redrawn and the pipeline ported (commits `e175053`, `9ce10b1`): 16×8 px tiles,
a 128² view (same 16-tile range), referents reduced to four distinct silhouettes (a
sharp-edged salt-lick cube, a pine, a star flower, the fox), and the drawing is a 32×32 px area
covering a sprite's footprint. Strokes are now black lines of fixed 2 px width, rasterised
directly at 32×32. Chance is 0.25 for accuracy and for the probes (four types).

Two loopholes had to be closed first, both found in the galleries, not the numbers (`6a6765f`):

- *Free dots.* With ink = length only, a zero-length stroke still draws a 2 px dot, and the
  sender drew dot constellations (a position code). Each stroke now costs a pen-lift price of
  4 px on top of its length (`stroke_cost`).
- *Faint strokes.* The sender then drew flower and fox as "blank" but separated them with
  half-on strokes (alpha < 0.5): the receiver could read them, a person could not. Strokes are
  now fully on or off in the forward pass with a straight-through gradient (`hard_gate`), the
  on/off gate the plan specified.

Cells: cost weight per px of ink {0.002, 0.01, 0.03} × bubble/stamp × resemblance loss on/off;
colour-dropout judge; 3 seeds.

| cell | acc | acc_blank | acc_swapped | CIC | judge_probe | ink (px) |
| --- | --- | --- | --- | --- | --- | --- |
| line_perc_bubble_ink002 | 1.00±0.00 | 0.25±0.01 | 0.00±0.00 | 5.5±0.8 | **0.80±0.01** | 29.4 |
| line_noperc_bubble_ink002 | 0.75±0.35 | 0.26±0.01 | 0.09±0.12 | 5.1±3.6 | 0.25±0.02 | 4.3 |
| line_perc_stamp_ink002 | 1.00±0.00 | 0.25±0.02 | 0.00±0.00 | 8.4±2.0 | **0.76±0.02** | 31.3 |
| line_noperc_stamp_ink002 | 1.00±0.00 | 0.24±0.01 | 0.00±0.00 | 3.0±0.3 | 0.26±0.03 | 8.8 |
| line_perc_bubble_ink01 | 0.88±0.15 | 0.24±0.03 | 0.02±0.03 | 4.4±0.5 | 0.27±0.02 | 6.1 |
| line_noperc_bubble_ink01 | 0.43±0.25 | 0.27±0.03 | 0.22±0.07 | 1.4±1.9 | 0.25±0.02 | 1.2 |
| line_perc_stamp_ink01 | 0.61±0.28 | 0.25±0.02 | 0.12±0.09 | 1.9±1.6 | 0.25±0.02 | 2.6 |
| line_noperc_stamp_ink01 | 0.88±0.14 | 0.25±0.02 | 0.03±0.04 | 1.8±0.5 | 0.25±0.01 | 4.5 |
| line_perc_bubble_ink03 | 0.26±0.01 | 0.26±0.01 | 0.28±0.02 | 0.0±0.0 | 0.25±0.02 | 0.0 |
| line_noperc_bubble_ink03 | 0.25±0.01 | 0.25±0.01 | 0.26±0.01 | 0.0±0.0 | 0.25±0.02 | 0.0 |
| line_perc_stamp_ink03 | 0.51±0.34 | 0.27±0.00 | 0.20±0.13 | 0.7±1.1 | 0.24±0.01 | 1.4 |
| line_noperc_stamp_ink03 | 0.48±0.20 | 0.27±0.01 | 0.20±0.07 | 0.6±0.5 | 0.24±0.01 | 1.1 |

What this shows:

1. **With the resemblance loss and a low cost (0.002), the drawings are line pictographs**,
   consistent across all four seeds inspected (bubble and stamp, seeds 0 and 1)
   *(judgement, from the galleries)*: the pine is one long near-vertical line (leaning `/`;
   one seed adds a branch, a `Y`); the fox is one long low horizontal line along the bottom;
   the flower is a small tick in the middle; the salt lick is two or three short strokes near
   the top, often a chevron like the cube's top edges. About 30 px of line per drawing, i.e.
   three to five strokes. Communication is perfect and causally used (swapped → 0.00).
2. **Without it, the code is positional**: tiny ticks in type-specific places, judge at chance.
   With line drawings the shared encoder alone produces no resemblance, in either mode.
3. **The cost weight is a cliff, not a dial.** At 0.01 senders often give up drawing and
   communication becomes unreliable; at 0.03 they draw nothing (ink 0) and accuracy is at
   chance. Likely mechanism *(judgement)*: once every gate is off, a straight-through gate gets
   no gradient that would turn a stroke back on at a profit, so the drawing cannot recover.
   The useful range is at or below ~0.005; finer steps there are untested.
4. **The judge probe is informative again** (0.80 vs 0.25, and at chance where drawings are
   positional), unlike the saturated 0.99 of the free-width blobs.

Galleries: `runs/m0/grid_line_galleries.png` (local).

### Grid 5 — instance variation, channel jitter, an invariant judge (2026-10-23)

Aim (owner): remove everything that is not the referent's shape, so a drawing must capture
what all trees (flowers, …) share rather than one remembered example, its place or its size.
Three switches, every combination, lines at cost 0.002 (commits `70a369f`, `081bbb6`):

- **variation**: six sprite variants per type (cube sizes; pines with 2–4 tiers and a broadleaf
  tree; flowers with 4–6 petals in five colours; foxes in three colours and two sizes), the last
  two of each **held out**: training never shows them; `acc` is measured on them, `acc_seen` on
  the training variants. Each object also gets a sub-tile offset (≤ 0.3 tile) and a scale
  (0.6–1.0), independently for the sender's referent and each candidate. Held-out variants as
  the test of type-level (not instance-level) drawings follow
  [Qiu et al. 2022](https://arxiv.org/abs/2111.14210), who define iconicity as accuracy on
  unseen instances.
- **jitter**: channel noise — the drawing is moved by up to 4 px within its frame when read or
  judged.
- **judge**: a *separate* judge encoder pre-trained with shift (±6 px), flip, blur and scale
  (0.5–1, the far view's range) and a term pulling augmented embeddings to the clean scene's.
  The sheep's own perception is pre-trained without it: making the shared encoder that
  invariant blinded the receiver (communication collapsed with or without the resemblance
  loss; diagnosed in single runs, not tabulated).

Two fixes were needed before the grid could run, both now default for these cells:

- **Delayed ink cost** (`ink_delay_steps=1500`). With variation and jitter the receiver learns
  to read slowly; meanwhile the ink cost was the only consistent gradient on the strokes and,
  under Adam, even its tiny ramp-start value switched every hard-gated stroke off within ~300
  steps, after which none recovered. The same mechanism plausibly explains grid 4's cliff.
- **Disjoint target cells.** Two candidate slots are 8 px apart on screen; with sub-tile offsets
  their target squares touched, and a receiver clicking a shared cell scored 0.35 with no
  message. Cells claimed by two candidates now count for neither.

| cell | acc (held-out) | acc_seen | acc_blank | acc_swapped | CIC | judge_probe | ink |
| --- | --- | --- | --- | --- | --- | --- | --- |
| var0 jit0 judge0 | 1.00±0.00 | – | 0.24±0.01 | 0.00±0.00 | 7.9±1.1 | 0.64±0.07 | 23.0 |
| var0 jit0 judge1 | 1.00±0.00 | – | 0.26±0.02 | 0.00±0.00 | 8.3±3.4 | 0.47±0.01 | 53.8 |
| var0 jit1 judge0 | 1.00±0.00 | – | 0.25±0.02 | 0.00±0.00 | 10.3±1.5 | 0.73±0.02 | 29.1 |
| var0 jit1 judge1 | 1.00±0.00 | – | 0.23±0.02 | 0.00±0.00 | 12.2±1.6 | 0.67±0.13 | 63.2 |
| var1 jit0 judge0 | 0.80±0.04 | 0.85±0.02 | 0.25±0.01 | 0.05±0.02 | 2.4±0.3 | 0.62±0.03 | 32.0 |
| var1 jit0 judge1 | 0.81±0.02 | 0.87±0.03 | 0.23±0.00 | 0.07±0.03 | 2.5±0.6 | 0.39±0.04 | 27.4 |
| var1 jit1 judge0 | 0.77±0.03 | 0.80±0.04 | 0.24±0.01 | 0.06±0.01 | 2.0±0.2 | 0.57±0.02 | 34.5 |
| var1 jit1 judge1 | 0.77±0.05 | 0.84±0.04 | 0.26±0.00 | 0.08±0.01 | 2.6±0.3 | 0.33±0.16 | 30.3 |
| all on, no resemblance loss | 0.78±0.04 | 0.86±0.04 | 0.25±0.03 | 0.07±0.01 | 2.7±0.8 | 0.24±0.08 | 34.3 |
| all on, stamp | 0.79±0.02 | 0.85±0.01 | 0.26±0.02 | 0.09±0.01 | 3.4±1.4 | 0.32±0.07 | 32.6 |

Judge probes compare only within a judge (judge0 vs judge1 are different encoders). The cells'
per-run files were lost to a case-insensitive directory collision (cell names differing only in
case); the metrics above were computed in-process and are intact, and galleries were
regenerated from seed 0 (`runs/m0/grid_inv3_gal/`, montage `runs/m0/grid_inv3_galleries.png`).

What this shows:

1. **Variation costs about 20 points of accuracy but generalises**: held-out variants are only
   ~5 points below seen ones (0.77–0.81 vs 0.80–0.87), so the drawings mostly stand for the
   type, not the example. Without variation every cell is perfect.
2. **The resemblance loss does not help communication** under variation (all on: 0.77 with,
   0.78 without), and jitter costs a few points.
3. **Galleries** *(judgement, seed 0)*: without variation each type gets one fixed glyph, and
   with jitter and the invariant judge it degenerates into horizontal strokes at different
   heights (a position code). With variation the marks vary by instance, but a few features
   recur across cells: the fox as a long low horizontal line (the most consistently iconic
   mark), the tree as a long diagonal, sometimes branched, the flower as small ticks. **Height
   in the frame still carries much of the code** — salt-lick marks at the top, fox marks at the
   bottom, although the cube stands on the ground — which motivates re-centring drawings on
   their ink before they are read (grid 6).

## What the plan predicts

From the literature (see `02-design-questions.md` Q2 and the survey §4): `shared_noperc` will
communicate (`acc` high) but produce drawings that are neither iconic by the judge nor readable;
`*_perc` cells will be iconic and readable at little cost to `acc`; `latent_*` will communicate
best and be least iconic; `pixels_*` will train poorly or not at all; `stamp` will be slower to
learn than `bubble`. Deviations from these predictions are the interesting result.
