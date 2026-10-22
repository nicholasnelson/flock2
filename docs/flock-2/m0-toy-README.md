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

## One run

```bash
python scripts/m0_run.py --out_dir runs/m0/shared_perc --perceptual_weight 1.0
```

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

## What the plan predicts

From the literature (see `02-design-questions.md` Q2 and the survey §4): `shared_noperc` will
communicate (`acc` high) but produce drawings that are neither iconic by the judge nor readable;
`*_perc` cells will be iconic and readable at little cost to `acc`; `latent_*` will communicate
best and be least iconic; `pixels_*` will train poorly or not at all; `stamp` will be slower to
learn than `bubble`. Deviations from these predictions are the interesting result.
