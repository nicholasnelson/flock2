# Review (2026-10-26): why the vision sheep's state does not hold where water is

Scope: reading only (code, docs, saved logs, probe and evaluation JSON, the render samples), plus a
few lines of numpy on the renderer's row table. No checkpoint was loaded and nothing was run on the
GPU. Every sentence marked *(judgement)* is mine; *(established)* points at a verified link in the
references. Numbers from the repository are quoted with their file.

## Summary

1. The probe's headline ("where water is, is not in the state") is only half established. What is
   established is that the direction to water is **not linearly readable** from (h, z) and that the
   MLP probe overfits. The MLP's negative R² values are an artefact (the MLP is below ridge on
   almost every continuous target, trained and untrained alike), and the "in view" set is far
   larger than the set of frames in which a pond is actually visible.
2. The strongest substantive hypothesis is that the pond, at the distances that matter (the flock
   starts 6–14 tiles from water), is a 1–4-row sliver of 20–30 columns worth 3–13 units of the
   ~100–130-unit per-frame reconstruction loss by day and **nothing at night** (contrast 69× lower
   at L = 0.12), and is often hidden behind flockmates in the horizon band. The loss therefore
   buys a precise pond position only within about 6–8 tiles by day. Trees are tall, dark on sky,
   never hidden by the ground, and read at R² 0.3–0.6: the same probe, the same state.
3. Second: whatever the state holds about the pond is held as a compressed picture, not as a
   bearing. The old meadow showed this directly (the decoder redrew water within 6–11° of the true
   azimuth while probes read left/right at R² 0.04). The policy's MLP has the same readout problem
   as the probe, plus a weak credit signal (turn toward a sliver, walk 80 steps, γ = 0.998).
4. Third, weaker: the training budget. The replay holds the last 16 iterations only (1,024
   sequences, 64 added per iteration), so the world model is trained on a sliding half-day window
   of 32 sheep; the KL carries 5–15 nats a step; reward plateaued by iteration 300 while
   reconstruction kept falling. Rung 2 (running) tests part of this.
5. Drinking is partly learned because "water at the mouth" and water within 1–3 tiles are large,
   fixed-position features (a pond with its edge 1 tile away is 13 % of the strip); the policy
   finishes the approach and drinks. What it cannot do is the 6–14-tile search.
6. The cheapest discriminating diagnostics are: decode the posterior and score water recall and
   azimuth by distance and by day/night (the v2 analogue of `m1_decoder_water.py`); re-probe with
   visibility taken from the renderer's own water mask, stratified by distance, with ridge as the
   headline and an early-stopped MLP; and probe the encoder embedding alone against (h, z).
7. Rung 3 as written fixes the column structure of the encoder and decoder and multiplies the
   per-step channel (272 categoricals against 32), both plausibly helpful, but it inherits the same
   loss, so if hypothesis 2 is right it will learn the same blind spot. Its 3 × 3 ConvGRU with zero
   padding cannot carry a map through a 22.5° turn unless the grid learns a polar layout; two
   learning-only additions (multi-step prediction and a pre-registered attention probe) would make
   its test informative.

## 1. Ranked hypotheses

### H1 (top). The pond earns too little of the loss at the distances that matter, and none at night

*What the loss rewards.* `wm.loss_fn` (RGB mode) sums squared colour error over 64 × 256 × 3 values
(as DreamerV3's image decoder does, *(established)* [1]). Late in training the logs give
`wm_recon` ≈ 100–130 per frame and `wm_pixel_acc` (mean absolute colour error) ≈ 0.02–0.03
(`meadow2-results/hybrid/s*_log.jsonl`, iterations 400–600), i.e. an RMS error of about 0.047 per
value: the frames are nearly flat colours and the residual sits on shape edges.

*What a pond costs.* Reimplementing `render._row_edges` (sinh rows, finest 0.6° at −8°) in numpy
and taking a pond of radius 2.75 tiles (the middle of `POND_RADIUS` = 2–3.5), with the colours of
`render.py`, haze 1 − exp(−d/30) toward the horizon colour, and the light scale:

| Edge distance (tiles; the probe's "distance") | Width (° / columns) | Rows | Pixels (% of strip) | Cost of omitting it, day | Same, night (L = 0.12) |
| --- | --- | --- | --- | --- | --- |
| 1 | 94 / 81 | 27 | 2,200 (13.4 %) | 523 | 7.5 |
| 2 | 71 / 60 | 20 | 1,200 (7.4 %) | 268 | 3.9 |
| 4 | 48 / 41 | 10.5 | 430 (2.6 %) | 84 | 1.2 |
| 6 | 37 / 31 | 5.7 | 179 (1.1 %) | 30 | 0.44 |
| 8 | 30 / 25 | 3.5 | 88 (0.5 %) | 13 | 0.19 |
| 10 | 25 / 21 | 2.3 | 49 (0.3 %) | 6.4 | 0.09 |
| 12 | 22 / 18 | 1.7 | 30 (0.2 %) | 3.5 | 0.05 |
| 16 | 17 / 14 | 1.0 | 14 (0.1 %) | 1.2 | 0.02 |

(Cost = pixels × squared colour distance between hazed water and hazed short grass; against lush
grass the per-pixel distance is 0.39 instead of 0.31, against sky 0.18.) Misplacing a pond with
its edge at 8 tiles by 10° (8.5 columns) costs about 10; the KL terms price a nat at 0.6.

Reading *(judgement)*: by day the loss pays well for ponds within 4–6 tiles, modestly at 8, and
hardly at all beyond 10. At night (40 % of steps, plus two 240-step ramps) the pond is invisible to
the loss at every distance: 0.12 × WATER_C against 0.12 × grass differs by 0.004 per pixel. The
flock starts 6–14 tiles from water (spec § 11: median 10.2), and the hybrid sheep that die are the
ones that never close that gap. So the regime the sheep must solve is exactly the regime where the
reconstruction loss is indifferent to where the pond is. DreamerV3's own text notes that "where
individual pixels may matter for the task, a weak regularizer is required to extract fine details"
*(established)* [1]; here the regulariser is already weak (β_rep = 0.1, KL 5–15 nats ≫ 1 free nat),
so the limit is the reconstruction objective itself, not the KL.

*Occlusion.* A neighbouring sheep 2 tiles away is 0.85 tiles tall and 0.9 wide: it covers rows
~16–46 and ~21 columns, i.e. the whole horizon band where a pond beyond 5 tiles lives (rows 22–31).
`render_samples.png`, first panel ("facing the flock's nearest pond"), shows exactly this: the pond
is a 3–4-row sliver half hidden by a flockmate. The scripted flock sees 8.5 others on average
(spec § 11), and it is the scripted flock that generates the probe data.

*Haze and colour.* Water (0.26, 0.50, 0.76) is closer in colour to the sky (distance² 0.18 to
SKY_DAY_HI) than to grass (0.31–0.39), and haze pulls both the pond and the ground toward the
horizon colour, so the far pond looks like a slightly darker bit of horizon: the cheapest thing for
a decoder to blur away.

*Evidence for:* the table; the untrained model's random features read cover distance at R² 0.41
(ridge) but water direction at −0.03, so even raw random projections of the image carry "where is a
tree" and not "where is the pond"; trees read at 0.3–0.6 in every trained seed. *Evidence
against:* on the old meadow the decoder did redraw water at the right azimuth in 56–75 % of frames
(after class balancing), so a reconstruction loss *can* carry water position when pushed; this says
H1 is about weight, not impossibility. Nothing in the v2 logs says whether the v2 decoder draws
ponds at all: diagnostic D1 is the test.

### H2. The position is in the state, but as a picture code, not as a bearing

*(judgement, with the old meadow as evidence.)* An RSSM trained by reconstruction represents what
is needed to repaint the strip. The pond's column is then a nonlinear function of many units
(which conv features at which of the 2 × 8 bottleneck cells, mixed by a 3,072 → 512 dense layer and
by the GRU). A ridge probe cannot read an angle from that; an MLP can in principle, but with 24
training meadows (48 ponds) and no regularisation it memorises layouts (see § 2). The PPO policy
is a 256-unit tanh MLP on the same 896 numbers, trained by a weak signal: the advantage of turning
toward a 25-column sliver materialises 50–100 walk-steps later. Privileged PPO gets the bearing as
two clean numbers and solves the meadow in 100 updates (spec § 13).

*Evidence for:* old-meadow decoder azimuth error 6–11° with probe R² 0.04 for left/right
(`m1-meadow-README.md`, hybrid table); cover direction reads at only 0.3 even though trees are
unmistakable, which suggests the state holds "picture" rather than "bearing" for everything, with
water the worst case. *Evidence against:* if H2 were the whole story, a well-regularised MLP probe
on day frames with the pond within 6 tiles should read direction well; nobody has run that split.

### H3. Data and training budget: a sliding half-day replay, 600 iterations, a 128-bit channel

Facts from `scripts/m2_hybrid.py`: 8 meadows × 4 recorded sheep × 2 sequences = 64 sequences enter
the replay per iteration; its capacity is 1,024, so it holds the last **16 iterations** (1,024
sim steps, 0.43 of a day) of 32 sheep. The 16 updates × 8 sequences per iteration sample each
sequence about twice in its life. The world model therefore never sees a mixture of day, night,
near-water and far-from-water experience at once; it sees whatever the last 16 iterations held.
DreamerV3 samples uniformly from a large replay *(established that it uses a replay; its size is
in the supplementary material, which I did not verify)* [1]. The per-step information channel is
32 categoricals × 16 classes (128 bits at most, straight-through); the logged KL is 5–15 nats a
step, i.e. the posterior corrects the prior by 7–20 bits per frame, which is enough to re-encode a
pond column (5 bits at bottleneck resolution) but says the model mostly trusts its prior.

*Evidence for:* reward plateaued by ~300 while `wm_recon` kept improving (160 → 100 from
iteration 100 to 600, seed 0); the old meadow's 2,000-iteration run improved reward but not the
left/right probe. *Evidence against:* the grid hybrid converged in 200 iterations with the same
budget; and more training cannot create loss where H1 says there is none. The running rung-2 test
(2,000 iterations) will settle part of this; my expectation *(judgement)* is a modest rise in
water distance R² and little change in direction.

### H4. Egocentric rotation: every turn shifts the code, and the prior does not predict the shift

A turn moves the panorama by 19.2 columns. In the old meadow turning was the most surprising
action (KL 37 against a routine 28, `m1-meadow-README.md` step 2), which says the prior does not
learn the shift well; everything about "where" must then be re-transmitted after each turn, and
h's memory of positions is short. Learned rotation of a vector code is a hard, global permutation
for a GRU *(judgement)*. This is the mechanism rung 3 is meant to help with, but see § 4: its
ConvGRU cannot rotate either. Cheap check: KL and reconstruction error by action on a v2
checkpoint (D6).

### H5. The probe's targets and sample are harsher than they look (see § 2)

Not a cause of the sheep's deaths, but a cause of the −0.2 to −0.5 numbers.

Ranking *(judgement)*: H1 > H2 > H3 ≈ H5 > H4. H1 and H2 are not exclusive: the far pond is
weakly encoded, and what is encoded is hard to read. D1 + D2 separate them.

## 2. Is the probe trustworthy?

It is trustworthy about what it reads well (felt numbers 0.99, mouth flags 0.86–0.92, cover
distance 0.5–0.7) and in its sign for water direction (no linear readout). It is **not** a clean
measurement of "is the bearing in the state", for these reasons:

1. **The MLP overfits.** In `probes/s0.json` the MLP is below ridge on water_fwd, water_left,
   water_dist, grass_fwd, grass_left, cover_fwd, cover_left (in view), and the same holds for
   seeds 1, 2 and the untrained model. A probe that is worse than the mean on held-out meadows
   has memorised the 24 training layouts (h carries meadow-level features). `mlp_probe` has no
   weight decay, no early stopping and 8 epochs at 1e-3. So a negative MLP R² means "overfit",
   not "no information", and "no better than untrained" compares two overfits. Headline numbers
   should be ridge (water: 0.00–0.17 in view), with an MLP early-stopped on held-out meadows.
2. **Out-of-view R² is meaningless as reported.** water_fwd out of view is −2.7 to −4.9 for every
   model: behind the sheep fwd ≈ −1 and beyond 16 tiles the distance is capped, so the target
   variance is tiny and any error is amplified. Report mean error against the constant baseline,
   or restrict to the 150–180° band and 16–32 tiles separately.
3. **"In view" is not "visible".** `IN_VIEW` = 16 tiles and ±150° gives 84 % of samples. By the
   table above, beyond 10 tiles the pond is ≤ 2 rows and ≤ 50 pixels, hazed 30–40 % toward the
   sky; at night it is invisible at every distance; and flockmates and trees occlude the horizon
   band. The share of frames in which a pond is discernible is probably 30–40 % *(judgement)*,
   and the R² over the whole in-view set is capped accordingly. The right definition is available
   for free, for evaluation only: the renderer's own `wet_px` mask (count water pixels in the true
   frame; "visible" = at least N by day).
4. **The target is the nearest water tile centre, as a unit vector.** `privileged._nearest` over
   `_TILES`; near the shore the nearest tile hops between neighbours, and dividing by a distance
   below 1 tile amplifies that into large swings of (fwd, left) while the picture barely changes.
   The scripted flock spends many steps at the shore (it tops hydration up to 0.95, about 16 sips,
   after walking the last tiles with the pond filling the view), so a sizeable share of the
   in-view variance of "left" is this near-shore noise, which no state can predict. Use the
   direction to the pond's centroid (or to the nearest water point of the smooth `water_at` field),
   or exclude samples under 1.5 tiles, or report by distance bin.
5. **Distribution shift.** The world model was trained on the hybrid's own behaviour (rest 20–35 %,
   trot 10–24 %, drink 2–12 %); the probe filters its state through the scripted flock, which
   walks straight at targets. Steering straight at water puts the pond at left ≈ 0 for long
   stretches, so the in-view variance of "left" is concentrated in the approach and shore phases.
   Collect probe data with the hybrid's own policy as well (D4).
6. **Sample size** is adequate for a bearing (headings vary within a meadow) but the number of
   distinct ponds (48) is what the MLP memorises; more meadows with fewer steps each would help the
   MLP, not ridge.
7. **The standardisation fix is right** (`_stats` floors sd at 0.05). The old `m1_probe_latent.py`
   values (sd + 1e-6) remain suspect, as the spec notes.

Probing with a linear readout "trained entirely independently of the model" is the standard
method *(established)* [2]; the caveat that a linear probe measures linear accessibility, not
presence, is the point of item 1 and of H2.

## 3. Why drinking is partly learned without a readable bearing

*(judgement, from the numbers.)* Drink success is 19–37 % against 2 % random
(`eval/hybrid_s*.json`), with drink at 2–12 % of actions. The state reads "mouth on water" at
balanced accuracy 0.86–0.92, and a pond whose edge is within 1–3 tiles fills 7–13 % of the strip
at a fixed place in the image (bottom rows, which the 2 × 8 bottleneck still resolves at 37.5° per
cell), so "water near, roughly ahead-left" is cheap to encode and the loss pays 270–520 per frame
for it. A policy that only drank when the mouth flag were on would score ~90 %, so the policy uses
the flag imperfectly and still drinks on dry ground (bad taste 4–11 % of steps). The last 1–3
tiles of the approach are therefore learnable from this state; the 6–14-tile search is not, and
sheep die of thirst on day 2 after the starting water (0.2–1.0, ~1.6 days from full at
NEED_SCALE 3) runs out. Day-2 rather than day-1 deaths also show that some sheep do reach water
once, by wandering or by the near-field cue, and then lose it again: nothing out of view is
remembered (every out-of-view R² ≤ 0).

## 4. Rung 3 (`flock/wm_spatial.py`): will it fix this?

What it changes, and my reading:

- **Column-preserving encoder** (3 stride-2 convs → 32 columns of 9.4° × 128 numbers) and
  decoder (learned column queries over cell keys). Good: the pond's column survives to the
  attention, instead of being mixed by a dense layer. A pond at 10 tiles is still only 2 columns
  of 32 and 2 rows of 64, so H1 applies unchanged.
- **Per-step channel**: 256 cell categoricals of 8 classes + 16 global = 272 variables (≤ 816
  bits) against 128. More room, same loss. Free bits stay at 1 nat over the sum; with 272
  variables the KL sum will be much larger in nats than today's 5–15, so the 0.1 representation
  weight bites harder in absolute terms; worth watching, not changing in advance.
- **Transition**: a 3 × 3 ConvGRU with zero padding. A turn of 22.5° moves content at radius 8
  cells by about 3 cells in one step; a 3 × 3 kernel moves it by at most 1. So the grid **cannot
  keep a Cartesian map (ego- or allocentric) consistent across turns**, whatever it learns. It can
  keep a polar one: 16 angular cells × 22.5° = 360° exactly, so a turn would be a one-cell shift
  along the angle axis, and a walk a non-uniform shift along the radius, both within reach of a
  3 × 3 kernel, but only with wrap-around along the angle axis, which zero padding denies. Nothing
  in the model points it toward that layout; the only pressure is reconstruction, which, as the
  doc's own risk section says, a cell-as-unordered-memory solution satisfies equally well.
  *(judgement)* The likeliest outcome is an egocentric "panorama memory": cells come to stand for
  view columns, the decoder's fixed queries read them one-to-one, and turns are handled by
  re-encoding, as now. That would be a real but modest gain (column structure), not a map.
- **The action is the only motion signal.** The commanded action is given at every cell; refused
  moves and jostling make it a noisy proxy for actual motion. The published results in which
  map-like codes emerge in recurrent networks feed **velocity** (self-motion) as input
  *(established)* [4]; here that is rung 4's first piece, deliberately withheld. So rung 3 is a
  harder test than the literature it leans on, which should be said when its result is read.
- **Neural Map** keeps a spatially structured 2-D memory with learned read/write; whether the
  agent's position is supplied is not stated in its abstract and I could not verify it from the
  abstract alone *(established only to that extent)* [3].
- Code: the shapes are consistent (`_global_summary`, `glob_in`, `k_dec`, `post_glob` inputs all
  match their MLP sizes); `cell_emb` is shared between the posterior queries and the decoder keys,
  which gives cells a stable identity, good. The decoder's column queries are static parameters,
  so re-routing after a turn relies entirely on the cell keys (h, z, embedding) changing, which is
  as intended.

Changes I would make before training, all learning-only and within the owner's principle:

1. **Add multi-step (open-loop) prediction to the loss** for both rungs 2 and 3: predict the view
   and felt numbers k = 1, 4, 8 steps ahead through the prior from a state inferred earlier. No
   privileged input; it is what DreamerV3's imagination already relies on. It makes the state
   carry what will come into view and what has left it, which is the whole point of memory.
   Without it, nothing rewards remembering a pond after turning away.
2. **Pre-register the attention probes** (evaluation only): (a) for each cell, the circular mean
   of its posterior attention over columns; after a left turn it should move by −2.4 columns (of
   32) for a cell that tracks the world, 0 for one that tracks the view; (b) does any cell's z
   predict "water visible at bearing b" across meadows. These say whether a map or a panorama
   memory emerged, which is the question rung 3 exists to answer.
3. **Optional, declared as a prior if used:** circular padding along one grid axis (a torus or
   cylinder topology). It is not geometry knowledge (nothing says which axis is angle or how far
   a step moves), but it is a shape prior, so I would run without it first and report it as a
   separate variant if the attention probe shows cells trying to wrap.
4. **Sanity test of the column autoencoder alone** (encoder + decoder, no recurrence) on a few
   thousand saved frames: does it reconstruct ponds at 8–12 tiles by day? If not, H1 dominates
   and rung 3's state cannot hold what its own loss discards.

What rung 3 does not address: H1 (the loss), H3 (replay), night. If D1 shows the v2 decoder
already ignores ponds beyond ~8 tiles, I would fix the loss or the world before spending three
seeds on rung 3.

## 5. Recommended diagnostics, in priority order

All use saved checkpoints; "GPU minutes" means after the running job finishes (the renderer and
world model compile per shape; keep batch sizes small).

| # | Diagnostic | Cost | What each result means |
| --- | --- | --- | --- |
| D1 | **Decode-from-posterior water check on v2.** Port `m1_decoder_water.py` to RGB: run the scripted flock (and the hybrid's policy) on 8 fresh meadows, filter (h, z), decode, and score a water mask (decoded pixel closer to the shaded WATER_C than to grass/sky, below the horizon) against the renderer's true `wet_px`. Report water-pixel recall, column IoU and azimuth error **by edge distance bin (< 2, 2–4, 4–8, 8–16) and by day/night**. | ~2 GPU minutes; CPU feasible with 2 meadows × 50 steps | Recall high and azimuth within ~10° at 4–8 tiles but collapsing beyond → **H1** (loss weight); fix the objective or the world. Recall high everywhere by day but probes still fail → **H2** (readout); fix the policy's input/probe. Recall low even at 2–4 tiles → the model is undertrained (**H3**) or something is broken in RGB mode. |
| D2 | **Re-probe with a visibility mask and stratification.** Same `m2_probe.py` collection, with: "visible" = ≥ 20 true water pixels by day (from the renderer); bins by distance; target = direction to the pond centroid as well as the nearest tile; exclude < 1.5 tiles; ridge headline; MLP with weight decay and early stopping on 4 held-out meadows. Also report R² of the **image column** of water (circular mean of true water columns) as a target, which removes label and occlusion issues entirely. | ~3 GPU minutes to collect; fits on CPU | Direction readable (R² > 0.5) in the visible-near bin and not beyond → H1. Readable nowhere even in the near bin → H2/H3. Column target readable but privileged direction not → label noise (H5). |
| D3 | **Probe the encoder embedding (512), h (384) and z (512) separately**, same targets. | same run as D2 | Embedding readable, (h, z) not → the posterior/KL channel drops it (raise z size, lower β_rep, or the column-preserving embedding). Embedding not readable → encoder/loss (H1). z readable, h not → no memory of position across steps (H4). |
| D4 | **Probe on the hybrid's own behaviour** (collect with `hy.sheep_step` instead of `scripted.policy`). | same | Large change in R² → the scripted-flock distribution is distorting the probe (H5), not the sheep. |
| D5 | **Policy-side check**: evaluate the saved hybrid with the current embedding appended to the policy input is not possible without retraining; instead fit, offline, an early-stopped MLP from (h, z) to "turn-left / turn-right toward water" using the privileged bearing as a label on the hybrid's own frames (evaluation only, never used by a sheep). | CPU, minutes | If a supervised readout can get the sign of the bearing from (h, z) at > 80 % on visible-near frames, the state is sufficient and PPO's credit assignment is the limit (H2); if not, the state is the limit. |
| D6 | **KL and reconstruction by action** on a v2 checkpoint (as the old `m1_wm_surprise.py`). | ~1 GPU minute | Turns ≫ other actions → the prior does not learn the panorama shift (H4); supports multi-step prediction and rung 3's column structure. |
| D7 | **Rung-2 result reading.** When the 2,000-iteration probe arrives, read it with D2's caveats (ridge, visible-near bin) before applying the pre-registered 0.3 rule; the rule as written compares against a measurement that is capped well below 1 by construction. | none | A visible-near R² above 0.3 with the whole-set value below it would mean the rule's threshold, not the model, decided. |

## 6. Fixes that respect the principle (and what each gives away)

Ordered by how little they build in. None uses privileged labels in a loss.

1. **Multi-step prediction loss** (rungs 2 and 3). Gives nothing away. Makes memory of
   out-of-view things pay.
2. **A column-preserving embedding in the existing RSSM** (replace the 3,072 → 512 dense with
   32 columns × 16 numbers, or simply keep the 2 × 8 × 192 bottleneck flattened without the dense
   layer). An architectural shape, not geometry. Cheap; a half-step toward rung 3 that isolates
   the encoder question.
3. **Larger per-step channel and a weaker representation loss**: z = 64 × 32, β_rep 0.05; or
   free bits per variable. Learning-only; DreamerV3's own remedy for "fine details" is a weak
   representation loss *(established)* [1]. Risk: more noise in the policy's input.
4. **Pixel weighting by colour rarity within the batch** (the RGB analogue of `recon_balance`:
   quantise colours to a small palette computed from the batch, weight pixels by inverse
   frequency, capped). Uses only the sheep's own pixels; it made the old meadow's decoder draw
   water (29–33 % → 56–75 %) without helping the probe much, so it is a partial fix for H1 and
   none for H2. Not privileged, but it is a designer's thumb on the scale toward rare colours;
   say so if used.
5. **Replay and budget**: a replay of 8–16k sequences (store the renderer's RGB at one byte per
   pixel through a palette, as § 5 of the spec planned, to keep memory at today's 1.6 GB), uniform
   sampling, and longer runs. Gives nothing away.
6. **The policy's input**: let PPO see the current embedding beside (h, z). The embedding is the
   sheep's own eyes, not privileged; it is a departure from DreamerV3's "policy on the model
   state", so record it as such. Cheap test of H2.
7. **World facts, not sheep knowledge** *(judgement, for the owner)*: at L = 0.12 the pond's
   contrast is 69× lower than by day; real water reflects the sky and is usually the brightest
   ground at night. Making water reflect the sky colour (brighter than grass at night) and raising
   the night floor are changes to the world, which the spec treats as legitimate ("facts of the
   world, not of the sheep"), and they would make night-time water findable at all. Larger ponds
   or a third pond were already on the table (spec § 10). These are the only changes here that
   make the task easier rather than the learner better.
8. **Last resort (rung 4)**, not recommended yet: proprioception, then shifting memory by motion.
   The emergent-map literature does give velocity *(established)* [4], so if rung 3 fails the
   first piece has a principled justification; but D1/D2 should come first, because if H1 holds, a
   map fed by a decoder that discards far ponds will be empty where it matters.

## 7. Other issues found

- **Replay horizon** (`m2_hybrid.py`): 1,024 sequences at 64 per iteration = the last 16
  iterations. The doc describes it as "replay of 1,024" without noting that this is a sliding
  window of under half a day; it belongs in the spec's § 14.
- **Loss weights vs DreamerV3**: `KL_DYN` = 0.5, `KL_REP` = 0.1. The DreamerV3 text I could verify
  gives β_dyn = 1, β_rep = 0.1 and free bits of 1 nat *(established)* [1]; the code's docstring
  says "all as in DreamerV3". A note in `wm.py` would do; the effect is small.
- **`wm_kl_dyn` and `wm_kl_rep` are logged as identical** (they are the same number with different
  stop-gradients). Harmless, but the log column is uninformative; log one.
- **`wm_pixel_acc` in RGB mode is a mean absolute error**, named like an accuracy; the spec says
  so, the log does not.
- **Stale docstrings in `wm.py`** (16 palette colours, 9 actions, "one-hot strip") now that
  `Config` carries RGB and 8 actions.
- **Rung-1 criteria differ between documents**: `m1-map-world-model.md` says "worked if every
  seed beats random by 5 SE and dawn water rises above 0.5"; the spec § 14 pre-registers "beats
  random by 5 SE with at most 20 % dying". The spec's is the one applied; the ladder doc should
  point at it.
- **The spec's § 5 row table** (three bands) is superseded by § 13's sinh curve; § 5 still reads as
  current. The mouth point's row (59 in § 5) also needs rechecking under the new curve: ground at
  0.6 tiles is row ~59.5 by the numpy above, so it still holds.
- **Probe in-view share (84 %) is reported as if it were visibility**; see § 2.
- **The rung-2 rule** ("better of MLP and ridge, 0.3 on forward or left") will be applied to a
  measurement whose ceiling is well under 1 for the reasons in § 2; it can fail a model that holds
  the bearing of every visible pond.
- **Evaluation**: `m2_eval.py` samples actions (not greedy), consistent with the baselines; drink
  success is computed from the pre-step mouth tile, which is correct since drinking does not move
  the sheep. No problem found. One caveat: `death_share` over 3 days from random starts with
  hydration as low as 0.2 is harsh on every policy that cannot find water in the first day; the
  per-day death curve would be more diagnostic than the total.
- **Scripted flock in the probe (`m2_probe.py`) uses `PRNGKey(7)` for cover preference inside
  `scripted.policy`** (stable per sheep, as intended); fine, noted only because the probe's
  meadows differ from the checks' and the preference table is shared.
- Nothing in `render.py` or `sim.py` looked wrong for this question. `wet` is computed on the
  smooth `water_at` field, consistent with drinking.

## 8. References (each link fetched and checked on 2026-10-26)

1. Hafner, Pasukonis, Ba, Lillicrap, "Mastering Diverse Domains through World Models" (DreamerV3),
   https://arxiv.org/abs/2301.04104 ; text at https://arxiv.org/html/2301.04104v2 , from which:
   free bits "clipping the dynamics and representation losses below the value of 1 nat";
   "loss weights β_pred = 1, β_dyn = 1, and β_rep = 0.1"; "where individual pixels may matter for
   the task, a weak regularizer is required to extract fine details". The abstract confirms a
   single configuration across domains and learning from pixels; the image decoder's exact loss
   form is in the paper body/appendix, which I did not verify beyond the HTML's "symlog squared
   loss" for the decoder.
2. Alain and Bengio, "Understanding intermediate layers using linear classifier probes",
   https://arxiv.org/abs/1610.01644 : linear probes trained independently of the model, measuring
   linear accessibility of features.
3. Parisotto and Salakhutdinov, "Neural Map: Structured Memory for Deep Reinforcement Learning",
   https://arxiv.org/abs/1702.08360 : a spatially structured 2-D memory with learned read/write.
   The abstract does not state whether the agent's position is given; I did not verify the body.
4. Cueva and Wei, "Emergence of grid-like representations by training recurrent neural networks
   to perform spatial localization", https://arxiv.org/abs/1803.07770 : grid-like, border and
   band cells emerge in RNNs trained on navigation **with velocity inputs**.
5. Hafner et al., "Learning Latent Dynamics for Planning from Pixels" (PlaNet),
   https://arxiv.org/abs/1811.04551 : the latent dynamics model "with both deterministic and
   stochastic transition components" that `wm.py` follows.
6. Deng, Jang, Ahn, "DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with
   Prototypical Representations", https://arxiv.org/abs/2110.14565 : reconstruction-based world
   models keep task-irrelevant detail and struggle with visual distraction (their abstract's
   framing; relevant to H1 as a known limitation of the objective, not as a prescription).
7. Seo et al., "Masked World Models for Visual Control", https://arxiv.org/abs/2206.14244 :
   end-to-end reconstruction-plus-dynamics training makes it "difficult to accurately model the
   interaction between robots and small objects" (abstract). The link from that to ponds at 10
   tiles is my inference.

Not verified and therefore not cited: Banino et al. 2018 (Nature; the page redirected to a login
gate), the DreamerV3 supplementary hyperparameters (PDF could not be rendered here), Kingma et
al. 2016 for free bits (the abstract does not mention them).
