# Meadow v2: specification for review (2026-10-26)

Status: **for the owner's review; nothing built yet.** The changes come from the owner's play-test
of the sheep viewer (`scripts/sheep_view.py`) on 2026-10-26, and from the discussion that followed.
Everything marked *(proposal)* is open to change. Numbers are first settings, to be confirmed by
the checks in § 7, not tuned.

The order of work is in § 9. In short: the world and the drives first, then a survival check of
the scripted flock, then the new rendering in the viewer for the owner to walk around in. No
learning happens until all three are signed off.

## 1. Why

- **Water is out of reach.** Over 200 starting meadows, the nearest water is a median 32 tiles
  from the flock (90th percentile 58, worst 92). A sheep that starts at hydration 0.2 has 250
  steps left: about 30 tiles at a walk. A sheep wandering at random covers roughly
  step × √steps, so reaching 32 tiles would take on the order of 70,000 steps. A learning sheep
  essentially never finds water by chance, and the owner, following the scripted flock, ran dry
  before reaching it.
- **The view hides what matters.** The tile being eaten is often out of view: the lowest row of
  the strip meets the ground about 0.48 tiles ahead. A pond 5–7 tiles away fills about 1.4 rows.
  Speckled ground texture fills much of the strip, can't be predicted, and costs the world model
  as much as anything important. That is my best guess at why the model drops water
  *(judgement)*.
- **"Head up" gives the encoder two perspectives**, and night is a hard switch from full view to
  a 5-tile range.

## 2. The meadow

- **Size: 48 × 48 tiles** (was 128 × 128). Owner's rule: from one edge, a tree at the far edge
  should be just visible. A tree 1 tile wide shrinks to one pixel at about 49 tiles, at 1.17° per
  column.
- **Contents, scaled to the area** *(proposal)*:
  - 2 ponds, 2–3.5 tiles in radius;
  - 14 trees: 8 along the north edge, 6 in two copses;
  - 1 hut, 1 salt lick, 6 flowers, 4 mushrooms;
  - grass with richer and barer patches, as now.
- **Edge:** a low post-and-rail fence around the meadow, so the edge is visible. Outside it, flat
  rough ground that can't be entered.
- **Flock: 12 sheep** *(proposal; was 40)*. Today's 40 sheep in 1/7 of the area would very likely
  overgraze it. 12 keeps more than one neighbour in view, and social satisfaction still saturates
  at 8 visible. Grass capacity is checked in § 7.
- **The flock always starts within reach of water.** Its starting centre is 6–14 tiles from the
  nearest water. Each sheep is placed on a spot that is dry and clear of trees and the hut.
  *(Changed while building, 2026-10-26.)* This replaces "every dry tile within 20 tiles of water",
  which two ponds cannot meet in 48 × 48: the best two-point cover leaves corners about 25 tiles
  away. Measured: none of 200 random two-pond layouts passed, nor 11 % of four-pond ones even at
  24 tiles.
- **Interest cells: 4 × 4 tiles** (12 × 12 cells; were 8 × 8 tiles, 16 × 16 cells), so novelty
  still has room to vary.

## 3. The sheep's body

### Actions (8; was 9)

stop, walk (0.12 tiles a step), trot (0.35), turn left or right (22.5°), graze, drink, rest.
**Head up is removed.**

### Mouth point

Grazing and drinking act on the tile under a fixed **mouth point, 0.6 tiles straight ahead** of
the sheep's centre. Its radius is 0.3, so the point sits just past its nose.
- **Graze** eats that tile.
- **Drink** works only if that tile is water. This replaces "anywhere within a tile of water": it
  is stricter, but the sheep can see whether it will work.
- The point is always at the same place in the view (§ 5), and the human viewer marks it with a
  reticle.
- A sheep can stand right at a pond's edge, so its mouth point reaches the water.

### Levels (felt numbers: 7, were 6)

- **Gut energy** (0–1.15). It falls with each action:
  - standing, grazing, drinking: 0.0008 a step;
  - turning: 0.0009;
  - walking: 0.0013;
  - trotting: 0.0026;
  - resting: 0.0005.

  A good bite adds 0.012, as now. Standing still from full now empties the gut in about 1,250
  steps, about one day; today it takes about 2,000 steps from the gut alone.
- **Body fat** (0–1, new). Full fat stores 4 units of gut energy, enough for about 4 days of
  standing.
  - *Storing:* while gut energy is above 0.95, up to 0.002 a step of gut energy moves into fat, at
    70 % efficiency.
  - *Releasing:* while gut energy is below 0.5, fat releases up to 0.0008 a step into the gut.
    That covers standing and grazing but not trotting.
  - Fat is felt, but has no drive term of its own *(proposal)*: it is a buffer. Hunger still comes
    from gut energy.
- **Hydration**, unchanged: 0.0008 a step, about 1,250 steps from full; a sip adds 0.04. The
  smaller meadow puts water within reach. The rate is revisited only if § 7 fails.
- **Warmth, social, interest**, unchanged, except "visible" for social (§ 5).

### Death

A sheep dies when **hydration reaches 0**, or when **gut energy and fat are both 0**. Warmth is not
lethal for now. A dead sheep is removed from the meadow (no longer drawn or counted) until the
meadow restarts. For learning, death is terminal: no bootstrap from the next state.

### Reward, with an alive bonus

**r = 2.5 − D − 0.05 × bad taste** while alive, and nothing once dead.

- D is the homeostatic drive, as now: D = (Σ|1−h|⁴)^(2/4).
- Its worst value with the five driven levels all empty is √5 ≈ 2.24. So every step alive earns at
  least +0.26, and dying always costs everything that would have followed.
- Without the bonus, every reward is ≤ 0, and dying would *end* the penalties. A learner would
  discover that dying pays.
- Reward centring (already used) removes the constant offset when learning.

## 4. Shapes: simple flat components

Every object is built from rectangles, circles, ellipses and triangles in flat colours: no
textures. Sizes are in tiles. Each is drawn as an upright billboard, as now.

| Thing | Shape | Size |
| --- | --- | --- |
| Sheep, side view (left or right) | Cream ellipse body; dark grey head circle at the front end; 2 dark leg rectangles | 0.9 long, 0.55 body, 0.85 to top |
| Sheep, front view (toward the viewer) | Cream circle body; head circle centred and lower | 0.6 wide |
| Sheep, back view (away) | Cream circle body; no head; small tail | 0.6 wide |
| Broadleaf tree | Brown trunk rectangle; dark-green canopy circle | trunk 0.15 × 0.8; canopy 1.0 across; 1.8 tall |
| Pine | Brown trunk; dark-green triangle | 0.9 wide, 2.0 tall |
| Hut | Wall rectangle; darker triangle roof; door rectangle; one window | 1.4 wide, 1.5 tall |
| Salt lick | Pale grey block | 0.4 × 0.3 |
| Flower | Thin green stem; small pink or yellow dot | 0.15 tall *(owner: smaller than now)* |
| Mushroom | Pale stem; tan half-circle cap | 0.15 tall |
| Fence | Thin posts every 2 tiles; one rail | 0.4 tall |

Facing uses the same 4 views as now (away, toward, left, right), because which way a neighbour
faces is how a sheep reads where it is going.

**Ground** is one flat colour per tile:
- bare (tan);
- short, medium and long grass (three greens, each darker and more saturated);
- water (blue);
- outside the fence (grey-brown).

## 5. The view

### Geometry

- **Strip:** 64 × 256 pixels, 300° wide, 1.17° per column (all unchanged).
- **Eye height: 0.8 tiles** (was 0.5).
- **Rows are no longer even in angle.** More rows go near the horizon, where the ground recedes:

  | Rows | Elevation | Per row | Shows |
  | --- | --- | --- | --- |
  | 0–7 | +34° to +10° | 3° | sky and treetops |
  | 8–47 | +10° to −10° | 0.5° | the horizon band (horizon between rows 27 and 28) |
  | 48–63 | −10° to −74° | 4° | the ground near the feet |

- **Effect on a pond 5–7 tiles away:** about 5 rows (was 1.4). The nearest visible ground is
  0.23 tiles ahead (was 0.48).
- **The mouth point** (0.6 tiles ahead, 53° below level) is at row 59, column 128. The owner asked
  for about a third up from the bottom. With rows packed near the horizon, the ground at the feet
  is squeezed into the bottom rows, so the reticle sits about 8 % up from the bottom. To put it a
  third up, the fine band would have to shrink, costing pond rows. This is a trade-off to decide.
- **One panoramic eye.** At this resolution two eyes about 0.15 tiles apart would see something 6
  tiles away only about one pixel apart. Distance comes from where things meet the ground (exact
  on flat ground at a known eye height), from size, and from motion parallax.

### No view range: haze instead

- Everything is drawn while it is at least one pixel wide and tall. A tree 1 tile wide reaches one
  pixel at about 49 tiles.
- **Distance haze** blends each pixel toward the horizon colour by 1 − exp(−d / 30), with d in
  tiles. That is a real depth cue, and it hides flicker from tiles smaller than a pixel.
- The 48-nearest-objects budget per view stays; the smaller meadow holds fewer objects in total.

### Light: a smooth day and night

- **Day:** 1,200 steps, 60 % light, as now.
- **Light level L:** 1.0 at midday and 0.12 at midnight, with smooth dawn and dusk ramps of 120
  steps each.
- **Sky:** blends from blue-black (#0b1020) to sky blue (#87c4ec) with L.
- **Everything else:** every colour is scaled by L, plus a slight blue cast at night. Contrast
  drops, so small and distant things fade naturally. There is no range cut.
- **The hut is a beacon:** its window glows at night, and the ground within 3 tiles of it is lit
  to at least 0.6.
- **Social "visible"** *(proposal)*: another sheep counts as seen if it is in the field of view,
  at least 2 pixels tall, and within 4 + 20 × L tiles: 24 tiles at midday, about 6 at midnight.
  This replaces the old range rule.

### What the sheep's eye receives

- **Colour (RGB) instead of 16 colour codes**, since lighting and haze are continuous. DreamerV3
  learns from RGB images ([Hafner et al. 2023](https://arxiv.org/abs/2301.04104)). The world
  model's view loss becomes squared error on RGB.
- **The water weighting goes away for now.** It needed colour classes, and using rendering labels
  in the training loss would break the no-privileged-labels rule. With flat shapes and no
  texture, water may no longer be drowned out. The decoder water check, which uses labels for
  evaluation only, will tell.
- **Storage stays at one byte a pixel.** The renderer writes an index into a 256-entry table: base
  colour × 4 haze levels × lit-by-hut, plus one light level per frame. The RGB is computed on the
  GPU when a batch is used. The replay is twice today's 4-bit size; if memory is short, it gets
  half as many sequences.

## 6. The human viewer

- Draws exactly what the sheep sees, from the same renderer.
- A reticle at the mouth point; the head-up key removed.
- The time of day shown as the light level.
- The minimap shows the whole 48 × 48 meadow.
- Free look is kept, for getting a feel for the view.

## 7. Checks before any learning (pre-registered)

The scripted flock is updated: no head up, the mouth point, and **walking unless a need is
urgent** (below 0.2) instead of trotting to far targets.

**Pass/fail checks.** Any failure means changing the meadow, not tuning sheep. Over 64 meadows,
6,000 steps each (5 days), from random starts with levels drawn from [0.2, 1.0]:

1. **Survival:** at most 2 % of scripted sheep die, and no meadow loses more than 1 sheep.
2. **Needs met:** after day 1, the scripted flock's mean gut energy and hydration both stay above
   0.4.
3. **Grass lasts:** mean grass biomass on day 5 is at least 70 % of day 0's.
4. **Water is near:** from the flock's start, the nearest water is at most 12 tiles away in the
   median meadow, and at most 20 at the 90th percentile.

**A measure to report, not a pass/fail.** The share of meadows in which a sheep that walks straight
on, turning at random about every 30 steps, reaches water within 500 steps. If this is under 30 %,
I will flag it before any learning.

## 8. What this changes elsewhere

- **Every result so far is a baseline for the old meadow.** The new meadow needs new baselines:
  - random, rest-always and scripted;
  - privileged PPO (the upper bound; the smaller meadow changes its features too);
  - the vector-model hybrid, as the comparison for the map world model.
- **`m1_eval.py`:** 12 sheep per meadow; reports deaths and days survived alongside reward. Since
  the alive bonus shifts reward, results are also reported as mean D.
- **The world model gets no free information** (owner, 2026-10-26: the sheep earn their
  knowledge; built-in geometry is an absolute last resort). The view's projection, eye height and
  row table are facts of the world, not of the sheep: no part of the sheep's model is told them.
  The plan is the ladder in `m1-map-world-model.md`: today's model on v2 first.
- **Unchanged:** the grid world and its results.

## 9. Order of work

1. **World and drives** (CPU, about a day):
   - 48 × 48 generator with its own checks;
   - the mouth point, 8 actions, fat, death, alive bonus;
   - the scripted flock updated;
   - unit tests.
2. **The § 7 checks** (CPU, an hour). Reported to the owner with any meadow changes they forced.
3. **Renderer v2 and the viewer** (about a day): shapes, rows, haze, light, hut light, RGB. **The
   owner walks around and signs off.**
4. **Baselines** (step 8's list), then the vector hybrid on the new meadow (GPU).
5. **The ladder of `m1-map-world-model.md`**, least built-in first: today's model on v2, then the
   same model better trained, then a state with spatial shape that is entirely learned. Built-in
   geometry only as a last resort, one piece at a time.

The six 2,000-iteration runs now on the GPU are on the old meadow. They still answer whether more
training helps the current hybrid, and they don't block steps 1–3, which run on the CPU.

## 10. Results of the § 7 checks (2026-10-26)

`scripts/meadow2_checks.py`: 64 meadows, 12 scripted sheep each, 6,000 steps (5 days), random
starts. Results are in `meadow2-results/checks_v{1,2,3}.json`. Diagnosis was by
`scripts/meadow2_deaths.py`, which traces every sheep that dies.

| Version of the scripted flock | Deaths (of 768) | Most in one meadow | Check 1 | Checks 2–4 |
| --- | --- | --- | --- | --- |
| v1: steering straight at targets | 289 (38 %): 232 thirst, 57 hunger | 12 | fail | pass |
| v2: picks the clear heading nearest the target | 3 (0.4 %), all thirst | 2 | fail | pass |
| v3: + look-ahead includes one step; short look-ahead in tight gaps; sheep placed off trees | **0** | **0** | **pass** | **pass** |

What the deaths were *(from the traces)*:

- **v1.** Sheep steered straight at the nearest water, and a tree or the hut in the way stranded
  them 2–3 tiles from the pond for hundreds of steps. Sheep with urgent hunger walked into the
  pond they stood at, and starved with their mouths on water. The old meadow's flock steered
  the same way; its size made such blocks rarer.
- **v2.** Sheep pressed against the hut, heading along its edge, saw the look-ahead points
  (0.4 tiles and beyond) clear while every 0.12-tile step clipped it. One sheep had been placed
  inside a tree's footprint and could never move. A few spun in tight gaps between treeline
  trees, where no heading was clear for 1.2 tiles.

v3 (all four checks pass):

| Check | Result | Criterion |
| --- | --- | --- |
| 1. Survival | 0 deaths | ≤ 2 %, ≤ 1 per meadow |
| 2. Needs met after day 1: gut energy / hydration, mean | 0.84 / 0.66 | > 0.4 |
| 3. Grass, day 5 against day 0 (dry tiles) | 185 % (1.51 → 2.80) | ≥ 70 % |
| 4. Flock start to water: median / 90th percentile | 9.8 / 13.1 tiles | ≤ 12 / ≤ 20 |

Other measurements:
- fat rose from 0.50 to 0.84;
- reward 2.24 per step;
- bad taste 0.7 % of steps;
- actions: walk 38 %, rest 30 %, graze 11 %, turns 16 %, drink 3 %, trot 1 %.

**Flagged (pre-registered threshold 30 %):** a blind wanderer, walking straight and turning a
random 22.5° once every 30 steps on average, gets its mouth onto water within 500 steps in
**15 %** of sheep. A learning sheep starts out wandering much like this, so finding water at first
will be rare. *(judgement)* Options:
- accept it, since curiosity and memory should beat a blind walker;
- add a third pond;
- make ponds larger.

Also noted *(not a criterion)*: grass grows rather than lasts, to 185 %, so 12 sheep are far from
grazing the meadow down. Regrowth could be slowed if scarcity is wanted later.

## 11. v2.1: changes after the owner's play-test (2026-10-26)

Agreed with the owner on 2026-10-26, after playing meadow v2 in the viewer.

**Time: two clocks.**
- A day is now 2,400 steps (`world.DAY_SCALE` = 2; was 1,200).
- Environment rates scale with the day, so a day or a night costs the same as before: light ramps,
  warmth by day and night, boredom.
- Body rates (gut energy, hydration, fat storage and release) are divided by `drives.NEED_SCALE`.
  It is set by a pre-registered balance rule: **a naive explorer must find water before it would
  die in at least half of starts.** The naive explorer is the wanderer of § 7: it walks straight
  and turns a random 22.5° about every 30 steps. Its first time to water is compared with each
  sheep's hydration budget (starting hydration ÷ loss per step).
- The owner's idea of making needs harsher as training goes on (a curriculum) is noted for later.
  If done, changes should be slow and logged, because a moving reward makes the critic's job
  harder.

**Grass.**
- **A smooth field.** `world.grass_at` blends the four nearest tile centres bilinearly. The mouth
  reads that field, and a bite takes 0.05 of biomass from the same four tiles by their weights.
  What the sheep sees, eats and depletes agree, and there are no tile edges.
- **Nutrition follows density.** A bite gives 0.008 × density of gut energy (0.012 at density 1.5,
  as before; 0.024 at full). Below a density of 0.5 a bite fails, with a bad taste.
- **Regrowth takes days.** It is 0.00013 of the gap to full per step, a time constant of about
  7,700 steps (3.2 days). Bare ground reaches the bite threshold (0.5) in about 1,400 steps
  (0.6 days) and a density of 1.0 in about 3,100 (1.3 days).
  Before, it was 0.0004 per step. Separately, the four-step display had made regrowth look
  instant: one bite at 2.0 showed as the step below and crossed back within about 60 steps.
- **No grass under water.**
- **Rendering (renderer v2, owner, 2026-10-26).** One continuous raised surface whose height and
  colour follow the smooth field: no blades, and no grid of tiles. Two constraints:
  - *Height at most 0.25 tiles.* From the eye at 0.8 tiles, the sight line to the mouth point (0.6
    tiles ahead) never dips below 0.4 tiles over the ground in between, so nearer grass can never
    hide the mouth spot.
  - *The reticle marks the grass surface at the mouth point.* What is drawn there is what a bite
    eats.

**Sheep are solid.** A move that would bring a sheep closer to another living sheep it would
overlap is refused, as walking into a tree is. Nobody is pushed. Before, overlapping sheep were
pushed apart by half the overlap each, which moved sheep that had not acted. Sheep are placed so
that none overlaps another at the start. The scripted flock treats other sheep as obstacles.

**Social is no longer a drive** *(owner: flocking should be learned because it helps, not
rewarded for itself)*.
- The driven levels are energy, hydration, warmth and interest. A sheep feels 6 numbers: those
  four, fat and taste.
- **Huddling:** while resting, each other resting sheep within 1 tile divides the loss of warmth by
  (1 + 0.5 n) and multiplies the gain from rest by (1 + 0.25 n), counting up to 4 neighbours.
- The number of other sheep in view is still recorded (`Flock.seen`), for analysis only.
- The worst drive is now √4 = 2, so the alive bonus of 2.5 still makes every living step worth
  at least +0.5.
- In nature the main reason for flocking is usually given as predation. A predator could be
  added later, if flocking fails to emerge from huddling and from following others to water.

### v2.1 check results (2026-10-26)

`meadow2-results/checks_v21.json`: 64 meadows, 12 sheep, 12,000 steps (5 days of 2,400), with
`NEED_SCALE` = 2.

| Check | Result | Criterion |
| --- | --- | --- |
| 1. Survival | 2 deaths of 768 (0.26 %, both of thirst, in different meadows) | ≤ 2 %, ≤ 1 per meadow: **pass** |
| 2. Needs after day 1: gut energy / hydration, mean | 0.85 / 0.71 | > 0.4: **pass** |
| 3. Grass, day 5 against day 0 | 172 % | ≥ 70 %: **pass** |
| 4. Flock start to water: median / 90th percentile | 10.2 / 13.2 tiles | ≤ 12 / ≤ 20: **pass** |

Other measurements:
- 50 % of resting sheep at night huddle with at least one neighbour;
- a sheep sees 8.5 others on average;
- grazing is 5 % of actions (was 11 % before density-based nutrition);
- fat rose from 0.50 to 0.84.

**The balance rule fails as written.** Each cell is the share of walkers that find water before
their hydration would run out (`walkers_v21.json`):

| Naive explorer | Never within 4,000 steps | NEED_SCALE 1 | 2 | 3 | 4 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| § 7's wanderer: turns only at random | 61 % | 15 % | 20 % | 25 % | 30 % | 35 % |
| Turns away when its next step is blocked | 25 % | 25 % | 40 % | **53 %** | 63 % | 70 % |

The § 7 wanderer spends much of its time pressed against the fence, trees and other sheep, since
it turns only by chance. A sheep that turns when blocked is still blind to water and has no
memory, but doesn't walk into walls. *(judgement)* That is the fairer stand-in for a newborn
learner, which should learn not to push against walls quickly.

**Open, for the owner:** either redefine the naive explorer as the one that turns when blocked and
set `NEED_SCALE` = 3 (hydration from full then lasts about 3,750 steps, 1.6 days), or keep the § 7
wanderer and change the meadow (more or larger ponds).

### Decision and recheck (2026-10-26)

The owner chose the naive explorer that turns when blocked, and `NEED_SCALE` = 3. Recheck
(`meadow2-results/checks_v21_n3.json`):

- all four checks pass;
- **no deaths** among 768 sheep over 5 days;
- the naive explorer finds water before it would die in 53 % of starts (rule: ≥ 50 %);
- 57 % of resting sheep huddle at night;
- fat rose from 0.50 to 0.85.

With `NEED_SCALE` = 3, hydration lasts about 3,750 steps from full (1.6 days), and gut energy
about the same standing still.

## 12. Renderer v2 as built (2026-10-26)

`flock/meadow2/render.py`. Samples: `meadow2-results/render_samples.png` (midday toward and near a
pond, facing away, dusk, night, dawn).

- **As specified:**
  - an RGB strip of 64 × 256, 300° wide, eye at 0.8 tiles;
  - the row table of § 5;
  - flat shapes at the sizes of § 4 (sheep from the side, front or back);
  - distance haze (30 tiles);
  - smooth light, and the hut's window and light pool;
  - the fence on the meadow's edge.
- **The ground is one continuous raised surface.** Each column marches outward over 192 distances
  from 0.15 to 90 tiles. A row shows the first sample whose top rises above its line of sight, so
  nearer grass hides what is behind it. Grass height is at most 0.25 tiles. The colour runs from
  tan (bare) through light green (density 1) to deep green (3).
- **Water is now a smooth field too** (`world.water_at`): the tile mask is blended between tile
  centres and cut at one half, so ponds have curved shores rather than tile staircases.
  Drinking, blocking and drawing all use it.
- **The reticle** marks the grass surface (or water) at the mouth point, using the grass height
  there (`render.reticle_row`). It sits about 8 % above the strip's bottom edge.
- **Stretched rows (for human eyes).** The horizon band's rows are 0.5° and the ground rows near
  the feet 4°, so things near eye level are magnified about 8× vertically in the stored strip.
  That is what the sheep receives. The viewer shows rows at their true angular heights by
  default, and as stored when you tick a box.
- **Storage (decided when training needs it):** the renderer returns RGB floats. § 5 planned a
  one-byte-a-pixel table, but continuous light and haze make that approximate. Plain RGB bytes are
  3 bytes a pixel (49 KB a frame).

## 13. Changes after the second play-test, and the first baselines (2026-10-26)

**Changes (owner, 2026-10-26):**
- **Jostling is back, but uneven.** A sheep that walks into another takes 70 % of the overlap and
  slides around it; the sheep it bumps takes 30 %. A push into water, a tree or out of the meadow
  is not taken. The brief "solid sheep" rule made sheep stop dead and could jam a crowd.
- **A bite crops a patch.** Grass is stored on half-tile cells (96 × 96), still drawn as one
  smooth surface. A bite takes 0.05 from every cell whose centre is within 0.4 tiles of the mouth
  point, and its energy follows the patch's mean density. A grazed patch now shows as a bare spot
  about a tile across. Before, a bite was spread over four whole tiles by their bilinear weights,
  so the spot under the mouth dropped by only a quarter to the whole of a bite and was blurred
  over.
- **Rows follow one smooth curve.** Row heights follow a sinh, finest (0.6° a row) at 8° below
  the horizon and growing smoothly to 5.4° at the bottom. This replaces the three hard-edged bands
  of 3, 0.5 and 4°. Rows per feature, before → after:
  - a pond 5–7 tiles away: 5.1 → 4.3;
  - a pond 10–12 tiles away: 1.5 → 1.1;
  - ground 2–3 tiles away: 1.7 → 6.6;
  - the sky: 28 → 18.

  The viewer has a pixel-grid overlay and a heatmap of rows per degree to judge it by eye.
- The PC froze completely once while the GPU, a CPU-heavy check and the viewer all ran together,
  with nothing in the Windows logs. Heavy jobs now run one at a time.

**Checks with these changes** (`meadow2-results/checks_v21_patch.json`):
- all four pass, with 0 deaths;
- grass ends at 176 % of its start;
- the naive explorer finds water before it would die in **65 %** of starts at `NEED_SCALE` 3
  (53 % before jostling; it no longer gets stuck against other sheep);
- 55 % of resting sheep huddle at night.

**Baselines on the fixed evaluation** (`scripts/m2_eval.py`: 16 meadows, seeds 50,000 + i, 12
sheep, 3 days of 2,400 steps; `meadow2-results/eval/`). Reward counts 0 after death, so dying
costs, and includes the alive bonus of 2.5. D is the mean drive among the living.

| Policy | Reward / step (± SE) | Mean drive D | Died | Days survived | Drink success | Bad taste per step |
| --- | --- | --- | --- | --- | --- | --- |
| rest always | 0.55 ± 0.01 | 0.80 | 100 % | 0.97 | — | 0 |
| random | 0.83 ± 0.04 | 0.71 | 86 % | 1.40 | 2 % | 13 % |
| scripted (privileged rules) | **2.36 ± 0.01** | 0.14 | 0 % | 3.0 | 57 % | 0.6 % |
| **PPO on privileged features, seeds 0 / 1 / 2** | **2.470 / 2.475 / 2.461** (SE ≤ 0.012) | 0.03 | 0 / 0 / 0.5 % | 3.0 | 98–100 % | ≤ 0.1 % |

*(judgement)* With the alive bonus, survival dominates the score: a sheep that lives scores about
2.2–2.4 a step, and one that dies early much less. The scale runs from about 0.5 (dead within a
day) to about 2.4 (alive, needs met).

PPO (`scripts/m2_ppo.py`, 1,500 updates, about 12 minutes per seed on the RTX 4080;
`meadow2-results/ppo/`, `meadow2-results/eval/ppo_s*.json`):
- It learns within about 100 updates and nearly solves the meadow. On the last day, gut energy is
  1.01–1.06, hydration 0.98–1.02, warmth 1.02–1.06, and fat 0.94–0.97.
- One sheep died in 576 sheep-runs.
- It beats the scripted flock (2.36), which waits until a need is pressing before acting.
- It rests 20–25 % of the time, but huddles less than the scripted flock (31–37 % of night rest,
  against 65 %).

*(judgement)* Meadow v2 is learnable from the right information, with clear headroom between
random (0.83) and PPO (2.47). The question for the next rung is how much of this a sheep can get
from its own eyes.

## 14. The vision hybrid on meadow v2: design and pre-registered criteria (2026-10-26)

Rung 1 of the ladder in `m1-map-world-model.md`: today's world model, given nothing for free.
`scripts/m2_hybrid.py`:
- **The world model** (`flock/wm.py`, new RGB mode) learns from birth, from each sheep's own view
  (renderer v2, 64 × 256 RGB) and its 6 felt numbers. It reconstructs the view by summed squared
  error on colour, as DreamerV3 does for images; otherwise it is unchanged (deterministic state
  384, stochastic 32 × 16).
- **The policy** is PPO on the model's state (h, z), trained from real experience, with death
  handled as in the PPO baseline.
- **Per iteration:**
  - 8 meadows of 12 sheep act for 64 steps;
  - 4 sheep a meadow feed the replay, in sequences of 32 steps kept only if the sheep was alive
    throughout;
  - 16 world-model updates on batches of 8 sequences;
  - PPO on every living sheep's steps, after 50 iterations of world-model-only infancy.

**Rendering cost:** every sheep needs its own view every step. Rendering the views one after
another inside the compiled program (`render.render_flock` and `render_meadows`) takes 0.22 ms a
view on the RTX 4080. Vectorising them with `vmap` took 2.7 ms a view, because its batched
intermediates are memory-bound.

**Pre-registered criteria**, on `m2_eval.py` (16 fixed meadows, 3 days), three seeds, 600
iterations each, all settings default:
- *Worked if* every seed beats random (0.83) by more than 5 SE, with at most 20 % of sheep dying.
- *Strong if* every seed scores at least 2.0 with at most 5 % dying. That is between random and
  the scripted flock (2.36); privileged PPO scores 2.47.
- *Also reported, not criteria:* drinking success, the last day's hydration, and the action mix.
  In the old meadow, drinking on dry ground was the signature failure.

### Results (2026-10-26): criteria not met

`meadow2-results/eval/hybrid_s*.json` and `meadow2-results/hybrid/s*_log.jsonl`. Each seed took
about 16 minutes to train.

| Seed | Reward / step (± SE) | Above random | Died | Days survived (of 3) | Drink success | Graze success | Last-day hydration (living) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.24 ± 0.07 | 5.9 SE | 76 % | 1.71 | 37 % | 89 % | 0.63 |
| 1 | 1.21 ± 0.08 | 4.6 SE | 68 % | 1.68 | 19 % | 93 % | 0.89 |
| 2 | 1.36 ± 0.07 | 7.8 SE | 58 % | 1.87 | 32 % | 84 % | 0.87 |

For reference: random 0.83 (86 % die), scripted 2.36, privileged PPO 2.47.

**Against the criteria:** *worked* fails. Seed 1 is only 4.6 SE above random, and every seed loses
far more than 20 % of its sheep. *Strong* fails.

**What the sheep did learn, from their own eyes alone** *(judgement)*:
- **Food is solved.** Grazing succeeds 84–93 % of the time; gut energy stays near 1.1 and fat near
  0.95.
- **Drinking is partly learned.** 19–37 % of drink attempts hit water, against 2 % for random. The
  old meadow's vision sheep drank on dry ground almost every time.
- **They die of thirst, mostly on day 2.**
- **They do not huddle at night** (1–5 % of night rest), unlike the scripted flock (65 %) and PPO
  (31–37 %).

**Training curves:** reward per living sheep-step rose to about 2.0–2.2 by iteration 300, then
stayed flat to 600. The share of sheep alive swung between about 35 % and 95 % with the meadow
restarts, so deaths were common in training too.

**Open:** whether the limit is what the world model's state holds about water (where it is), or
the policy and the amount of training.

### Probe of the world model's state (2026-10-26)

`scripts/m2_probe.py`; results in `meadow2-results/probes/{s0,s1,s2,untrained}.json`. 32 fresh
meadows (seeds 70,000 + i) are run with the scripted flock. Each sheep's state is filtered from its
own view and felt numbers. Probes are fitted on 24 meadows (86,400 samples) and scored on 8 (28,800).
"In view" means inside the 300° field and within 16 tiles; water is in view in 84 % of samples. The
probe floors the spread it standardises inputs by: without that, a latent unit constant in the
training meadows exploded the MLP probe's predictions (R² around −10⁵ in a smoke test). The old
meadow's `m1_probe_latent.py` lacks the floor, so its strongly negative R² values may be distorted.

MLP probe, held-out meadows:

| Target | Trained (seeds 0–2) | Untrained model |
| --- | --- | --- |
| felt numbers (R²) | 0.98–0.99 | −0.29 to 0.36 |
| mouth on water (balanced accuracy) | 0.86–0.92 | 0.54 |
| mouth on biteable grass (balanced accuracy) | 0.90–0.92 | 0.54 |
| cover distance, in view (R²) | 0.63–0.68 | 0.14 |
| cover direction in view, forward / left (R²) | 0.15–0.30 / 0.25–0.34 | −0.67 / −0.60 |
| **water direction in view, forward / left (R²)** | **−0.20 to −0.29 / −0.41 to −0.51** | −0.55 / −0.47 |
| **water distance in view (R²)** | **−0.06 to −0.35** | −0.45 |
| grass direction in view, forward / left (R²) | −0.41 to −0.48 / −0.49 to −0.58 | −0.61 / −0.67 |
| anything out of view (R²) | all below 0 | all below 0 |

Ridge probes agree: at best 0.17 for water distance in view, and about 0 for water direction.

Reading *(judgement)*:
- The probe works. It reads the body almost perfectly and large objects reasonably.
- The state holds what is at the mouth, and the body's levels, so grazing and drinking where the
  sheep stands are learnable. That matches what the hybrid learned.
- **Where water is, is not in the state**, even when it is in view, and nothing is remembered once
  it is out of view. A policy on this state can only find water by wandering, which matches the
  deaths from thirst. The old meadow's model had the same gap.
- This is the evidence the ladder (`m1-map-world-model.md`) asked for: the vector state does not
  organise "where". Rung 2 (more training, a larger model) is untested.

### Rung-2 test: pre-registered (2026-10-26, owner's decision)

The same hybrid, seed 0, trained for 2,000 iterations instead of 600, otherwise unchanged; then
evaluated (`m2_eval.py`) and re-probed (`m2_probe.py`, same 32 meadows).

**Rule:** take the better of the MLP and ridge probes for each component of the direction to water
in view. If both the forward and the left components stay below R² 0.3, move to rung 3 (a state
with spatial shape, all learned). If either reaches 0.3, rung 2 (more training, a larger model)
gets a full three-seed test first. The evaluation score is reported but does not decide.

### Rung-2 test: result (2026-10-26)

Seed 0, 2,000 iterations: `meadow2-results/eval/hybrid_long_s0.json`, `hybrid/long_s0_log.jsonl`,
`probes/long_s0.json`.

| Seed 0 | 600 iterations | 2,000 iterations |
| --- | --- | --- |
| Reward / step (± SE) | 1.24 ± 0.07 | 1.45 ± 0.10 |
| Died | 76 % | 60 % |
| Drink success | 37 % | 18 % |
| Water direction in view, forward / left: better of ridge and MLP (R²) | 0.00 / −0.15 | **0.06 / −0.11** |
| Water distance in view (R²) | 0.14 | 0.18 |
| Cover direction in view, forward / left (R²) | 0.29 / 0.34 | 0.48 / 0.58 |
| Water at the mouth (balanced accuracy) | 0.92 | 0.94 |

**Rule outcome:** both components of the direction to water stay below 0.3, so the pre-registered
rule says move to rung 3.

*(judgement)*
- Training longer improved survival and made big objects (cover) clearly more readable, but not
  water.
- The Fable 5.1 review (`review-2026-10-26-water.md`) cautions that this probe cannot reach 1 by
  design: its "in view" is not true visibility, and its nearest-tile targets are noisy at the
  shore. It recommends a decode-from-posterior water check (its D1) before acting on the rule.

## 15. The diagnosis gate (pre-registered 2026-10-26, before any of its results)

Adopted from the plan review (`review-2026-10-26-plan.md`), agreed with the owner. The rung-2 rule's
outcome stands as recorded, but rung 3 is **deferred pending this gate**: rung 2 tested more training
of what the loss already pays for, not what rung 3 changes.

**Measurements** (on the 600-iteration seed 0 and the 2,000-iteration seed 0, and the new arm):
- **D1, water reconstruction.** Views are decoded from the model's posterior. A true water pixel
  counts as redrawn if the decoded colour is closer to its true colour than to the colour the same
  pixel shows with the water removed (rendered again without water). Reported as the share redrawn
  ("recall"), by the pixel's distance (< 4, 4–8, 8–16, ≥ 16 tiles) and by day or night.
- **D2/D3, a better probe.** "Visible" means water pixels are actually on screen (from the
  renderer's own mask). Targets are the image column of the visible water's centre (its bearing),
  and its distance. Results are binned by distance, and "just out of view" means not visible now
  but visible within the last 30 steps. Ridge is the headline; an MLP with early stopping on held-out
  meadows is reported beside it. The encoder's embedding, h and z are probed separately.
- **The policy-reads-encoding arm.** PPO sees (h, z) and the current view's embedding (stop-gradient,
  so PPO does not train the encoder). One seed, 600 iterations, `m2_eval`.
- **Dispersion and the information gap**, added to `m2_eval`: nearest-neighbour distance, other sheep
  in view, and the share of thirsty steps (hydration < 0.5) in which a sheep cannot see water but a
  flockmate within 10 tiles saw water in the last 300 steps. "Sees water" here is a geometric proxy:
  a water tile within 16 tiles inside the field of view, ignoring occlusion.

**Branches**, decided by these written criteria, for water by day within 8 tiles (D1) and visible water
(D2):
- **A. "The loss is the limit"** if D1 recall within 8 tiles by day is below 0.5. Then the fixes
  are to the loss and the data (multi-step prediction, the KL weight, the replay), and the night-water
  world change goes to the owner. Rung 3 waits, because it would inherit the blind spot.
- **B. "The state is the limit"** if D1 recall is at least 0.5 and the embedding's ridge probe for
  bearing is at least 0.5, but h and z read at most 0.3. Then the bearing is seen and drawn but not
  carried in a readable form; rung 3 (and a column-preserving state) is justified.
- **C. "Perception is the limit"** if D1 recall is at least 0.5 but the embedding cannot read bearing
  either (below 0.3). Then the encoder bottleneck is the problem: a column-preserving encoder first.
- If the policy-reads-encoding arm clearly beats the plain hybrid (more than 5 SE), the bearing is
  usable from the eyes. That supports B or C over A.

**The week's success criterion** for Track A (pre-registered): at most 20 % of sheep die on `m2_eval`,
**and** the ridge R² for water bearing is at least 0.5 on frames where the pond is visible by day
within 8 tiles, **and** at least 0.3 for frames within 30 steps after it leaves view. Reported for
embedding, h and z separately, with time to first drink and deaths per day.

### Gate results (2026-10-26)

`meadow2-results/diag/{s0,long_s0,embed_s0}.json` (`scripts/m2_diag.py`: 32 meadows, seeds 70,000 + i,
scripted flock; D1 on every 10th step) and `meadow2-results/eval/hybrid_embed_s0.json`.

**D1, share of true water pixels redrawn from the model's state:**

| Model | Day, < 4 tiles | Day, 4–8 | Day, 8–16 | Night, < 4 | Night, 4–8 | **Day, within 8** |
| --- | --- | --- | --- | --- | --- | --- |
| 600 iterations | 0.73 | 0.33 | 0.00 | 0.18 | 0.04 | **0.63** |
| 2,000 iterations | 0.86 | 0.66 | 0.06 | 0.58 | 0.19 | **0.81** |
| arm (policy reads the encoding) | 0.75 | 0.43 | 0.00 | 0.03 | 0.01 | **0.67** |

**D2, water bearing with water visible by day within 8 tiles** (ridge R², early-stopped MLP in
brackets):

| Model | Encoder's embedding | h | z | h and z |
| --- | --- | --- | --- | --- |
| 600 iterations | **0.69** (0.44) | 0.27 (0.31) | 0.10 (0.00) | 0.27 (0.23) |
| 2,000 iterations | **0.64** (0.40) | 0.30 (0.35) | 0.06 (−0.04) | 0.31 (0.31) |
| arm | **0.68** (0.40) | 0.30 (0.28) | 0.08 (−0.01) | 0.30 (0.26) |

By distance, the embedding's ridge R² for bearing is 0.63–0.71 under 4 tiles, 0.29–0.35 at 4–8, and
negative at 8–16. **D3:** within 30 steps after water leaves view, every input reads its bearing
below 0, so nothing is remembered. (The R² for distance beyond 16 tiles, about −10¹¹, comes from
2,258 frames with almost constant targets and should be ignored.)

**The arm** (seed 0, 600 iterations) scores 1.75 ± 0.11 against 1.24 ± 0.07, with 43 % dying
against 76 %. That is 3.9 SE: suggestive, short of the 5 SE the gate asked for. First dispersion
and gap numbers for learned sheep:
- nearest neighbour 6.1 tiles away;
- 3.5 other sheep in view;
- thirsty sheep see water 39 % of the time;
- **information gap 24 %**.

**Branch, by the written rules:** D1 within 8 tiles by day is at least 0.5 for all three models, so
**not A**. The embedding reads bearing at 0.64–0.69 (at least 0.5), and h and z read at most about
0.30 (exactly at the boundary for the 2,000-iteration model: h 0.302, h and z 0.305). So this is
**branch B, "the state is the limit"**, cleanly for the 600-iteration model and the arm, and at the
boundary for the 2,000-iteration one. The arm's gain points the same way.

*(judgement)* The picture is mixed by distance and light:
- **Near water by day:** the eyes see it and the decoder draws it, but the model's memory keeps only
  part of its bearing. Branch B, so a column-preserving state and rung 3 are justified.
- **Far water (8+ tiles), and water at night:** neither drawn nor encoded. That is branch A's
  problem (the loss, and the night-water world fact), even though the rule's threshold, set within
  8 tiles, did not trigger it.
- **No memory once out of view.**

## 16. Branch B's fixes and the loss fixes: screening (pre-registered 2026-10-26)

**World change (owner, 2026-10-26):** water reflects the sky. A water pixel mixes in 30 % of the
sky at the mirrored elevation, and is never lit below 45 % of its day brightness, so ponds stay
visible at night (`render.REFLECT`, `WATER_FLOOR`). Night contrast had been about 70× lower than
by day, and the models barely redrew water at night. This changes the world, so every
vision-based result after it is on the new rendering. Scripted and privileged-PPO results are
unaffected: they do not use the view.

**Arms** (seed 0, 600 iterations, `m2_hybrid.py`; the policy reads (h, z) only, since the probe is
about the state):

| Arm | World model | Loss | Replay |
| --- | --- | --- | --- |
| **N0**, the new baseline | vector RSSM | dynamics KL weight 1 (DreamerV3's), was 0.5 | 2,048 sequences; half of each iteration's added, about 64 iterations of history (was 1,024, all added, 16) |
| **N1** | vector RSSM | as N0, plus multi-step prediction (weight 1; horizons 2, 4, 8; starts every 8 steps) | as N0 |
| **S1** | rung 3, spatial (`flock/wm_spatial.py`) | as N1 | as N0 |

Each is evaluated (`m2_eval.py`) and diagnosed (`m2_diag.py`).

**Screening rule** (written before the runs):
- A fix is **kept** if, against N0, it raises the ridge R² for water bearing from h and z (visible
  by day within 8 tiles) by at least 0.1, **or** brings the bearing just after water leaves view
  (D3, the better of sine and cosine) to at least 0.3.
- The arm that does best on those, if any is kept, goes to three seeds.
- The week's criterion (§ 15) stays the target. Evaluation scores are reported but do not decide
  the screening, which is about what the state holds.

**Deviations before launch (2026-10-26):**
- **S1 uses batches of 4 sequences per world-model update, not 8,** with the same number of
  updates. With 8, the spatial model ran out of GPU memory.
- **One compiled world-model update is called 16 times per iteration,** instead of one compiled
  call unrolling all 16. With multi-step prediction, the unrolled version took 8 minutes to compile
  for N1. For S1 it reached 15 GB of host RAM while compiling, which killed the process and then
  crashed WSL. The arithmetic is unchanged.
- **Training runs go through `runs/watchdog.sh`,** which kills a job above 12 GB of RAM.

### Screening results (2026-10-26)

`meadow2-results/eval/hybrid_{n0,n1}.json`, `eval/hybrid_spatial_s1arm.json`, `diag/{n0,n1,spatial_s1arm}.json`, plus `hybrid/n0_log.jsonl`, `n1_log.jsonl` and
`spatial_s1arm_log.jsonl`. Peak RAM was 6.8, 7.5 and 8.2 GB, all under the 12 GB watchdog.

| Arm | Reward / step | Died | Water redrawn by day, within 8 tiles | Night, < 4 / 4–8 | Bearing from h and z, day within 8 (ridge) | Bearing just after leaving view (D3) |
| --- | --- | --- | --- | --- | --- | --- |
| N0 (KL 1, longer replay) | 1.41 ± 0.10 | 53 % | 0.70 | 0.73 / 0.55 | 0.20 | below 0 |
| **N1** (N0 + multi-step) | 1.38 ± 0.08 | 62 % | 0.68 | 0.74 / 0.50 | **0.35** | below 0 |
| S1 (spatial + multi-step) | 0.78 ± 0.04 | 97 % | 0.04 | 0.01 / 0.00 | −0.97 | below 0 |

**By the rule:**
- **N1 is kept.** Multi-step prediction raises the bearing from h and z by 0.15 over N0 (needed:
  0.1). Most of the gain is within 4 tiles (0.46 against 0.39). N1 goes to three seeds (seeds 1
  and 2 to add).
- **S1 is not kept.** The spatial model's decoder redraws almost no water, its state reads no
  bearing, and it evaluates at random's level (0.78; random scores 0.83). Its training reward per
  living step (about 2.0) and the share alive in training (0.57 against N1's 0.65) look normal,
  and its checkpoint loads completely (all 91 parameter arrays), so this is a real failure, not a
  loading fault.
- Its column encoder does read bearing: the embedding's MLP probe gives 0.56, the best MLP score
  so far. The failure is in the grid state and the attention decoder, at this budget. *(judgement)*
- **The week's criterion is not met:** bearing from the state is 0.35 (needed 0.5), and nothing is
  remembered after water leaves view.

**The night-water change worked.** Water at night within 4 tiles is now redrawn 73–74 % of the
time, against 3–18 % before.

**A naming collision:** the S1 arm wrote into `~/flock2_data/m2_hybrid/s1`, the folder of the
earlier vector hybrid's seed 1. It appended its log there and overwrote that checkpoint. Seed 1's
evaluation and log were already saved under `meadow2-results/`, so no recorded result is lost.
Future arm names will not reuse seed folder names.

### N1 on three seeds (2026-10-26)

`meadow2-results/{eval/hybrid_,diag/}n1{,_seed1,_seed2}.json`.

| N1 seed | Reward / step | Died | Water redrawn, day within 8 | Bearing from h and z (ridge) | Bearing from the embedding | D3 (best of sin, cos) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.38 ± 0.08 | 62 % | 0.68 | 0.35 | 0.68 | −0.54 |
| 1 | 1.11 ± 0.07 | 79 % | 0.69 | 0.26 | 0.64 | −0.14 |
| 2 | 1.65 ± 0.09 | 45 % | 0.71 | 0.34 | 0.68 | −0.64 |
| N0 (seed 0, reference) | 1.41 ± 0.10 | 53 % | 0.70 | 0.20 | 0.66 | −0.65 |

*(judgement)*
- Multi-step prediction raises the bearing held in the state consistently: 0.26–0.35 on three
  seeds, against 0.20 for N0. But it stays well short of the week's 0.5.
- Nothing is remembered once water leaves view.
- Survival varies widely between seeds (45–79 % die), and on average is not better than N0.
- **The week's criterion is not met by any arm so far.**

## 17. One more learned fix, aimed at memory (pre-registered 2026-10-27, owner's decision)

**Arm M** (seed 0, 600 iterations): N1, plus three changes.
- **Multi-step prediction reaches further:** horizons 2, 4, 8, 16 and 32 steps (was up to 8), starts
  every 16 steps.
- **Training sequences of 64 steps** (was 32). The replay holds 1,024 sequences, the same memory as
  2,048 of 32, with half of each iteration's added.
- **The policy also reads the view's embedding** (stop-gradient), the arm that gave the best
  survival so far.

Run name `mem_s0` (no reuse of seed folder names). Evaluated and diagnosed as before.

**Rule:**
- If the bearing just after water leaves view (D3: the better of sine and cosine, ridge, from h and
  z) reaches **at least 0.3**, learned memory is working. M goes to three seeds and the week's
  criterion is re-checked.
- If it stays **below 0.3**, the learned options for memory are exhausted at this budget, and the
  decision returns to the owner: rung 4 (a body sense of movement, and the memory shifted by it), or
  move on with a view-reading policy and "calling", not describing.
- The bearing from h and z in view, and the evaluation, are reported beside it.

Deviation before launch: arm M uses **batches of 4 sequences** of 64 steps (8 of 64 ran the GPU out
of memory in the smoke test). That is the same number of frames per update as N1's 8 of 32. Smoke
test: 6.3 GB peak RAM.

### Arm M: result (2026-10-27)

`meadow2-results/{eval/hybrid_,diag/}mem_s0.json`. Peak RAM 8.8 GB.

| | Arm M | N1, seed 0 |
| --- | --- | --- |
| Reward / step (± SE) | 1.54 ± 0.08 | 1.38 ± 0.08 |
| Died | 57 % | 62 % |
| Water redrawn, day within 8 | 0.62 | 0.68 |
| Bearing in view from h and z (ridge) | 0.26 | 0.35 |
| **Bearing just after leaving view, from h and z (best of sin, cos)** | **−0.52** | −0.54 |

**By the rule:** D3 stays far below 0.3, so the learned options for memory are exhausted at this
budget and **the decision returns to the owner.**

**A caveat to weigh first** *(judgement)*: the D3 probe has not been validated on an input that is
known to carry memory. Every model so far, trained or not, scores below 0 on it. Its target is the
true current bearing to the nearest water tile, after the scripted sheep has kept moving and
turning for up to 30 steps. A check that a probe of this design can read memory when memory exists
would take minutes: for example, an input of the bearing at the moment water was last seen,
together with the turns made since.

### Validating the D3 (memory) probe: pre-registered (2026-10-27, owner's decision)

On the same frames as D3 (water not visible now, visible within the last 30 steps, nearest water
more than 1.5 tiles away), the D3 probe is applied to inputs whose content is known:
- **oracle:** the sine and cosine of the true current bearing (its own target). Should be about 1.
- **ideal memory:** the bearing when water was last seen (from the image), corrected for the
  heading change since: the direction a perfect memory would hold if the sheep had turned but not
  walked. Given as its sine and cosine.
- **raw memory parts:** the last-seen bearing's sine and cosine, the heading change's sine and
  cosine, and the steps since. The MLP can combine these; ridge cannot.
- **noise:** random numbers of the same width (the floor).

**Rule:**
- If ideal memory reaches a ridge R² of **at least 0.5**, with the oracle at **0.95 or more**, the
  probe can detect memory. "No memory" stands, and the owner decides between rung 4 and moving on.
- If ideal memory scores **below 0.3**, the probe's design cannot detect memory even when it is
  present. The probe is redesigned before that decision.

### D3 probe validation: result (2026-10-27)

`meadow2-results/diag/d3_validation.json`: the 8,021 D3 frames of arm M's collection, with 6,849
used for fitting and 1,172 for testing.

| Input | Ridge R² (sin / cos) | Early-stopped MLP R² (sin / cos) |
| --- | --- | --- |
| oracle (the target) | 1.00 / 1.00 | 1.00 / 1.00 |
| **ideal memory** (bearing when last seen, turned by the heading change since) | **1.00 / 1.00** | 1.00 / 1.00 |
| raw memory parts | 0.85 / 0.97 | 0.99 / 1.00 |
| noise (5 numbers) | −0.02 / −0.01 | −0.09 / −0.03 |
| arm M's h and z | −0.74 / −0.46 | −0.86 / −0.38 |

**By the rule:** ideal memory scores 1.00, at least 0.5, and the oracle 1.00, at least 0.95. The
probe can detect memory, so **"no memory" stands.**

*(judgement)*
- **Correcting for turns alone is enough:** within 30 steps of losing sight of water, the bearing
  when last seen, turned by the sheep's own heading change, gives the current bearing almost
  exactly. The steps walked barely matter. So the memory a sheep needs is a remembered direction
  rotated by its own turns.
- **The world model has the information it needs.** It is given the actions, including every turn,
  but it has not learned to rotate a remembered direction by them. This is a learning failure, not
  missing information.
- That is why, of rung 4's pieces, a body sense of movement alone would add little: the actions
  already carry the turns. The useful piece would be a memory that is rotated by the turns.
- **A limit of this validation:** the noise control was 5 numbers wide, not as wide as h and z
  (1,408). Ridge on 1,408 inputs fitted on 6,849 samples can overfit. That may explain why h and z
  score below the noise rather than at it, though it cannot turn a real signal into a negative R².

## 18. Rung 4, minimal: a memory rotated by the sheep's own turns (pre-registered 2026-10-29, owner's decision)

**Why this piece only.** § 17's validation showed that, within 30 steps of losing sight of water,
the bearing when last seen, turned by the sheep's own heading change, gives the current bearing
almost exactly. The world model is given the turns but has not learned to rotate anything by them.
So rung 4 adds that one operation, and nothing else: no body sense of movement, no map, no
translation by steps walked.

**The ring** (`flock/wm.py`, `rot_bins`; `m2_hybrid.py --model rotmem`):
- 32 bins × 16 numbers, carried in h beside the GRU's 384. One bin is 11.25°, so a 22.5° turn rolls
  the ring by exactly 2 bins.
- Each step, the ring is first **written** from the stochastic state z and the GRU state through a
  learned gate (biased toward keeping, like the GRU's), then **rolled** 2 bins one way if the action
  was "turn left", 2 bins the other way if "turn right", and left in place otherwise.
- The ring is **read** by the GRU's input and by everything that reads h (prior, posterior, decoder,
  reward and felt-number heads, and the policy).
- **Built in:** only the roll, i.e. that a turn is a turn of 22.5°. **Learned:** what is written,
  into which bins, how much is kept, and how it is read. Nothing tells the model which bin is
  "ahead" or that the view covers 300°.
- In open-loop multi-step prediction the ring is rolled by the planned turns and written from the
  model's own predicted z. That is where the learning signal for it should come from: predicting
  what reappears when the sheep turns back.

Tests (`tests/test_wm_rotmem.py`): a left turn rolls by 2 bins, a right turn undoes it, no other
action moves it, 16 left turns make a full circle, the write receives gradient through the losses,
and a model without the ring is unchanged.

**Arm R** (seed 0, 600 iterations): arm M (§ 17: multi-step horizons to 32, sequences of 64,
batches of 4, the policy also reading the view's embedding) **plus the ring**. That is the only
change. Run name `rot_s0`. Evaluated (`m2_eval.py`) and diagnosed (`m2_diag.py`) as before.
h and z are now 1,920 numbers wide (was 1,408).

**Rule** (written before the run):
- **Memory works** if D3 (the bearing just after water leaves view; the better of sine and cosine,
  ridge, from h and z with the ring) reaches **at least 0.3**. R then goes to three seeds, the
  week's criterion is re-checked, and the communication work builds on it.
- **Validity:** D3 is also run on noise as wide as h and z (1,920 numbers), answering § 17's caveat.
  If that noise scores **0.1 or more**, the probe is overfitting at this width and the D3 result is
  not counted until the probe is fixed.
- **Not working** if D3 stays below 0.3. The decision returns to the owner.
- **Reported, not deciding:** D3 from the ring alone and from the GRU state alone, the
  early-stopped MLP's D3, the bearing in view, water redrawn (D1), reward and deaths against arm M
  (1.54 ± 0.08, 57 % died).

### Arm R: result (2026-10-29)

`meadow2-results/{eval/hybrid_,diag/}rot_s0.json`, `hybrid/rot_s0_log.jsonl`. Peak RAM 9.3 GB.

| | Arm R (M + ring) | Arm M |
| --- | --- | --- |
| Reward / step (± SE) | 1.54 ± 0.11 | 1.54 ± 0.08 |
| Died | 48 % | 57 % |
| Water redrawn, day within 8 | 0.70 | 0.62 |
| Bearing in view from h and z (ridge) | 0.37 | 0.26 |
| … from the ring alone / the GRU state alone | 0.41 / 0.38 | — |
| **D3 from h and z, ridge (sin / cos)** | **−1.08 / −0.26** | −0.74 / −0.46 |
| D3 from the ring alone (sin / cos) | −1.10 / −0.56 | — |
| D3, early-stopped MLP on h and z (sin / cos) | −1.43 / −0.37 | — |
| D3 on noise 1,920 wide (sin / cos) | −0.28 / −0.24 | — |

**By the rule:**
- **Validity:** the wide noise scores −0.28 / −0.24, below 0.1, so the D3 result counts.
- **D3 stays below 0.3** (best −0.26), so **the ring did not give memory.** The decision returns to
  the owner.

*(judgement)*
- **The ring is used, but for the present, not the past.** It carries the bearing of water in view
  better than the GRU state (0.41 against 0.38), and the state as a whole now reads it better than
  any earlier arm (0.37). But once water leaves view, nothing in the ring points to it.
- The likeliest reason is the **write**: every bin is rewritten from z each step through one learned
  gate. Nothing in the design ties a bin to a direction in the view, so there is no reason for the
  bins behind the sheep to be left alone, and the gate can simply learn to refresh everything.
- **The wide noise answers § 17's caveat:** ridge at this width overfits to about −0.25 on these
  frames. h and z's cosine score (−0.26) is at that floor; the sine scores (about −1.1, also for the
  current view's embedding) are well below it, which suggests the probe meets meadow-specific
  patterns that do not carry to the test meadows.
- Survival is a little better (48 % against 57 % died) at the same reward, on one seed: not a
  difference to rely on.

### Where this leaves Track A (2026-10-29)

Paused for the owner's decision. The summary of every arm, the review's reading (the meadow may not
demand this kind of memory) and the proposed staged plan are in [`status-2026-10-29.md`](status-2026-10-29.md)
and [`review-2026-10-29-staged.md`](review-2026-10-29-staged.md).

## 19. Stage 0: does the meadow demand memory? (pre-registered 2026-10-29, before any run)

The owner adopted the staged plan ([`status-2026-10-29.md`](status-2026-10-29.md) § 5) and chose to run
stage 0 first. Stage 0 trains no learning sheep. It measures what the *world* demands, with privileged
policies, which the owner's principle allows in evaluation and probes. Nothing from it is carried into
a sheep. Two parts, the free one first.

### 19.1 Death forensics (free: the existing checkpoints, re-evaluated with a trace)

`m2_eval.py --deaths <file>` re-runs the fixed evaluation (16 meadows, 3 days) and, for every sheep
that dies, records the day of death, its levels, the distance to the nearest water and the steps since
water was last inside an **8-tile** cone (the eyes' range, spec § 15) and since it was last within 8
tiles at all. The summary also gains, for every policy: deaths per day; the share of living steps
within 8 tiles of water; and `thirsty_sees_water` and the information gap recomputed with an 8-tile
cone beside the 16-tile ones (the review's § 5.2). Run on: the scripted flock, privileged PPO
(`ppo_s0`), and the hybrids `s0`, `long_s0`, `embed_s0`, `n1` (three seeds), `mem_s0`, `rot_s0`.

**Reading, written before running:** if the median hybrid death comes more than **300 steps** after
water was last in the 8-tile cone, a 30-step memory could not have saved most of them, and the review's
first branch (§ 19.2) is already the likely one. If instead most deaths come within 300 steps of losing
sight of water, the D3 capacity was plausibly the right target. The forensics decide nothing by
themselves; the ladder does.

### 19.2 The myopic-oracle ladder

`m2_ppo.py` unchanged in its learning, with two options in `privileged.observe`:

- **`--water_range r`:** the water features (forward, left, distance) describe the nearest water tile
  that lies **within r tiles and inside the 300° field of view**, and are masked to (0, 0, 1) when there
  is none. (Not "the nearest water, hidden when far": a sheep with a pond ahead and a nearer one behind
  would see the one ahead.) Occlusion by trees and night are ignored, so each arm is an *upper bound* on
  what eyes of that range could give.
- **`--memory kind --window W`:** four extra inputs, an ideal memory of the last water seen under the
  same mask: forward, left, distance / 32 and age / W, masked to (0, 0, 1, 1) once the age exceeds W
  or if water was never seen. Two kinds. **`rot`:** the bearing when last seen, turned by the sheep's
  heading change since (the capacity the D3 probe measures; distance held), which ignores the sheep's
  own travel. **`place`:** the position of the last-seen water tile, re-expressed in the sheep's frame
  each step from the true pose (perfect path integration). The memory is carried through the rollout
  and the evaluation and reset with the meadow.

**Arms**, each 1,500 updates as `ppo_s*` (about 7 minutes a seed), **three seeds each**, evaluated by
`m2_eval.py` on the same 16 meadows with the same observation function:

| Arm | Range | Memory |
| --- | --- | --- |
| `r4`, `r8`, `r16` | 4, 8, 16 tiles | none |
| `rinf` | unlimited | none (the existing `ppo_s0–s2`, not rerun) |
| `r8_rot30` | 8 | rot, 30 steps |
| `r8_place30` | 8 | place, 30 steps |
| `r8_place1000` | 8 | place, 1,000 steps |

**Rule** (the deciding statistic is the death share on `m2_eval.py`, mean of three seeds, every seed
reported):

1. **`r8` loses ≤ 20 %:** the meadow does not demand memory at the eyes' present range. The vision
   sheep's limit is reading what they see and using it. Stage 1 (the reactive sheep) follows, and
   stage 2a's pressure must be built, not assumed.
2. **`r8` loses ≥ 40 % and `r8_rot30` brings it to ≤ 20 %:** the D3 capacity was the right target
   and the failure was learning, not demand. Stage 1 still runs (it is the demand check's "previous
   brain"), and stage 2a's module is aimed at exactly this memory on the present world.
3. **`r8` loses ≥ 40 %, `r8_rot30` does not rescue it, and `r8_place1000` does:** the demanded
   capacity is a place memory lasting hours (or far perception), and D3 has been measuring the wrong
   thing. Stage 2a's probe and pressure are redesigned before it runs. `r8_place30` against `r8_rot30`
   says whether path integration or duration made the difference.
4. **`r8` between 20 and 40 %:** memory is weakly demanded. Proceed to stage 1; whichever memory arm
   rescues `r8` names stage 2a's target.
5. **No arm rescues `r8`:** the deaths are not about water knowledge at all (the forensics say what
   they are about), and the plan is revised before stage 1.

**Reported, not deciding:** `r4` against `r8` against `r16` against `rinf` (how much range buys, and
whether willows must be seen from 8 or 16 tiles); reward per step; deaths per day; the forensics of
§ 19.1 for every arm.

## 20. Stage 1: the reactive sheep in a meadow that can be seen (pre-registered 2026-10-29, before any run)

Runs only if § 19.2's rule lands on branch 1, 2 or 4 (branch 5 revises the plan first; branch 3
redesigns stage 2a but stage 1 still runs). Written before stage 0's ladder had finished; the forensics
of § 19.1 for the embed hybrid were in (median death 2,300 steps after water was last within 8 tiles;
41 % of the dead never had water within 8 tiles; privileged PPO spends 96 % of its living steps within
8 tiles of water).

**World change (one, a fact of the world):** `--willows 3`: three willows (`world.WILLOW`, a new kind:
a 1.6-tile trunk, a wide pale crown reaching 2.45 tiles, a skirt to the ground) stand 0.9 tiles outside
each pond's edge at random angles. They block movement like trees, are **not** cover for warmth (so a
night by the pond gains nothing), and are told to nobody. A distinct kind rather than more broadleaf
trees, because the meadow already has two copses of three trees: "three trees" would be an ambiguous
cue and the test is of far perception, not of disambiguation. Gate: `meadow2_checks.py --willows 3` on
64 meadows for 5 days must still pass (≤ 2 % scripted deaths, no meadow losing more than one).

**Model:** `scripts/m2_reactive.py`: the world model's RGB conv encoder (5 strided convolutions,
64 × 256 → 512) plus a linear map of the 6 felt numbers, layer-normed, then the felt numbers again,
into the actor and critic MLPs (256 wide). Trained end-to-end by PPO (the reward trains the eyes); no
world model, no recurrence, no replay. Rollouts of 8 meadows × 12 sheep × 64 steps, 4 epochs of 8
minibatches, Adam 3e-4, γ 0.998, λ 0.95, clip 0.2, entropy 0.01, death handled as `m2_ppo.py`.
Budget **2,000 iterations** (about 8 M sheep-steps, the `long` hybrid's budget).

**Arms** (seed 0 to screen; the stage arm then at three seeds):

| Arm | World | Brain | Question |
| --- | --- | --- | --- |
| A1 | willows | reactive | the stage proper |
| A2 | no willows | reactive | does far perception matter, or was the world model the cost? |
| A3 | willows | the `embed` hybrid (`m2_hybrid.py --policy_input state_embed --willows 3`, 2,000 iterations) | does the RSSM cost or buy survival once water can be seen? |

**Pass (pre-registered):** every seed of A1 ≤ 20 % dead on `m2_eval.py` (16 meadows, 3 days, willows 3)
and reward ≥ 2.0. *Strong:* ≤ 5 %. **Demand check for stage 2a** (run with the pass): A1's brain on
the stage-2a world must lose ≥ 40 %, or that pressure is not a pressure.

**Reported:** deaths per day; the forensics of § 19.1; the share of living steps within 8 tiles of
water (the route-around measure); `thirsty_sees_water_8` and the gap at 8; the D2 bearing probe on A1's
embedding by distance band (expect ≥ 0.5 at 8–16 tiles by day with willows, where the hybrids read 0).

**Route-around list:** (i) stay-within-sight-of-water: the 8-tile share. The privileged oracle does it
96 % of the time, so a reactive learner that finds it is doing what the upper bound does; it is allowed
at stage 1, noted, and it is the first thing stage 2a's bare banks must break. (ii) Camping at the pond
and starving: graze success and gut energy on the last day. (iii) Following: nearest-neighbour distance
and others-in-view against the hybrids' (6 tiles, 0.4).

**If A1 fails:** the problem is perception or policy learning from pixels, not memory and not the
world model. The next change is to the encoder (a column-preserving change, as the 2026-10-26 review's
fix 2) or to PPO's budget, pre-registered first, and nothing about memory until A1 passes.

**If A2 passes too:** willows did not matter; the world model was the cost, and stage 2a's pressure
must come with its own demand check against A2's brain.
