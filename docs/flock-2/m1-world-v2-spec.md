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
