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

### 20.1 A1 at seed 0, and the search measurement (pre-registered 2026-10-29, after A1 s0, before any new training)

**What A1 s0 did.** On the fixed evaluation: 29.7 % dead (max 10 in one meadow), reward 1.89, so it fails
§ 20's rule at the first seed. Deaths by day 22 / 18 / 17; hydration 0 and energy 1.13 at death; 12 to 20
tiles from water; a median 1,880 steps since water was last within 8 tiles, and 28 % of the dead *never*
had water within 8 tiles in their lives. Living sheep spend 68 % of steps within 8 tiles (the oracle: 96 %).
The D2_true probe decodes the true water bearing from the embedding only within 4 tiles by day (R² ≈ 0.5)
and not at all beyond 4 tiles or at night: this sheep does not read the willows at distance. Training
reward reached 2.3 with 85 % alive, but training windows are short and reset.

**What the never-within-8 share says.** A day is 2,400 steps, a trot covers 0.35 tiles a step, the meadow
is 48 tiles across with two ponds. A sheep that lives two or three days and never comes within 8 tiles of
a pond has not searched and failed; it has not searched. The action mix agrees (graze 23 %, rest 19 %,
stop 6 %): the dead ate where they stood until hydration hit zero. The other three quarters found water,
left it to graze (right), and did not come back as thirst rose (the same missing behaviour). So the
dominant failure is neither memory nor perception at the pond: a thirsty sheep with no water in view does
nothing about it. The owner's reading, adopted here: A2 (no willows) will come out the same, since the
willows were never the lever in this regime; the chain runs on for the record.

**Two candidate causes, which the measurement separates.**

- *Credit*: PPO with γ = 0.998 values about 500 steps ahead, a fifth of a day. If a searching sheep
  takes longer than that to reach water, the drink never reaches the decision to search.
- *Exploration*: PPO's entropy exploration is an uncorrelated walk (A1's mix: moving 28 % of steps, a
  turn every ~5 steps), whose net drift over 500 steps is a few tiles. A sheep 12 tiles from water
  then never samples "walk far, find water, drink", so there is nothing to reinforce even when the
  horizon would cover it. The uniform-random sheep (§ 17's baseline) loses 86 % and lives 1.4 days.

**The measurement** (`scripts/meadow2_search.py`, CPU). Sheep are placed at uniform random dry
positions and headings in the willow world, levels at MAX_LEVEL so nobody dies inside the window, and
run for one day (2,400 steps) under memoryless search rules; the time to first have water within 8 tiles
inside the 300° field of view (§ 19's criterion, occlusion ignored) is recorded per sheep, binned by the
start distance to water (4–8, 8–12, 12–16, 16–24, 24+ tiles), 32 meadows × 3 seeds × 12 sheep. Rules:

- `uniform`: each action with probability 1/8 (PPO's exploration at the start).
- `a1_mix`: A1 s0's evaluation action marginals drawn independently each step (its exploration as
  trained, without its reactions).
- `walk_p`: trot every step; with probability *p* per step turn one notch (π/8) in a random direction,
  for *p* in {0.02, 0.05, 0.1, 0.25, 0.5}: the correlated walk, from near-straight to near-diffusive.
  `_wall` variants also turn whenever a trot was blocked (the meadow's edge or an object).
- `lazy_p0.05`: A1's movement budget (walk 12 %, trot 16 %, else stop) with a persistent heading
  (*p* = 0.05): does persistence alone rescue a sheep that moves as little as A1 does?

Reported per rule and bin: n, share found, median and 90th percentile steps, and the share found within
500 steps (γ's horizon) and within 1,100 steps (the hydration a sheep at level 0.3 has left:
HYDRATION_DECAY is 0.0008 / 3 per step, so full lasts ~4,300 steps and 0.3 lasts ~1,100).

**Reading rule (pre-registered).** For starts 8–16 tiles from water:

1. If a correlated walk's 90th percentile is ≤ 500 steps while `uniform` and `a1_mix` are mostly
   censored or above 500: exploration is the bottleneck. The one change for the next A1 arm is
   temporally correlated exploration during training (action persistence or correlated noise), γ
   unchanged, pre-registered in § 20.2 before it runs; three seeds; § 20's pass rule unchanged.
2. If the best correlated walk's 90th percentile is between 500 and 1,100 steps: the horizon is the
   bottleneck. The one change is γ such that 1 / (1 − γ) ≥ 1.5 × that percentile (0.9995 covers 2,000),
   with the rollout and value targets checked for variance on one seed before three.
3. If above 1,100 steps even for the best correlated walk: the meadow cannot be searched by a
   memoryless sheep in the time thirst allows, and the world is eased (more or larger ponds) as the
   last resort, noting that stage 2a's mud ring will push the sheep back out to 8 tiles, so a world
   eased here must still pass the gate of § 21 afterwards.

Whatever the branch, the encoder is not changed in the same arm: the probe's 4-tile limit is real but
a sheep that does not move toward anything would not be saved by seeing further.

**Results of the search measurement (2026-10-29, 32 meadows × 3 seeds × 12 sheep, one-day cap;
`meadow2-results/search/`).** Starts 8–12 / 12–16 tiles from water (n = 196 / 188):

| rule | found in a day | median steps | 90th pct | ≤ 500 | ≤ 1,100 |
| --- | --- | --- | --- | --- | --- |
| uniform | 0.84 / 0.73 | 253 / 820 | censored / censored | 0.59 / 0.36 | 0.74 / 0.58 |
| A1's mix | 0.89 / 0.75 | 250 / 966 | censored / censored | 0.66 / 0.37 | 0.77 / 0.55 |
| trot, turn p 0.05 | 0.84 / 0.78 | 404 / 793 | censored / censored | 0.52 / 0.39 | 0.66 / 0.58 |
| trot, turn p 0.05, turn at walls | 0.99 / 1.00 | 100 / 246 | 692 / 884 | 0.81 / 0.77 | 0.95 / 0.94 |
| trot, turn p 0.1, turn at walls | 0.99 / 1.00 | 90 / 254 | 609 / 864 | 0.87 / 0.73 | 0.97 / 0.96 |
| A1's movement budget, p 0.05, walls | 0.92 / 0.87 | 250 / 654 | 1,962 / censored | 0.60 / 0.44 | 0.78 / 0.64 |

From 24+ tiles the wall-aware walks still find water 99 % of the time (median ~320 steps, 90th
percentile ~1,000); the uncorrelated rules find it 28–34 % of the time.

*Reading, by the rule above:* **branch 2**, the horizon. The best memoryless search has a 90th percentile
of 600–900 steps from 8–16 tiles, above γ's 500 and below the 1,100 a thirsty sheep has; the
pre-registered consequence is γ ≈ 0.9993 (horizon 1,333 ≥ 1.5 × 884). Exploration is implicated too,
short of branch 1: the uncorrelated rules leave 11–27 % of sheep unfound after a whole day where the
correlated walks leave none, and a straight walker without the wall rule pins itself to the meadow's
edge and does no better than uniform. A1's own movement budget with a persistent heading is not
enough; it has to move more. None of this is the critical path any more, for the reason below.

### 20.2 The seed-0 screening, and the plan adjustment (2026-10-29)

| arm (seed 0) | world | brain | dead | max / meadow | reward | within 8 | thirsty sees water (8) | D2_true 8–16 day (sin / cos) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | willows | reactive | 29.7 % | 10 | 1.89 | 68 % | 0.28 | 0.05 / −0.05 |
| A2 | none | reactive | 59.4 % | 12 | 1.44 | 46 % | 0.17 | — |
| A3 | willows | embed hybrid | **4.7 %** | 6 | **2.34** | 86 % | 0.18 | **0.59 / 0.38** |
| embed hybrid (§ 19.1, 26 Oct) | none | embed hybrid | 39.6 % | — | 1.80 | 35 % | 0.08 | ≈ 0 (D2_visible) |
| scripted | willows | rules | 0 % | 0 | 2.36 | 75 % | 0.91 | — |

**The assumption in § 20.1 was wrong.** A2 did not match A1: without willows the reactive sheep loses
twice as many, so far perception was never irrelevant in this regime. What A1 could not do was learn
to read the willows from the reward gradient alone (its probe: nothing beyond 4 tiles).

**A3 passes § 20's strong bar at seed 0.** The same embed hybrid that lost 39.6 % on 26 October loses
4.7 % with willows on the banks, one change, and its embedding decodes the true water bearing at 8–16
tiles by day (0.59 / 0.38) where § 20 asked for ≥ 0.5 and the old hybrids read 0. The world model's
prediction loss learned to represent a pale crown at distance; PPO's reward gradient did not. Its 9
deaths: 6 on day 1, 3 on day 2, 0 on day 3; 8 of 9 never had water within 8 tiles; 17–26 tiles from
water at death. Those are far-start search deaths, the residual the search measurement sizes (from 24+
tiles even a good memoryless search has a 90th percentile near 1,000 steps, and a sheep starting at
hydration 0.2 has ~750). Route-around list: (i) within-8 share 86 % (the oracle's 96 %), allowed and
noted; (ii) not camping-and-starving: graze success 0.87, last-day gut 1.09, fat 0.96; (iii) following:
nearest neighbour 2.6 tiles and 6.2 others in view against the old hybrids' 6 and 0.4, so the flock is
much tighter and following may carry some of the survival; D3_recent and a lone-sheep evaluation are
owed before stage 2a leans on this brain.

**Adjustment (the owner's standing permission: adjust when a result shows the plan no longer makes
sense).** Stage 1's bet was that a sheep which reads the willows survives; A3 wins it, A1 does not.
So:

1. The stage-1 brain becomes the embed hybrid with willows (A3's recipe: `m2_hybrid.py --willows 3
   --iters 2000 --kl_dyn 1.0 --replay 1024 --add_frac 0.5 --policy_input state_embed`), to be confirmed
   at seeds 1 and 2 under § 20's pass rule as written (every seed ≤ 20 % dead, reward ≥ 2.0; strong ≤ 5 %).
   Three seeds or none.
2. Stage 2a's demand check runs with A3's brain on the 8-tile mud world (must lose ≥ 40 %).
3. A1 is reported at its remaining seeds as the record of the reactive sheep; its registered fix
   (branch 2: γ 0.9993, one change) is parked as a side arm, run only if GPU time is free, since the
   reactive sheep is no longer on the critical path. If it ever runs, wall-aware correlated exploration
   is a second, separate arm, not folded in.
4. The D2 probe and the stage-2a column build on the hybrid's embedding (`embed`, 512) rather than a
   reactive encoder; the stage-2a pre-registration (§ 21's learning stage) is written against that.

Chain order (owner's approval, 2026-10-29 15:10): after A1 s1 finishes, the first chain is stopped before A1 s2 and `runs/m2_stage1b_chain.sh` runs A3 s1, A3 s2, then A1 s2 for the record.

**Queued after it (`runs/m2_stage1_demand.sh`, pre-registered 2026-10-29 15:30, before any run):**

- *Stage 2a's demand check*: each A3 seed on the 8-tile mud world (`m2_eval.py --inner 8`, the § 21
  gate's terrain: no grass or regrowth within 8 tiles of water, regrowth ramping to full over 2 more),
  16 meadows, 3 days. The mud is a pressure if the stage-1 brain loses ≥ 40 % of its sheep there, as
  § 20 set; the scripted flock runs on the same world through the same script for the comparison (it lost
  0 of 768 in the § 21 gate). Below 40 %, the brain already copes without memory (the probes say it has
  none: A3's D3_recent is negative) and the ramp's end point must move out before stage 2a's column is
  built. Expected: the deaths rise at night, when the willows cannot be seen.
- *Lone-sheep evaluation*: each A3 seed with one sheep per meadow, 64 meadows, 3 days, the willow world.
  If the lone death share exceeds 20 % while the flock's is under 5 %, following carries survival, and
  stage 2a's demand check and pass rule are read on the lone evaluation as well as the flock's.
### 20.3 Stage 1 result: the embed hybrid with willows passes at three seeds (2026-10-29, 18:20)

| arm | seed | dead | max / meadow | reward | within 8 | nearest neighbour | deaths by day | never within 8 | water at death (quartiles) | D2_true 8–16 day sin / cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A3 | 0 | 4.7 % | 6 | 2.34 | 86 % | 2.6 | 6 / 3 / 0 | 0.89 | 17 / 24 / 26 | 0.59 / 0.38 |
| A3 | 1 | 6.8 % | 7 | 2.31 | 95 % | 1.3 | 8 / 4 / 1 | 0.15 | 8 / 9 / 10 | 0.50 / 0.45 |
| A3 | 2 | 9.9 % | 3 | 2.28 | 76 % | 4.2 | 13 / 1 / 5 | 0.47 | 11 / 18 / 23 | 0.40 / 0.36 |
| A1 | 0 | 29.7 % | 10 | 1.89 | 68 % | 3.6 | 22 / 18 / 17 | 0.28 | 12 / 18 / 20 | 0.05 / −0.05 |
| A1 | 1 | 20.8 % | 9 | 2.03 | 79 % | 2.9 | 21 / 12 / 7 | 0.50 | 10 / 12 / 20 | — |
| A1 | 2 | 52.1 % | 10 | 1.48 | 55 % | 5.5 | 43 / 42 / 15 | 0.43 | 12 / 17 / 22 | — |

**Verdict.** The embed hybrid with willows passes § 20's rule at every seed (≤ 20 % dead, reward ≥ 2.0):
4.7, 6.8 and 9.9 % dead, mean 7.1 %, rewards 2.34, 2.31, 2.28 against the scripted flock's 2.36. The
strong bar (≤ 5 %) is met at seed 0 only. The reactive sheep fails at two of three seeds (29.7, 20.8,
52.1 %; mean 34 %) with a wide spread, so § 20's registered arm is closed as a fail and the stage's bet
is won by the hybrid, as § 20.2 adopted. Stage 1 passes with the embed hybrid as its brain.

**What the three seeds add.** The embedding decodes the true water bearing at 8–16 tiles by day at
every seed (sin 0.59 / 0.50 / 0.40, cos 0.38 / 0.45 / 0.36), around the 0.5 § 20 asked for and far from
the old hybrids' 0; at 4–8 tiles 0.78 at every seed; at night nothing. D3_recent (water seen in the last
300 steps, now out of view) is negative at every seed: this brain has no memory, so stage 2a's demand
check is a fair test of the world. Deaths are mostly by day (night holds 40 % of the steps and 11, 15
and 42 % of the deaths), so the willows are not a night-only cue failing at night; they are far-start
search deaths and, at seed 1, something else: its dead died 8–10 tiles from water, with 85 % having had
water within 8 tiles, in a flock so tight (1.3 tiles to the nearest neighbour, 95 % of steps within 8
tiles of water) that it looks like a herd that camps at the pond and loses a member who does not drink
in time. That seed is the one the lone-sheep evaluation will be hardest on.

**Route-around reading across seeds.** (i) within-8 share 76–95 %, the oracle's 96 %; allowed at
stage 1, and the first thing the mud must break. (ii) Not camping-and-starving: graze success 0.81–0.95,
last-day hydration 1.04–1.10. (iii) Following: nearest neighbour 1.3–4.2 tiles against the old hybrids'
6; open, pending the lone evaluation queued below.

- A3's probes so far (seed 0): D1 water-pixel recall 0.92 / 0.73 / 0.09 by band (0–4 / 4–8 / 8–16 tiles)
  by day, so the pond itself is still not segmented beyond 8 tiles; the bearing at 8–16 is read from the
  willows. D3_recent (bearing of water seen in the last 300 steps, now out of view): negative, no memory.

### 20.4 The demand check and the lone-sheep evaluation (2026-10-29, 18:30)

**Lone sheep** (one per meadow, 64 meadows, 3 days, the willow world):

| seed | lone dead | flock dead (§ 20.3) | lone reward | within 8 | deaths by day | never within 8 | water at death |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 7.8 % | 4.7 % | 2.27 | 86 % | 3 / 2 / 0 | 1.00 | 19 / 21 / 24 |
| 1 | 17.2 % | 6.8 % | 2.11 | 92 % | 7 / 4 / 0 | 0.91 | 13 / 15 / 17 |
| 2 | 14.1 % | 9.9 % | 2.20 | 76 % | 6 / 1 / 2 | 0.56 | 10 / 12 / 19 |

Alone, every seed still passes § 20's rule (≤ 20 %, reward ≥ 2.0), so the survival is the sheep's own,
but the flock helps: 13.0 % mean dead alone against 7.1 % in a flock, and most for seed 1, the tight
herd (17.2 against 6.8). Every lone death is a day-1 or day-2 far-start search death: the dead never
had water within 8 tiles and died of thirst 10–24 tiles from it. The flock's help is the obvious one: a
sheep that cannot see water follows one that can. Route-around (iii) is therefore real but partial,
and § 20.2's rule for it (lone > 20 % with flock < 5 %) is not triggered. Stage 2a is read on both.

**Demand check** (each seed on the 8-tile mud, 16 meadows, 3 days; the scripted flock through the
same script lost 0 of 192, reward 2.31, within-8 share 43 %, last-day hydration 0.70):

| seed | dead | max / meadow | reward | within 8 | graze success | last-day gut / fat | deaths by day | cause | water at death |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 14.6 % | 4 | 1.98 | 95 % | 0.04 | 0.30 / 0.28 | 3 / 3 / 22 | starved by the pond (gut 0, hydration 1.04) | 0.8 / 1.3 / 2.1 |
| 1 | 17.2 % | 4 | 2.02 | 91 % | 0.16 | 0.29 / 0.28 | 5 / 1 / 27 | starved by the pond (gut 0, hydration 1.14) | 0.7 / 1.1 / 2.5 |
| 2 | 25.5 % | 9 | 1.94 | 50 % | 0.45 | 0.78 / 0.56 | 27 / 12 / 10 | thirst beyond the mud (hydration 0) | 8.5 / 12.2 / 15.4 |

**Reading, by the rule as registered: not a pressure at 3 days** (14.6, 17.2, 25.5 %, all under 40 %).
**Reading of the evidence: the rule's window was wrong for this pressure.** § 20 set the 40 % line
with thirst deaths in mind, which arrive within a day. The mud kills by hunger, and a sheep's fat
(FAT_CAPACITY 4 gut-units, starts at 0.2–0.8) carries it for about two days without grass. Seeds 0 and 1
do exactly what stage 2a predicted of a memoryless brain: they stay within 8 tiles of water (95 and
91 % of steps, the route-around § 20 named first) where there is now no grass, graze successfully on 4
and 16 % of attempts, and start dying on day 3 (22 and 27 of their 25 and 33 deaths) with the last day's
gut at 0.30 and fat at 0.28, which is a flock a day from collapse. Seed 2 wanders instead (within 8:
50 %) and dies of thirst 8–15 tiles out, the other failure of a brain without memory: it leaves water to
eat and does not find its way back. Both are the failure the pressure was built to produce; the 3-day
evaluation simply ends before the first style has finished killing.

**Pre-registered before running (18:35): the demand check's window is 5 days,** the same window § 20's
gate used for the scripted flock (`meadow2_checks.py --days 5`). The 40 % line is kept. Each seed and
the scripted flock run again on the 8-tile mud for 5 days through `m2_eval.py --days 5 --inner 8`.
If the 5-day share is ≥ 40 % at every seed the mud is a pressure and stage 2a's learning stage is
written (§ 21's column, the ramp 0 → 8, D3 ≥ 0.3); if the scripted flock also loses sheep at 5 days the
end point moves in; if a seed stays under 40 % at 5 days with gut and fat still falling, the window,
not the world, is reported and the owner decides. Nothing about the window is changed after this run.

**Result at 5 days (18:38; `meadow2-results/eval/*_mud8_d5.json`):**

| flock | dead | max / meadow | reward | within 8 | graze success | last-day gut / fat | deaths by day | cause | water at death |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scripted | 0 of 960 | 0 | 2.32 | 46 % | 0.92 | 0.85 / 0.70 | 0 / 0 / 0 / 0 / 0 | — | — |
| A3 seed 0 | **69.3 %** | 11 | 1.59 | 96 % | 0.04 | 0.17 / 0.12 | 5 / 5 / 20 / 48 / 55 | 126 starved, 7 thirst | 0.8 / 1.1 / 1.6 |
| A3 seed 1 | **67.7 %** | 12 | 1.60 | 93 % | 0.13 | 0.19 / 0.12 | 10 / 3 / 26 / 41 / 50 | 118 starved, 12 thirst | 0.7 / 1.0 / 1.7 |
| A3 seed 2 | **41.7 %** | 11 | 1.76 | 48 % | 0.48 | 0.81 / 0.67 | 29 / 7 / 21 / 14 / 9 | 78 thirst, 2 starved | 10 / 15 / 19 |

**The mud is a pressure: every seed loses ≥ 40 % at 5 days, and the scripted flock loses nobody.**
Seeds 0 and 1 die as predicted, camped at the pond on bare ground, 48 to 55 deaths on each of days 4 and
5 with the survivors' gut at 0.17 and fat at 0.12. Seed 2 clears the line narrowly (41.7 %) by the other
route, thirst beyond the mud, and its deaths are front-loaded (29 on day 1) rather than climbing, so it
is the seed least pressed: it keeps itself fed (gut 0.81) and loses the sheep that do not get back.
The stage-1 brain cannot live on this world without something it does not have; what the mud asks
for is to leave sight of water to eat and come back, which is stage 2a's memory. The demand check
passes, and § 21's learning stage can be pre-registered against this brain, the 8-tile end point,
and the 5-day window.

**Two notes for the stage 2a design.** (i) The two failure styles are the two things the column must
beat: a brain that will not leave the water (seeds 0, 1) and a brain that leaves and cannot return
(seed 2). The ramp 0 → 8 is meant to carry the first kind out gradually; the D3 probe and the
route-around list catch whether the second kind is solved by memory or by wall-following and
path-following. (ii) The 3-day evaluation stays the stage's pass window (§ 20's rule) but the mud
world is judged at 5 days, since its deaths arrive on days 3–5.

## 21. Stage 2a's pressure: depletion and trampling (owner's design, pre-registered 2026-10-29)

**The idea (owner).** Instead of the review's hand-set bare banks (no grass within R tiles of water),
let the flock make the bare ring itself: once stage 1's brain has peaked on the world as it is, slow the
grass's regrowth so the bank is eaten faster than it recovers, and make grass lose growth where sheep
tread, so the bank wears bare and paths form along the routes the flock uses. The ring then grows until
grazing has to happen out of sight of the willows, which is where memory starts to pay; the pressure
ramps continuously and adjusts itself to how much the flock camps. The paths are worth watching in
themselves.

**Mechanics (built now; neutral by default).** Two numbers carried in every meadow (`Meadow.regrow`,
`Meadow.trample`), so a curriculum can change them on live meadows without recompiling:
- `regrow`: the share of the gap to full regrown per step (was the constant `REGROW` = 0.00013, a time
  constant of about 7,700 steps; unchanged by default);
- `trample`: biomass lost per step by every grass cell within `PATCH_RADIUS` of a sheep that **moved**
  this step (walk or trot; a resting or grazing sheep wears nothing). Default 0.

**The ring-growth measurement (CPU, scripted flock, before any learning).** `scripts/meadow2_ring.py`:
the scripted flock (privileged water knowledge, so it never needs memory) on 16 meadows for 5 days at
each setting of a grid, regrowth × {1, 1/2, 1/4, 1/8} and trample ∈ {0, 0.005, 0.02, 0.05}. Per day
and setting: the mean grass density by distance band from water (0–2, 2–4, 4–8, 8–12, 12–16, > 16
tiles); the **bare radius** (the outermost band, counted from the water, whose mean density is still
below `MIN_BITE`, i.e. ungrazeable); the mean distance from water at which successful bites happen;
deaths, by cause (hydration or gut energy at death); and the share of dry cells below 0.5 biomass more
than 8 tiles from water (trails and over-grazing beyond the ring).

**Reading, written before running:**
- The pressure is usable if some setting drives the bare radius beyond **8 tiles** (the willows' sure
  range by day) within 3 days while the scripted flock still passes its survival check (≤ 2 % die), and
  those that die do so of thirst, not hunger. That setting (or the mildest such) is the ramp's end
  point; the ramp's start is the world as it is.
- If no setting gets the ring past 8 tiles without starving the scripted flock, depletion alone is too
  weak with 12 sheep and the pressure needs help (a wider trample radius, or the review's fixed bare
  bank as a floor), decided then.

**The learning stage (to be pre-registered after stage 1, with its numbers):** continue from stage 1's
brain, add the recurrent column progressive-style, ramp `regrow` down and `trample` up on a schedule
toward the end point, with the demand check (the stage-1 brain at the end point loses ≥ 40 %, of
thirst) and the scripted gate at every setting. **Route-around list:** (i) grazing at the ring's edge
where the willows are still in view (bite distance against the willows' readable range; the ring
should keep growing if they do); (ii) **following the paths**: a trampled path to the pond is persistent
information a sheep that never met its makers can read (stigmergy, the kind of mark stage 4 wants). A
named capacity if it appears, not a cheat, but it is not memory: the swap test erases or scrambles the
worn ground at evaluation and survival must not fall for a sheep said to remember; (iii) following
other sheep (nearest-neighbour distance, others in view). D3 stays the memory probe.

### Ring-growth measurement, grid 1: result (2026-10-29)

`meadow2-results/ring_grid1.json`: 16 meadows with willows, 5 days, the scripted flock, regrowth ×
{1, ½, ¼, ⅛} × trample {0, 0.005, 0.02, 0.05}.

**The ring did not form at any setting.** The bare radius stayed at 0 everywhere: the 0–2-tile band's
mean density fell from 1.54 to **0.80** at the strongest setting (regrowth ⅛, trample 0.05) over five
days, still above `MIN_BITE` (0.5). Nobody died. Trampling did more than slower regrowth (at full
regrowth, trample 0.05 took the bank to 1.82; at ⅛ regrowth with no trampling, 1.61). The scripted
flock bites at 5.6–8.7 tiles from water, and at baseline the meadow is so under-grazed by 12 sheep that
grass rises from 1.5 to 2.7 in five days. More than 8 tiles from water, trampling bared 0.7–7 % of
cells (the beginnings of trails).

**By the rule: the second branch.** Depletion at these settings is too weak with 12 sheep. Two
caveats before reaching for the fixed bank: (i) the scripted flock wanders out to graze and keeps only
69 % of its steps within 8 tiles of water, where the privileged learner camps 96 % of the time, so a
learned flock would wear the bank far harder; (ii) the bank was still falling at day 5. Grid 2
(pre-registered here): regrowth × {¼, 1/16, 0} × trample {0.05, 0.15, 0.4}, **8 days**, with the
scripted flock and with a **camping** scripted flock (the same rules, but when idle it drifts back
toward water beyond `CAMP_DIST` = 6 tiles, as the learner does). Same reading as before: usable if
some setting bares the bank past 8 tiles within 3 days of the camping flock's arrival while the
scripted flocks still survive. If the camping flock bares it and the wandering one does not, the
pressure is real but conditional on camping, which is exactly what the stage-1 brain is expected to do.

### Ring-growth measurement, grid 2: result (2026-10-29)

`meadow2-results/ring_grid2_{wander,camp}.json`: 16 meadows with willows, 8 days, regrowth ×
{¼, 1/16, 0} × trample {0.05, 0.15, 0.4}, the wandering and the camping scripted flocks.

**Depletion alone cannot make the ring.** Even with **no regrowth at all** and trample 0.4, the bare
radius reached only 2–4 tiles by day 7–8 (both flocks), and the meadow thinned everywhere instead: the
8–12-tile band fell to 0.8–1.0 and hunger deaths rose (wandering flock at regrowth 0: 2, 26 and 34
of 192 as trampling rose; camping flock: 5–8). The bank band plateaus at about 0.5 because a bite
needs patch density ≥ `MIN_BITE` = 0.5: grazing cannot push grass below the bite floor, so the flock
half-eats every patch it reaches and moves on, and only trampling takes a cell lower. Camping did not
change the picture (its bank reached 0.41–0.45 against the wanderers' 0.45–0.50); the camping flock's
bites moved out to 10–13 tiles by day 8 because the near grass was ungrazeable, which is the pressure
working, but on a meadow that was by then thin everywhere.

**By the rule: the pressure needs help.** The help that keeps the owner's mechanism emergent is
**muddy banks**: each cell's regrowth is scaled by its distance from water over `bank` tiles (0 at
the water's edge, full beyond `bank`; `Meadow.regrow_map`, set by `with_terrain(bank=…)`; `bank` 0 is
the world as it was). Near the pond, grazing and trampling beat regrowth and the bank bares; far away
the grass recovers at full rate, so nobody starves; the ring's size depends on how hard the flock camps,
and `bank` is the one dial the curriculum ramps. This is a world fact (mud by the water), not a rule
about the sheep.

**Grid 3 (pre-registered here):** full regrowth, `bank` ∈ {4, 8, 12, 16} × trample ∈ {0, 0.05,
0.15}, 8 days, both flocks. Same reading: usable if some setting bares the bank past 8 tiles within 3
days of the camping flock's arrival while both scripted flocks still survive (≤ 2 % die), and those
that die do so of thirst. The mildest such setting is the ramp's end point.

### Ring-growth measurement, grid 3: result (2026-10-29)

`meadow2-results/ring_grid3_{camp,wander}.json`: full regrowth, bank ∈ {4, 8, 12, 16} × trample
∈ {0, 0.05, 0.15}, 8 days, both flocks.

**Muddy banks keep everyone alive (0 deaths of 192 in all 24 runs) but still make no ring.** The bare
radius never reached 2 tiles; at bank 16 with trample 0.15 the 0–2-tile band's mean density settled at
0.71 (camping) / 0.81 (wandering) and the 2–4 band at about 1.0, still above the bite floor. The
camping flock's bites moved out from 4.7 to 7–8 tiles, the wanderers' to 10–12. Part of this is the
scripted flock's own rule (it grazes only patches of density ≥ 1.0 and walks to a richer one three
tiles away, so it leaves about 1.0 where a learned sheep could bite down to 0.5). The rest is
arithmetic *(estimate)*: 12 sheep eat about 100 bites each a day to cover their energy use, about 200
biomass units for the flock, whatever their policy; the standing crop within 8 tiles of two ponds is
about 1,600 cells × 2 ≈ 3,200 units, and at half regrowth it recovers about 375 units a day. The flock
cannot carve an 8-tile ring out of standing grass in days; it would take weeks with no regrowth at all.

**Reading:** depletion by this flock cannot *create* the ring; it can only *maintain* one. Two ways to
get the ring, for the owner to choose between:
- **A. Mud sets the starting grass too.** The same bank map scales the initial grass as well as the
  regrowth (nothing grows well on the muddy bank, a world fact), so the ring is there from the start at
  the width the dial sets; trampling and slow regrowth keep it bare and wear the paths. The ramp is the
  bank width, as planned. The ring's creation is then designed, its maintenance emergent. No compute
  cost. Grid 4 (pre-registered here) measures it: bank ∈ {8, 12, 16} × trample ∈ {0, 0.05}, both
  flocks, 8 days; usable if the bare radius is ≥ 8 tiles from day 1 and holds, with ≤ 2 % scripted
  deaths.
- **B. A bigger flock.** Consumption scales with the flock: 40 sheep (the installation's "tens") would
  eat about 700 units a day and could carve the ring in about a week at a 16-tile bank. The ring would
  be fully emergent, but rendering cost per meadow triples, the fixed evaluation and every baseline
  change, and it is a decision about the whole world, not one dial.

### Ring-growth measurement, grid 4: result (2026-10-29)

`meadow2-results/ring_grid4_{camp,wander}.json`: mud sets the starting grass too, bank ∈ {8, 12, 16}
× trample ∈ {0, 0.05}, 8 days, both flocks. Nobody died in any run.

**The linear ramp cannot hold a ring either.** The bare radius starts at 2 tiles (bank 8) to 4 tiles
(bank 16) and then *shrinks*: at 4 tiles from water a 16-tile ramp still regrows at a quarter rate,
more than the flock eats, so by day 6–8 the ring is 0–2 tiles with no trampling and 2 tiles with it.
Bites moved out to 7–10 tiles (camping) and 9–11 (wandering) on day 1 and drifted back in. The ring
never reached 8 tiles.

**Reading.** With 12 sheep, a ring at the willows' range needs terrain with essentially no grass and
no regrowth out to that radius and a short edge: the review's fixed bare bank, with mud as its
mechanism. The owner's trampling and slow regrowth remain as the ring's *maintenance* and as the
source of paths; its *creation* is a terrain fact set by one dial, the mud radius. **Grid 5
(pre-registered here):** `bank_map(water, bank, inner)`: zero within `inner` tiles of water, ramping to
1 at `bank` = `inner` + 2; mud sets the starting grass; `inner` ∈ {6, 8, 12} × trample ∈ {0, 0.05},
8 days, both flocks. Usable if the bare radius holds ≥ `inner` for 8 days with ≤ 2 % scripted deaths.
Since at `inner` 16 two ponds would leave only about a third of the meadow grassed, 12 is the largest
radius measured; if 12 passes, the ramp for stage 2a runs from 0 to 12 and the demand check decides
where on it the stage-1 brain fails.

### Ring-growth measurement, grid 5: result (2026-10-29)

`meadow2-results/ring_grid5_{camp,wander}.json`: mud radius `inner` ∈ {6, 8, 12} (edge 2 tiles), mud
sets the starting grass, trample ∈ {0, 0.05}, 8 days, both flocks.

**The mud radius holds the ring exactly, for all 8 days, at every setting** (bare radius = `inner`;
the bands inside it read 0.0). Bites move out with it: 8–9 tiles from water at `inner` 6, 10–11.5 at
8, 14–16 at 12 (camping flock; the wanderers 1–2 tiles farther). Trampling adds 1 tile of bite
distance and the first trails (2–5 % of far cells bared at `inner` 8).

**Survival sets the ceiling.** Hunger deaths of 192 over 8 days: `inner` 6: 0–1; **`inner` 8: 2–4
(1.0–2.1 %, at the 2 % gate)**; `inner` 12: 18–28 (9–15 %), a clear fail, though 60 % of the meadow
is still grassed: the walk between water and grass is what the scripted flock's rules cannot afford,
not the grass. All deaths were of hunger, none of thirst (the scripted flock knows where water is).

**Reading.** Stage 2a's dial is the mud radius, ramped from 0 toward **8 tiles**, the largest radius
the scripted gate allows with 12 sheep (10 is untested; 12 fails). The full 64-meadow, 5-day gate
(`meadow2_checks.py --inner 8 --mud --trample 0.05 --willows 3`) is run below; the end point is the
largest radius ≤ 8 that passes it. Trampling 0.05 comes with it for the paths. Whether 8 tiles is a
*pressure* is the demand check's question (the stage-1 brain must lose ≥ 40 %, of thirst): by day a
sheep grazing at 10–11 tiles may still see the willows (readable to 8–16), so the demand is expected
to come at night, when the eyes reach 4–5 tiles; if the stage-1 brain survives the 8-tile mud, the
pressure needs night (willows unlit) or a bigger flock, decided then with the owner.

### The full scripted gate on the mud (2026-10-29)

`meadow2_checks.py --willows 3 --trample 0.05 --inner R` (64 meadows, 5 days; `meadow2-results/checks_mud{6,8}.json`):

| Mud radius R | Died of 768 | Of hunger / thirst | Worst meadow | Needs met | Gate |
| --- | --- | --- | --- | --- | --- |
| 6 | 4 (0.5 %) | 3 / 1 | 1 | yes | **pass** |
| 8 | 32 (4.2 %) | 32 / 0 | 7 | yes (energy after day 1: 0.72) | **fail** |

So the ramp's end point is **6 tiles** (7 is being measured; 8 fails on the full gate, where the
16-meadow grid had it at the line). At 6 tiles the camping scripted flock bites 8–9 tiles from water,
at the edge of the willows' sure daytime range; the demand is expected to come at night.

### The scripted flock's far sense for grass, and the gate re-run (2026-10-29)

The owner asked why a privileged flock should starve with grass a 70-step walk away. It should not:
an 8-tile round trip costs about 0.06 energy and one bite returns 0.012. The cause was the scripted
flock's grass sense, which looked only at eight points three tiles away (all zero on mud), so a hungry
sheep steered in an arbitrary direction. It now heads for the nearest grass of density ≥ 1.5 anywhere
when nothing worth eating lies within three tiles (`scripted.GOOD_GRASS`), a privileged sense like
its water sense; on the base world this rarely triggers. Gate re-run (`checks_v4_mud{0,8,12}.json`,
64 meadows, 5 days, willows 3, trample 0.05):

| Mud radius R | Died of 768 | Of hunger / thirst | Reward / step | Gate |
| --- | --- | --- | --- | --- |
| 0 (base world) | 0 | 0 / 0 | 2.355 (2.373 before the change; trampling on) | pass |
| 8 | 0 | 0 / 0 | 2.303 | **pass** |
| 10 | 9 (1.2 %) | 9 / 0 | 2.285 | fail (two meadows lost 2) |
| 12 | 17 (2.2 %) | 17 / 0 | 2.274 | fail, at the line |

So the ramp's end point is **8 tiles**, past the willows' sure daytime range (10 passes the 2 % rule
but not the one-per-meadow rule). The fixed flock's own baseline on the willow world (`m2_eval.py
--policy scripted --willows 3`, `eval/scripted_v4_willows.json`): 2.357 per step, 0 deaths, 75 % of
living steps within 8 tiles of water, thirsty-sees-water at 8 tiles 0.59, gap at 8 tiles 0.34. The water view at 4/8/12/16 tiles (`meadow2-results/water_samples.png`) shows why 8 tiles
is "far" for the eyes though not for the legs: a pond is a flat surface seen from an eye 0.8 tiles
high, a two-pixel sliver on the horizon at 8 tiles, gone at 12–16 and at night beyond 8.

### 21.1 The learning stage: pre-registration (2026-10-29 evening; approved by the owner as drafted, 2026-10-29 night, with the chain already running)

**Where it starts.** From each of stage 1's three brains (A3 seeds 0, 1, 2: the embed hybrid with
willows, § 20.3), each continued in its own run, so the stage has three seeds by construction. The demand
check is done (§ 20.4): on the 8-tile mud at 5 days the stage-1 brain loses 69 / 68 / 42 %, in two styles,
camping on bare ground and starving, or wandering beyond the mud and dying of thirst.

**The brain change (one, progressive-style).** A recurrent column beside the stage-1 network:

- A GRU of 256 units, state **g**, whose input each step is the current view's embedding (512, the
  world model's encoder, gradient stopped so the column cannot retrain the eyes), the six felt numbers
  and the one-hot previous action. It is trained by PPO's reward only, through time, which is what
  stage 1 lacked: a state shaped by what pays rather than by what predicts the next frame (the RSSM's
  h is such a state, the policy already reads it, and D3 says it holds no water bearing once water is
  out of view).
- The policy's input grows from [h, z, embed] to [h, z, embed, g]. The first layer of the actor and of
  the critic is widened by 256 columns initialised to **zero**, so at iteration 0 the policy is exactly
  stage 1's and the world is stage 1's (mud radius 0): there is nothing to regress from, and the column
  earns its way into the decision. The world model keeps training as before (its loss is unchanged and
  the mud is new data for it); the stage-1 policy weights keep training under PPO. The review's "freeze
  the old until the regression tests hold" is met by the zero start plus the regression test below, with
  a freeze held as the first fallback.
- Credit through time: the rollout grows from 64 to **128 steps**, and PPO's update runs the column
  forward over each sheep's whole rollout from its stored starting state (one sequence per sheep per
  rollout, minibatches of sequences), so gradients flow across the 128 steps. γ 0.998 (a 500-step
  horizon) covers an out-and-back trip (8 tiles at a walk is ~70 steps each way plus grazing). The
  column's state is carried across rollouts like h and reset with the meadow.
- Everything else as A3's recipe: 8 meadows × 12 sheep, replay 1,024, add 0.5, kl_dyn 1.0, 4 epochs,
  Adam 3e-4, clip 0.2, entropy 0.01, reset every 2 days. Budget **2,000 iterations** of 128 steps
  (twice stage 1's samples; about 2 hours a seed with its evaluations).

**The world change (one, ramped).** Mud radius `inner` 0 for iterations 1–200 (the stage-1 world while
the column settles), linear to **8 tiles** over iterations 200–1,400, held at 8 for 1,400–2,000; edge 2
(regrowth back to full over 8–10 tiles). A meadow takes the current radius when it is (re)made and
keeps it until its reset, so live meadows lag the ramp by at most two days. **No trampling in this arm**:
the demand check and the 5-day scripted gate through `m2_eval.py` were run without it, and without worn
paths route-around (ii) cannot arise, which keeps the first test of memory clean. Trampling (0.05, the
owner's paths) is added as its own world fact once this arm is read, with § 21's swap test then. This is
a departure from the owner's original pairing of mud and paths, flagged for review.

**Evaluation (pre-registered).** `m2_eval.py`, 16 meadows (seeds 50,000+), the willow world:

| test | world | window | rule |
| --- | --- | --- | --- |
| survival on the mud | `--inner 8` | 5 days | ≤ 20 % dead at every seed, reward ≥ 2.0 (scripted: 0 of 960, 2.32) |
| memory | the same frames | | D3 from **g** (bearing of water seen within the last 300 steps, now out of view and > 1.5 tiles away; the better of sine and cosine, ridge) **≥ 0.3** at every seed; noise as wide as g scores < 0.1 |
| stage 1 still passes | no mud | 3 days | ≤ 20 % dead at every seed (the regression test) |
| alone on the mud | `--inner 8 --sheep 1 --meadows 64` | 5 days | reported; passes alone if ≤ 20 % |

**Pass:** survival and memory and the regression test, all three seeds. **Reported, not deciding:** D3
from h and z (does the world model's state pick up memory once the world pays for it?) and from
[h, z, g] together; D2_true from g; deaths per day and their cause (starved / thirst); within-8 and
within-16 shares of living steps; bite distance by day and by night; the within-8 share at night
against the day; nearest neighbour and others in view; the action mix; the lone result.

**Route-around list, with the readings that catch each.** (i) *Grazing within sight of the willows*:
with the willows readable to ~16 tiles by day (§ 20.3's probe), a sheep can eat at 9–15 tiles and still
see water, needing no memory by day. This is expected and allowed: it is the stage-0 route-around
(§ 19.3), and the ring is where the grass is. Night is when it fails, since the probe reads the willows
at 0 at night, so the readings are the night-time within-8 share and the night deaths. A brain that
passes survival with D3 < 0.3 has lived by (i): the stage then fails on memory, the result is written,
and the decision returns to the owner, with two pre-named next pressures: a radius beyond the willows'
readable range (12–16, with the scripted gate re-tuned, since the scripted flock camps by design), or
stage 3's hiding of the willows pulled forward. (ii) *Paths*: none in this arm. (iii) *Following*: the
lone evaluation on the mud.

**Fallbacks, in order, one change each, pre-registered:** (1) the regression test fails (stage 1
forgotten): freeze the world model and the stage-1 policy weights, train only the column and the
zero-initialised slices. (2) Survival fails at the end point in the camping style (deaths starved,
within-8 high): hold the ramp longer (end point reached at iteration 1,000, held 1,000 more). (3)
Survival fails in the wandering style (deaths of thirst beyond the mud, D3 < 0.3): **2a-ii, the owner's
compass**: two or three fixed peaks on the horizon at distinct bearings, a world fact readable in one
frame, so "water is north-east" can be stored without integrating turns (the design sketched in
conversation 2026-10-29: a one-dimensional skyline indexed by heading, composited behind the ground
before the renderer's march; the sun and moon stay scenery until the fixed peaks are read). It comes
with a heading probe (allocentric heading from one frame's embedding). (4) D3 ≥ 0.3 but survival fails:
the memory is there and the policy cannot use it; PPO budget doubled once.

**Code (built and smoke-tested 2026-10-29 evening, before any run; `tests/test_column.py` checks the zero start numerically, the ramp and the update through the column; a 5-iteration GPU run at full width peaked at 4.4 GB):** `m2_hybrid.py --column 256 --continue_from <stage-1 dir>
--mud_ramp 200,1400,8 --steps 128` with the recurrent PPO update; `m2_eval.py` and `m2_diag.py` carry g
for a `column` config and probe it (D3, D2_true, the noise floor at g's width); `runs/m2_stage2a.sh`
with the three seeds in one chain, each followed by the four evaluations. Smoke test on CPU, the
regression test's zero-start checked numerically (the widened policy's logits equal stage 1's at
iteration 0), then the chain.

## 22. World v3: a hex lattice and terrain (owner's direction, 2026-10-29; planned, not built)

**When.** At the next world change where it makes sense: stage 3's (the information gap, where hills
that hide water from some sheep and not others are the pressure wanted). Both the lattice and the
terrain machinery are built then, together, because they rewrite the same five functions; the height
field may be left flat at first (`height` all zero) and switched on when stage 3 needs it. Not before
stages 1 and 2 are passed on meadow v2: every baseline and demand check is a v2 number, and v3 restarts
them.

**Why it is cheap for the sheep.** Nothing a sheep sees or does touches the lattice: positions and
headings are continuous, grass and water are smooth fields point-sampled at float positions, the
renderer marches rays and draws a surface, objects stand at float positions. The lattice is an
implementation detail of five functions in `world.py`: `cell_centres`, `grass_at`, `water_at`,
`patch` and `dry_cells` (plus the generator's blobs and the tests). Nothing in `render.py`,
`sim.py`'s movement, the drives, the learners or the evaluation changes for the lattice.

**Hex lattice.** Cell centres on a triangular lattice stored as a rectangular array with offset rows
(axial coordinates); reads by three-point barycentric interpolation (isotropic; simpler than
bilinear); a bite's patch is the cells within `PATCH_RADIUS` of the mouth point, as now; the human
display draws hexes if it draws cells at all. Benefits: isotropy for every distance-based rule
(regrowth by moisture, trampling, the muddy bank) and a more organic top-down look. Cost: about a day,
and the scripted checks re-run, expected unchanged within noise.

**Terrain.** A height field at tile resolution (same lattice), water wherever the height is below a
water level (lakes of any shape; shorelines from the field), the eye at the terrain height under the
sheep plus `EYE`. The renderer's ground pass already marches `K` distances per column and takes a
running maximum of elevation angle, which is the height-map algorithm: terrain adds the ground height
to each sample's top, and hills occlude for free; the sky pass and the billboards take the ground
height under each object. Movement: an uphill multiplier on energy use and a steepness limit in the
blocked-move test (and in the scripted flock's look-ahead). Generation must still guarantee a flock
start 6–14 tiles from water, reachable on foot. Expected rendering cost: unchanged (about 0.2 ms a
view). **Grass stays a density grid** on the same lattice: terrain and grass are two fields over the
same ground; grass sits on the height for rendering and the sim eats it without knowing the height.
**Moisture replaces the mud map**: regrowth as a continuous function of height above the water level
(or distance to water), so banks recover slowly and hills are dry by one rule of the terrain, and the
stage-2a mud radius becomes a special case of it.

**Costs that matter.** A new world (v3): scripted checks, the privileged upper bound, the fixed
evaluation, forensics and the staged ladder's demand checks are all re-measured (about a week).
Perception gets harder (a rolling horizon, hills that hide ponds), so stage 1's pass is re-checked on
v3 with the willows. What the world demands changes: hills hide water, which raises the demand for
memory and for telling, which is the point of doing it at stage 3.

### 19.3 Stage 0: results (2026-10-29)

`meadow2-results/eval/ppo_r*.json` (the ladder), `eval/*_forensics.json` and `deaths/*.json` (the
forensics), `ppo/*_log.jsonl` (training logs). 16 meadows × 12 sheep, 3 days; "died" is the share of
192.

**The ladder** (three seeds each; `rinf` is `ppo_s0–s2`):

| Arm | Range | Memory | Died, by seed | Mean | Reward / step | Deaths by day (seed 0) | Living steps within 8 tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rinf | ∞ | — | 0.0, 0.0, 0.5 % | **0.2 %** | 2.47 | 0, 0, 0 | 0.96 |
| r16 | 16 | — | 1.6, 1.6, 0.5 % | **1.2 %** | 2.44 | 1, 2, 0 | 0.98–0.99 |
| r8 | 8 | — | 12.5, 15.1, 20.8 % | **16.1 %** | 2.16 | 19, 2, 3 | 0.42–0.77 |
| r4 | 4 | — | 33.9, 22.4, 24.0 % | **26.7 %** | 1.96 | 43, 17, 5 | 0.41–0.59 |
| r8_rot30 | 8 | rot, 30 steps | 18.8, 16.1, 15.1 % | 16.7 % | 2.17 | 16, 12, 8 | 0.68–0.76 |
| r8_place30 | 8 | place, 30 steps | 24.0, 20.3, 22.9 % | 22.4 % | 2.06 | 24, 20, 2 | 0.72–0.74 |
| r8_place1000 | 8 | place, 1,000 steps | 12.5, 19.3, 17.7 % | 16.5 % | 2.13 | 17, 6, 1 | 0.69–0.83 |

**By the rule: branch 1.** `r8` loses 16.1 % (one seed at 20.8, two under 16): **the meadow does
not demand memory at the eyes' present range.** No memory arm rescues `r8`; none is better than `r8`
itself (16.5–22.4 %), and the ideal 30-step rotated memory, the capacity the D3 probe measured for
five arms, changes nothing (16.7 %). The deaths that remain at 8 tiles come on **day 1** (13–19 of
each seed's 24–40), from the lowest hydration starts (0.2 = 750 steps) 6–14 tiles from a pond the
sheep cannot see: a *search* problem in the first hour of life, not a memory problem. **Range is what
the world pays for:** 4 → 8 → 16 tiles of sight takes deaths from 27 % to 16 % to 1 %, and 16 tiles
is as good as seeing everything. So the willows must be readable from about 16 tiles by day, and a
reactive sheep that reads them should survive. Stage 1 follows; stage 2a's pressure is being built
(§ 21), as branch 1 requires.

**The forensics** (every hybrid arm, re-evaluated with the trace; the embed arm's numbers are typical):
- The dead died of **thirst with a full gut** (median hydration 0.0, energy 1.0–1.15 at death).
- They died **far from water** (median 15–21 tiles) and **long after last seeing it**: the median death
  came 840–2,300 steps after water was last within the 8-tile cone, 65–84 % of deaths more than 300
  steps after, and 19–45 % of the dead never had water within 8 tiles at all. A 30-step memory could
  not have saved most of them (§ 19.1's reading).
- Deaths come on **every day**, day 1 the most (random low starts), then days 2 and 3 (the "lost the
  pond" deaths).
- The hybrids keep only 29–58 % of their living steps within 8 tiles of water; privileged PPO keeps 96 %
  and the scripted flock 69 %. The sheep that survive are the ones that stay near water.
- **The 16-tile proxies were generous** (the review's § 5.2): thirsty-sees-water is 0.08–0.18 with an
  8-tile cone against 0.31–0.57 at 16; the information gap at 8 tiles is 0.13–0.28 (0.17–0.25 at 16);
  for the scripted flock the gap at 8 is 0.32 (0.08 at 16), because what a flockmate "knows" is now
  what it could actually have seen.
