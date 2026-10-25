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
