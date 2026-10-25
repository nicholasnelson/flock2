# Design note: a world model with a map (proposal, revised 2026-10-26)

Status: **proposal; nothing built yet.** First version 2026-10-26. Revised the same day after the
owner's principle (below). Diagram: `map-brain.html`, published as
https://claude.ai/artifact/KEQwzib48kyri492esSh2b. It shows the last-resort design of § 4.

## The owner's principle

> "We want the sheep to earn their knowledge for themselves, as much as possible. Giving them
> shortcuts like map translation on movement should be an absolute last resort." (2026-10-26)

So the plan uses the best model available that gets **no free information**.

- **Allowed:**
  - what the sheep senses: its view and its felt numbers;
  - the action it chose;
  - generic, learned machinery: convolutions, recurrent memory, attention, latent variables,
    losses that predict its own senses.

  An architecture may have a *shape* (a 2-D grid of latent cells, say), but every operation on
  it is learned.
- **Not allowed except as a last resort, one piece at a time, each justified by a failed rung:**
  hand-coded knowledge of the world's geometry. That covers:
  - placing what is seen at map positions by projection (splat);
  - drawing the view by casting rays through a map;
  - shifting or rotating a memory by the known effect of an action;
  - a "body sense" that reports how far the sheep actually moved.
- **Privileged labels** (where water really is, and so on) remain for evaluation only, as before.

## Why a map at all

Four results so far point at how the state is organised. *(Judgement: suggestive, not proof.)*

- **The grid's view is already map-like, and there the hybrid works:** −0.017 to −0.023 per step
  on three seeds, beating the scripted sheep (−0.098).
- **The meadow model holds where water is, but not in a usable form.** Its decoder redraws water
  within 6–11° of the right bearing. Yet with 32 meadows, an MLP probe reads the left/right
  component of the direction to water at only R² 0.04.
- **Map-like privileged features make the meadow learnable:** PPO on them scored −0.39.
- **The vector-state model does not rank actions.** In the grid it ranked drink first beside
  water, when thirsty, in 10 % of states; the truth is 95 %.

Meadow v2 (`m1-world-v2-spec.md`) also removes several obstacles that had nothing to do with the
model: water out of reach, speckled ground texture, and a pond filling about 1.4 rows. Some of
the failures above may go away on v2 with no change to the model at all. So the first rung is to
find out.

## The ladder: least built-in first

Each rung is tried on meadow v2, with three seeds and criteria written before the run. A rung is
climbed only if the one below fails its criteria.

1. **Today's model on meadow v2.** The DreamerV3-style RSSM (one vector state h, z), learning
   from birth, with PPO on its state from real experience (the hybrid). Nothing new is built in.
   *Worked if* every seed beats random by 5 SE, and dawn water rises above 0.5 (toward privileged
   PPO's level on v2).
2. **Same model, better trained.** Learning only, no new structure:
   - longer runs;
   - a larger model (DreamerV3's model sizes scale the same design;
     [Hafner et al. 2023](https://arxiv.org/abs/2301.04104));
   - multi-step prediction of future views and felt numbers, so that the state must carry what
     stays out of view.
3. **A state with spatial shape, all of it learned.** The single vector state becomes a 2-D grid of
   latent cells, say 16 × 16 × 32, plus a global vector:
   - **encoder:** convolutions plus learned attention from the strip's columns into the cells;
   - **transition:** a ConvGRU with the action as input;
   - **decoder:** learned attention from the cells back to the strip's columns.

   Nothing tells the model which cell means "ahead-left", or how a step moves things. If a map
   emerges, it is the sheep's own. Probes (evaluation only) test whether it has: do cells come to
   stand for places, and does the grid shift consistently when the sheep walks?

   The nearest published idea I know of is the Neural Map
   ([Parisotto and Salakhutdinov 2017](https://arxiv.org/abs/1702.08360)). It has a learned write
   operator, but there the agent's position in the memory is given. Here it would not be.
4. **Last resort: built-in geometry, one piece at a time.** Each piece is separately justified,
   in this order (least knowledge first):
   1. a body sense of actual movement (proprioception);
   2. shifting and rotating the memory by that movement;
   3. projecting the view onto the map, and casting rays back.

   This is the design in § 4 and the diagram.

## 4. The last-resort design (fixed geometry)

Kept for reference; used only if rungs 1–3 fail, and then only the pieces needed.

1. **See.** A conv encoder gives 64 columns. Each column gets a depth distribution over 24 bins
   and a 32-number content vector; their outer product lifts each column along its line of sight
   ([Philion and Fidler 2020, Lift, Splat, Shoot](https://arxiv.org/abs/2008.05711)).
2. **Place (fixed).** Each (bearing θ, distance d) lands at x = d·sin θ, y = d·cos θ on an
   egocentric grid around the sheep.
3. **Remember and predict.** The memory is shifted and rotated by the sheep's movement (fixed),
   then a learned ConvGRU prior predicts the rest, and a learned posterior merges the new view.
   An egocentric map warped by the agent's own motion is the approach of the Cognitive Mapper and
   Planner ([Gupta et al. 2017](https://arxiv.org/abs/1702.03920)), which updates its belief
   "using the ego-motion to transform the belief from the previous time step into the current
   coordinate frame".
4. **Visualise (the owner's idea).** Rays are cast back through the map (fixed), and a learned
   per-cell appearance is composited front to back. A learned upsampler draws the strip, which is
   compared with the real view now and with the next view (prior).
5. **Act and feel.** A CNN policy on the map, plus heads for the felt numbers and the reward.

A learned mapper inside a modular exploring agent is also in
[Chaplot et al. 2020, Active Neural SLAM](https://arxiv.org/abs/2004.05155).

## Training (all rungs)

- On the sheep's own life: reconstruct this view and predict the next (squared error on RGB in
  meadow v2), the KL terms with free bits as DreamerV3, the felt numbers and the reward.
- No privileged labels in any loss. Probes and the decoder water check are for evaluation only.

## Grid first?

The grid's 8 × 8 view is top-down. A spatial latent there would get a map-shaped *input* for free,
which partly undercuts rung 3's test. *(judgement)* Rung 3 is better tested on meadow v2 directly,
using the grid only to debug code.

## Risks

- Rung 3 may learn a spatial grid that is not a map: cells used as an unordered memory. The
  probes will show it. If so, rung 3 has failed, which justifies rung 4's first piece and no more.
- Each rung costs runs. Three seeds of a 2,000-iteration meadow run take about 7 hours on the
  RTX 4080 at today's speed. Meadow v2 has 12 sheep, not 40, which should make that faster.
