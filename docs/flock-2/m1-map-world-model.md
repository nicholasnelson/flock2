# Design note: a world model with a map (proposal, 2026-10-26)

Status: **proposal for the owner's review; nothing built yet.** Diagram: `map-brain.html`
(pan and zoom), published as https://claude.ai/artifact/KEQwzib48kyri492esSh2b.

## Why

The idea came from the owner on 2026-10-26: turn what the sheep sees into a map, then put the
world model on the map. Four results so far point the same way.

- **The grid's view is already a map, and there the hybrid works.** It sees an 8 × 8 top-down
  window. PPO on the world model's state scored −0.017 to −0.023 per step on three seeds, beating
  the scripted sheep (−0.098) (README, "Grid hybrid: results").
- **In the meadow, the model holds where water is, but in a form a policy cannot easily use.** Its
  decoder redraws water within 6–11° of the right bearing. Yet with 32 meadows, an MLP probe reads
  the left/right component of the direction to water at only R² 0.04 (README, "Meadow hybrid with
  class-balanced reconstruction").
- **A map-like input makes the meadow learnable.** PPO on privileged egocentric features
  (direction and distance to the nearest water, grass and cover) reached −0.39, the best meadow
  result.
- **The vector-state model does not rank actions.** In the grid it ranked drink first beside water,
  when thirsty, in 10 % of states; the truth is 95 % (S3). On a map, most of an action's effect is
  geometric: walking shifts the map and turning rotates it. Only the rest needs learning.

*(judgement)* Together these suggest that the bottleneck is how the state is organised, not what
it is trained to predict.

## The design

All sizes are a first proposal, not tuned. The numbered parts match the diagram. In each part,
*learned* means the weights are trained, and *fixed* means geometry with no weights: innate
structure, like a lamb knowing that near things look big.

1. **See: lift the view (learned).** A small conv encoder turns the 64 × 256 strip into 64
   columns of 128 features. Each column covers about 4.7° of the 300° field. From each column it
   reads a depth distribution over 24 distance bins (0.25–16 tiles, finer near the sheep) and a
   32-number content vector. Their outer product spreads each column's content along its line of
   sight, in proportion to how likely each distance is: 64 bearings × 24 distances × 32.
2. **Place it on a map (fixed).** Each (bearing θ, distance d) cell lands at x = d·sin θ,
   y = d·cos θ on a grid of 0.5-tile cells. The sheep sits at the centre, facing up. The result is
   the observation map Oₜ, 64 × 64 × 32, plus a "seen" channel recording how much evidence each
   cell received.
3. **Remember and predict: the world model.**
   - **Memory.** The map Mₜ₋₁ (64 × 64 × 32; 32 × 32 tiles around the sheep), plus a global GRU
     state g (256) for things with no place on the map: time of day and feelings.
   - **Move the map (fixed).** Rotate it by ∓22.5° for a turn, and shift it by 0.24 or 0.7 cells
     for a walk or trot. Bilinear resampling; cells entering at the edge start as "unknown". The
     move is driven by a body sense of how the sheep actually moved (turn, and distance, which is
     0 if blocked). This is proprioception, not information about the meadow.
   - **Prior (learned).** A 3 × 3 ConvGRU over the moved map, given the action and g, predicts what
     moving cannot explain: grass eaten, other sheep walking, light fading. A categorical z per
     8 × 8 block (16 classes) keeps DreamerV3's stochastic state
     ([Hafner et al. 2023](https://arxiv.org/abs/2301.04104)), and so its ability to imagine.
   - **Posterior (learned).** A ConvGRU merges Oₜ into the prior map, gated by "seen". z is
     inferred per block, and g is updated with the felt numbers. KL(posterior ‖ prior) per block is
     the surprise, and it can be shown on the map.
4. **Visualise: draw the view back (owner's idea).**
   - Cast 64 rays and read the map at 24 distances: the splat run in reverse (fixed).
   - A per-cell MLP gives an opacity and an appearance vector (learned). These are composited front
     to back, so nearer things hide farther ones.
   - Apparent height ∝ 1/distance (fixed), then a learned ConvT upsampler gives the 64 × 256 × 16
     logits.
5. **Act and feel.** A small CNN over the map, plus g, gives 9 action logits and a value, trained
   with PPO on real experience as in the hybrid. On an egocentric map, "water ahead-left" is always
   in the same region, which is what a convolutional policy reads easily *(judgement)*. Felt
   numbers and reward are predicted from the pooled map and g, as in Milestone 1.

## Training: no privileged labels

Per step, on sequences from the sheep's own life:

- class-balanced cross-entropy of the strip drawn from the posterior map (this view);
- the same, drawn from the prior after the move, before the view is seen (next view);
- KL(posterior ‖ prior) with free bits, as DreamerV3;
- felt numbers and reward, as Milestone 1.

**Why the map has to become a real map** *(judgement)*. Drawing the view back from the map only
proves that the map keeps the information: an encoder and a decoder could agree on any private
code. The next-view term closes that loophole. The map is moved by fixed geometry before the next
view is drawn. A private code garbles when shifted or rotated, while a true layout still draws the
right view. Motion also separates near from far: things near the sheep move across the view faster.

**Privileged information is for evaluation only.** The map can be compared directly with the true
meadow (water, grass, cover cells), and drawn as an image for a viewer.

## Prior work (verified links)

- **Egocentric map warped by the agent's own motion:** Gupta et al. 2017, *Cognitive Mapping and
  Planning for Visual Navigation* ([arXiv:1702.03920](https://arxiv.org/abs/1702.03920)). It
  updates a top-down belief "using the ego-motion to transform the belief from the previous time
  step into the current coordinate frame". Their mapper is trained end to end for navigation
  together with a planner, not by predicting views.
- **Lifting image features into a top-down grid with a per-pixel depth distribution:** Philion and
  Fidler 2020, *Lift, Splat, Shoot* ([arXiv:2008.05711](https://arxiv.org/abs/2008.05711)). Parts
  1–2 follow this. Theirs is supervised with bird's-eye-view labels; ours is not.
- **A spatially structured 2-D memory for an RL agent:** Parisotto and Salakhutdinov 2017,
  *Neural Map* ([arXiv:1702.08360](https://arxiv.org/abs/1702.08360)).
- **A learned mapper inside a modular exploring agent:** Chaplot et al. 2020, *Learning to Explore
  using Active Neural SLAM* ([arXiv:2004.05155](https://arxiv.org/abs/2004.05155)).

What would be new here *(judgement, not a literature search)*: training the map only by drawing
the current and next first-person views, inside a stochastic world model with homeostatic
reward, with no map labels. Whether this has been done before is not checked.

## Risks and open questions

- **Forgetting.** The map edge is 16 tiles away, and a pond further off falls off the map. Options:
  a coarser outer ring, or a second allocentric map later.
- **Depth from a strip.** One glance cannot separate a small far tree from a big near one. Motion
  should resolve it, but a learned depth that never sharpens would smear everything along rays.
  Check: the map's water channel against the true ponds, by distance.
- **How much geometry to build in.** The fixed parts use the renderer's field of view and
  perspective, which the sheep "knows" innately. Learning them instead is purer but is one more
  thing that can fail. Start fixed.
- **Cost.** 64 × 64 × 32 per sheep for 640 sheep (16 meadows × 40) is 84 M floats per step of
  state, before activations. If memory is tight, cut the channels or the number of sheep per
  batch.
- **Scope.** This drops the imagination actor-critic and curiosity for now. Both can return on top
  of the map model, once it is shown to work.

## Build order and pre-registered criteria

Three seeds per step, one change per comparison, criteria fixed before the runs.

1. **Grid, 8 × 8 top-down view.** The lift is the identity here, so this tests the memory, the
   fixed move and the prior. *Worked if* a probe of the pond's direction from the map, when the pond
   is out of view, reaches R² > 0.9 (the vector model: 0.59–0.67), and the model ranks drink first
   beside water when thirsty in > 80 % of states (now 10 %). The hybrid's score should stay at
   least as good as now (−0.017 to −0.023).
2. **Grid with a first-person strip** (a new 1-D view of the grid). This tests the lift. *Worked
   if* the hybrid beats random by more than 5 SE with food and water above 0.3, and the map's water
   cells match the true pond (IoU > 0.5) for ponds seen in the last 50 steps.
3. **Meadow.** *Worked if* the hybrid's dawn-3 water rises toward privileged PPO's 1.09 (above 0.5
   on all seeds), and its reward beats the vector-model hybrid run for the same number of
   iterations.

Rough effort *(judgement)*: step 1 is one to two days, step 2 one day, step 3 two to three days,
plus runs.
