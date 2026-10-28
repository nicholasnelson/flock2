# Review (2026-10-29): does the meadow demand memory, and should we build the brain in stages?

Scope: reading only (the spec through § 18, the two 2026-10-26 reviews and the plan, the ladder, Track B,
`flock/wm.py`, `scripts/m2_hybrid.py`, `scripts/m2_eval.py`, `flock/meadow2/*`, the results JSON). Nothing
was run and no code was changed. *(judgement)* marks my own reading; *(established)* points at a link in
§ 8 that I opened and checked today. Project numbers are quoted with their file.

## Summary

1. **The owner is mostly right, with a twist.** The meadow does not demand the memory the project has
   been measuring (a bearing held for 30 steps after water leaves view). A memory-free policy with
   complete perception solves the meadow outright (privileged PPO is a plain MLP on the current
   features: 0 deaths, `eval/ppo_s*.json`). What kills the vision sheep is that their *usable* water
   perception reaches about 4–8 tiles by day and nothing beyond, in a 48 × 48 meadow where they must
   re-find a pond every 1.6 days. The capacity the world actually demands is either far perception, or
   a search strategy, or "stay within sight of water", or a place memory lasting hours: not a 30-step
   bearing. So the learned memory did not appear because the reward and the reconstruction loss pay
   almost nothing for it, which is itself the owner's thesis in action. One half-day measurement
   settles it (§ 1.4).
2. **The staged "forced evolution" approach is sound and is the field's own direction**, provided each
   stage has (a) a *demand check*: the previous stage's brain must demonstrably fail on the new world,
   (b) a *route-around audit*: pre-registered cheap strategies that would satisfy the pressure without
   the capacity, and (c) all earlier stages' evaluations kept as regression tests. The literature says
   curricula and co-evolved environments work (POET, PAIRED/ACCEL, hide-and-seek), that "start small"
   is not a law (Rohde & Plaut reversed Elman), that memory does emerge in plain recurrent nets when a
   task needs it (Wijmans et al.; Cueva & Wei), that adding stages brings forgetting and loss of
   plasticity (Kirkpatrick; Dohare), and that agents route around pressures with great reliability
   (specification gaming; Kottur; Lowe).
3. **Hand-design the module slots; search only small things.** Neuroevolution of the meadow brain is
   out of reach on one RTX 4080 (rendering alone is ~27 minutes per generation at today's 0.22 ms a
   view). Evolving a handful of discrete choices, or policies in the grid world, is cheap with
   EvoJAX-style tooling. Progressive-network style "add a column, keep the old one" is the published
   form of "bolting on a network".
4. **First two stages (§ 4):** Stage 1 is a *reactive* sheep (conv encoder + PPO, no world model,
   no recurrence) on a meadow where water is visible from far (a world fact: willows mark the ponds).
   Pass: ≤ 20 % die on `m2_eval`, three seeds. Stage 2 adds the first real memory pressure: the banks
   are bare, so grazing happens at the edge of sight of the landmark and a few steps too far loses it.
   The module is recurrence; the probe is the validated D3; the demand check is that the stage-1 brain
   dies on the stage-2 world. The fox and the alarm call can come beside stage 2, because an alarm
   needs only perception, so signalling can start before water memory exists.
5. **Other things I think are off (§ 5):** the information-gap metric uses a 16-tile "sees water" proxy
   when the eyes read water to about 8; the week's criterion and the D3 rule measure a capacity the world
   barely rewards; the reconstruction loss is a designer's choice too, so letting reward train the
   encoder at stage 1 is not a violation of the principle; non-lethal drives (warmth, huddling) will
   not force anything.

## 1. Does the meadow demand memory? (Q1)

### 1.1 What the numbers say

| Quantity | Value | Source |
| --- | --- | --- |
| Hydration budget from full / from the lowest start (0.2) | 3,750 / 750 steps | spec § 11 (`NEED_SCALE` 3) |
| Flock start to nearest water, median / 90th pct | 10.2 / 13.2 tiles = 85 / 110 walk-steps | `checks_v21_patch.json` |
| Blind explorer (turns when blocked, no eyes, no memory): first water, median | 1,434 steps; 14 % never within 4,000 | `checks_v21_patch.json` |
| Same explorer: finds water before it would die | 65 % of starts | same |
| Privileged PPO (reactive MLP on current features, water bearing at any range) | 0 deaths, 2.47 / step | `eval/ppo_s*.json` |
| Scripted flock (reactive rules on true state) | 0 deaths, 2.36 / step | `eval/scripted.json` |
| Vision hybrids, best arms (embed, n1 s2, rot, mem) | 43–57 % die; days survived 2.0–2.3 | `eval/hybrid_*.json` |
| Thirsty hybrid sheep with water in the geometric 16-tile cone | 39–53 % of thirsty steps | same |
| Water redrawn from the state by day: < 4 / 4–8 / 8–16 tiles | 0.81 / 0.37 / 0.00 (arm R) | `diag/rot_s0.json` D1 |
| Bearing readable from the eyes (embedding, ridge R²): < 4 / 4–8 / 8–16 tiles | 0.63–0.74 / 0.29–0.35 / < 0 | spec § 15, `diag/rot_s0.json` |
| Bearing 1–30 steps after water leaves view (D3), every arm | ≤ 0 (probe validated: ideal memory 1.00) | spec § 17–18 |
| Information gap (thirsty, cannot see water, a flockmate within 10 tiles saw it in 300 steps) | 20–26 % | `eval/hybrid_*.json` |

Three readings follow *(judgement)*:

- **Reactive suffices if perception is complete.** The privileged policy has no recurrence at all: it is
  `m2_ppo.py`'s MLP on `privileged.observe`, which hands it the bearing and distance of the nearest water
  through walls, at night, at any range. It never dies. So the meadow has a memory-free solution.
- **The vision sheep's effective water range is ~4–8 tiles.** D1 recall is zero beyond 8 tiles in every
  arm and the embedding's bearing probe is negative beyond 8. In a 48 × 48 meadow with two ponds, a
  sheep more than 8 tiles from both ponds is blind to water. The thirsty-sees-water proxy (16 tiles,
  no occlusion) overstates what they can use; at 8 tiles it would be well under 0.4.
- **The deaths are day-2 deaths.** Days survived are 2.0–2.3 with 43–57 % dying, and hydration from full
  lasts 1.6 days. That is the signature of sheep that drink once (or start full), wander off to graze,
  and cannot re-find a pond a day later. A 30-step bearing memory would not save them: by then the pond
  is a day's wandering away. What would save them is any of: (a) seeing the pond from far, (b) never
  grazing out of sight of it, (c) a search that covers ground (walk straight, follow the fence), or
  (d) a place memory lasting a day. The world rewards all four equally and the project has been
  probing only a short version of (d).

### 1.2 Why that explains the non-emergence

The reconstruction loss pays for a far pond at about 13 units of a ~100-unit frame at 8 tiles and 1 at
16 (`review-2026-10-26-water.md`, § 1), and the reward's gradient toward "remember the pond you left
an hour ago" passes through hundreds of steps at γ = 0.998 from a state that cannot even read the
bearing of a visible pond at 8 tiles. There is no stage at which remembering is the cheapest way to
score. Memory emerges in plain recurrent agents when the task leaves them no other way: blind PointGoal
agents with only egomotion and a goal vector build map-like memories lasting ~1,000 steps and shortcut
through them *(established)* [21]; recurrent nets given velocity and asked to localise grow grid and
border cells *(established)* [22]; on Memory Gym's endless tasks a GRU beats Transformer-XL
*(established)* [24]; and a carefully tuned recurrent model-free agent is a strong baseline across 18 of
21 POMDP tasks *(established)* [25]. The common thread is that the task pays for memory on every step
and nothing cheaper works. The meadow does not do that.

### 1.3 Where the owner's hypothesis is wrong, or at least incomplete *(judgement)*

The meadow does demand *something* the current sheep lack, and a view-reactive policy with the eyes as
they are is not "close to enough": the best arm still loses 43 % of its sheep. The gap is not memory,
but it is not nothing: it is perception range plus a strategy for being out of range. A reactive sheep
can survive only if it either sees water from everywhere or learns the "stay within sight of water"
strategy. Grass grows to 172 % of its start with 12 sheep (`checks_v21_patch.json`), so a ring of
8 tiles around each pond (about 12 % of the meadow) would probably feed the flock *(judgement, not
measured)*; a reactive learner could find that strategy, and it would then have nothing to remember and
nothing to tell. That is the route-around to watch for in § 4.

### 1.4 The quick measurement that settles it (pre-registerable; half a day; no new learning sheep)

**M1. The myopic-oracle ladder.** Train `m2_ppo.py` unchanged except that the water features in
`privileged.observe` are masked (bearing 0, distance 1) when the nearest water is outside the 300° field
or farther than *r* tiles, for *r* ∈ {4, 8, 16, ∞}. Three seeds each at *r* = 8 (12 minutes a seed; about
2.5 hours in all). Then two memory arms at *r* = 8: (i) the **ideal 30-step memory** from the D3 validation
(the last-seen bearing rotated by the turns since, fading to 0 after 30 steps) as two extra inputs; (ii)
the same fading over 1,000 steps. All of this is privileged and allowed: it probes the world, not a sheep.

Rules, written before running:
- If reactive *r* = 8 loses ≤ 20 %: **the meadow does not demand memory at the eyes' present range.** The
  vision sheep's limit is reading what they see, and the policy's use of it; no memory module is
  justified by this world. Proceed to § 4 stage 1.
- If reactive *r* = 8 loses ≥ 40 % and the 30-step memory rescues it to ≤ 20 %: the D3 capacity was the
  right target after all, and the owner's hypothesis is wrong; the problem is learning, not demand.
- If reactive *r* = 8 fails and only the 1,000-step memory rescues it: the demanded capacity is a place
  memory lasting hours (or far perception), and the D3 probe has been measuring the wrong thing.
- *r* = 4 against *r* = 8 says how much far perception alone buys; *r* = 16 says whether the geometric
  proxy's range would be enough.

**M2. Free, from the existing evaluation traces (CPU minutes):** for every hybrid death, the distance to
the nearest water and the steps since water was last in the 8-tile cone; the per-day death curve; the
share of living steps within 8 tiles of water. If most deaths come more than 300 steps after water was
last near, no 30-step memory could have helped, and M1's first branch is already likely.

## 2. The staged "forced evolution" approach: what the literature says (Q2)

### 2.1 What works

- **Curricula and co-evolved environments.** The curriculum-learning survey for RL frames the ordering of
  tasks so that problems too hard from scratch become tractable, and classifies dozens of methods by their
  assumptions *(established)* [6]. POET pairs environments with agents, evolves both, and transfers
  solutions between environments as stepping stones; the authors report it solves challenges that direct
  optimisation or a direct-path curriculum cannot, and that transfer is essential *(established)* [1],
  with Enhanced POET adding a domain-general novelty measure and a flexible environment encoding
  *(established)* [2]. PAIRED has an adversary propose environments that maximise the protagonist's
  regret against an antagonist, yielding "a natural curriculum of increasingly complex environments"
  *(established)* [3]; Robust PLR curates random levels with a Nash guarantee and improves PAIRED
  *(established)* [4]; ACCEL edits levels to stay "at the edge of an agent's abilities", starting simple
  and growing *(established)* [5]. These are the owner's idea, automated.
- **Pressure from other agents.** Hide-and-seek produced six phases of strategy (shelters, ramps) from
  competition alone, each phase forcing the other team to adapt *(established)* [18]; simple physics
  arenas with self-play produced tackling, feinting and defending, with the competition and not the
  environment supplying the complexity *(established)* [19]; XLand's open-ended task generation gave
  agents that transfer zero-shot to hide-and-seek, capture-the-flag and tag *(established)* [20].
  Caution *(judgement)*: all three used compute far beyond one GPU; what transfers to this project is the
  mechanism (an opponent is a self-adjusting pressure), not the scale.
- **Memory and maps emerge when demanded**: § 1.2, refs [21], [22], [24], [25]. Memory Maze shows the
  other side: current agents succeed on small mazes and fall well short of humans on large ones
  *(established)* [23], so demand alone does not guarantee emergence at scale.
- **Modular growth.** Progressive networks add a new column per task, keep prior columns, connect
  laterally, are "immune to forgetting", and transfer in both sensory and control layers on Atari and 3D
  mazes *(established)* [9]. NEAT starts from minimal networks and complexifies, with speciation
  "protecting structural innovation" so new topologies have time to be optimised *(established)* [13];
  the 2019 review covers evolving architectures, novelty and diversity, and indirect encoding
  *(established)* [14]. Novelty search beats objective search on deceptive maze and biped tasks by
  rewarding new behaviour instead of the goal *(established)* [15]. Clune's AI-GA framing names
  "generating effective learning environments" as one of three pillars *(established)* [17].
- **Communication under environmental pressure.** A referential game produces a code because the receiver
  can succeed only by using the message *(established)* [26]; goal-directed multi-agent environments
  produce symbol streams with vocabulary and syntax, and pointing and guiding when speech is unavailable
  *(established)* [27]; cooperative gridworld navigation produces signals that refer to locations and
  directions ("left", "upper left room") *(established)* [28]. In evolved robot colonies, communication
  about food evolved readily with genetically similar colonies under colony-level selection; under
  individual selection among unrelated robots, signalling turned deceptive and lowered performance
  *(established)* [31]. That last result bears directly on a flock whose reward is individual
  homeostasis (§ 4, stage 2a).
- **Developmental framing.** Oudeyer & Kaplan's typology of intrinsic motivation is the standard
  developmental-robotics reference for staged, self-driven learning *(established)* [33]; the project's
  curiosity ensemble already sits in that tradition.

### 2.2 What is known to fail

- **"Start small" is not a law.** Elman's recurrent nets learned an embedded grammar only when input or
  memory was initially restricted *(established)* [7]; Rohde & Plaut, with more natural-looking languages,
  found restriction unnecessary and, as languages became more English-like, harmful *(established)* [8].
  Lesson *(judgement)*: an easy first world is a bet, not a guarantee; it pays when the hard world's
  gradient is flat (which § 1 says it is here), and it costs when the easy world teaches a habit the hard
  world must undo (the stay-near-water strategy).
- **Forgetting and loss of plasticity when stages stack.** Catastrophic forgetting is the default when
  tasks are learned in sequence; EWC slows learning on weights important to old tasks *(established)*
  [10]; progressive nets avoid it by freezing *(established)* [9]. Beyond forgetting, networks trained
  continually *lose the ability to learn new things*: on an ImageNet task sequence accuracy fell from 89 %
  to 77 % by the 2,000th task, across architectures and optimisers, mitigated by L2 plus weight
  perturbation, and by re-initialising a small share of the least-used units *(established)* [11].
  Continual World argues the field over-focuses on forgetting and should measure forward transfer
  *(established)* [12]. For a plan that keeps one brain and stacks pressures, all three apply: every stage
  must keep every earlier evaluation as a regression test, and the plan must say in advance whether old
  modules are frozen.
- **Agents route around pressures.** Specification gaming (flipping the block instead of stacking it;
  circling for points instead of finishing the race) is the general form *(established)* [32]. In
  emergent communication specifically: agents reach near-perfect reward with languages that are neither
  interpretable nor compositional unless the channel is constrained *(established)* [30]; messages can be
  informative about the speaker's next action while having no effect on the listener *(established)*
  [34]; without inductive biases, communication often fails to emerge at all in temporally extended
  settings *(established)* [29]. M0 needed nine interventions (`m0-toy-README.md`). Lesson: each pressure
  needs a written list of the cheap ways to satisfy it, with a measurement that tells them apart.
- **Hand-designed curricula carry the designer's guess.** ACCEL's authors note that evolutionary
  environment methods "usually depend on domain-specific heuristics and large computational resources"
  *(established)* [5]. A hand-made ladder is exactly such a heuristic; the demand check (§ 4) is how we
  find out when the guess was wrong.
- **Lock-in of early solutions.** Floreano et al. observed that once one signalling system became
  established it tended to block the evolution of better ones *(established)* [31]; novelty search exists
  because objectives are deceptive *(established)* [15]. A stage that is passed by a crude strategy may
  block the better one the next stage needs.

### 2.3 My assessment *(judgement)*

The staged approach is the right correction to a programme that has spent five arms on a capacity the
world does not pay for. Its strength is that it turns "why didn't memory appear?" into "does this world
kill sheep that lack memory?", a question with a yes/no measurement. Its risks are the three above:
routing around, forgetting, and early lock-in. All three have cheap guards (§ 4). The one thing I would
not do is restart from a world *simpler than the meadow*: the grid world already works and tells us
nothing about first-person perception, which is where the meadow sheep actually fail. "Easy" should
mean "the meadow with its perception demand lowered", not "a different world".

## 3. Hand-designed modules or searched? What fits one GPU (Q3)

**Costs today** (spec § 14, § 16): a view is 0.22 ms; a 600-iteration hybrid run is ~16 minutes and
2,000 iterations about an hour; privileged PPO solves the meadow in 100 updates, 12 minutes a seed;
three seeds × 2,000 iterations of a meadow run is an afternoon. RAM is the binding limit (watchdog at
12 GB; the spatial model crashed WSL at compile).

**Neuroevolution of the meadow brain is not feasible** *(arithmetic)*: a population of 32 policies, each
scored on 8 meadows × 12 sheep × one day (2,400 steps), renders 7.4 M views, 27 minutes per generation
before any network cost; a few hundred generations is a week per experiment. NEAT-style topology search
on top multiplies that. EvoJAX shows neuroevolution runs "within minutes on a single accelerator" for
its tasks *(established)* [16], but those are small observation spaces; here the renderer is the cost.

**What search can do here:**
- Evolve policies (not perception) in the **grid world**, where there is no renderer and the hybrid
  converges in 200 iterations; good for the channel bench and for testing a pressure's logic before the
  meadow.
- Evolve a **small gene set** (a dozen discrete choices: which felt numbers, which memory block, GRU size,
  whether the policy reads the embedding), scored by meadow runs: that is a 2–3-day sweep, not an
  algorithm. It is honest to call it hyperparameter search.
- Evolve **world parameters** rather than brains (POET's other half): haze length, bank width, pond count,
  to keep each stage "at the edge of ability" (ACCEL's idea [5]) is cheap because checking the scripted
  flock on a world is CPU minutes.

**Recommendation** *(judgement)*: hand-design the slots (what a module reads, what it writes, which probe
must light up), keep gradient learning inside them, and reserve evolution for the grid and for the world.
When a module is added, do it progressive-network style first (keep the old column, add the new, lateral
reads, old column frozen) and un-freeze only if the regression tests hold [9]. This keeps "one change per
comparison" meaningful: the module is the change.

## 4. The staged plan (Q4)

Conventions for every stage: three seeds or none; criteria written before runs; every earlier stage's
evaluation is re-run as a regression test and must still pass; each new pressure has a **demand check**
(the previous stage's winning brain is run on the new world and must fail its criterion, or the pressure
is not a pressure) and a **route-around list** (cheap strategies that satisfy the pressure without the
capacity, each with the measurement that detects it). Probes stay evaluation-only.

### Stage 0 (prerequisite, half a day): M1 and M2 from § 1.4

Not a learning stage; it decides whether stage 2's pressure is the right first one.

### Stage 1: the reactive sheep in a meadow that can be seen

**World (one change, a world fact):** ponds are marked by **willows**: 2–3 trees of the existing broadleaf
shape standing on each pond's bank. Trees are tall, dark on sky, never hidden by the ground, and already
read at R² 0.3–0.6 at the same probe where water reads 0 (spec § 14; `review-2026-10-26-water.md` § 1).
Nothing is told to the sheep: it must learn that those trees mean water. Alternative if the owner prefers
not to add objects: reeds (a tall light-green fringe) on the shore. Everything else as meadow v2.1.

**Model (the smallest thing that should clearly work):** a conv encoder on the 64 × 256 RGB view plus the
6 felt numbers, an MLP policy and value, trained end-to-end by PPO with death handled as `m2_ppo.py`
does. **No world model, no recurrence.** This is the sheep's own eyes trained by the sheep's own reward:
it gives nothing away. Keep `m2_hybrid.py`'s loop (8 meadows × 12 sheep × 64 steps, staggered resets);
drop the replay and the world-model updates. Budget: 2,000 iterations (about 40 minutes a seed at
today's speed, less without the world model).

**Comparison arms (one change each, seed 0 first, then the best at three seeds):**
- A1 reactive, willows (the stage proper);
- A2 reactive, no willows (does far perception matter, or was the world model the cost?);
- A3 the current best hybrid (arm "embed"), willows (is the RSSM costing us?).

**Pass (pre-registered):** every seed of A1 at most 20 % dead on `m2_eval.py` (16 meadows, 3 days) and
reward ≥ 2.0. *Strong:* ≤ 5 % dead. **Reported:** per-day death curve, time to first drink, share of
living steps within 8 tiles of water, thirsty-sees-water with the 8-tile cone, the D2 bearing probe on the
encoder (expect ≥ 0.5 for willows at ≥ 8 tiles by day).

**Route-around list:** (i) stay-within-sight-of-water (the 8-tile share above; if > 0.8, note it, it is
allowed here but it will be the first thing stage 2 must break); (ii) camping at the pond and starving
(graze success and energy on the last day).

**If A1 fails:** the problem is perception or policy learning from pixels, not memory, not the world
model; the next change is to the encoder (column-preserving, as the earlier review's fix 2) or to PPO's
budget, and the plan says so before anything about memory.

### Stage 2: the first memory pressure (and, in parallel, the first signal)

**2a. World pressure: the banks are bare.** Grass does not grow within *R* tiles of water (muddy,
trampled banks; a world fact), with *R* set so that grazing happens at the edge of the willows'
readable range: start at *R* = 8 and adjust by the scripted-flock checks (0 deaths must still hold for
the scripted flock, which has privileged water knowledge). Now every sheep must leave sight of the
landmark to eat and come back, and a few steps too far loses the landmark behind haze or trees.

**Demand check:** the stage-1 brain on the stage-2 world must lose ≥ 40 % (if it does not, raise *R* or
shorten haze until it does, logging each change; if no setting makes it fail without killing the scripted
flock, the pressure is wrong and memory is not demanded by this route).

**Module:** recurrence in the policy: a GRU (DreamerV3's layer-normalised GRU from `wm.py` is fine)
between encoder and policy, trained by PPO through time (truncated backprop over the 64-step rollout,
which is what Memory Maze found current agents need [23]). Progressive style: keep the stage-1 encoder
and policy column, add the GRU column reading the encoder's embedding and the previous action, policy
reads both. The RSSM is a second candidate for the same slot (arm M's settings), compared one change at a
time; if the GRU-only policy passes, the world model is not needed for memory and the ladder says so.

**Pass:** ≤ 20 % dead on the stage-2 world, three seeds, **and** D3 (the validated probe: bearing of
water 1–30 steps after it leaves view, ridge from the recurrent state, with the wide-noise control
≤ 0.1) ≥ 0.3, **and** stage 1's evaluation still passes (regression).

**Route-around list:** (i) grazing only on the bank edge where the willows are still in view (the
8-tile share; the demand check already prices this); (ii) wall-following or straight-line search that
re-finds water without memory (D3 stays ≤ 0 while deaths drop; then the capacity that appeared is
*search*, which is worth having, but must be named); (iii) following other sheep (nearest-neighbour
distance falls, others-in-view rises, deaths drop, D3 flat): also worth having, also to be named, and the
reason signalling will later have to beat following.

**2b. World pressure, in parallel on its own branch: the fox, and the alarm.** A scripted fox visits
about once a night (the brief's design), takes the most isolated sheep it reaches, and is visible as a
fourth billboard. Cohesion becomes worth it (dilution), huddling already pays warmth, and an **alarm
channel** (Track B's 2-number signal, carried into the meadow through `sim.py` as a broadcast heard
within earshot, entering the sheep through its felt numbers) is worth using the moment a sheep that sees
the fox benefits from flockmates bunching toward it. This is the earliest signal that can mean something,
because "fox in view" needs only perception, not memory; it is also the only pressure here under which an
individually rewarded sender gains from being heard (the caller's own risk falls when others come), which
is the condition Floreano et al. found necessary to keep signalling honest [31] *(the mapping to this
world is judgement)*.

**Pass:** causal influence by the Track B tests (swap what is heard across worlds; listener's movement
and survival change, p < 0.01, ≥ 5 of 8 seeds in the grid first, then 3 of 3 in the meadow), and the
beacon baseline (a scripted sender that signals when the fox is in view) is matched or beaten by the
learned channel. **Route-around list:** signals that predict the sender's own next action but move no
listener (Lowe's pitfall [34]); cohesion without any signal (the no-signal condition is the control);
the fox being avoided by all sheep standing still (action mix at night).

### Stage 3: water calling (needs 2a and 2b)

**Pressure:** make the information gap real and consequential: one pond dries at a random time each
day and refills later, so the sheep near it know and the others do not; or the willows are removed at
night (the beacon only by day) so a sheep that saw water by day knows what a night-thirsty flockmate
does not. Keep the gap metric with an 8-tile cone and require it ≥ 0.3 of thirsty steps before any
channel work (the plan review's go/no-go). **Module:** the same channel as 2b, now carrying content that
only a sheep with the stage-2 memory can hold. **Pass:** the Track B criteria (time to water for blind
thirsty sheep ≥ 20 % below the no-signal condition; causal influence; beats the beacon that signals
"I am drinking"; survives the sender being out of view). **Route-around:** following (sender visible),
presence-only calls (the beacon ceiling).

### Stage 4: marks (persistent information)

**Pressure:** information whose value outlives the caller's presence: the dry pond's schedule, a fox
den, a rich patch that regrows over 3 days. Sound decays at once; a drawn mark persists and is seen
through the ordinary view. M0's medium (black strokes, pen-lift cost, re-centring, jitter, clutter) is
built. **Pass:** a listener that never met the marker changes its route and outcome because of the mark;
blank and swapped marks do not. This is where the brief's sound-vs-marks specialisation hypothesis
finally gets its test.

### Stage 5: generations (far later)

Lambs inheriting weights with mutation, learning by observing elders; iterated-learning literature in the
project's own survey. Not before stage 3 has produced something worth inheriting.

### What is reused

| Existing piece | Use in the staged plan |
| --- | --- |
| `flock/meadow2/` world, drives, sim, render, scripted flock, checks | Kept as the base world at every stage; each pressure is one change to it, checked by the scripted flock first |
| `scripts/m2_eval.py` (deaths, days, dispersion, information gap) | The fixed evaluation and regression test at every stage; add an 8-tile cone and per-day deaths |
| `scripts/m2_diag.py` D1–D3 and the validated D3 probe | Stage 2's memory probe; D2 on the encoder at stage 1 |
| `scripts/m2_ppo.py` (PPO with death) | Stage 1's learner, with the encoder in front of it; stage 0's myopic oracle |
| `scripts/m2_hybrid.py` loop | The rollout loop; the replay and world-model updates become optional |
| `flock/wm.py` RSSM (and its GRU) | One candidate for stage 2's memory slot; the ring (`rot_bins`) is retired |
| `flock/wm_spatial.py` | Shelved; nothing in the plan demands a spatial state until a long-range place memory is shown to be demanded (M1's third branch) |
| Track B (`flock/grid/flock.py`, trainer, evaluation, beacon and no-signal conditions, causal tests) | Run as designed in the grid first (CPU, 8 seeds), then its channel and metrics move into the meadow at 2b |
| Grid world and its hybrid | The cheap bench for channel logic and for any evolutionary search of policies; not a stand-in for the meadow |
| M0's drawing medium and the independent examiner | Stage 4 |

## 5. What else I think is being got wrong (Q5)

1. **The target of the last four arms was never shown to matter.** The week's criterion (bearing ≥ 0.5
   in view, ≥ 0.3 just after) and the D3 rule were pre-registered as *measurements of the state*, not
   as capacities the world demands. Pre-registration guarded against moving the goalposts but not
   against aiming at the wrong goal. Every future criterion should come with its demand check.
2. **The information-gap and thirsty-sees-water proxies use 16 tiles and no occlusion** (`m2_eval.py`,
   `SEE_RANGE`). The eyes read water to about 8 tiles. Recompute both at 8; the gap will shrink or
   grow, and the plan for a channel rests on it.
3. **Deaths from random starts at hydration 0.2** make the primary measure harsh for any learner that
   needs a day to find water, as the plan review said; the per-day death curve and time-to-first-drink
   are still not reported. They are free.
4. **The reconstruction loss is a designer's choice, not a law of the sheep.** The principle forbids
   privileged information and built-in geometry; it says nothing about which of the sheep's own
   signals train the encoder. Letting the reward train the encoder (stage 1) is as honest as letting the
   pixels do it, and DreamerV3's own authors note the reconstruction objective needs a weak regulariser
   to keep fine detail *(established)* [35]. Recording the choice is enough.
5. **Non-lethal drives will not force anything.** Warmth is not lethal; huddling pays only warmth; the
   hybrid sheep huddle 1–5 % of night rest. Under the staged logic that is expected: there is no pressure.
   Either warmth becomes lethal at some stage or huddling is not a capacity we should expect.
6. **The ring's write (arm R) is a design that could not have worked as a memory** (the owner's reading
   in § 18 is right: every bin is rewritten from z each step). Fixing it would mean telling the model
   which bins are "ahead", which is more geometry. Retiring it is the right call; the staged plan makes
   it unnecessary.
7. **"Interesting and meaningful" still has no bar that presence cannot pass.** The plan review's four
   conditions (causal, referential on held-out worlds, beats the beacon and following, survives the sender
   out of view) should be written into stage 3's criterion now, not after the first channel run.
8. **Compute hygiene is good** (watchdog, one job at a time, no name collisions); keep the stage runs at
   600 iterations for screening and 2,000 for claims, as now.

## 6. Top risks of the staged approach, and the guard for each *(judgement)*

| Risk | Guard |
| --- | --- |
| A pressure is routed around (stay-near-water, wall-following, following others) and the stage "passes" without the capacity | The route-around list and its measurements, written with the criterion; the capacity probe (D2, D3, causal influence) is a *pass condition*, not a report |
| The pressure does not bite (the old brain survives the new world) | The demand check, run before the new module is trained |
| Forgetting or loss of plasticity as stages stack | Every earlier evaluation is a regression test; add modules progressive-style first; log the plasticity remedies if ever used [11] |
| Stage 1 fails (reactive from pixels does not solve the seen meadow) | Then the problem is perception from pixels, and the plan says the encoder is the next change, not memory and not the world |
| Early lock-in: a crude stage-2 solution (search, following) blocks memory | Name it as a capacity when it appears; raise the pressure (bare banks wider, haze shorter) only with a new demand check |
| Compute: each stage is 3 seeds × 2,000 iterations plus arms | Screen on one seed and on probes (minutes), claim on three; evolution only in the grid |

## 7. Answers in one line each

- **Q1.** Mostly yes: the meadow has a memory-free solution (privileged PPO is reactive) and the deaths
  are day-2 "lost the pond" deaths that a 30-step bearing would not fix; what the world demands is far
  perception or a strategy for being out of range. M1 (the myopic-oracle ladder, half a day) settles it.
- **Q2.** Staged pressures are what the open-ended and multi-agent literature does; the known failures are
  routing around, forgetting/plasticity, and early lock-in, each with a cheap guard; "start small" is a
  bet, not a law.
- **Q3.** Hand-design the slots, learn inside them by gradient, add them progressive-style; evolve only in
  the grid or over a dozen discrete choices; the renderer makes meadow neuroevolution infeasible.
- **Q4.** Stage 1 reactive conv + PPO on a meadow with willows (≤ 20 % die); stage 2a bare banks → a
  GRU, D3 ≥ 0.3 with the demand check; stage 2b fox + alarm channel, the earliest meaningful signal;
  stage 3 water calling; stage 4 marks; stage 5 generations.
- **Q5.** Criteria measured a capacity nobody showed the world demands; the gap proxy is too generous; the
  recon loss is a choice; non-lethal drives force nothing.

## 8. References (each link opened and checked on 2026-10-29)

1. Wang, Lehman, Clune, Stanley, "POET", https://arxiv.org/abs/1901.01753
2. Wang et al., "Enhanced POET", https://arxiv.org/abs/2003.08536
3. Dennis et al., "Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design" (PAIRED), https://arxiv.org/abs/2012.02096
4. Jiang et al., "Replay-Guided Adversarial Environment Design" (Robust PLR), https://arxiv.org/abs/2110.02439
5. Parker-Holder et al., "Evolving Curricula with Regret-Based Environment Design" (ACCEL), https://arxiv.org/abs/2203.01302
6. Narvekar et al., "Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey", JMLR 2020, https://jmlr.org/papers/v21/20-212.html
7. Elman, "Learning and development in neural networks: the importance of starting small", Cognition 1993, https://pubmed.ncbi.nlm.nih.gov/8403835/
8. Rohde & Plaut, "Language acquisition in the absence of explicit negative evidence: how important is starting small?", Cognition 1999, https://pubmed.ncbi.nlm.nih.gov/10520565/
9. Rusu et al., "Progressive Neural Networks", https://arxiv.org/abs/1606.04671 (abstract: immune to forgetting, lateral transfer; the column mechanics are in the body, which I did not re-read)
10. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", https://arxiv.org/abs/1612.00796
11. Dohare et al., "Maintaining Plasticity in Deep Continual Learning", https://arxiv.org/abs/2306.13812 (the Nature 2024 version is behind a cookie gate)
12. Wołczyk et al., "Continual World", https://arxiv.org/abs/2105.10919
13. Stanley & Miikkulainen, "Evolving Neural Networks Through Augmenting Topologies" (NEAT), Evolutionary Computation 2002, https://nn.cs.utexas.edu/?stanley:ec02
14. Stanley, Clune, Lehman, Miikkulainen, "Designing neural networks through neuroevolution", Nature Machine Intelligence 2019, https://www.nature.com/articles/s42256-018-0006-z
15. Lehman & Stanley, "Abandoning objectives: evolution through the search for novelty alone", Evolutionary Computation 2011, https://pubmed.ncbi.nlm.nih.gov/20868264/
16. Tang, Tian, Ha, "EvoJAX: Hardware-Accelerated Neuroevolution", https://arxiv.org/abs/2202.05008
17. Clune, "AI-GAs", https://arxiv.org/abs/1905.10985
18. Baker et al., "Emergent Tool Use From Multi-Agent Autocurricula", https://arxiv.org/abs/1909.07528
19. Bansal et al., "Emergent Complexity via Multi-Agent Competition", https://arxiv.org/abs/1710.03748
20. Open-Ended Learning Team et al., "Open-Ended Learning Leads to Generally Capable Agents" (XLand), https://arxiv.org/abs/2107.12808
21. Wijmans et al., "Emergence of Maps in the Memories of Blind Navigation Agents", ICLR 2023, https://arxiv.org/abs/2301.13261
22. Cueva & Wei, "Emergence of grid-like representations by training recurrent neural networks to perform spatial localization", https://arxiv.org/abs/1803.07770
23. Pasukonis, Lillicrap, Hafner, "Evaluating Long-Term Memory in 3D Mazes" (Memory Maze), https://arxiv.org/abs/2210.13383
24. Pleines et al., "Memory Gym: Towards Endless Tasks to Benchmark Memory Capabilities of Agents", https://arxiv.org/abs/2309.17207
25. Ni, Eysenbach, Salakhutdinov, "Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs", https://arxiv.org/abs/2110.05038
26. Lazaridou, Peysakhovich, Baroni, "Multi-Agent Cooperation and the Emergence of (Natural) Language", https://arxiv.org/abs/1612.07182
27. Mordatch & Abbeel, "Emergence of Grounded Compositional Language in Multi-Agent Populations", https://arxiv.org/abs/1703.04908
28. Kajić, Aygün, Precup, "Learning to cooperate: Emergent communication in multi-agent navigation", https://arxiv.org/abs/2004.01097
29. Eccles et al., "Biases for Emergent Communication in Multi-agent Reinforcement Learning", https://arxiv.org/abs/1912.05676 (the 28 % / 0 % no-bias emergence rates are from the project's own survey `research/emergent-communication.md`, which read the body; the abstract I opened states only that the biases improve performance)
30. Kottur et al., "Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog", https://arxiv.org/abs/1706.08502
31. Floreano, Mitri, Magnenat, Keller, "Evolutionary conditions for the emergence of communication in robots", Current Biology 2007, https://pubmed.ncbi.nlm.nih.gov/17320390/
32. Krakovna et al., "Specification gaming: the flip side of AI ingenuity", DeepMind blog 2020, https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/
33. Oudeyer & Kaplan, "What is intrinsic motivation? A typology of computational approaches", Frontiers in Neurorobotics 2007, https://www.frontiersin.org/articles/10.3389/neuro.12.006.2007/full
34. Lowe et al., "On the Pitfalls of Measuring Emergent Communication", https://arxiv.org/abs/1903.05168
35. Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3), https://arxiv.org/abs/2301.04104 (the "weak regularizer" remark is from the HTML text as quoted in `review-2026-10-26-water.md`; today I re-opened the abstract page only)

Not cited because I could not open them today: Bengio et al. 2009 "Curriculum learning" (PDF
unreadable; the Narvekar survey covers the idea), POPGym's model comparison (body not readable; Memory
Gym is cited instead).
