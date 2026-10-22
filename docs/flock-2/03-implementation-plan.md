# FLOCK 2.0 — staged implementation plan

Each milestone is watchable or testable on its own, has a defined success criterion, a compute
estimate, named risks and a fallback. Milestone 0 is the riskiest-assumption toy and is the
first build step after this plan is reviewed. Compute figures are **Estimates** unless linked.

The milestone order differs slightly from the brief (which listed sound, fox, ground marks). The
fox is placed before sound because the "sound specialises for urgency" hypothesis cannot be
tested without a time-critical event, and because a scripted fox is cheap and makes the meadow
watchable early. If you prefer the brief's order, M2 and M3 swap with no dependency problems
beyond that test moving to M4.

Design decisions referenced below are in [`02-design-questions.md`](02-design-questions.md);
evidence in [`01-research-survey.md`](01-research-survey.md) and [`research/`](research/).

---

## Overview

| # | Milestone | Headline question | Watchable output | Est. compute |
| --- | --- | --- | --- | --- |
| 0 | Toy drawing game | Do recognisable, meaningful drawings emerge? | Gallery of learned marks + ablation table | CPU, hours |
| 1 | Meadow, drives, world model | Does a drive-driven sheep with a world model look alive? | Single flock grazing/drinking/resting on a day/night cycle, with "expected view" overlay | CPU + ~10 GPU-h |
| 2 | Fox, fear, flocking | Does flocking/vigilance emerge from a confusable fox? | Nightly fox visits; tightening, flight, limps | ~10–20 GPU-h |
| 3 | Sound | Can a continuous bleat channel carry meaning and be listened to? | Bleats that move neighbours; per-channel CIC dial | CPU + ~5 GPU-h |
| 4 | Ground marks and channel specialisation | Do marks carry information; does sound take urgency? | Persistent glyphs on the meadow; the two-channel experiment | ~20 GPU-h |
| 5 | Relationships | Do reciprocity, grudges, honesty/deception emerge? | Trust graph; betrayals | ~10 GPU-h |
| 6 | Generations | Does the language structure and persist across cohorts? | Lamb cohorts learning from elders; dialect plots | ~20–40 GPU-h |
| 7 | Long run on the desktop | Does it run unattended for weeks with a viewer window? | The meadow running live on the desktop | GPU, weeks of wall-clock |
| 8 | Deferred | Installation hardening, documentary camera/curator, narration, frozen learned fox, migrants | — | — |

GPU-hours are on the desktop's RTX 4080 (16 GB), which is within ~20–30% of the RTX 4090 the
published throughput figures use (**Estimate**; the 4080 has fewer SMs and less bandwidth, and 16
GB vs 24 GB bounds batch size, not feasibility). No cloud resources are used. The whole plan's
offline compute is on the order of 100–200 GPU-hours, i.e. a few weeks of overnight runs on one
machine; developer time, not compute, is the constraint. The Milestone 6 evolution sweeps are the
only place the single-GPU budget could bite; see that milestone's fallback.

---

## Tooling decisions that apply throughout

Revised at plan review (see Q10): the desktop (13700KF + RTX 4080) is both the training machine
and the installation host, there is no cloud, and v0.4's Rust/Pi requirements are discarded.

- **One vectorised GPU simulation, used for training and for the installation.** The meadow is
  written once as a batched, pure-functional environment; offline it runs thousands of copies in
  parallel, live it runs one copy at a few Hz with the same code. There is no second
  implementation, no ONNX export and no Rust ML runtime.
- **Framework: JAX** (decided at plan review). Vmapped gridworld + DreamerV3 lineage; the
  differentiable rasteriser, auxiliary losses and metrics are all plain JAX.
- **Perception at 4 px/tile, 64×64 egocentric heading-aligned view, 16×16 canvas** from M0
  onward (Q8, Q9), so every milestone's result transfers. Rendering 64×64 views inside the
  vectorised sim is the throughput bottleneck to budget for; views are rendered from a palette
  index map by gather, not by a rasteriser.
- **Viewing is a simple desktop window**, not an installation. A viewer process (or an in-process
  window) draws the meadow from a state stream (positions, headings, sprites, marks, bubbles,
  bleat parameters, telemetry) and synthesises bleat audio live from the learned parameters. The
  state stream is kept as the boundary so that an installation renderer, a Pi kiosk or a
  documentary camera can be added later without re-architecting. Installation concerns
  (kiosk, secondary display, degradation ladder) are deferred.
- **Replay and checkpoints on disk.** 64×64 palette views are 2 KB per step; buffers of ~10⁶
  steps per flock in RAM, older data on disk.
- **Every milestone logs the same telemetry schema** from day one (per-sheep drives, surprise
  scalars, per-channel CIC, topographic similarity, trust tables, population), because the
  documentary camera in M7 is driven by these and retrofitting is painful.

---

## Milestone 0 — the toy drawing game

**Goal.** Test the riskiest assumption: that two small agents sharing a 16×16, 16-colour canvas,
with the receiver decoding the canvas through the *same* visual encoder it uses for the world,
learn drawings that are (a) causally used, (b) recognisable to a human, and (c) iconic rather
than arbitrary — and measure how much a shared encoder contributes versus an explicit
resemblance loss.

**Approach.**

- *World.* A ~24×24 tile grid rendered at 4 px/tile, 16-colour palette, with a handful of
  distinct object types (e.g. grass patch, pond, salt lick, fox, tree, bare ground) drawn as
  multi-pixel sprites (≈6×6 px for animals, textured tiles for ground). Both agents see
  egocentric, heading-aligned 64×64 views through one encoder — the same scale as the eventual
  meadow (Q8, Q9), so the results transfer. The sprite set is designed here and reused.
- *Game.* The sender is spawned where it can see the target (one of K referents, or a referent at
  one of D relative directions). The receiver starts elsewhere with the target out of view, sees
  the sender's 16×16 canvas rendered into its own view at native scale (as a bubble beside the
  sender, or as a stamp on the ground), and must navigate to the target within T steps. Both are
  rewarded on success. Distractor objects are present so that "which thing" and "which way" both
  matter.
- *Drawing head.* Per Q2: K = 6–8 line primitives with position, 16-colour softmax, on/off gate;
  distance-field rasteriser; ink cost; optional perceptual loss in the shared encoder's feature
  space. Receiver differentiable; sender trained through the rasteriser with straight-through
  colours, plus REINFORCE for the discrete gate if needed.
- *Auxiliary losses.* Eccles-style positive-signalling and positive-listening terms, because the
  base rate of non-emergence in decentralised multi-step games is high.
- *Ablations (the point of the milestone).*
  1. Shared encoder vs separate message encoder.
  2. With vs without the perceptual (resemblance) loss.
  3. Stroke head vs VQ-latent decoder vs pixel-categorical head (the last expected to fail; cheap
     to confirm).
  4. Canvas presented as a bubble (transient, beside the sender) vs a stamp (persistent, on the
     ground, read later) — the first persistent-vs-transient data point, since none is published.

**Success criteria.**

- *Used*: navigation success well above the no-canvas control, **and** a causal-influence test
  (scramble or blank the canvas at test time) changes the receiver's action distribution. A
  correlation alone does not count.
- *Readable*: you (or a few people) can match drawings to referents at better than 2× chance on
  a held-out set — the Mihai & Hare human-study format. Target: ≥ 35% at 10-way, which is roughly
  what the perceptual loss achieved there.
- *Iconic*: a **fresh encoder-only probe** (the perception encoder with no communication
  training) ranks the true referent patch above distractors from the drawing's features above
  chance; and shared-encoder feature similarity between drawing and referent exceeds that to
  distractors. Both must hold without the receiver being in the loop.
- *Ablation signal*: the table shows which of shared encoder / perceptual loss / action space
  moves readability and iconicity. The hypothesis "shared encoder makes iconicity emerge" is
  confirmed only if the shared-no-perceptual-loss cell is readable; the literature predicts it
  will not be.

**Compute.** Small CNN (~0.3–0.5M parameters at 64×64 input) and ~10⁴–10⁵ episodes per cell;
with the toy vectorised on the 4080, minutes per cell; the full ablation grid (≈24 cells × 3
seeds) in a few hours. This is also where the vectorised-sim scaffolding (batched env, 64×64
view rendering by gather, rasteriser) is first built and profiled.

**Risks and fallbacks.**

- Nothing emerges (0/N seeds): raise the auxiliary-loss weights, start with the centralised
  (fully differentiable) receiver, reduce K, shrink the referent set to 4.
- Emerges but unreadable private codes in *every* cell: this is the Mihai & Hare game-loss result;
  increase perceptual-loss weight; try iconic initialisation (sender initialised to copy its view
  of the referent patch).
- Readable but not causally used (positive signalling without listening): the Lowe pathology;
  separate the message head from the action head; add the listening loss.
- Sprites too small to be drawable at 16×16: enlarge the sprites (the canvas can hold one or two
  ≈6–8 px icons); the sprite set is a design variable here.

**What a pass buys.** The drawing head, the encoder-sharing decision, the sprite set and the
perceptual-loss weight all become fixed inputs to M4, and the ablation is a publishable result
because it does not exist in the literature.

---

## Milestone 1 — meadow, drives and world model

**Goal.** A single flock (20–40 sheep) on the 128×128 meadow with grazing patches, pond, salt
lick, treeline and day/night, driven by homeostatic drives and a shared RSSM world model, that
looks alive to a viewer and whose "expected view" and surprise are displayable.

**Approach.**

- *Drives* per Q-section 2 of the survey: HRRL drive-difference reward over hunger, thirst,
  warmth, social, interest (fear added in M2); continuous decay; non-substitutable norm;
  over-satiation penalties; warmth decays faster at night and away from cover and recovers only
  while resting; social set-point saturating at ~6–10 visible conspecifics with allelomimetic
  coupling to neighbour state changes.
- *Curiosity*: hashed counts over static-world features only, routed into the interest drive;
  blind to other sheep.
- *Perception*: 64×64 heading-aligned view at 4 px/tile with a rear blind wedge; night shrinks
  the lit radius; "head up" extends it (Q9).
- *World model*: shared RSSM at DreamerV3's native 64×64 input, sized between the repo's
  `size1m` and the paper's XS (≈2–8M parameters; measure), per-sheep recurrent state plus a trait
  embedding (the trait embedding is the first seed of the genome). Categorical decoder with
  cross-entropy over 16 colours. Pre-trained self-supervised on logged rollouts from
  scripted/random sheep (Ha & Schmidhuber's recipe), then actor-critic in imagination with short
  horizons.
- *Vectorised sim*: the full 128×128 meadow, batched across many instances on the 4080, grown
  from the M0 scaffolding. Profile agent-steps/s with view rendering here; this number sizes
  every later milestone.

**Success criteria.**

- Behavioural: a dawn/dusk grazing rhythm with rest at night emerges without a night penalty
  (the ethological target: ~2-hour rest/graze alternation at night); no sheep stands still > X%
  of the time; no corner-standing; no sheep-chasing.
- World model: decoded expected frame keeps single-pixel objects (no IRIS-style erasure);
  posterior‖prior KL spikes on genuinely novel events and not on routine neighbour motion.
- Displayable: a side-by-side of actual vs expected view that a viewer can read.
- Stability: a 24-hour continuous run with learning on shows no behavioural-metric regression
  (watchdog thresholds defined here and reused in M7).

**Compute.** World-model pretraining on ~10⁷ logged steps: hours on the 4080 (~5–10 GPU-h
including a hyperparameter sweep); drives and policy tuning mostly CPU.

**Risks and fallbacks.**

- Rest does not emerge from warmth/energy alone (nobody has shown it does): add a fatigue state
  in which actions are unavailable (Laurençon's mechanism), which is still drive-driven.
- Posterior collapse at ~1M parameters on small data: DreamerV3's free bits / KL balancing /
  unimix; more pretraining data (it is free).
- Drive gaming (parking on a patch, endless wandering): structural fixes first (norm exponent,
  over-satiation, interest drive cap), never a hand-tuned weight sum.
- View rendering dominates sim throughput (image-observation JAX envs run ~50× slower than
  symbolic): render by gather from a palette index map, reduce parallel meadows, or drop to
  3 px/tile (48×48) if the M1 profile demands it.

---

## Milestone 2 — the fox, fear and flocking

**Goal.** Add the scripted fox (Q4), injury/death (Q3), the danger buffer and fear model, and
show that flocking, vigilance and flight-to-cover emerge from survival pressure.

**Approach.** Parametric fox with isolation-weighted targeting, 1/N confusion, target inertia,
leaps, satiation (≤ 1 catch per visit), ≥ 120° vision, and a success-rate governor. Injury state
with visible limp; death on second catch while injured. A small refuge. Offline fox-dense
curriculum (visits every few minutes → nightly). Never-flushed danger buffer; fear model F(s)
shaping the value target with −λF(s′); fear as a thresholded top-priority drive. A "head up"
vigilance action that costs foraging and extends fox-detection range.

**Success criteria.**

- Flocking emerges: nearest-neighbour distance and swarm density versus a no-fox control and a
  no-confusion-rule control (the Olson ablation). Expected: tightening with confusion, none
  without.
- Vigilance falls with group size (the sheep ethology result); edge sheep are more vigilant.
- Flight to refuge then freezing; the fox takes isolated/edge sheep.
- Fox success rate held in the governor's band; population never hits the floor in a simulated
  month.
- One-shot-ish avoidance: after a single catch at a location, the flock's occupancy of that
  location at night drops for several nights and then recovers (fear extinction is untuned in
  the literature; tune by eye and record).

**Compute.** Fox-dense offline training with many parallel meadows: ~10–20 GPU-h, or days on the
desktop CPU with the PyO3/Python meadow.

**Risks and fallbacks.**

- Prey learn "turn away" and never flock: the capture rule is wrong; strengthen the confusion
  rule and isolation weighting before touching the learner.
- Fear dominates (sheep never graze at night): lower λ (the λε bound tells you the cost);
  hysteresis on the fear threshold.
- Population drifts to the floor: tighten satiation, widen the governor, enlarge the refuge
  slightly (not a lot — large refuges destabilise).

---

## Milestone 3 — sound

**Goal.** A learned bleat channel — a few continuous parameters (pitch contour, duration,
timbre), broadcast in a radius, no line of sight — that carries meaning and is causally listened
to.

**Approach.**

- Listeners receive the sender's *parameters* plus distance-scaled noise (DIAL-style); audio is
  rendered only for the audience. This keeps the channel differentiable and sidesteps the
  spectrogram-listener route, which needed DQN-class training and pretraining to work.
- Pre-train offline in a referential/recruitment game (10⁴–10⁵ interactions): "food here",
  "fox", "where are you". Positive-signalling and positive-listening losses; a repulsion/entropy
  term against continuous-channel collapse; a learned "whether to bleat" gate.
- Drop into the meadow from M2 and continue learning in-world.

**Success criteria.**

- Per-channel **causal influence**: muting or scrambling a bleat at test time changes neighbours'
  movement distribution; this metric is logged continuously from here on.
- Speaker consistency / context independence: bleat parameters cluster by context (fox present,
  food found, isolated) in a way that a 2-D plot shows.
- Collapse rate across seeds reported (expect some; keep the best seeds).
- Audible to a viewer as *different* bleats for different situations, even if not decodable.

**Compute.** Referential-game pretraining is minutes to hours on CPU; in-world fine-tuning ~5 GPU-h.

**Risks and fallbacks.**

- Collapse to one bleat: stronger repulsion bias; more channel noise (noise *increased*
  compositionality in the acoustic study).
- Signalling without listening: the listening loss; make listening pay (a sheep that heeds a fox
  bleat gets a free detection step).
- Nothing emerges in-world after emerging offline: freeze the pretrained channel longer; lower the
  in-world learning rate on it.

---

## Milestone 4 — ground marks and channel specialisation

**Goal.** Add persistent drawn marks (stamps) to the meadow using the M0 drawing head, and run
the two-channel experiment: does sound take urgency and drawing take information?

**Approach.**

- Stamps rendered into the 4×4-tile footprint (or whatever M0 found legible) with ForMIC-style
  exponential decay; the automatic close look at full resolution when adjacent; bubbles at 16×16.
- Curriculum: bubbles first (immediate feedback), then stamps; seed early training with
  hand-authored iconic marks (grass glyph near grass, fox glyph at the treeline) so readers learn
  to trust marks before writers learn to make them.
- Drawing is slow (the sheep stands still for several ticks) and costs ink; bleats are
  instantaneous and cheap. Rescale augmentation on the message path.
- Rotating referent set so a degenerate language costs reward.

**Success criteria.**

- Stamps are read: a sheep arriving at a mark later changes behaviour (CIC on the stamp channel,
  measured with a delay).
- Information vs urgency: per-channel mutual information with *event type* and with *event
  timing*. The hypothesis passes if bleat usage is concentrated on time-critical events (fox,
  separation) and stamp usage on static, spatial information (food location, water, danger
  zones), relative to a control where the costs are equalised. A clean null is a legitimate
  outcome and should be reported as such.
- Readability: marks on the meadow pass the M0 human-readability bar on a held-out sample; a
  perimetric-complexity trace shows simplification over time without loss of readability.
- Community vs dyad: marks seen by many sheep stay more iconic than bubbles exchanged between
  pairs (the Fay community prediction).

**Compute.** ~20 GPU-h for the two-channel experiment with seeds and the equal-cost control.

**Risks and fallbacks.**

- Marks are ignored (the stigmergy bootstrap problem): longer seeded-mark phase; slower decay
  early.
- Marks drift to unreadable codes in-world though readable in M0: raise the perceptual-loss
  weight in-world; the shared encoder may need to be frozen while the drawing head adapts.
- No channel specialisation: report the null; the installation still has both channels.

---

## Milestone 5 — relationships

**Goal.** Reciprocity, alliances, grudges, and honesty/deception emerging from repeated
interaction between identifiable individuals.

**Approach.**

- Fixed per-sheep ID: distinct wool pattern rendered into the observation plus an index channel;
  IDs retired on death with a cool-down.
- Per-partner social memory: two discounted scalars per living partner — niceness (partner's
  action-advantage for me, Eccles-style) and trust (signal-vs-outcome accuracy: did "food here"
  turn out true?) — fed to the policy alongside observations of that partner.
- Consequence channels: partner choice with real payoff (approach/avoid; sharing a patch,
  grooming/warmth huddles, challenging at the salt lick); heterogeneous social value orientation
  across the population so sharers, exploiters and liars coexist; a mostly-cooperative baseline
  with occasional scarcity so lies are tempting.
- Exploration kept high enough (the ε=0.01 freeze result).

**Success criteria.**

- Trust tables become non-uniform and *predictive*: a sheep's trust in a partner predicts whether
  it acts on that partner's marks.
- Reciprocity: a partner-choice graph that thickens into cliques; tit-for-tat-like conditional
  behaviour in the sharing task; relapse when reciprocators are removed (a known result that
  would be a strong sanity check).
- Deception, if it emerges: false marks measured by outcome; scepticism measured as a smaller
  behavioural response to marks from low-trust senders. If it does not emerge, report it.
- Watchable: the N×N trust matrix and partner graph as a documentary overlay.

**Compute.** ~10 GPU-h; much of this is CPU-cheap because the social tables are tabular.

**Risks and fallbacks.**

- Everyone cooperates, no grudges: increase scarcity/heterogeneity slightly.
- Everyone defects, no communication value: back toward cooperative; communication needs mutual
  benefit to emerge.
- Behavioural-only identification fails early in life ("stranger danger"): expected; the index
  channel makes the ID available regardless.

---

## Milestone 6 — generations

**Goal.** Lambs inherit a genome (priors) and learn language and traditions from elders through
a bottleneck; language structure emerges and persists across cohorts; dialects diverge in
isolation.

**Approach.**

- *Genome* (Baldwinian, mutated): trait embedding, drive weights and set-points, initial weights
  or a low-rank seed of the policy and communication heads, imitation budget, bleat timbre/pitch
  range. Never the parent's trained language weights.
- *Lamb life-cycle* (NIL/SIL hybrid): (1) a short supervised imitation phase on a *sample* of
  nearby elders' (context → bleat) and (referent → drawing) pairs, with recurrent memory and an
  elder-prediction auxiliary loss; (2) interaction in referential/coordination games with
  flockmates; (3) adulthood, when the sheep's outputs become what new lambs sample. Elders are
  frozen or very-low-LR during a lamb's imitation window.
- *Turnover*: fixed target population; births in cohorts of 3–5 coincident with deaths (old age,
  fox); elders occasionally out of a lamb's view (expert dropout).
- *Offline evolution*: an ES/GA outer loop (evosax or a seed-list GA) over genomes across many
  meadows, with a quality-diversity archive keyed on cheap behaviour descriptors (distance to
  centroid, graze/vigilance ratio, communication rate, fox-avoidance latency) as both gene bank
  and reseed source.
- *Anti-freeze*: rotating referents, slow prior mutation, occasional migrants from a separately
  evolved flock.

**Success criteria.**

- Topographic similarity (bleat-parameter space vs referent space) rises over the first ~5–10
  cohorts and holds; ease-of-teaching (a probe lamb's learning speed) improves.
- Language persists across a full population turnover without collapse to a few all-purpose
  signals (the expressivity-pressure check).
- Drawing perimetric complexity falls over cohorts while readability holds.
- Two isolated flocks become mutually less intelligible; a migrant causes a visible
  reconciliation or split.
- Baldwin effect in the outer loop: later-generation lambs reach competence faster than
  first-generation ones.

**Compute.** Evolution across many meadows: ~20–40 GPU-h for a few hundred generations at
population 256–1,000 (the Hamon anchor is 1,000 agents × 1M steps in ~20 min on one GPU); the
in-world cohort mechanism is CPU-cheap. This is the one milestone where a single 4080 could be
limiting; if sweeps take more than a few overnight runs, shrink the genome (ES is unreliable past
~100k parameters anyway), reduce population, or run the outer loop on CPU cores in parallel with
GPU work.

**Risks and fallbacks.**

- Language collapses at each cohort (the ~20% success crash at full reset): smaller cohorts; a
  longer elder-frozen window; never reset all at once.
- Too-gradual turnover loses the structuring pressure (Li & Bowling's staggered-N=10 result):
  cohorts, not trickle.
- Imitation budget wrong (the most sensitive knob in Ren et al.): sweep it offline; it is a genome
  parameter, so evolution can also tune it.
- ES unreliable if the evolved portion grows past ~100k parameters: keep the genome small (traits,
  drive weights, low-rank seeds).

---

## Milestone 7 — long run on the desktop

**Goal.** The M6 system running live on the desktop for weeks: one meadow at a wall-clock tick,
a viewer window with audio, persistence with rollback, and a soak test of the learning population.

**Approach.**

- *Process shape.* One Python/JAX process (`flock-live`) runs the single live meadow, batched
  inference for all sheep, the social tables, the lamb imitation step, optional online
  fine-tuning and telemetry, with periodic checkpoints. It is the same code path as training with
  one meadow and a wall-clock tick.
- *Viewer.* A window drawing the meadow from the state stream, with overlays for the internal
  scalars (expected-view ghost for a selected sheep, surprise, trust graph, language metrics) and
  live bleat synthesis from the learned parameters. Whole meadow, no cutting; the documentary
  camera is deferred and plugs into the same stream later.
- *Persistence.* Snapshot the whole state (weights, recurrent states, social tables, danger
  buffer, genome archive, marks on the ground, population) on a schedule, with checkpoint
  rollback driven by the behavioural watchdog from M1.
- *Soak.* A two- to four-week unattended run with learning on, watching the watchdog,
  plasticity remedies (weight decay + LayerNorm, dormant-unit reinitialisation, periodic head
  resets with buffer retained) and rollback. This is the only place months-long behaviour of a
  learning population can be measured; nobody has published it. The desktop is also the training
  machine, so the soak coexists with further offline runs.

**Success criteria.**

- Two weeks unattended with no watchdog rollback more than once a week, population within
  bounds, per-channel CIC and topographic similarity not degrading, fox success in band.
- Process restart from checkpoint loses less than one in-game hour of state.
- The viewer is watchable for an hour without the viewer feeling they have seen everything
  (informal, but it is the point).

**Compute.** The desktop GPU, shared with any remaining offline work; wall-clock weeks.

**Risks and fallbacks.**

- Online learning degrades behaviour over weeks: turn the online train ratio down or off; the
  system is designed to work frozen.
- GPU contention between the live process and offline runs: pin the live process to a fixed
  memory budget; the live tick needs a tiny fraction of the GPU.
- Python process stability over weeks: supervisor + checkpoints; the viewer is separable so a
  brain restart is a brief freeze, not a blank screen.

---

## Milestone 8 — deferred

Not planned now; all are additive on top of the state stream and telemetry, none needs
re-architecting.

- **Installation hardening**: kiosk/TV display path (desktop-direct or Pi), secondary display,
  graceful-degradation ladder, supervisor.
- **Documentary camera and curator** (Q6): telemetry-triggered cuts, a Motif-style distilled
  interest scorer over ground-truth captions, caption cards.
- **Grounded narration** via a local small LLM, fed captions that only assert what the telemetry
  supports.
- **Frozen learned fox**: trained offline against a league of sheep snapshots, then frozen and
  governed like the scripted one.
- **Migrants and seasons** as scheduled non-stationarity.
- **A "lab" overlay** showing the live metrics plainly.

---

## Decisions taken at plan review

1. **Hardware.** The desktop (13700KF + RTX 4080) is both the training machine and the
   installation host; all training is local, no cloud; use the GPU as much as possible.
2. **v0.4 is not a constraint.** Its crate structure, Rust renderer, persistence and camera
   designs are discarded as requirements; 2.0's architecture is decided on its own merits (Q10).
3. **Milestone order.** Fox before sound, as planned.
4. **Perception scale.** 4 px/tile, 64×64 heading-aligned view, 16×16 canvas, multi-pixel
   sprites (Q8, Q9), from M0 onward.
5. **M0 referent set.** K ≈ 6 object types × 4 directions; no specific objects requested.
6. **Human-readability test.** At least two people (you plus one other).

7. **Framework.** JAX.
8. **Scope.** Not an installation for now: it runs on the desktop with a viewer window. Kiosk,
   Pi, secondary display and degradation ladder are deferred (M8).
9. **Documentary camera** deferred (M8); the telemetry schema and state stream it needs are built
   from M0 so it can be added without re-architecting.
10. **Sound.** Live synthesis of bleats from the learned parameters.

## Still open (none blocks M0)

- **Persistence cadence** for M7: acceptable loss on a restart, and whether the live run should
  resume mid-night with the fox in the field.
- **Sprite set and palette**: designed during M0; you may want a say in the look before M1.
