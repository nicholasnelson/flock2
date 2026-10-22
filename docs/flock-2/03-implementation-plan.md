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
| 7 | Integration and soak | Does it run for weeks on the installation hardware? | The installation | CPU, weeks of wall-clock |
| 8 | Optional extras | Narration; frozen learned fox; migrants | — | small |

GPU-hours are on the desktop's RTX 4080 (16 GB), which is within ~20–30% of the RTX 4090 the
published throughput figures use (**Estimate**; the 4080 has fewer SMs and less bandwidth, and 16
GB vs 24 GB bounds batch size, not feasibility). No cloud resources are used. The whole plan's
offline compute is on the order of 100–200 GPU-hours, i.e. a few weeks of overnight runs on one
machine; developer time, not compute, is the constraint. The Milestone 6 evolution sweeps are the
only place the single-GPU budget could bite; see that milestone's fallback.

---

## Tooling decisions that apply throughout

- **M0–M3 are built in Python/PyTorch** with a small purpose-built gridworld, *not* in the Rust
  workspace. The differentiable rasteriser, the auxiliary losses and the metrics are all easier
  to iterate in Python, and nothing in these milestones needs the renderer.
- **The vectorised meadow question is deferred to the end of M1.** Two routes exist: (A) a JAX
  port of the meadow (10⁵–10⁶ agent-steps/s on the 4080; a second implementation to keep in
  sync), or (B) the Rust `flock-sim` exposed via PyO3 with batched stepping on the CPU (one
  implementation; ≈10⁴–10⁵ agent-steps/s on the 13700KF) with the networks on the 4080. With a
  local GPU, route B is viable for every milestone except possibly M6's population-scale
  evolution; the M1 compute profile and the answers about `flock-sim` (below) decide.
- **Installation inference is Rust via `ort` (ONNX)**, with the recurrent cell exported as a
  single-step function with hidden state as explicit I/O. If online learning in Rust is wanted,
  `burn` with safetensors. A half-day benchmark spike precedes M7 (export a ~2M-param cell; time a
  batch-60 forward and a 512-sample train step on both i7s, and on the 4080 via `ort`'s CUDA
  provider if the desktop hosts the installation).
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

- *World.* A 12×12 tile grid, 16-colour palette, with a handful of distinct object types (e.g.
  grass patch, pond, salt lick, fox, tree, bare ground) drawn as small 2×2 or 3×3 sprites. Both
  agents see egocentric 16×16 views through one encoder.
- *Game.* The sender is spawned where it can see the target (one of K referents, or a referent at
  one of D relative directions). The receiver starts elsewhere with the target out of view, sees
  the sender's canvas (as a 16×16 frame through the shared encoder), and must navigate to the
  target within T steps. Both are rewarded on success. Distractor objects are present so that
  "which thing" and "which way" both matter.
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
  4. Receiver sees the canvas at full 16×16 vs downsampled to 8×8 and 4×4 (the Q1 footprint
     question), with and without rescale augmentation during training.

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

**Compute.** Small CNN (~100k parameters) and ~10⁴–10⁵ episodes per cell; minutes to an hour per
cell on the 13700KF; the full ablation grid (≈24 cells × 3 seeds) in a day or two of CPU, or an
hour or two on the 4080. No GPU required, but it is there.

**Risks and fallbacks.**

- Nothing emerges (0/N seeds): raise the auxiliary-loss weights, start with the centralised
  (fully differentiable) receiver, reduce K, shrink the referent set to 4.
- Emerges but unreadable private codes in *every* cell: this is the Mihai & Hare game-loss result;
  increase perceptual-loss weight; try iconic initialisation (sender initialised to copy its view
  of the referent patch).
- Readable but not causally used (positive signalling without listening): the Lowe pathology;
  separate the message head from the action head; add the listening loss.
- Downsampled views unreadable: accept 8×8 footprints instead of 4×4 and record the number.

**What a pass buys.** The drawing head, the encoder-sharing decision, the footprint size and the
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
- *World model*: shared RSSM, ≈1–2M parameters, per-sheep recurrent state plus a trait embedding
  (the trait embedding is the first seed of the genome). Categorical decoder with cross-entropy
  over 16 colours. Pre-trained self-supervised on logged rollouts from scripted/random sheep
  (Ha & Schmidhuber's recipe), then actor-critic in imagination with short horizons.
- *Vectorised sim*: the small Python meadow, batched across many instances. Profile it here.

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
- Framework overhead dominates at this model size: measured here; informs the JAX-vs-PyO3
  decision.

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

## Milestone 7 — integration and soak

**Goal.** Move the trained system into the FLOCK Rust workspace and renderer, add persistence,
the documentary camera with an FM curator, graceful degradation, and prove it runs for weeks on
Scarif or the desktop with the Pi rendering.

**Approach.**

- *Crates (provisional — v0.4 code not seen).* `flock-types` and `flock-sim` carry over in role;
  `flock-sim` gains the fox, injury, marks, drives and population rails if they are not already
  there. `flock-brain` is rewritten: it loads ONNX/safetensors models and runs batched inference
  per tick via `ort` (or `burn` if online learning stays in Rust), holds per-sheep recurrent
  state and the tabular social memory, runs the lamb imitation step, and emits telemetry.
  `flock-render` carries over, plus the new overlays (expected-view ghost, surprise, trust graph,
  caption cards). The ESP32 secondary display, if kept, is a natural home for the trust matrix
  or the language-metric plots.
- *Persistence.* Snapshot the whole state (weights, recurrent states, social tables, danger
  buffer, genome archive, marks on the ground, population) on a schedule, with checkpoint
  rollback driven by the behavioural watchdog from M1.
- *Documentary camera.* Candidate shots scored by a Motif-style distilled interest model over
  ground-truth captions, with event triggers on the telemetry (surprise spikes, cohort birth,
  trust drop, dialect split, fox entry); an optional OMNI-style periodic LLM judge. ASAL-style
  CLIP novelty as one feature among several, not the driver.
- *Graceful degradation.* The installation must keep running with learning off, with the curator
  off, with the fox off, and with any single channel disabled; each is a flag.
- *Soak.* A two- to four-week unattended run on the target hardware with learning on, watching
  the watchdog, plasticity remedies (weight decay + LayerNorm, dormant-unit reinitialisation,
  periodic head resets with buffer retained) and rollback. This is the only place months-long
  behaviour of a learning population can be measured; nobody has published it.

**Success criteria.**

- Benchmark spike passed: batch-60 forward + a train step within budget on the chosen host.
- Two weeks unattended with no watchdog rollback more than once a week, population within
  bounds, per-channel CIC and topographic similarity not degrading, fox success in band.
- Camera picks shots a viewer agrees are interesting (a small informal study is enough).
- Every degradation flag exercised.

**Compute.** CPU on the installation host; wall-clock weeks.

**Risks and fallbacks.**

- Framework overhead at ~1M parameters on CPU is worse than estimated: batch everything into one
  call per tick; drop the decoder from the live path and compute the expected view only for the
  sheep the camera is watching.
- Online learning degrades behaviour over weeks: turn the online train ratio down or off; the
  system is designed to work frozen.
- The GRU ONNX export pitfalls: single-step cell export with explicit hidden state.

---

## Milestone 8 — optional extras

- **Grounded narration** via the local small LLM (Q6), fed captions that only assert what the
  telemetry supports.
- **Frozen learned fox**: trained offline against a league of sheep snapshots, then frozen and
  governed like the scripted one.
- **Migrants and seasons** as scheduled non-stationarity.
- **A "lab" overlay** showing the live metrics plainly, for the viewer who wants the science.

---

## What carries over from v0.4 (provisional)

The v0.4 design document and code were not available; this is based only on the crate names and
the brief, and is flagged for your review.

| v0.4 part | 2.0 status (provisional) |
| --- | --- |
| Cargo workspace layout (`flock-types`, `flock-sim`, `flock-brain`, `flock-render`) | Keep the layout. `flock-brain` changes from an LLM client to a model runtime; the others extend. |
| Rendering: Rust → WASM in a Chromium kiosk on the Pi 5 | Carries over unchanged in principle; gains overlays. |
| ESP32 secondary display | Carries over; a good home for the trust matrix / language metrics. |
| Persistence | Carries over in role; the state to persist grows (weights, recurrent states, social tables, danger buffer, marks, genome archive). |
| Documentary camera | Carries over in role; shot selection changes to telemetry triggers + distilled interest scorer. |
| Graceful degradation | Carries over as a principle; the degradation ladder is now learning → curator → fox → channels. |
| The sim's world rules (grazing, pond, salt lick, treeline, day/night) | Carry over; add fox, injury, marks, refuge, population rails, drives. |
| LLM per sheep (Gemma 4 E2B via llama-server) | Removed from the agent loop. The server may be reused for optional narration. |

---

## Decisions taken at plan review

1. **Hardware.** The desktop has an RTX 4080; all training is local, no cloud. Offline work runs
   on the desktop; the installation runs on Scarif or the desktop with the Pi rendering.
2. **v0.4 code** is not available to this planning environment; questions about it are answered
   on request. Specific questions still open are listed below.
3. **Milestone order.** Fox before sound, as planned.
4. **M0 referent set.** K ≈ 6 object types × 4 directions; no specific objects requested.
5. **Human-readability test.** At least two people (you plus one other).

## Still open (needed by the end of Milestone 1, not before Milestone 0)

- `flock-sim` v0.4: approximate cost per agent-step; whether stepping is deterministic given a
  seed; whether the world state can be stepped in batch (many meadows) from one process; how
  observations are produced (is the 16×16 egocentric view already computed in the sim or in the
  renderer?).
- Which host runs the installation brain — Scarif (i7-7820X) or the desktop (13700KF + 4080)? If
  the desktop, online fine-tuning becomes much cheaper and the soak-test design changes.
