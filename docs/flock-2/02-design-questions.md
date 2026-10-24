# FLOCK 2.0 — the seven open design questions

Each question gets a recommendation, the reasoning, and the evidence. Labelling follows the
survey: **Established** (linked result), **Estimate** (derived arithmetic), **Judgement** (my
reading, or speculation). Where a recommendation is a bet rather than a conclusion, it says so,
and the plan in [`03-implementation-plan.md`](03-implementation-plan.md) names the milestone that
tests it.

Full citations: [`research/`](research/). Synthesis: [`01-research-survey.md`](01-research-survey.md).

---

## Q1. Mark scale — a 16×16 ground mark at 1 px/tile fills the whole view

> **Superseded at plan review by Q8.** The recommendation below assumed 1 px per tile. At 4 px per
> tile (Q8) a 16×16 canvas stamped at native scale covers a 4×4-tile footprint and is seen at its
> own resolution, so neither the downsampling nor the "close look" is needed. The capacity
> reasoning (≈5–8 usable bits) still holds. Kept for the record.

### Recommendation

Render ground marks into a **4×4-tile footprint by default** (so a mark occupies a 4×4 patch of
the receiver's 16×16 view), and give every sheep an automatic "close look": when a sheep is on or
adjacent to a mark, the mark is additionally presented at full 16×16 resolution as a second
frame through the *same* encoder. Speech bubbles stay at full 16×16, presented the same way.
Train the message path with random rescale augmentation. Do **not** add a discrete "look
closely" action in the first version. Budget for ≈5–8 usable bits per mark.

### Reasoning

A mark that fills the whole view breaks two things at once: the receiver cannot see the mark
*and* its context (which direction is "food here" pointing?), and the sender's cheapest iconic
solution becomes "copy my entire view", which is not a drawing of a thing. A reduced footprint
restores context and forces selection — the Qiu/Hawkins "draw the diagnostic part" regime.

Two scales are needed because the two functions of a ground mark differ. At 4×4 the mark is a
*glyph you notice* while grazing; at 16×16 it is a *glyph you read* once you have walked to it.
Making the close look automatic rather than an action avoids adding an exploration problem
(a learned "look" action has to be discovered before it pays, and the base rate for discovering
communicative actions is poor — see Q7 and the Eccles numbers). The cost of the automatic look
is one extra encoder pass for sheep standing near marks, which is negligible. *(Judgement.)*

The capacity budget follows from the literature's anchors rather than from the canvas: 32-bit
binary messages already saturate in-domain referential accuracy, 11 bits suffice for 128-way
discrimination with a strong encoder, and sketch-game success was flat from 5 to 50 lines.
*(Estimate from Established anchors.)* A 4×4 footprint at 16 colours is 64 raw bits, so even the
downsampled glyph carries more than the usable channel. Design the referent set small — a
handful of concepts × a handful of directions or identities — so that iconicity, not compression,
is the binding constraint. The surplus canvas is what makes resemblance affordable; icons are
less bit-efficient than codes and that is the point.

Legibility of learned marks under downsampling is unstudied, so it is a measured outcome of
Milestone 0, not an assumption. The mitigation that the literature does support is rescale
augmentation on the message path, which is also what stops from-scratch encoders latching onto
low-level pixel statistics.

### Evidence

- Success flat across 5–50 lines; capacity bottleneck was the latent, not the canvas —
  [Mihai & Hare 2021](https://arxiv.org/abs/2106.02067). **Established.**
- Simplification keeps diagnostic parts; one-step drawing fails; agents revert to iconic for
  unfamiliar referents — [Qiu et al. 2022](https://arxiv.org/abs/2111.14210). **Established.**
- 32-bit binary messages saturate in-domain accuracy — [Evtimova et al. 2018](https://arxiv.org/abs/1705.10369);
  11 bits suffice for 128-way with augmentation — [Dessì et al. 2021](https://arxiv.org/abs/2106.04258).
  **Established.**
- From-scratch CNNs exploit low-level statistics unless augmented — [Dessì et al. 2021](https://arxiv.org/abs/2106.04258),
  [Lazaridou et al. 2018](https://arxiv.org/abs/1804.03984). **Established.**
- Capacity and downsampled-legibility arithmetic: [`research/graphical-communication.md` KQ7](research/graphical-communication.md). **Estimate.**
- No published work on multi-scale perception of learned marks. **Gap.**

---

## Q2. Action space for drawing — pixels, strokes/stamps, or a latent decoder?

### Recommendation

**A differentiable stroke/stamp rasteriser.** The drawing head emits K primitives (start with
K = 6–8 line segments, optionally a small library of stamp shapes), each with continuous
position parameters, a softmax over the 16-colour palette, and an on/off gate; a distance-field
rasteriser renders them onto the 16×16 canvas; gradients flow from the receiver's loss through
the rasteriser to the sender. Add a per-primitive **ink cost** and a small **perceptual loss**
between the mark and the referent patch in the shared encoder's feature space. One extra learned
bit decides *said* (bubble) vs *drawn* (stamp).

Fallback, if the stroke head does not train: a quantised latent (VQ / straight-through) → small
decoder → 16×16×16 logits, *with* the perceptual loss. Do not start with a pixel-level
categorical head trained by REINFORCE.

### Reasoning

Only two families have published drawing agents that train: differentiable rasterisers
(Mihai & Hare's distance-field lines; Qiu's neural renderer) and RL against a non-differentiable
painter (SPIRAL, the Signification Game with PPO), and the second is consistently reported as
slower. Nobody has trained a pixel-categorical head for communication; 256 cells × 16 colours is
a 1,024-bit joint discrete action, and REINFORCE over it has no precedent and every reason to be
a variance disaster. *(Judgement, from Established.)*

Strokes also carry an inductive bias the project wants for free: an "ink" prior. Every result in
which drawings simplified while staying readable (Qiu's rabbit → ear; human Pictionary) operated
on a stroke substrate with a cost on strokes or time. A 16×16 canvas with 16 colours makes the
analytic distance-field rasteriser trivial to implement and cheap to run.

The latent-decoder route is the easiest to optimise but is exactly the case Carmeli et al.
identify as "trivial": with a shared encoder the sender just transmits its embedding and nothing
iconic emerges. It is kept as a fallback *only* with the perceptual loss attached.

The perceptual loss is not a nicety. It is the one ingredient that separated human-unreadable
from human-readable drawings in Mihai & Hare (humans 8.3% → 38%) at essentially no cost to agent
success. It is also the term that makes the shared-encoder hypothesis testable rather than
assumed: Milestone 0 ablates it.

### Evidence

- Distance-field rasteriser and results — [Mihai & Hare 2021](https://arxiv.org/abs/2106.02067),
  [Differentiable Drawing and Sketching](https://arxiv.org/abs/2103.16194). **Established.**
- Neural renderer with policy gradients; REINFORCE baseline slower and high-variance —
  [Qiu et al. 2022](https://arxiv.org/abs/2111.14210). **Established.**
- Gumbel-Softmax / straight-through converge faster than REINFORCE; quantised channels train
  near-continuous — [Havrylov & Titov 2017](https://arxiv.org/abs/1705.11192),
  [Carmeli et al.](https://arxiv.org/abs/2211.02412). **Established.**
- Shared encoder + high-capacity channel is trivially non-iconic — [Carmeli et al. App. A.2](https://arxiv.org/abs/2211.02412). **Established.**
- No pixel-categorical communication head in the literature. **Gap.**

---

## Q3. Death — what should death or injury mean, and how to avoid collapse?

### Recommendation

**Injury first, death rare, population railed.**

- A fox catch produces *injury*: a large immediate drive signal (fear/pain), then a temporary
  state lasting a few in-game days — slower, worse foraging, a visible limp. Injury is the
  default outcome of a catch.
- *Death* occurs on a second catch while injured, and from old age (which generational turnover
  needs anyway). Dead sheep leave the simulation; their last pre-attack window stays in the
  shared danger buffer so survivors learn from it.
- *Population*: births track deaths (birth rate = f(target − current)), lambs born in small
  cohorts from healthy ewes; a hard floor (e.g. below 60% of target, the fox returns
  empty-handed) and a hard ceiling (fertility → 0). Fox satiation: at most one catch per visit.
  A small refuge (barn/hedge tiles the fox will not enter) holding a *constant, small* number of
  sheep.

### Reasoning

v0.4 had death off; 2.0 needs stakes, turnover and a predator, so death has to exist, but the
learning system needs a dense signal and the exhibition needs a flock that is still there in
month three. Injury gives roughly 5–10× more negative samples than death alone *(Estimate)*,
keeps the agent in the batch (removal creates value-estimation and posthumous-credit problems
that the MARL literature documents), and reads clearly to a viewer. There is existence proof that
a non-lethal contact penalty alone suffices for flocking to evolve.

Population stability should be a *tunable*, not an emergent property. The field's de facto
standard for stable long runs with learners is a constant population via respawn; the one
reset-free eco-evolutionary system with free-floating reproduction lost 3 of 5 seeds to collapse
despite tuning, and finite stochastic predator-prey systems hit extinction eventually. Because the
fox is a single external agent with no population of its own, the Lotka-Volterra loop is
one-sided — only the sheep count can crash — so a satiated fox with a governor plus births that
track deaths is sufficient. The refuge is both an ecological stabiliser and the documented sheep
behaviour (flight to cover, freezing at refuge); it must be small, or the fox becomes useless.

Old-age death matters for a different reason: iterated learning needs turnover (§Q5 and the
generations milestone). A flock where nothing dies of age has no bottleneck and its language
will not structure.

### Evidence

- Non-lethal "bleeding" penalty sufficient for flocking — [Li et al. 2023](https://arxiv.org/abs/2308.12624). **Established.**
- Absorbing-state and dead-agent problems — [Cohen et al.](https://arxiv.org/abs/2111.05992),
  [MAPPO](https://arxiv.org/abs/2103.01955), [Health-Informed PG](https://arxiv.org/abs/1908.01022). **Established.**
- Constant population by respawn — [SELFish](https://arxiv.org/abs/1905.04077),
  [Neural MMO](https://arxiv.org/abs/2001.12004). **Established.**
- 3 of 5 seeds collapsed under free-floating reproduction — [Hamon et al. 2023](https://arxiv.org/abs/2302.09334). **Established.**
- Refuges protecting a constant small number stabilise; large refuges destabilise —
  [Krivan 1998](https://fix.prf.jcu.cz/~krivan/clanky/Krivan-TPB-1998.pdf). **Established.**
- Type II functional response (handling time/satiation) — [Olson et al. 2013](https://arxiv.org/abs/1209.3330). **Established.**
- No injury-vs-death comparison exists. **Gap.**

---

## Q4. Fox learning — scripted vs learned, and getting signal from a rare event

### Recommendation

**Ship a scripted, parametric fox. Never co-learn the fox live.** Optionally, later, pre-train a
fox offline against a league of sheep snapshots and *freeze* it. Get the sheep's fox competence
from offline volume, not from nightly encounters.

Fox design (the capture rule is the main lever for lifelike flocking):

- Enters from an edge tile at night; targets the nearest sheep **weighted by isolation**
  (marginal predation).
- **Confusion rule** P(capture) = 1/N_nearby (sheep within radius r of the target); target
  inertia; occasional "leap" bursts; vision angle ≥ 120°.
- **Satiation**: ≤ 1 catch per visit, then leaves.
- 4–6 exposed parameters (visit probability, speed ratio, confusion radius, persistence, vision)
  auto-tuned slowly to a target success rate of ~25–40% of visits.

Sheep-side learning signal:

- **Offline**: train in a vectorised copy of the meadow with fox visits every few minutes rather
  than nightly; curriculum from a slow, always-visible fox to a fast nightly one.
- **Permanent danger buffer + fear model**: store the ~20–50 ticks before every fox contact in a
  buffer that is never flushed; train a fear predictor F(s) on balanced danger/safe batches; add
  −λF(s′) to the value/drive target. The bound λε caps how much grazing it costs.
- **Live**: freeze or heavily down-weight the fox-response head; keep foraging, social and
  communication learning live.

### Reasoning

Every clean prey-flocking result in the literature used a scripted or frozen predator at least
as scaffolding. Live co-learning arms races (hide-and-seek, AlphaStar) needed league machinery
and large compute and still drifted; a learned predator learned restraint only with careful
curriculum. For a months-long exhibition a drifting fox is a liability in both directions — a
massacre or a toothless visitor — and nobody is on hand to retune it at 3 a.m. A governor on a
parametric fox replaces the arms race with a controlled one. *(Judgement.)*

On signal: one fox visit per in-game night is ≈30 encounters per month per flock *(Estimate)*,
and nothing in the RL literature learns robust avoidance from that; the emergent anti-predator
behaviours in the literature came from 1,200 generations × 180 replicates (Olson) or 500k
training steps (SELFish). The danger buffer is the mechanism for "replaying fox encounters": it
is the one published method for one-shot-ish avoidance of rare catastrophes with a bound on its
cost. World-model imagination can rehearse fox encounters only once the model has real fox
trajectories; safe-RL imagination results are all on frequent costs, so expect hallucination
rather than extrapolation from a handful of samples. Seed the live world model with the
offline-trained one so the fox is "known" from day one.

### Evidence

- Confusion rule necessary for swarming (density 0.69 → 12.48; cohesive swarms in ~70% vs 0%) —
  [Olson et al. 2013](https://arxiv.org/abs/1209.3330). **Established.**
- Scripted predator with distraction and leaps; flocking with no flocking reward —
  [SELFish](https://arxiv.org/abs/1905.04077). **Established.**
- Fixed-rule predator used to speed prey learning — [Li et al. 2023](https://arxiv.org/abs/2308.12624). **Established.**
- Self-play cycling; league exploiters degrade late — [self-play survey](https://arxiv.org/abs/2408.01072). **Established.**
- Learned predator learned restraint only with curriculum — [Ritz, Hahn et al. 2020](https://mobile.ifi.lmu.de/team/carsten-hahn). **Established** (abstract level).
- Intrinsic Fear: never-flushed danger buffer, fear model, λε bound —
  [Lipton et al.](https://arxiv.org/abs/1611.01211). **Established.**
- Edge/spaced animals are the ones taken — [selfish herd field data](https://pmc.ncbi.nlm.nih.gov/articles/PMC1634896). **Established.**
- No controlled scripted-vs-learned predator comparison. **Gap.**

---

## Q5. Offline vs online — what is pre-trained or evolved, what keeps learning live?

### Recommendation

| Component | Offline (desktop RTX 4080, once per release) | Online (installation, continuous) |
| --- | --- | --- |
| Perception encoder + RSSM world model | Self-supervised on logged sim video; then imagination training | Frozen, or fine-tuned at train ratio ≈1–2 with plasticity remedies and rollback |
| Policy (actor-critic heads) | Pre-trained in imagination / PPO under fox-dense curriculum | Slow per-lineage adaptation; fox-response head frozen |
| Drives, traits, priors (the genome) | Evolved (ES/GA outer loop over drive weights, trait vector, initial heads, imitation budget) | Inherited with mutation at birth; not learned |
| Bleat and drawing heads | Pre-trained in offline referential games with auxiliary signalling/listening losses | Continue learning in-world; cultural transmission at birth |
| Fear model + danger buffer | Trained on offline encounters | Buffer keeps growing; model updated rarely |
| Per-sheep recurrent state | — | Always live (it *is* the individual's memory) |
| Per-partner social memory (niceness, trust) | — | Always live; tabular, cheap, no plasticity issue |
| Lamb imitation phase | — | Live; a short supervised budget per lamb |
| Language-structure metrics, CIC tests | Validated offline | Logged live for the documentary layer |

Feasibility on the stated hardware: inference for 60 sheep at 5 Hz is < 5% of either i7; a few
SGD steps/s on the shared model is realistic on either; the Pi stays a renderer. *(Estimate.)*

### Reasoning

The published sample complexities are the argument. Dreamer-class world models use train ratios
of 32–512; referential games take 10⁴–10⁵ interactions and many seeds fail outright in
decentralised settings; anti-predator behaviour came from thousands of encounters; iterated
learning needs tens of generations. The installation delivers ≈500 environment steps/s and ≈2
updates/s for one shared model — a train ratio of ≈2. Online-from-scratch would be an order of
magnitude slower than anything that has worked in the literature, and the audience would watch
generation 0 for weeks. So the exhibition should open with a competent, already-speaking
generation N produced offline, and the live run's job is to *maintain and drift*, not to learn
from nothing.

What stays live is chosen by two criteria: it must be *individual* (recurrent state, social
tables, the lamb's imitation of its particular elders) or it must be *the thing the audience is
watching change* (language drift, relationships, generations). Everything else is a liability
over months: continual deep learning loses plasticity, non-episodic PPO degrades after ~1M of
10M steps in the one long-run study, and a fox-response head that keeps learning can unlearn.
The social tables are tabular and the lamb imitation phase is a bounded supervised step, so
neither inherits the deep-continual-learning pathology.

The split also matches the three inheritance channels (genome / lifetime learning / culture): the
genome is only ever changed offline and by mutation; the language is only ever changed by
in-world use and transmission.

### Evidence

- Train ratios and GPU-days — [DreamerV3](https://arxiv.org/abs/2301.04104). **Established.**
  Online train ratio ≈2 on this hardware — [`research/world-models.md` KQ7](research/world-models.md). **Estimate.**
- Referential-game sample complexity and failure base rates — [Li & Bowling 2019](https://arxiv.org/abs/1906.02403),
  [Chaabouni et al. 2022](https://arxiv.org/abs/2204.12982), [Eccles et al. 2019](https://arxiv.org/abs/1912.05676). **Established.**
- Pretraining the acoustic channel raised eval accuracy 0.789 → 0.958 — [Eloff et al.](https://arxiv.org/abs/2111.02827). **Established.**
- Loss of plasticity and remedies — [Dohare et al. 2024](https://www.nature.com/articles/s41586-024-07711-7),
  [Nikishin et al. 2022](https://arxiv.org/abs/2205.07802). **Established.**
- Non-episodic PPO degrades — [Jelly Bean World](https://arxiv.org/abs/2002.06306). **Established.**
- Evolve priors, learn in life — [Fernando et al. 2018](https://arxiv.org/abs/1806.07917),
  [Singh, Lewis, Barto & Sorg 2010](https://ieeexplore.ieee.org/document/5471106). **Established.**
- CPU feasibility — [`research/tooling-and-hardware.md` KQ5](research/tooling-and-hardware.md). **Estimate.**

---

## Q6. The LLM's role — none, narration, or something else?

### Recommendation

**Not a brain, and not in the loop of any agent's behaviour or reward.** Two legitimate roles,
both on the *observer* side:

1. **Camera curator (recommended).** Generate structured captions from the simulator's ground
   truth ("ewe 12 left the flock at dusk with two lambs; fox 30 tiles east; her trust in ram 4
   fell after a false food mark"), have an LLM rank caption pairs for interestingness *once*,
   offline (Motif-style), distil to a tiny scorer, and run that scorer live at zero marginal cost
   to pick shots. Optionally an OMNI-style periodic judge ("given the last hour, which of these
   ten storylines is most worth showing?") a few times an hour.
2. **Narration (optional, later).** A small local model (the v0.4 Gemma 4 E2B via llama-server
   is already there) turning the same ground-truth captions and logged metrics into caption-card
   prose. Never fed pixels; never allowed to assert what a sheep "thinks" beyond what the
   telemetry supports.

### Reasoning

The question is really "where does a language model add value without faking the thing the
project is about?" The whole point of 2.0 is that inner life is learned, so an LLM generating
behaviour or dialogue would undo the premise; and used as *reward*, foundation models produce
documented Goodhart effects (OMNI's "misalignment by composition", Motif's agent hallucinating
the oracle). Used only to steer the camera, the Goodhart risk is confined to the lowest-stakes
place in the system. *(Judgement.)*

The evidence that an FM filter beats a heuristic at choosing what is interesting is real (14 vs
9 vs 3 tasks in OMNI), and the Motif recipe makes the live cost zero. The commentary literature
is unanimous that feeding structured data beats feeding frames — VLMs hallucinate state — and a
simulation owns its ground truth, so there is no reason to make a model infer it.

Narration is the riskier role for a different reason: it invites the narrator to claim meaning
that the causal-influence tests may not support. If it is used, the captions must be grounded in
the logged metrics (CIC, trust tables, topographic similarity) so that "Dot warned the flock"
only appears when the intervention test says the bleat changed neighbours' behaviour. That
constraint is also the honest answer to Q7.

### Evidence

- FM interestingness filter: 14 vs 9 vs 3 — [OMNI](https://arxiv.org/abs/2306.01711). **Established.**
- LLM preferences distilled to a small reward; prompt steering +17% to +150%; misalignment by
  composition — [Motif](https://arxiv.org/abs/2310.00166). **Established.**
- VLM success detection only 72.7% human-aligned — [OMNI-EPIC](https://arxiv.org/abs/2405.15568). **Established.**
- Structured data beats raw frames for commentary; VLMs hallucinate state —
  [INLG 2021](https://aclweb.org/anthology/2021.inlg-1.11.pdf), [`research/evolution-alife-and-curation.md` KQ5](research/evolution-alife-and-curation.md). **Established.**
- CLIP novelty "hard to sustain" on a flocking substrate — [ASAL](https://arxiv.org/abs/2412.17799). **Established.**
- No published FM-driven camera in an ALife installation. **Gap.**

---

## Q7. Theatre versus science — where the learned approach risks being dull, and mitigations

### Where it will be dull (Judgement, grounded in the base rates above)

1. **Generation 0.** From-scratch agents wander, stand still, or chase each other. Communication
   has a high base rate of not emerging at all, and when it emerges it is often un-listened-to.
2. **The long middle.** Language structure plateaus within ~10 generations; a flock's visual
   novelty saturates fast (ASAL's Boids warning); months of a stable meadow are, by default, a
   screensaver.
3. **Invisible cognition.** Anticipation, surprise, trust and grudges are internal scalars. A
   sheep that "expects the fox" looks exactly like one that does not, unless the display shows it.
4. **Honest failure.** Some of the hypotheses (iconicity from a shared encoder, sound-for-urgency)
   may not hold. A show that depends on them holding is fragile.

### Mitigations

**Open at generation N, not 0.** Everything in Q5 points here. The exhibition's first day is the
flock's hundredth; offline pretraining and evolution produce the opening cast. Viewers never see
generation 0 unless the documentary chooses to show archive footage of it — which, framed as
history, is interesting rather than dull.

**Make the internal scalars visible, cheaply.** The RSSM decodes an *expected next view* for free;
the KL and reconstruction error are per-step surprise scalars. Render them: a faint "expected"
ghost of the sheep's view in a caption card, a startle animation on a surprise spike, the N×N
trust matrix as a relationship graph, a lamb's recall curve after its teacher dies. These are the
hooks the papers already use as figures (ToMnet's 2-D character embeddings, Eccles' niceness
values, topographic-similarity plots); the documentary camera's job is to cut on *events in these
scalars* — a cohort birth (speaker entropy spike), a dialect split (inter-flock intelligibility
drop), a glyph being conventionalised (perimetric complexity falling), a betrayal (trust
dropping after a false mark).

**Engineer reliable drama where emergence is unproven.** The fox is scripted precisely so that
every night has a beat. Injury gives a visible limp. Cohort births give a recurring event. The
refuge gives a place to run to. None of these depend on learning working.

**Seed culture.** Hand-authored iconic ground marks in early offline training (a grass glyph near
grass, a fox glyph at the treeline) are what the stigmergy literature says is needed to bootstrap
*reading* before *writing*; they also mean the opening flock has a legible proto-language even if
its own innovations are slow. Frame them as an elder tradition.

**Keep the world slowly non-stationary.** Seasons, a new pasture opening, the fox's entry side
changing, occasional migrant sheep with a different dialect, a rotating referent set. Open-endedness
is observer-relative; novelty must be re-supplied on a cadence shorter than the plateau.

**Design for the null result.** Each channel is instrumented with a per-channel causal-influence
test. If drawings turn out to be private codes, the show still works: the marks are visible,
persistent and *mysterious*, and the caption layer can say honestly that the flock's marks are
not yet readable by us. If sound does not specialise for urgency, bleats are still bleats. The
science stays honest and the theatre degrades gracefully — which is v0.4's "cheapest machinery"
pillar restated: the cheapest believable thing is the real signal, shown plainly.

**Where to spend the "appearance of inner life" budget.** The v0.4 pillar was appearance with the
cheapest machinery. In 2.0 the machinery is not cheap, so the pillar becomes: *show the machinery*.
A viewer who can see what a sheep expected and how it was wrong, who it trusts and why, gets inner
life from the evidence rather than from a performance. *(Judgement.)*

### Evidence

- Non-emergence base rates — [Eccles et al. 2019](https://arxiv.org/abs/1912.05676); signalling
  without listening — [Lowe et al. 2019](https://arxiv.org/abs/1903.05168). **Established.**
- Structure plateaus before generation 10 — [Beckner et al. 2017](https://doi.org/10.1093/jole/lzx001). **Established.**
- Novelty saturates on flocking; open-endedness is observer-relative and AdA plateaued at ~1 month —
  [ASAL](https://arxiv.org/abs/2412.17799), [Hughes et al. 2024](https://arxiv.org/abs/2406.04268). **Established.**
- Stigmergy needs seeded trustworthy marks — [ForMIC](https://arxiv.org/abs/2006.08152). **Established.**
- Legible internal representations: ToMnet embeddings and belief maps — [Rabinowitz et al. 2018](https://arxiv.org/abs/1802.07740);
  topographic-similarity plots — [Li & Bowling 2019](https://arxiv.org/abs/1906.02403). **Established.**
- Documentary hooks per topic: [`research/iterated-learning-and-generations.md` KQ7](research/iterated-learning-and-generations.md),
  [`research/theory-of-mind-and-social.md` KQ7](research/theory-of-mind-and-social.md). **Judgement.**

---

## Q8. Perception scale and sprite legibility — is 1 px per tile enough?

Raised at plan review: for drawings to be interpretable by humans, the things being drawn (grass,
fox, sheep, pond) need sprites that are simple but recognisable, and a 3–4-pixel sprite probably
is not.

### Recommendation

*Superseded (2026-10-23/24): the art is 16 × 8 px isometric tiles at a 128² view, the drawing a
32 × 32 stroke canvas, and agents see a first-person strip — [`CURRENT-DESIGN.md`](CURRENT-DESIGN.md).
The reasoning below is kept as history.*

**Change the perception scale to 4 px per tile with a 64×64 egocentric view (16-tile range),
and keep the drawing canvas at 16×16.** Objects become multi-pixel sprites: a sheep ≈ 6×6 px, a
fox ≈ 6×6, a tree ≈ 8×8 over 2×2 tiles, grass/pond/salt as textured tiles. A 16×16 canvas stamped
onto the ground at native pixel scale then covers exactly a **4×4-tile footprint and is seen at
its own resolution** inside the 64×64 view — which dissolves Q1's scale problem: no downsampling,
no "look closely" action, no second frame. Speech bubbles are also rendered into the view at
native scale beside the speaker. The "same encoder" idea becomes literal: a drawing is just part
of the world image.

Cheaper fallback: 3 px/tile, 48×48 view, same canvas (a stamp then covers ~5×5 tiles).

### Reasoning

At 1 px/tile a grass tile is a colour, not a shape, and so is a fox. An "iconic" drawing of a fox
is then one orange pixel, indistinguishable from any orange thing; iconicity collapses to colour
plus relative position — a *map*, not a *picture*. Maps are a legitimate iconic convention (and
sheep may well draw them anyway), but they make the human-readability test weak and the
shared-encoder hypothesis nearly vacuous, because resemblance has nothing to work with. With
sprites of ~6×6 px the encoder's features respond to shape, a drawn fox can look like a fox, and
a 16×16 canvas has room for one or two sprite-sized icons plus a relation (an arrow, a gap).
*(Judgement.)*

Three things make 64×64 the natural size rather than an arbitrary one:

- It is **DreamerV3's native input resolution**: every published image config, regulariser and
  hyperparameter is validated at 64×64, so the world model is de-risked rather than extrapolated
  from the tiny-input regime nobody has published.
- 16 px / 4 px = 4 tiles: the canvas-to-footprint ratio falls out exactly.
- Human sprite legibility is well-trodden ground: 8-bit game sprites at 8×8 to 16×16 are
  recognisable; 3–4 px are not. The 16×16 canvas is the size of a classic icon.

Cost: the encoder needs four stride-2 convolutions instead of two and the decoder reconstructs
64×64×16 logits; per-step compute rises by roughly an order of magnitude over the 16×16 design
(to tens of MFLOP per sheep per step), which is nothing on the 4080 and still well under the CPU
budget if the brain ever moves. *(Estimate.)* The real cost is **replay memory**: 64×64 palette
indices at 4 bits are 2 KB per step, so 10⁷ steps is 20 GB — store replay on disk or keep buffers
at ~10⁶ steps per flock. Night vision shrinks the lit radius of the view exactly as before.

What changes elsewhere: Q1's recommendation is superseded (no footprint downsampling, no close
look); Q2 is unchanged (the rasteriser draws on the 16×16 canvas); M0 must use the same scale so
results transfer; and the world's visual design — the palette and the sprite set — becomes a
design task in its own right, since the sprites are the referents the sheep learn to draw.

### Evidence

- DreamerV3 image configs and encoder ("stride 2 convolutions ... until resolution 4×4") are all
  at 64×64 — [Hafner et al. 2023](https://arxiv.org/abs/2301.04104). **Established.**
- IRIS small-object erasure at low token counts — [IRIS](https://arxiv.org/abs/2209.00588).
  **Established.** (Argues for more pixels per object, not fewer.)
- Shape-biased backbones give more shape-like sketches; readability depends on the encoder having
  shape features to match — [Mihai & Hare 2021](https://arxiv.org/abs/2106.02067). **Established.**
- Image-observation JAX envs run ~20k steps/s on one GPU vs ~1M for symbolic —
  [XLand-MiniGrid](https://arxiv.org/abs/2312.12044). **Established.** (Rendering 64×64 views in
  the vectorised sim is the throughput cost to budget for.)

---

### Addendum: isometric rendering (after plan review)

The world is rendered **isometrically** (2:1 diamond tiles, 8×4 px at the current scale) with
four grass growth states as ground tiles, water as a ground state, single-tile camera-facing
objects, and sheep/fox facing left or right. An earlier note here claimed a fixed camera was
unavoidable with isometric art; that was wrong. The tile *layout* is rotated about the sheep by
its heading before projection and every sprite is drawn upright (billboarded), so the view is
still heading-aligned (Q9 stands), the human display is the same renderer at rotation 0, and a
drawing of a tree is the same picture in every frame. "Ahead" projects to the up-right screen
diagonal. An agent's screen facing depends on the view rotation, so sheep and fox have four facings
(front/back × left/right), indexed by the heading in the rotated frame.
Resolution stays at 64×64 for M0/M1 (decision: keep for now; see the plan for the cost table).

Implementation: `flock/iso.py`, `flock/sprites_iso.py`; tests in `tests/test_iso.py`.

### Addendum: foveated vision (full detail near, less far) — recommended for M1

Raised at plan review. Recommendation: **yes, as a two-scale view**, not true log-polar
foveation. The sheep gets its 64×64 near view at full resolution (16 tiles across) plus a
second 64×64 "far" view rendered at half scale (32 tiles across), stacked as extra input
channels. *(Judgement.)*

Why: it doubles the detection range for ≈2× pixel cost instead of the 4× of a 128² view;
it is the cheapest way to let a fox be noticed far off while a drawing is only *readable*
close up — which revives Q1's "glyph you notice, then walk to and read" without any action
or second frame, because a stamp in the far channel is a blurred 8×8 blob and in the near
channel a full 16×16 mark; and it matches the night-vision and blind-wedge masks, which
simply apply to both scales. Prey ungulates have a broad low-acuity field with a horizontal
high-acuity band rather than a fovea, so "wide and coarse plus a sharp region" is the right
shape biologically too (not verified against a primary source here). The cost is one extra
render per sheep per tick and a 32-channel instead of 16-channel encoder input; both are
small. It is an M1 item, not M0: the toy game needs no range.

Implementation sketch: render a 128×128 view with the same gather at tile scale 8×4 and
stride-2 subsample (or render directly with 4×2 tile boxes); concatenate one-hot channels.

## Q9. Visibility — whole world, local window, or a cone?

### Recommendation

**A local, heading-aligned window with a wide field of view and a small rear blind wedge; not the
whole world, and not a narrow cone.** Concretely: the 64×64 view is centred on the sheep and
rotated so "ahead" is up; a rear wedge of ~60° is masked (sheep have near-panoramic vision, so
the blind spot is small); night shrinks the lit radius; the "head up" vigilance action extends the
lit radius at a foraging cost. The **fox** gets a genuine forward cone, because it is a pursuit
predator and because an asymmetric detection geometry (sheep see wide, fox sees far-but-narrow)
is what makes flanking, hiding behind trees and the selfish herd meaningful.

### Reasoning

Full-world visibility removes the information asymmetry that communication exists to bridge: if
every sheep can see every patch and the fox, there is nothing to tell. Every emergent-communication
result relies on the sender knowing something the receiver does not. Partial observability is
also what makes the world model do work — anticipation is only interesting when the next frame
is uncertain — and what makes vigilance a real trade-off. *(Judgement, from Established.)*

A narrow cone is wrong for sheep biologically (prey ungulates have ~270–320° fields with a small
posterior blind spot) and bad for the installation: a sheep that cannot see a fox beside it
reads as stupid, not vigilant. But some heading-dependence is valuable: it gives the encoder a
consistent frame (ahead is always up), makes "turn to look" a visible behaviour, makes directional
drawings ("that way") meaningful relative to the drawer's heading, and gives the rear wedge a
dramatic function (the fox approaches from behind).

### Evidence

- Communication needs an information gap; non-emergence base rates when it is weak —
  [Eccles et al. 2019](https://arxiv.org/abs/1912.05676), [Lazaridou & Baroni 2020](https://arxiv.org/abs/2006.02419). **Established.**
- Swarming favoured only when predator vision ≥ 120° — [Olson et al. 2013](https://arxiv.org/abs/1209.3330). **Established.**
  (The fox's cone must not be too narrow or grouping stops paying.)
- Sheep vigilance behaviour: scanning bouts, group-size effect —
  [Michelena & Deneubourg 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3078120). **Established.**
- Sheep visual field width: widely reported as ~270–320° with a small rear blind spot; not
  verified against a primary source in this research. **Unverified.**

### Addendum: first-person view, ground reading, and steps toward language (M1+, 2026-10-23)

Agreed with the owner in a side discussion; **for M1 onward, not M0** (M0 finishes on the
isometric world). Items 1–3 revise Q8/Q9; item 4 is scope for M1 and later. Status: design
direction, not yet tested; item 1 needs a benchmark before it is adopted.

1. **Agents perceive a first-person view with billboard sprites** (Doom-style), not the rotated
   isometric view; humans keep the isometric display, rendered from the state stream.
   - Gains *(judgement)*: distance shrinks sprites by itself, which replaces the two-scale far
     view in the foveated-vision addendum above; real occlusion (hiding behind trees, a fox
     behind the flock); a real view cone.
   - Precedents for learning from first-person pixel views: DreamerV3 on DeepMind Lab —
     [Hafner et al. 2023](https://arxiv.org/abs/2301.04104), [Beattie et al. 2016 (DMLab)](https://arxiv.org/abs/1612.03801);
     ViZDoom — [Kempka et al. 2016](https://arxiv.org/abs/1605.02097). **Established** (that
     agents learn from such views; not that it suits this project).
   - Cost: about 1.5–2× the isometric render (a guess, not benchmarked). Before deciding, a
     short JAX ray-cast spike with a throughput benchmark against the current renderer.
     **Spike result (2026-10-23)**: `flock/fpv.py` (cylindrical 300° strip, 256×64, ground by
     per-pixel ray to the ground plane reusing the isometric tile art, billboards scaled by
     1/distance and painted far to near), benchmarked by `scripts/bench_fpv.py` on the 4080 at
     batch 512 against `iso.render_view` (128×128) on the same worlds:

     | world | objects | isometric views/s | first-person views/s | first-person cost |
     | --- | --- | --- | --- | --- |
     | 24×24 | 8 | 654,000 | 1,477,000 | 0.44× |
     | 64×64 | 40 | 233,000 | 400,000 | 0.58× |
     | 128×128 | 120 | 98,000 | 141,000 | 0.70× |

     The first-person view is *cheaper*, not dearer: the isometric renderer pads and copies the
     whole image once per sprite it pastes. Neither culls objects out of range, so both have
     headroom. Either way rendering will not limit M1 *(judgement)*. Visual check: distance
     shrinks objects as intended — something 8 tiles away is a few pixels tall, so far
     objects lose detail without a separate far view; ground right in front of the viewer is
     blocky (the tile art magnified), a cosmetic issue for later.
     **Decision (owner, 2026-10-23): adopted for M1 onward.** The isometric view stays for
     humans (the display) and as the tested fallback for agents.
   - Relevant M0 finding: a perception encoder that must recognise objects at several scales
     is what a first-person view forces; M0 had to add scale variation by hand
     (`m0-toy-README.md`, the instance-variation work).
2. **Field of view ≈ 300°, as a wide strip** (e.g. 256×64, the same pixel count as 128×128).
   The owner explicitly does not want 360°: the ~60° blind wedge behind each sheep is intended,
   because it makes warnings valuable, and no wrap-around padding is needed. A non-square image
   is fine for convolutions; only flatten sizes and the receiver's choice grid change.
3. **Ground marks are read by "looking down".** An action temporarily replaces world vision with
   a top-down image of the ground at full resolution, fed through the *same* encoder with a mode
   flag. This revives Q1's "look closely" idea. Reading costs vigilance, a good trade-off with
   the fox; a mark is first noticed as a smudge in the 3D view, then walked to and read.
   Speech bubbles stay in normal vision. *(Judgement.)*
4. **Toward an emergent sheep language: consider folding into M1 now** *(judgement)*:
   - (a) uneven information that pays off for both sides: occlusion, the blind wedge,
     out-of-sight food or fox (communication needs an information gap — see Q9's evidence);
   - (b) displacement: telling about absent, past or moved things (needs memory);
   - (c) sequences of glyphs or bleats rather than single images, so syntax has somewhere to
     live.

   Later milestones: meaning spaces built from parts (what × where × how many × how urgent);
   generations and communities — iterated learning makes transmitted languages more learnable
   and structured ([Kirby, Cornish & Smith 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2504810)),
   and communities converge on shared systems where isolated pairs develop local ones
   ([Fay et al. 2010](https://onlinelibrary.wiley.com/doi/10.1111/j.1551-6709.2009.01090.x));
   a reply channel for repair — drawings simplified only when the partner could give feedback
   ([Fay et al. 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2607338/)); a resemblance
   pressure that fades over time; structure metrics logged from the start (topographic
   similarity — [Li & Bowling 2019](https://arxiv.org/pdf/1906.02403) — held-out
   generalisation, dialect divergence).

---

## Q10. GPU-first architecture — the desktop runs everything

Raised at plan review: the installation runs on the desktop (13700KF + RTX 4080), the GPU should
be used as much as possible, and v0.4's Rust/Pi requirements no longer constrain the design.

### Recommendation

**One implementation of the meadow, written as a vectorised GPU simulation, used for both
training and the installation.** Offline, thousands of meadows run in parallel on the 4080;
live, one meadow runs at a few Hz with the same code, with inference and online learning on the
same GPU. The renderer is a separate process fed by a state stream, so the display path is
decoupled from the brain (and can be Rust/WASM on the Pi, a native window on the desktop, or
both). The installation becomes a supervised Python process rather than a Rust binary.

Framework: **JAX** for the sim and the world model (the vmapped-gridworld and DreamerV3 lineage
is JAX; Craftax/XLand/NAVIX-class throughput is the proven path), unless you are substantially
more at home in PyTorch — in which case PyTorch with tensor-batched environments is workable at
lower throughput and is the better choice for developer speed. This is a question for you below.

### Reasoning

With no cloud and no v0.4 constraint, the two-implementation problem (Rust sim for the
installation, Python/JAX sim for training, kept in sync) has no remaining justification. The
things that argued for Rust in the installation — a CPU-only host, a Pi in the loop, an existing
codebase — are all gone. A single vectorised implementation means the offline-trained models run
unchanged live, there is no ONNX export and no Rust ML crate risk, online learning is as cheap as
the GPU allows, and every milestone's test harness *is* the installation code. *(Judgement.)*

The costs are real and should be named: a long-running Python/JAX process needs supervision
(systemd restart, periodic checkpoints — which the plan already requires), the 64×64 view
rendering inside the vectorised sim is the throughput bottleneck (image-observation JAX envs run
~50× slower than symbolic ones), and JAX's pure-functional style is a learning curve if
unfamiliar.

### Evidence

- Vectorised JAX gridworlds and their throughput — [Craftax](https://arxiv.org/abs/2402.16801),
  [XLand-MiniGrid](https://arxiv.org/abs/2312.12044), [JaxMARL](https://arxiv.org/abs/2311.10090). **Established.**
- JAX at batch 1 is no faster than Python; the win is only in batch on an accelerator —
  [JaxMARL Table 3](https://arxiv.org/abs/2311.10090). **Established.** (Live at a few Hz the
  framework is irrelevant; the point is sharing code with training.)
- PureJaxRL meta-evolution of 512 agents × 1024 generations on one GPU in ~9 h —
  [PureJaxRL](https://chrislu.page/blog/meta-disco/). **Established.** (M6's evolution loop fits a
  single GPU in this stack.)
- Rust ML crate status and ONNX GRU export pitfalls that this avoids —
  [`research/tooling-and-hardware.md` KQ3–4](research/tooling-and-hardware.md).

---

## Summary table

*As recommended at plan review (2026-10-22). Rows 1, 8 and 9 are superseded: the drawing is
32 × 32 strokes, the isometric view 16 × 8 px tiles at 128², and agents see a first-person 300°
strip — see [`CURRENT-DESIGN.md`](CURRENT-DESIGN.md).*

| # | Question | Recommendation (short) | Confidence | Tested at |
| --- | --- | --- | --- | --- |
| 1 | Mark scale | Superseded by Q8: at 4 px/tile a 16×16 stamp *is* a 4×4-tile footprint at native resolution; ≈5–8 usable bits | High (given Q8) | M0, M4 |
| 2 | Drawing action space | Differentiable stroke/stamp rasteriser + ink cost + perceptual loss; VQ-decoder fallback | High that strokes train; medium that iconicity follows | M0 |
| 3 | Death | Injury first; death on second catch or old age; births track deaths with floor/ceiling; small refuge | High | M2, M7 |
| 4 | Fox | Scripted parametric fox with confusion rule and governor; offline fox-dense training; frozen fox head live | High | M2 |
| 5 | Offline vs online | Learn offline, maintain online; genome / lifetime / culture kept separate | High on feasibility; medium on how much online fine-tuning is safe | M1, M7 |
| 6 | LLM | Camera curator from ground-truth captions; optional grounded narration; never a brain or reward | High | M7 |
| 7 | Theatre vs science | Open at generation N; show the internal scalars; scripted drama where emergence is unproven; design for the null | — | all |
| 8 | Perception scale | 4 px/tile, 64×64 view, 16×16 canvas; multi-pixel sprites; DreamerV3-native resolution | Medium-high; replay memory is the cost | M0, M1 |
| 9 | Visibility | Local heading-aligned window, wide FOV, small rear blind wedge; night shrinks radius; fox gets a cone | High | M1, M2 |
| 10 | Architecture | One vectorised GPU sim (JAX by default) for training and installation; renderer decoupled by a state stream | High on shape; framework choice is yours | M1, M7 |
