# FLOCK 2.0 — research survey (synthesis)

This document condenses the nine per-topic research notes in [`research/`](research/) into
what matters for FLOCK 2.0. Each section ends with a one-line verdict. The notes carry the
full citations and the per-claim provenance; this file links the load-bearing ones.

Labelling convention, used throughout this folder:

- **Established** — a published result, linked.
- **Estimate** — arithmetic derived from published numbers; assumptions stated in the notes.
- **Judgement** — my reading of the evidence, or speculation. Not a literature claim.

A note on coverage: the researchers read primary sources where they could. Where a claim rests
on an abstract, a secondary summary or a paywalled paper, the notes say so. Several
"gaps" recur across topics and are collected in §10 — these are things nobody has published
and that FLOCK would have to measure for itself.

---

## 1. World models — what is the minimum that gives believable anticipation and surprise?

Notes: [`research/world-models.md`](research/world-models.md)

**Established.** DreamerV3 is an RSSM (GRU + categorical latents) with a small set of
regularisers (free bits, KL balancing, 1% unimix, symlog) that make it stable across domains
without per-task tuning ([Hafner et al. 2023](https://arxiv.org/abs/2301.04104)). The smallest
published image config is XS at 8M parameters; the official repo ships a `size1m` config used
for proprioceptive inputs ([configs.yaml](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml)).
Its encoder halves resolution until 4×4, so a 16×16 input needs only two stride-2 convolutions
and most of the CNN cost that is sized for 64×64 disappears. Every published Dreamer run uses
one V100/A100 and a train ratio (replayed steps per environment step) of 32–1024. The closest
tiny precedent is Ha & Schmidhuber's VAE + 0.42M-parameter MDN-RNN + 867-parameter linear
controller ([World Models, 2018](https://arxiv.org/abs/1803.10122)), which also documents the
main failure mode — the policy exploiting a flawed dream — and its remedy (sampling temperature).

Dreamer 4 (2B parameters, TPU pods; [arXiv:2509.24527](https://arxiv.org/abs/2509.24527)),
IRIS, STORM and DIAMOND (3–7 GPU-days per Atari game) are far outside FLOCK's envelope.
TD-MPC2's 1M config is small but decoder-free and plans with ~9k latent rollouts per action
([arXiv:2310.16828](https://arxiv.org/abs/2310.16828)); it cannot show the audience an "expected
frame".

Multi-agent Dreamer variants (MAMBA, MARIE) all share world-model parameters across agents and
keep one recurrent state per agent ([MAMBA](https://arxiv.org/abs/2205.15023),
[MARIE](https://arxiv.org/abs/2406.15836)). Nothing published trains tens of independent
per-agent world models.

**Estimate.** A FLOCK-sized RSSM (16×16×16 one-hot input, 16×16 categorical latent, GRU 256,
categorical decoder) is ≈1–2M parameters and ≈9 MFLOP per step with the decoder. Fifty sheep at
10 Hz is ≈4.5 GFLOP/s — trivial on either i7. One gradient update is ≈50–70 GFLOP, so a single
shared model can take ≈1.5–4 updates/s on 8 cores; fifty independently-trained models cannot.
The implied *online* train ratio is ≈2 against the 32–512 the literature uses, so online-only
learning from scratch would be far slower than any published Dreamer result.

**Judgement.** The minimum credible model is a DreamerV3-style RSSM at ≈1–2M parameters,
**shared parameters, per-sheep recurrent state plus a per-sheep trait embedding**, pre-trained
offline on logged simulation video (as Ha & Schmidhuber and Dreamer 4 both do) and only lightly
fine-tuned online. Surprise for the audience comes free from the posterior‖prior KL and the
decoded-frame error; curiosity for the agent should use a small Plan2Explore-style disagreement
ensemble rather than raw prediction error ([Sekar et al. 2020](https://arxiv.org/abs/2005.05960)).
Reconstruct in one-hot colour space with cross-entropy, not MSE, or single-pixel objects
(fox, grass) get erased — the IRIS small-object failure ([IRIS](https://arxiv.org/abs/2209.00588)).

**Verdict:** one shared ~1–2M RSSM with per-sheep state, pre-trained offline. Independent
per-sheep world models are not affordable and have no precedent.

---

## 2. Drives and intrinsic motivation

Notes: [`research/intrinsic-motivation-and-drives.md`](research/intrinsic-motivation-and-drives.md)

**Established.** Homeostatic RL defines reward as the *reduction* of a drive function (distance
from set-point) and proves that maximising discounted reward is the same as minimising
discounted drive; it yields risk aversion, anticipatory regulation and avoidance of extreme
states ([Keramati & Gutkin 2014](https://elifesciences.org/articles/04811)). Any additive reward
term outside the drives shifts the effective set-point and causes over-consumption (their
"palatability" pathology). Continuous internal-state decay (allostasis) and fatigue-forced rest
are the published extensions ([Laurençon et al. 2021](https://arxiv.org/abs/2109.06580)); in deep
RL, "difference of drive" was the best-performing reward form but still produced deaths,
stereotypy and habit carry-over ([Yoshida et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11635831/)).
Drives must be non-substitutable (a norm with exponent > 1, inverted-U penalties), and
safety-like objectives should be thresholded-lexicographic ([Pihlakas & Pyykkö](https://arxiv.org/abs/2410.00081),
[Skalse et al. 2022](https://www.ijcai.org/proceedings/2022/476)).

Prediction-error curiosity is attracted to stochastic transitions — the "noisy TV"
([Burda et al. 2019](https://arxiv.org/abs/1808.04355)) — and two curious agents will prolong
interaction with each other (the Pong-rally result in the same paper). Other sheep and the fox
are built-in noisy TVs. In gridworlds the term that actually drives exploration is a count-based
episodic bonus; removing it from NovelD/RIDE/AGAC breaks learning
([Henaff et al. 2022](https://arxiv.org/abs/2210.05805)). SMiRL (surprise *minimisation*) gives
bracing/dodging/escaping behaviour but trivially collapses to standing still in a stable world
([Berseth et al. 2021](https://arxiv.org/abs/1912.05510)).

Deep networks lose plasticity under continual training; remedies are continual backprop
(reinitialise dormant units), weight decay + LayerNorm, periodic head resets with buffer
retained, shrink-and-perturb ([Dohare et al. 2024](https://www.nature.com/articles/s41586-024-07711-7),
[Nikishin et al. 2022](https://arxiv.org/abs/2205.07802), [Lyle et al. 2024](https://arxiv.org/abs/2402.18762)).
Tabular/linear learners sidestep this entirely.

Sheep ethology gives concrete numbers: vigilance and cortisol fall with group size; imitation of
scanning (ε≈0.92) is far stronger than imitation of foraging (ε≈0.47)
([Michelena & Deneubourg 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3078120)); flocks
alternate long dispersed grazing with fast packing events driven by allelomimesis
([Ginelli et al. 2015](https://abdn.elsevierpure.com/files/58184052/PNAS_2015_Ginelli_12729_34.pdf)).
No published RL agent has had rest *emerge* from an energy drive alone; the one circadian result
used an explicit night penalty ([Labash et al. 2023](https://proceedings.mlr.press/v202/labash23a.html)).

**Judgement.** Reward = HRRL drive difference over (hunger, thirst, warmth, social, fear, interest)
with continuous decay, nothing additive outside it. Fear as a thresholded top-priority objective.
Curiosity as hashed counts over *static-world* features only, routed into an "interest" drive,
blind to other agents. Warmth decays faster at night and away from cover and recovers only while
resting, which is the closest the literature gets to drive-driven sleep. Keep the online learner
small, apply the plasticity remedies, and run a behavioural watchdog with checkpoint rollback.

**Verdict:** drives are well-grounded and cheap; the risks are gaming (fixed by structure, not
weights) and the noisy-TV pathology (fixed by making curiosity blind to agents).

---

## 3. Emergent communication — general results and the bleat channel

Notes: [`research/emergent-communication.md`](research/emergent-communication.md)

**Established.** The default outcome of decentralised, temporally extended multi-agent RL is
*no* communication: without an inductive bias, 0/50 runs succeeded on Summing-MNIST and 28% on
Treasure Hunt; adding positive-signalling and positive-listening auxiliary losses raised this
to 100% and 94% ([Eccles et al. 2019](https://arxiv.org/abs/1912.05676)). Messages can be
perfectly correlated with the sender's state and still have zero causal effect on the receiver:
in matrix games, causal influence of communication was within 2% of its minimum in 89–99.9% of
games ([Lowe et al. 2019](https://arxiv.org/abs/1903.05168)) — so "the sheep are talking" needs
an intervention test, not a correlation. Length costs fix anti-efficient encoding but slow
convergence and reduce the success rate ([Chaabouni et al. 2019](https://arxiv.org/abs/1905.12561)).
Population size alone does not help; heterogeneity does ([Rita et al. 2022](https://arxiv.org/abs/2204.12982)).
Resetting listeners makes languages more teachable ([Li & Bowling 2019](https://arxiv.org/abs/1906.02403)).
Continuous channels train more easily but ~25% of seeds collapsed in one study vs 0% discrete
([arXiv:2604.03266](https://arxiv.org/abs/2604.03266)), and need a repulsion bias to escape
"same message, two states" ([Villanger & Alexandersen 2024](https://proceedings.mlr.press/v233/villanger24a/villanger24a.pdf)).

The only substantial RL work on a genuinely acoustic channel renders phone sequences through
TTS and room noise; channel noise induced redundancy and *higher* compositionality, and DQN beat
REINFORCE by a wide margin ([Eloff et al.](https://arxiv.org/abs/2111.02827)). Nobody has
published agents learning pitch/duration/timbre parameters. Costly signalling for honesty did
*not* emerge under tabular RL in the Sir Philip Sidney game
([PLOS Comput Biol 2025](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013302)),
and modern signalling theory holds that honesty needs costly *cheating*, not costly signals
([Lachmann, Számadó & Bergstrom 2001](https://pmc.ncbi.nlm.nih.gov/articles/PMC60846)). No MARL
paper has two coexisting channels of different cost that specialise.

**Judgement.** The bleat is a continuous channel with a handful of parameters: expect the
continuous-channel failures (collapse, receiver dominance), mitigate with distance-scaled noise
and a repulsion/entropy term, and let listeners receive the *parameters* with noise while audio is
rendered only for the audience. Pre-train the channel in an offline referential game (10⁴–10⁵
interactions) before dropping it into the world. The "cheap sound = urgency" hypothesis has no
precedent and only indirect support (information-bottleneck and length-cost results push
frequent, high-value meanings onto short signals); design it as a measured experiment with
per-channel causal-influence metrics.

**Verdict:** feasible, but non-emergence is the base rate. Auxiliary losses (or centralised
training) and a live causal-influence test are not optional.

---

## 4. Graphical communication and iconicity

Notes: [`research/graphical-communication.md`](research/graphical-communication.md)

**Established.** In the two canonical sketching games, iconicity never came for free.
Mihai & Hare used the *same* frozen VGG16 for photos and sketches on the receiver side; with
the game loss alone, drawings were "indecipherable to a human" (humans at 8.3%, chance 10%);
adding a perceptual (feature-matching) loss lifted humans to 38% at almost no cost to agent
success ([Mihai & Hare 2021](https://arxiv.org/abs/2106.02067)). Qiu et al. seeded the sender
with a pretrained photo-to-sketch module and used time discounting as the only cost; drawings
simplified to diagnostic parts (rabbit → ear) while staying classifiable, and agents reverted to
longer, more iconic drawings for unfamiliar referents; one-step drawing failed outright
([Qiu et al. 2022](https://arxiv.org/abs/2111.14210)). Human Pictionary data shows the same
pattern — fewer strokes, but diagnostic features preserved, not drift to arbitrary marks
([Hawkins et al. 2023](https://arxiv.org/abs/2109.13861)) — and icons become symbols only with
addressee feedback; communities keep more residual iconicity than isolated pairs
([Fay et al. 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2607338/)).

From-scratch CNN agents exploit low-level pixel statistics unless augmented — a shared CNN
"succeeded" on Gaussian-noise blobs at 84.7% ([Dessì et al. 2021](https://arxiv.org/abs/2106.04258)).
A shared encoder plus a high-capacity continuous channel makes the game trivially (and
non-iconically) solvable, because the sender can just transmit its embedding
([Carmeli et al.](https://arxiv.org/abs/2211.02412)). The one system where signals and stimuli go
through one perception network reports iconic starts drifting to abstract symbols under
size/curvature penalties, with no ablation isolating the sharing
([Signification Game, 2025](https://arxiv.org/abs/2502.01568)). Stigmergic marks in MARL work
with exponential decay but need a curriculum: early marks are noise and agents learn to ignore
them ([ForMIC](https://arxiv.org/abs/2006.08152)). No paper trains a pixel-level categorical
drawing head for communication; stroke rasterisers and neural renderers are what train.

**Judgement.** The shared-encoder hypothesis is *supported in direction, unproven in mechanism*:
sharing makes iconic marks decodable for free and encoder mismatch hurts, but everywhere
iconicity appeared it was injected (perceptual loss, pretrained sketch init, classifier-entropy
reward). FLOCK should treat the shared encoder as necessary but not sufficient and add a cheap
resemblance term in the shared encoder's feature space. Usable capacity with small co-trained
nets is tens of bits (≈5–8 bits per mark), not the 1,024 raw bits of the canvas; the surplus is
what makes resemblance affordable. The shared-vs-separate ablation does not exist in the
literature and is exactly what Milestone 0 should produce.

**Verdict:** drawings can emerge and be readable, but only with a resemblance signal in the
objective and a stroke/stamp action space. This is the riskiest and most novel part of FLOCK.

---

## 5. Iterated learning, cultural transmission and generations

Notes: [`research/iterated-learning-and-generations.md`](research/iterated-learning-and-generations.md)

**Established.** Structure in iterated learning needs *both* pressures: a learning bottleneck
gives compressibility but alone collapses languages to degenerate forms; communication adds
expressivity; together they yield compositionality
([Kirby, Tamariz, Cornish & Smith 2015](https://doi.org/10.1016/j.cognition.2015.03.016),
[Kirby, Cornish & Smith 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2504810)). Compositionality
rises then plateaus before generation 10 ([Beckner et al. 2017](https://doi.org/10.1093/jole/lzx001)).
The neural version — reset agents, short imitation phase on a sample of the elder's outputs
(the bottleneck), then interaction — gave topographic similarity 0.93 vs 0.23 without resets and
held-out accuracy 0.85 vs 0.14; the imitation budget is the most sensitive knob
([Ren et al. 2020](https://arxiv.org/abs/2002.01365)). Staggered one-at-a-time replacement in a
population of 10 loses the effect; simultaneous cohort reset restores it; every full reset
crashes success to ~20% ([Li & Bowling 2019](https://arxiv.org/abs/1906.02403)). Compositional
languages are not *produced* by communication pressure but are *retained* because they are easier
for new learners ([Chaabouni et al. 2020](https://arxiv.org/abs/2004.09124)).

Model-free agents do not imitate spontaneously; social learning needs memory, an intermittently
absent expert, an auxiliary loss attending to the expert, and a curriculum
([Bhoopchand et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10684502/),
[Ndousse et al. 2021](https://arxiv.org/abs/2010.00581)). For inheritance, Baldwinian (evolve
priors and learning rules, re-learn each life) beats Lamarckian (copy learned weights) when the
task distribution is broad or changing ([Fernando et al. 2018](https://arxiv.org/abs/1806.07917)),
and evolved plasticity rules let networks adapt from random weights
([Najarro & Risi 2020](https://arxiv.org/abs/2007.02686)).

**Judgement.** Keep two inheritance channels strictly separate: the *genome* carries priors
(initial weights or a low-rank seed, drive weights, imitation budget, timbre/pitch range) and is
mutated; the *language* is learned culturally through a short supervised imitation phase on a
sample of elder bleats and drawings, then interaction. Copying a parent's trained language weights
would bypass the bottleneck that produces structure. Births in small cohorts, not a trickle.
Expect structure to saturate within ~10 generations; the long tail of the show is dialect
divergence and glyph conventionalisation, which needs slow non-stationarity (rotating referents,
migrants, prior mutation).

**Verdict:** unusually good fit; the mechanism is well-specified by the literature. Nobody has
run neural iterated learning over a pixel channel, so expect tuning.

---

## 6. Theory of mind, reciprocity, honesty and deception

Notes: [`research/theory-of-mind-and-social.md`](research/theory-of-mind-and-social.md)

**Established.** ToMnet builds a character embedding from several past episodes of the *same*
agent; 2-D embeddings are legible when plotted and the belief head produces spatial maps
([Rabinowitz et al. 2018](https://arxiv.org/abs/1802.07740)). Every positive reciprocity result
adds a channel through which a partner's past becomes consequential: partner selection with one
bit of memory per partner turns 20 Q-learners from defection into a tit-for-tat society, while
random pairing stays at defection and ε=0.01 freezes it
([Anastassacos et al. 2020](https://arxiv.org/abs/1902.03185)); a per-partner discounted
"niceness" scalar matched by intrinsic reward yields reciprocity in gridworld dilemmas, and
removing the reciprocators causes relapse ([Eccles et al. 2019](https://arxiv.org/abs/1903.08082));
heterogeneous social value orientation makes populations richer and more general
([McKee et al. 2020](https://arxiv.org/abs/2002.02325)). Shared reputation (indirect reciprocity)
is cognitively heavy and collapses under private information or assessment error
([Nowak & Sigmund 2005](https://www.nature.com/articles/nature04131),
[Uchida & Sasaki 2013](https://ideas.repec.org/a/eee/chsofr/v56y2013icp175-180.html)).
Deception and scepticism co-emerge when agents model the listener *and* incentives are misaligned
([Alon et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10575559/)); under strong competition,
cheap-talk communication fails to emerge at all
([Cao et al. 2018](https://arxiv.org/abs/1804.03980), [Noukhovitch et al. 2021](https://arxiv.org/abs/2101.10276)).
No paper implements per-individual social-memory slots keyed by a perceptual tag.

**Judgement.** Give each sheep a fixed, perceptually distinct ID (wool pattern + index channel);
keep a tiny per-partner table of two discounted scalars — niceness (partner's action-advantage
for me) and trust (signal-vs-outcome accuracy) — fed to the policy alongside observations of that
partner; make partner choice have real payoff; mix social value orientations; keep exploration
high. Honesty cost comes from memory (a liar is remembered and avoided), not from signal cost,
which is what the modern signalling literature says. Lies need a mostly-cooperative baseline with
occasional conflict to be *meaningful*.

**Verdict:** every ingredient has support; the per-partner trust table is FLOCK's own
contribution. The ID requirement is hard: without it none of this works.

---

## 7. The fox, death and population stability

Notes: [`research/predator-prey-death-and-population.md`](research/predator-prey-death-and-population.md)

**Established.** Flocking emerges from survival pressure alone *only when the capture rule makes
grouping pay*: with a 1/N confusion rule, swarm density rose from 0.69 to 12.48 and cohesive
swarms formed in ~70% of replicates; with no confusion, none did
([Olson et al. 2013](https://arxiv.org/abs/1209.3330)). Deep-RL prey with a fully scripted
predator (random-target distraction, leaps) and immediate respawn produced Boids-like flocking
with no flocking reward ([Hahn et al. 2019, SELFish](https://arxiv.org/abs/1905.04077)); a
non-lethal "bleeding" contact penalty was sufficient, and a fixed-rule predator was used to speed
prey learning ([Li et al. 2023](https://arxiv.org/abs/2308.12624)). No source shows alarm calls
emerging from learning in a predator-prey simulation. Live co-learning arms races need
league/fictitious-self-play machinery and still drift ([self-play survey](https://arxiv.org/abs/2408.01072));
a learned predator learned restraint only with curriculum and long-horizon discounting
([Ritz, Hahn et al. 2020](https://mobile.ifi.lmu.de/team/carsten-hahn)).

For rare catastrophes, the relevant mechanism is a never-flushed danger buffer plus a learned fear
model that shapes the value target, with a bound (λε) on the return it costs
([Lipton et al., Intrinsic Fear](https://arxiv.org/abs/1611.01211)). Stable long runs with
learners hold population constant by respawn (SELFish; Neural MMO spawn cap,
[arXiv:2001.12004](https://arxiv.org/abs/2001.12004)); free-floating energy-based reproduction
collapsed in 3 of 5 seeds despite tuning ([Hamon et al. 2023](https://arxiv.org/abs/2302.09334)).
Refuges protecting a constant *small* number of prey stabilise; large refuges destabilise
([Krivan 1998](https://fix.prf.jcu.cz/~krivan/clanky/Krivan-TPB-1998.pdf)). Removing dead agents
creates value-estimation and credit-assignment problems
([Cohen et al.](https://arxiv.org/abs/2111.05992)). Real sheep respond with vigilance, flocking,
flight to cover and freezing at the refuge; vigilance falls with group size; edge/spaced animals
are the ones taken ([Animal Welfare review 2023](https://www.cambridge.org/core/journals/animal-welfare/article/how-has-the-risk-of-predation-shaped-the-behavioural-responses-of-sheep-to-fear-and-distress/0005C66D41B3ECC47267627F47EAD70B)).

**Judgement.** A scripted parametric fox with 1/N confusion, isolation-weighted targeting,
target inertia, ≤1 catch per visit and a governor holding success at ~25–40% is the stable
choice; a frozen offline-learned fox is an optional later richness. Sheep learn fox response
offline under a fox-dense curriculum with a permanent shared danger buffer; the live run freezes
that head. Injury-first (visible limp, slower, poorer foraging), death only on a second catch
while injured or at old age; births track deaths with a floor and ceiling; a small refuge.

**Verdict:** well-charted. The capture rule, not the learner, is the main lever for lifelike
flocking. Emergent alarm calls are a stretch goal, not a plan.

---

## 8. Evolution, open-endedness and foundation models as curators

Notes: [`research/evolution-alife-and-curation.md`](research/evolution-alife-and-curation.md)

**Established.** ES/GA match policy-gradient RL on small nets, tolerate sparse delayed reward,
long horizons and non-differentiable parts, at 3–10× the data cost on hard continuous control
([Salimans et al. 2017](https://arxiv.org/abs/1703.03864), [Such et al. 2017](https://arxiv.org/abs/1712.06567));
a 3,667-parameter evolved agent beat 4.7M-parameter baselines on CarRacing generalisation
([Tang, Nguyen & Ha 2020](https://arxiv.org/abs/2003.08165)); ES struggles beyond ~100k parameters
([evosax](https://arxiv.org/abs/2212.04180)). Evolving the *starting point* (initial weights,
plasticity rules, reward/drive weights) with lifetime learning on top is the consistent winning
pattern, with the Baldwin effect appearing within ~10 generations
([Fernando et al. 2018](https://arxiv.org/abs/1806.07917), [DERL](https://arxiv.org/abs/2102.02202),
[Singh, Lewis, Barto & Sorg 2010](https://ieeexplore.ieee.org/document/5471106),
[Niekum et al. 2010](https://gpbib.cs.ucl.ac.uk/gp-html/Niekum_2010_ieeeTAMD.html)). Classic ALife
worlds "stop producing anything interesting within days"
([Stanley, Lehman & Soros 2017](https://www.oreilly.com/radar/open-endedness-the-last-grand-challenge-youve-never-heard-of/));
the strictness of the minimal criterion decides between stagnation and orderly divergence
([Soros, Cheney & Stanley 2016](https://www.uvm.edu/neurobotics/pubs/pdf/2016_SorosCheneyStanley_HowTheStrictnessOfTheMinimalCriterionImpactsOpenEndedEvolution_ALIFE.pdf));
open-endedness is observer-relative, needing novelty *and* learnability, and AdA plateaued at
about a month ([Hughes et al. 2024](https://arxiv.org/abs/2406.04268)).

ASAL scores simulations with CLIP for target match, historical novelty and diversity, and warns
that novelty was "hard to sustain" on Boids — a flocking substrate
([Kumar et al. 2024](https://arxiv.org/abs/2412.17799)). As a curriculum filter, an FM model of
interestingness learned 14 interesting Crafter tasks vs 9 for learning-progress and 3 for
uniform ([OMNI](https://arxiv.org/abs/2306.01711)). Motif distils LLM preferences over 500k
caption pairs into a small reward model; intrinsic-only agents beat score-trained ones, with
documented "misalignment by composition" ([Klissarov et al. 2023](https://arxiv.org/abs/2310.00166)).
VLMs are unreliable at fine-grained state (72.7% human agreement as success detectors,
[OMNI-EPIC](https://arxiv.org/abs/2405.15568)); commentary systems fed structured data beat
those fed raw frames.

**Judgement.** Use evolution for the slow, population-level layer (priors, drives, plasticity
rules, trait vectors), lifetime learning for the fast layer. Expect collapse and boredom by
default and build rails: population floor with archive reseeding, resource regrowth floor,
tuned minimal criterion, difficulty scheduling, a quality-diversity archive. Use a foundation
model as *camera curator* fed ground-truth captions, never as agent reward and never as a
pixel-level state estimator. Pre-training compute is tens of dollars on a rented 4090.

**Verdict:** evolution-outer/learning-inner is the right shape; the real risks are ecological
collapse and observer boredom, not optimisation.

---

## 9. Tooling and hardware

Notes: [`research/tooling-and-hardware.md`](research/tooling-and-hardware.md)

**Established.** JAX gridworlds reach 10⁵–10⁷ steps/s on one GPU when thousands of environments
are vmapped (Craftax 267k steps/s on a 4090, 1B steps < 1 h,
[arXiv:2402.16801](https://arxiv.org/abs/2402.16801); XLand-MiniGrid ~0.8M PPO steps/s on a
4090, [arXiv:2312.12044](https://arxiv.org/abs/2312.12044)); at batch 1, JAX is often *slower*
than the original Python env ([JaxMARL](https://arxiv.org/abs/2311.10090)), and no CPU-only
numbers are published. PufferLib's compiled-env pattern (C header + binding) is the closest
template for driving a Rust sim from Python ([PufferLib docs](https://puffer.ai/docs.html));
PyO3 precedents exist but are small (rlox, rlgym-sim-rs). On the Rust side as of October 2026:
`ort` 2.0.0-rc.14/15 (ONNX Runtime 1.31, "production-ready", has on-device training) and
`tract` 0.23.9 (pure Rust, WASM, long-standing GRU/LSTM ops) for inference; `burn` 0.22
(autodiff, CPU JIT, safetensors loading) for training in Rust; `candle` has backprop and GRU but
is inference-focused by its own statement; `dfdx` is pre-alpha. PyTorch GRU ONNX export has
well-documented pitfalls; the robust pattern is to export a single-step cell with hidden state as
explicit I/O. Cloud GPU: RTX 4090 ≈ $0.33–0.89/h, A100 80GB ≈ $1.19–2.79/h, H100 ≈ $1.99–4.29/h
(October 2026, links and dates in the notes).

**Estimate.** 60 sheep × 5 Hz of a ~2M-parameter CNN+GRU ≈ 6 GFLOP/s, under 5% of either i7;
one SGD step on 512 samples ≈ 30 GFLOP, so ≈5–10 steps/s on either x86 box. The Pi 5 could do
inference (~12% CPU) but not training (~0.6 s/step) and already runs the renderer. A PyO3-wrapped
Rust sim on the 13700KF reaches perhaps 10⁴–10⁵ agent-steps/s with learning — 10–100× below the
JAX-on-GPU regime for population-scale evolution.

**Judgement.** Keep the Rust sim as the single source of truth and the Pi as renderer only.
Offline training in Python; PyTorch + PyO3-wrapped batched sim by default, with a JAX port of the
meadow only if thousands-of-sheep evolution becomes central. Installation inference in Rust via
`ort` (ONNX) or `burn` (safetensors, if online learning in Rust is wanted). A half-day benchmark
spike (export a 2M-param cell, time batch-60 forward and a 512-sample train step on both CPUs)
should precede architecture lock-in.

**Verdict:** the installation side is comfortably feasible on either x86 box; the offline side is
cheap in dollars and expensive in developer time only if a second (JAX) meadow is built.

---

## 10. Gaps the literature does not fill (FLOCK would measure these itself)

These are collected from the "Gaps" sections of the notes. They are, in effect, the research
contributions FLOCK could make, and they shape what the milestones must test.

1. **Shared vs separate encoder** for image-shaped messages, on an iconicity metric. No ablation
   exists. (Milestone 0.)
2. **Legibility of learned marks viewed downsampled** (a 16×16 mark seen at 4×4 or 8×8 inside a
   larger view). Nothing published. (Milestone 0 / 4.)
3. **Pixel-categorical drawing heads** for communication. Untested; stroke/stamp rasterisers are
   what train. (Milestone 0 ablation.)
4. **Persistent vs transient messages** for learning speed and protocol quality. No controlled
   comparison. (Milestone 4.)
5. **Two channels of different cost specialising** (cheap sound vs costly drawing). No MARL
   precedent. (Milestone 4.)
6. **Learned acoustic parameters** (pitch, duration, timbre) in RL. Nothing published.
   (Milestone 3.)
7. **Emergent alarm calls** in predator-prey RL. Not demonstrated anywhere. (Stretch goal.)
8. **Per-individual trust tables keyed by a visual tag** in learning agents. No paper.
   (Milestone 5.)
9. **Neural iterated learning over a pixel/sketch channel across generations.** Open ground.
   (Milestone 6.)
10. **Months-long wall-clock runs** of learning populations; fear-extinction rates;
    injury-vs-death comparisons. No data. (Milestone 7 soak test.)
11. **CPU throughput** of ~1M-param RSSMs at 16×16 in any framework. Must be benchmarked.
    (Pre-integration spike.)
12. **An FM steering a camera in an ALife installation.** No precedent. (Milestone 7.)

---

## 11. Cross-cutting conclusions

Three things come up in almost every topic and should be treated as design principles:

- **The default is nothing happens.** Communication does not emerge without auxiliary pressure;
  flocking does not emerge unless the capture rule rewards it; imitation does not happen unless
  the expert is informative and intermittently absent; populations collapse or stagnate unless
  railed. Every milestone needs an explicit mechanism for its headline behaviour and a metric that
  can distinguish "emerged" from "looks like it emerged".
- **Offline is where learning happens; online is where it is maintained.** The hardware can run
  the flock and adapt it slowly, but the published sample complexities (train ratios of 32–512,
  10⁴–10⁵ referential-game interactions, thousands of fox encounters) are only reachable in
  vectorised offline simulation. The exhibition should open with an already-competent,
  already-speaking generation N, not generation 0.
- **Separate what evolves from what is learned from what is taught.** Genome: priors, drives,
  traits. Lifetime learning: world model fine-tuning, social memory, fox response (frozen
  live). Culture: the language, through a bottleneck. Collapsing these channels (Lamarckian copying
  of trained language weights, for instance) removes the pressure that produces structure.
