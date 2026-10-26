# Review (2026-10-26, evening): the proposed plan from here

Companion to `review-2026-10-26-water.md`. Read against `00-brief.md`, `03-implementation-plan.md`,
`CURRENT-DESIGN.md`, `m0-toy-README.md`, the rung-2 record at the end of `m1-world-v2-spec.md`,
and the proposed plan as the coordinator stated it. Reading only; nothing run. *(judgement)* marks
mine; *(established)* points at a verified link in § 8.

## Summary

1. The ordering of Track A is right in outline (diagnose → learning-only fixes → rung 3) but the
   plan commits to the fix list before the diagnosis has spoken. Make the diagnosis a gate with
   pre-written branches: the fixes that make sense if the loss discards far ponds (H1) are not
   the fixes that make sense if the state holds them in an unreadable form (H2). One branch is
   missing entirely: changes to the world that are facts of the world (night water visibility,
   pond size), which the owner's principle permits and the data point at.
2. The "policy reads the current encoding" arm is the cheapest discriminating experiment in the
   whole plan and should run first, with stop-gradient, as a control that answers "is the state
   or the readout the limit". It is not a cheat. Letting PPO's gradient into the encoder later
   is a different matter: it makes one encoder serve two losses and quietly turns the world model
   into a feature extractor for the policy; keep that as a named, separate arm if at all.
3. The one-week box is feasible only if the diagnosis takes two days and the fixes are screened
   on the probe (minutes) rather than the evaluation (hours of sheep-time per seed); rung 3 should
   start only if the diagnosis says the state, not the loss, is the limit. The criterion needs a
   measurement defined now: "holds water location" must name the probe (visible-near, ridge, by
   day) and the number; "shortly after looking away" must name the window (1–3 turns) and the
   number. Otherwise the box cannot fail.
4. Track B is worthwhile and cheap, but as designed it tests the machinery, not emergence in the
   meadow: the grid's top-down view already holds "where" for free, which is exactly what the
   meadow lacks. Design it so the signal enters as an observation (through the world model, like
   the felt numbers), the sender is visible so that *following* is the competing strategy, the
   gap is real (the listener cannot find water alone in time), and the baseline is a scripted
   beacon. Pre-register which interventions are "medium" and which are "encouragement" before
   any run; M0 needed nine, and none of them was planned.
5. Putting a minimal water channel before the fox is defensible, with two conditions. The
   plan's own argument (urgency needs a time-critical event) only concerns the sound-vs-marks
   specialisation test, which moves later rather than vanishing. But the fox is also the only
   pressure for cohesion in the plan, and a broadcast channel needs sheep within earshot: the
   hybrid sheep do not huddle (1–5 % of night rest) and nothing measures their dispersion.
   Measure it first; if they scatter, cohesion (fox or otherwise) comes before the channel.
6. Missing: a definition of "meaningful" that cannot be satisfied by "I am drinking, come here";
   the information-gap metric (planned in M1, never built) as the go/no-go for any channel; a
   guard against communication papering over perception (a channel can carry only what the
   sender's state holds, so if the state lacks "where", the emergent referent will be presence,
   not place); per-day death curves and time-to-first-drink as the water-finding measures; and
   the recognition that a view-reading fallback policy has no memory and therefore nothing to
   *tell*.

## 1. Track A: ordering and gates

### What to keep

The sequence diagnose → learning-only fixes → rung 3 respects the ladder and the principle. The
rung-2 result (cover direction 0.29/0.34 → 0.48/0.58, water 0.00/−0.15 → 0.06/−0.11, water at
mouth 0.94) says that longer training improves exactly what the loss already pays for and leaves
water where it was. That is consistent with H1 (the far pond earns too little loss) and does
not distinguish it from H2 (unreadable code). The pre-registered rule says "rung 3"; I would
read it as "rung 3 is not yet justified, because rung 2 did not test the thing rung 3 changes".

### What to change

1. **Run the diagnosis as a gate with written branches**, two days at most, GPU minutes each:
   - D1 (decode-from-posterior water recall and azimuth by distance bin and day/night);
   - D2/D3 (probe with the renderer's own visibility mask, distance bins, image-column target,
     ridge headline, early-stopped MLP; embedding vs h vs z);
   - the policy-reads-encoding arm (below), one seed, 600 iterations, probed the same way.
   Branches, written before running:
   - *Decoder draws ponds to ~8 tiles by day and the embedding reads direction, (h, z) does not*
     → readout/channel: column-preserving embedding, larger z, weaker representation loss,
     policy-reads-encoding. Rung 3 is justified.
   - *Decoder does not draw ponds beyond ~4–6 tiles, embedding does not read direction either*
     → the loss: multi-step prediction, rarity weighting, **world facts** (below). Rung 3 with the
     same loss would inherit the blind spot; do not start it until the decoder draws far ponds.
   - *Neither reads anything even near* → the model is undertrained or RGB mode is broken; check
     reconstructions by eye before anything else.
2. **Add the world-facts branch.** At L = 0.12 water's contrast with grass is 69× lower than by
   day, so no loss on the sheep's own senses can learn where water is at night; the spec already
   treats the renderer as "facts of the world, not of the sheep" (§ 8). Water that reflects the
   sky (brighter than grass at night, as real water is) and a slightly higher night floor are
   changes to the world, not to the learner, and they are the only change on the list that makes
   night-time water findable at all. Larger ponds and a third pond were already on the owner's
   table (spec § 10). Put these to the owner as a branch, not as a fallback after rung 3.
3. **Screen fixes on the probe, not the evaluation.** A 600-iteration run is 16 minutes; the
   probe is minutes more; the evaluation is longer and noisier (± 0.07–0.10). For screening, the
   primary readout should be the visible-near direction R² and D1's recall, with the evaluation
   reported but not deciding, exactly as the rung-2 rule did.
4. **KL_DYN → 1** is a one-line alignment with the verified DreamerV3 text *(established)* [1];
   fold it into the baseline rather than spend a seed on it.
5. **Replay.** Bigger and uniformly sampled is right; store one byte per pixel through a palette
   so memory stays flat (the spec § 5 plan). Without this, every other fix is trained on a
   sliding half-day window.
6. **Rung 3 only after the gate**, with the two learning-only additions from the earlier review
   (multi-step prediction; the pre-registered attention-shift probe) and the encoder-only test.
   Circular padding stays an owner decision.

### The policy-reads-encoding arm

The coordinator's answer to the owner is right on all four points: the (h, z)-only policy is
inherited from training in imagination; the hybrid does not need it; the encoding is the sheep's
own eyes; the risks are hiding the world model's weakness, no memory in the encoding, and the
encoding sharing the loss's blind spot. Two additions:

- **It is the cleanest test of H2.** If PPO on (h, z, e) finds water and PPO on (h, z) does not,
  the information is in the sheep's senses and the posterior/GRU drops or scrambles it; if it
  does not help, the encoder itself (and so the loss) is the limit. Run it first.
- **Keep the stop-gradient.** With PPO's gradient in the encoder, the encoder serves the policy's
  loss and the world model's; the world model's probes then measure a representation shaped by
  reward, which the ladder's question ("what does a sense-only model hold") can no longer be
  asked of. It is not a privileged-information violation, but it is a change of what the world
  model *is*. If it is ever run, name it as its own arm and do not pool its probes with the
  others.
- **For the end goal it matters which wins.** A view-reading policy survives but has nothing to
  tell: communication about where water *was* needs the recurrent state to hold it after the
  pond leaves the view. So "accepting a view-reading policy" is a survival fallback, not a route
  to the brief. Say that in the plan.

### The time box and criterion

- **Feasibility.** One week is enough for: diagnosis (2 days), one or two fixes on one seed then
  three (2 days: 600-iteration runs are 16 minutes, 2,000-iteration runs about an hour), and a
  rung-3 run if justified (2 days: the code exists and was tested on CPU; three seeds × 2,000
  iterations is one afternoon of GPU). It is not enough for all four fixes factorially plus
  rung 3 plus a world change. Order by the gate and stop when the box ends.
- **The criterion as stated is not yet measurable.** Propose: (a) ≤ 20 % deaths on `m2_eval.py`
  over 3 days (the § 14 "worked" level), **and** (b) on the D2 probe, ridge R² ≥ 0.5 for the
  direction to water on frames where the pond is visible by day with its edge within 8 tiles,
  **and** (c) R² ≥ 0.3 on the same target 1–3 turns after the pond has left the view (the
  memory half, measured on the scripted flock's turn-aways, which are frequent). Report (b)
  and (c) for the embedding, h and z separately so that a pass by the embedding alone is not
  mistaken for memory.
- **Deaths is a harsh primary.** Random starts give hydration as low as 0.2 and a 1.6-day budget
  from full; a sheep that needs a day to find water dies regardless of what it knows. Add
  time-to-first-drink and the per-day death curve as reported measures; they separate "cannot
  find water" from "found it too late".
- **The fallback choice** ("velocity input, or accept a view-reading policy"): commanded actions
  are already given at every step, and proprioception adds only the refused and jostled
  corrections, so the first rung-4 piece is a small step in information and unlikely to change
  the outcome on its own *(judgement)*. The second piece (shifting memory by motion) is the one
  that does work in the literature, where emergent spatial codes are trained with velocity
  input *(established)* [2]. If rung 3 fails, the owner should be offered pieces 1 and 2
  together as the next rung, with the reason.

## 2. Track B: the grid-world communication bench

### Worthwhile, with its scope stated

The grid hybrid works (−0.017 to −0.023, beats PPO on the true state), is CPU-cheap, and the
channel machinery (continuous parameters, a gate, distance noise, the metrics) is identical
whatever the world. Building it there is the right call. But the grid's 8 × 8 top-down view is a
map: the state holds "where" for free, which is precisely what the meadow's state lacks. So a
grid result shows that the channel, the losses and the metrics work; it says nothing about
whether meadow sheep will have anything to say. Write that in the pre-registration.

### Design so that results transfer

1. **The signal is an observation, not a feature.** Give the listener the received parameters
   through the world model's input, as the felt numbers enter (a linear layer added to the
   embedding), so the state must learn to carry it and PPO reads it from (h, z). Feeding the
   signal straight to the policy would be a second free channel and would not transfer to a
   meadow whose policy reads (h, z).
2. **The sender is visible.** Following is the simplest form of "where's water" communication and
   is probably what will emerge first in the meadow (the plan's own fallback already leans on
   it). In the grid, the listener must be able to see the sender, so that the *signal's* added
   value can be measured against following. The ablation that matters is: learned signal vs
   signal replaced by the sender's position and heading only.
3. **The gap is real.** Partial views such that some sheep see water and others do not; water far
   enough that a listener who ignores the signal usually dies of thirst or pays heavily. Without
   this there is no pressure; with too much, the sender has no reason to stay. Measure the
   information-gap metric here first (it is cheap on a grid) and tune the world until the gap is
   frequent and consequential.
4. **Frames of reference.** "Water is to my left" is useless to a listener who does not know the
   sender's heading. In the grid, decide whether the signal can carry an allocentric direction
   (the grid has a fixed frame; the meadow has none that a sheep knows) or whether the listener
   must read the sender's facing. In the meadow a sheep sees others' facing as one of four
   billboards, so a direction in the sender's frame is learnable but hard; distance is
   frame-free. Expect the first meaningful content to be distance or presence, not direction.
5. **Baselines that can refute.** A scripted beacon (the sender emits a fixed signal while
   drinking) is the floor: if the learned channel does no better than the beacon on listener
   time-to-water, it carries presence only. A muted channel is the ceiling-of-nothing. Both
   belong in every table.
6. **Interventions, pre-registered.** M0 reached emergence only with nine channel interventions,
   each closing a cheat, and the owner's closing frame was "the medium, no encouragement". Decide
   now which items are medium (noise, cost, gate, a delay before the cost bites, re-centring)
   and which are encouragement (positive-signalling and positive-listening losses *(established
   as published inductive biases)* [3], social-influence reward *(established)* [4]). Run the
   medium-only cells first; the encouragement cells are a diagnostic arm, reported as such.
7. **Collapse and seeds.** Report the collapse rate across seeds from the first run (the survey
   notes ~25 % collapse for continuous channels in one study); the grid is cheap enough for
   eight seeds per cell.
8. **Metrics**: causal influence by muting and scrambling at test time, measured on the listener's
   *movement* and on its *outcomes* (time-to-water, deaths), not on its action distribution
   alone; context clustering of signal parameters on held-out worlds; and the Lowe check: a
   signal that predicts the sender's next action but does not change the listener is not
   communication *(established)* [5].

## 3. Communication before the fox

The plan's reason for fox-before-sound was that the sound-vs-marks specialisation hypothesis
needs a time-critical event. A minimal water channel does not test that hypothesis, so moving it
earlier does not break the argument; the specialisation test moves to after the fox, and the
plan should say so rather than leave M3/M4 implicitly reordered.

Two conditions *(judgement)*:

1. **Cohesion.** A broadcast channel needs listeners in range. The only cohesion pressures now
   are huddling (not learned: 1–5 % of night rest) and nothing by day; the fox was the plan's
   cohesion pressure. `Flock.seen` is recorded by the simulator but not reported by `m2_eval.py`.
   Add mean sheep-in-view and nearest-neighbour distance to the evaluation for every policy and
   look at the hybrid's numbers. If its sheep scatter, a meadow channel will fail for a reason
   that has nothing to do with communication, and some cohesion pressure (the fox, or a learned
   benefit of proximity) comes first after all.
2. **The information gap must exist and be measured.** The M1 plan item "visibility record and
   information-gap metric" was never built. It is CPU-cheap (who sees water, who needs it, per
   step) and it is the go/no-go for any channel: if every thirsty sheep can already see water,
   there is nothing to say; if the ones who know are always at the pond, the message is
   "I am here". Build it now, run it on the scripted flock and on the hybrid's evaluation, and
   report the gap before any channel is designed.

## 4. What is missing

1. **A definition of "meaningful" that presence cannot satisfy.** Proposed, for any channel in
   any world: (i) causal (muting or scrambling changes the listener's movement *and* outcomes);
   (ii) referential (parameters cluster by context on held-out worlds, with the clustering
   predictive of the referent, not of the sender's own next action); (iii) beats the scripted
   beacon and the follow-only ablation on listener time-to-water; (iv) survives the sender being
   out of the listener's view (otherwise it is following with extra steps). "Interesting" is a
   further bar (variety, context dependence, a human can hear the difference) that the brief
   sets; it should be reported, not required.
2. **The papering-over risk, stated as a test.** A channel can carry only what the sender's state
   holds. If Track A ends with a view-reading policy and a state that still lacks "where", the
   channel will converge on presence/urgency ("I am drinking", "I am thirsty"), which may be
   watchable but is not the brief's referential communication. The guard is the beacon baseline
   plus (iv) above; the honest report is "the sheep learned to call, not to describe".
3. **The M0 lesson, applied.** Nine interventions were needed, none planned; one seed in six
   still failed. Budget for the same: a list of candidate interventions with their status
   (medium / encouragement), a log of each one added and why, and a seed count that can show a
   failure rate. The plan's "positive signalling/listening losses" in Track B are already an
   encouragement item; decide their status before the first run, as above.
4. **Memory as a prerequisite for telling.** Make explicit in Track A's criterion (c) that the
   recurrent state, not the embedding, must hold water's position after the pond leaves the view.
   Without that the whole communication track rests on sheep that can only report what they
   currently see.
5. **Evaluation detail.** Per-day death curve; time-to-first-drink; `seen` and nearest-neighbour
   distance; and for the probe, the visibility mask from the renderer. All cheap, all missing.
6. **The owner's principle, applied to Track B.** The grid's top-down view, the signal as a
   policy input, and auxiliary losses on the message are each a form of free information or
   encouragement. None is forbidden on a test bench, but the pre-registration should say which
   ones the meadow version will not have, so that the transfer question is asked honestly.

## 5. Recommended order (concrete)

1. Day 1–2: D1, D2/D3, the policy-reads-encoding arm (stop-grad), `seen`/NN-distance in the
   evaluation, the information-gap metric on scripted and hybrid runs. Decide the branch.
2. Day 3–4: the branch's fix(es), one seed each at 600 iterations, probe-screened; the best at
   three seeds × 2,000. Baseline includes KL_DYN = 1 and the larger uniform replay. If the branch
   is "loss", put the world-facts change (night water) to the owner in parallel.
3. Day 5–7: rung 3 only if the gate said "state, not loss", with multi-step prediction and the
   attention-shift probe; otherwise spend the days on the loss branch.
4. Track B in parallel on CPU from day 1, medium-only cells first, beacon and follow-only
   baselines, eight seeds, information gap measured before the channel is added.
5. After the box: the meadow channel only if (a) Track A's criterion (c) passed, (b) the hybrid's
   sheep are within earshot, (c) the gap metric says there is something to tell. If any fails,
   the fox (cohesion) or the memory problem comes first, and the plan says which.

## 6. On the rung-2 decision specifically

The rule was followed correctly and its outcome ("rung 3") should stand as the ladder's record.
But the rule compared against a measurement capped well below 1 (visibility, nearest-tile noise,
night), and rung 2 changed training length only, not the loss or the channel, so it did not test
what rung 3 changes. My recommendation is to record "rung 2 failed its rule; rung 3 deferred
pending D1/D2, which decide whether the loss or the state is the limit", and to re-run the
rung-2 probe through D2 before the three-seed decision. That costs minutes and keeps the ladder
honest.

## 7. Answers to the questions, in one line each

- *Ordering right?* Yes in outline; make the diagnosis a gate with written branches, add the
  world-facts branch, start rung 3 only on the "state, not loss" branch.
- *One week and its criterion?* Feasible if screened on the probe; the criterion needs its
  measurements named now (visible-near ridge ≥ 0.5 by day; ≥ 0.3 one to three turns after
  looking away; embedding/h/z reported separately) and softer water-finding measures added.
- *Track B premature?* No, if scoped as a machinery bench with the signal as an observation, a
  visible sender, a real gap, beacon and follow-only baselines, and pre-registered interventions.
- *Communication before fox?* Acceptable for a minimal water channel, provided dispersion and the
  information gap are measured first; the specialisation test moves later explicitly.
- *Missing?* A presence-proof definition of "meaningful"; the gap metric; the papering-over test;
  the M0 intervention ledger; memory as prerequisite; the evaluation details.

## 8. References (each link fetched and checked on 2026-10-26)

1. Hafner et al., DreamerV3, https://arxiv.org/html/2301.04104v2 : "loss weights β_pred = 1,
   β_dyn = 1, and β_rep = 0.1"; free bits of 1 nat.
2. Cueva and Wei, https://arxiv.org/abs/1803.07770 : grid-like codes emerge in RNNs trained for
   localisation with velocity inputs.
3. Eccles, Bachrach, Lever, Lazaridou, Graepel, "Biases for Emergent Communication in
   Multi-agent Reinforcement Learning", https://arxiv.org/abs/1912.05676 : inductive biases for
   positive signalling and positive listening.
4. Jaques et al., "Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement
   Learning", https://arxiv.org/abs/1810.08647 : counterfactual influence reward, computable in a
   decentralised way, "more meaningful learned communication protocols".
5. Lowe, Foerster, Boureau, Pineau, Dauphin, "On the Pitfalls of Measuring Emergent
   Communication", https://arxiv.org/abs/1903.05168 : agents can "appear to communicate" while
   messages have no effect on the receiver; metrics can mislead.
6. Lazaridou and Baroni, "Emergent Multi-Agent Communication in the Deep Learning Era",
   https://arxiv.org/abs/2006.02419 : survey (background only).
