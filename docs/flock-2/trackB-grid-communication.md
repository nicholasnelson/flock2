# Track B: communication machinery in the grid world (design, pre-registered 2026-10-26)

Status: **design; nothing built or run yet.** Agreed with the owner on 2026-10-26, following the
plan review (`review-2026-10-26-plan.md`, its point 5).

## Purpose

This is a bench for the communication machinery, **not a test of emergence**. The grid's top-down
view hands the sheep "where" for free, which the meadow lacks (spec § 15). The point is to have a
working, measured channel, its baselines and its metrics ready for when the meadow's sheep know
where water is. Results here say whether the machinery works, not whether meadow sheep will talk.

## The world (`flock/grid/flock.py`, new; the one-sheep grid is unchanged)

- **Layout:** a 24 × 24 grid (was 16), one pond, and **4 sheep** per world. Each sheep sees its own
  8 × 8 top-down window, so the pond is usually out of view, and **other sheep are visible** in it
  (a new cell class). Following is therefore always possible, and it is the strategy signalling has
  to beat.
- **Needs:** as the one-sheep grid (food and water). Rates are set so that a sheep that has lost the
  pond usually cannot find it alone before it is in trouble, but can if led: a real information
  gap, measured, not assumed.
- **The signal:** each step every sheep emits 2 numbers in [−1, 1]. A sheep hears the mean signal of
  the others within ``EARSHOT`` tiles (Chebyshev distance 6), plus how many it hears. Sound carries
  no direction: the listener learns where a caller is only by seeing it. Expect presence or
  distance to be encoded before direction; direction would need the sender's frame.
- **The sheep:** the hybrid that works in the grid (world model from the 8 × 8 view and felt
  numbers; PPO on its state). What is heard enters **through the world model**, as extra felt
  numbers, so the listener's model must learn what calls predict. The policy outputs a move (one of
  7) and a signal (a Gaussian with learned spread, squashed to [−1, 1]).

## Conditions (8 seeds each)

1. **Medium only:** the channel exists; no extra losses. Run first.
2. **No signal** (heard values set to 0): following alone.
3. **Beacon** (scripted senders): a sheep's signal is (1, 0) while it drinks and (0, 0) otherwise;
   listeners learn. This is the ceiling for "calling".
4. **Encouragement**, only if 1 fails: positive-signalling and positive-listening losses, each
   logged as an intervention.

## Measurements

- **Outcome:** reward per step, deaths, and time to water for thirsty sheep that cannot see the
  pond, compared across conditions.
- **Causal influence:** replace what each listener hears by what a listener in another world heard
  at the same step, and measure the change in its move distribution (mean KL) and in time to
  water.
- **Context:** how well the signal separates contexts. Mutual information (binned) between the
  signal and the sender's situation (drinking, thirsty, sees the pond, saw it in the last 50 steps).
- **Collapse:** the share of seeds whose signal variance across contexts falls below 0.01.
- **Pitfall check** (Lowe et al. 2019): a signal can carry information without being used. Report
  both what the signal says (context) and what it does (causal influence).

## Pre-registered criteria

- **The machinery works** if, in the medium-only condition, at least 5 of 8 seeds show causal
  influence above the no-signal baseline's noise (permutation test, p < 0.01) **and** a mean time to
  water for out-of-view thirsty sheep at least 20 % below the no-signal condition.
- **"Calling", not describing:** we expect the signal to mean roughly "I'm at water / come". That is
  to be reported as calling. "Describing" would need context information about something the
  sender is not doing (such as "water is far, behind").

## Intervention ledger

Every change made to get a result is logged in this file with the date, the reason, and the seeds
that failed before it. Milestone 0 needed nine unplanned interventions; the count is part of the
result.

| Date | Change | Kind (medium or encouragement) | Seeds failing before |
| --- | --- | --- | --- |
| | | | |
