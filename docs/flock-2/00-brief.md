# FLOCK 2.0 — working brief

This is the design brief as given at the start of the project, recorded here so the research
and plan have a fixed reference point. Everything in it is pending research and tuning.

## What FLOCK is

A generative-art installation: a wall-mounted TV showing tens of simulated sheep living in a
meadow, designed to be watched like an aquarium.

The previous design (v0.4) drove each sheep with a small LLM (Gemma 4 E2B via llama-server)
and was built as a Rust Cargo workspace: `flock-types`, `flock-sim`, `flock-brain`,
`flock-render` (WASM in a Chromium kiosk on a Raspberry Pi 5, plus an optional ESP32
secondary display). v0.4's pillar was "the appearance of inner life with the cheapest
machinery". Death was off in v0.4.

The v0.4 design document and code were not available during this planning phase. Only the
crate names and the facts above are known; anything else said about v0.4 in these documents
is provisional.

## Direction change for 2.0

Sheep behaviour, instincts, communication and relationships should be genuinely *learned*,
not borrowed from an LLM. Whether the LLM keeps any role (narration, nothing) is open.

## Hardware

- **Scarif**: Unraid homelab server, Intel i7-7820X (8C/16T, Skylake-X, AVX-512), no
  discrete GPU assumed.
- **Desktop**: Intel i7-13700KF (16C/24T). GPU presence not stated — treated as an open
  question.
- **Raspberry Pi 5**: rendering only.
- **Rented cloud GPU** time for one-off offline training is acceptable if justified.

## Working design

- **World.** A single shared 128×128 tile meadow. Grazing patches, pond, salt lick, treeline,
  day/night cycle.
- **Perception.** Each sheep sees an egocentric 16×16 pixel, 16-colour view of the meadow
  (roughly 1 pixel per tile). At night the effective view shrinks.
- **World understanding.** A small per-sheep world model (DreamerV3-style or simpler) trained
  on its own experience, with internal drives (hunger, warmth, social, fear) as reward plus
  curiosity from prediction error.
- **Visual communication.** A sheep can produce a 16×16, 16-colour image. One learned bit
  decides whether it is *said* (a transient speech bubble seen by nearby sheep) or *drawn*
  (stamped persistently onto the ground, where others later perceive it through their normal
  view). Drawing is slow and costly; it can carry more and it persists.
- **Sound.** Fast, cheap, transient bleats with a few learned parameters (pitch contour,
  duration, timbre), broadcast in a radius, no line of sight needed. Hypothesis: sound
  specialises for urgency, drawing for information.
- **Fox.** A predator that visits about once per night. Start scripted; possibly a learning
  agent later.
- **Key idea to test.** Process received drawings with the *same* visual encoder used for the
  world view, so iconic drawings (ones that look like the thing) are easier to understand
  than arbitrary codes.
- **Relationships.** Sheep learn models of specific individuals (who shares, who challenges,
  who lies), with reciprocity, alliances and grudges emerging from repeated interaction.
- **Generations.** Lambs inherit from parents (evolved priors, possibly mutated) and learn
  language and traditions by observing elders.

## Open questions to resolve

1. Mark scale — a 16×16 ground mark at 1 px per tile fills a sheep's whole view.
2. Action space for drawing — pixels, strokes/stamps, or a latent-vector decoder.
3. Death — what death or injury means with a fox and generations; avoiding flock collapse.
4. Fox learning — scripted vs learned; getting signal from a rare nightly event.
5. Offline vs online — what is pre-trained/evolved, what keeps learning live; feasibility.
6. The LLM's role — none, narration, or something else.
7. Theatre vs science — where the fully learned approach risks being dull, and mitigations.

## Ground rules

- Research findings and plan as Markdown in `docs/flock-2/`, with citations for every claim
  from the literature.
- Established results clearly separated from speculation.
- No production code before the plan is reviewed; the toy prototype is the first build step.
- Stop and ask rather than guess or quietly compromise.
