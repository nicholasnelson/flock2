# FLOCK 2.0 — research and plan

FLOCK is a generative-art installation: a wall-mounted TV showing tens of simulated sheep
living in a meadow, watched like an aquarium. FLOCK 2.0 re-imagines it so that sheep
behaviour, instincts, communication and relationships are genuinely *learned* rather than
borrowed from an LLM.

This folder holds the research, the staged implementation plan and the record of each
milestone as it is built. Milestone 0 (the toy drawing game) is closed; Milestone 1 (the meadow,
drives and world model) is in progress — start with [`HANDOFF.md`](HANDOFF.md) for where things
stand.

## Documents

| File | What it is |
| --- | --- |
| [`00-brief.md`](00-brief.md) | The working design brief for 2.0, as given, plus hardware and constraints. |
| [`01-research-survey.md`](01-research-survey.md) | Synthesis of the literature survey: what is established, what fits FLOCK, what is still open. |
| [`02-design-questions.md`](02-design-questions.md) | Recommendations for the seven open design questions, with reasoning and evidence. |
| [`03-implementation-plan.md`](03-implementation-plan.md) | Staged milestones, each watchable or testable on its own, starting with the Milestone 0 toy. |
| [`research/`](research/) | Per-topic research notes with citations. The survey above is built from these. |
| [`CURRENT-DESIGN.md`](CURRENT-DESIGN.md) | The design as it stands in the code, one page, with a dated decision log. Where other documents disagree, this one is current. |
| [`review-2026-10-24.md`](review-2026-10-24.md) | An outside review of progress at 12abf7d (reading only) and its recommendations. |
| [`HANDOFF.md`](HANDOFF.md) | Where things stand, what to read first, practical gotchas. Newest update at the top. |
| [`m0-toy-README.md`](m0-toy-README.md) | Milestone 0: the toy drawing game — design, every experiment grid, the result. |
| [`m1-meadow-README.md`](m1-meadow-README.md) | Milestone 1: the meadow, drives, world model and actor-critic — what was built, every run, the diagnosis, what is next. |
| [`m1-model.svg`](m1-model.svg) | Diagram of the M1 sheep brain: world model, heads, actor-critic, planned curiosity ensemble. |
| [`m1-results/`](m1-results/) | M1 run logs, evaluation numbers and figures cited in the M1 README. |

## Conventions

- Every claim taken from the literature carries a link (arXiv, DOI or official repository).
- Established results, estimates and speculation are labelled as such. Where a section is
  the author's own judgement rather than a cited result, it says so.
- v0.4 (the LLM-driven FLOCK) is not a constraint on 2.0. Its design document and code were not
  available when this was written, and at plan review its requirements (crate structure, Rust
  renderer on the Pi, persistence and camera designs) were discarded; 2.0's architecture is
  decided on its own merits (design question Q10 and the architecture questions in the plan).
