# FLOCK 2.0 — research and plan

FLOCK is a generative-art installation: a wall-mounted TV showing tens of simulated sheep
living in a meadow, watched like an aquarium. FLOCK 2.0 re-imagines it so that sheep
behaviour, instincts, communication and relationships are genuinely *learned* rather than
borrowed from an LLM.

This folder holds the pre-build research and the staged implementation plan. Nothing here
is production code; the first build step is the Milestone 0 toy prototype described in
[`03-implementation-plan.md`](03-implementation-plan.md), after the plan is reviewed.

## Documents

| File | What it is |
| --- | --- |
| [`00-brief.md`](00-brief.md) | The working design brief for 2.0, as given, plus hardware and constraints. |
| [`01-research-survey.md`](01-research-survey.md) | Synthesis of the literature survey: what is established, what fits FLOCK, what is still open. |
| [`02-design-questions.md`](02-design-questions.md) | Recommendations for the seven open design questions, with reasoning and evidence. |
| [`03-implementation-plan.md`](03-implementation-plan.md) | Staged milestones, each watchable or testable on its own, starting with the Milestone 0 toy. |
| [`research/`](research/) | Per-topic research notes with citations. The survey above is built from these. |

## Conventions

- Every claim taken from the literature carries a link (arXiv, DOI or official repository).
- Established results, estimates and speculation are labelled as such. Where a section is
  the author's own judgement rather than a cited result, it says so.
- v0.4 (the LLM-driven FLOCK) is referenced only as described in the 2.0 brief; its design
  document and code were not available when this was written, so statements about what
  carries over from v0.4 are provisional and flagged for review.
