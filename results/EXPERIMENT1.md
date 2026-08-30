# Experiment 1 receipt — stigmergic continuous field

This experiment removed Experiment 0's four named sampling channels and its
explicit crowding penalty.

Four latent temporal processes were smoothly mixed across a 64-position ring.
Eight initially identical learners started at the same position and could move
only locally. Source identity existed only for world generation and post-hoc
scoring; no learner received source labels.

Three conditions were run for 5,000 steps across 24 matched seeds:

- **NO_TRACE** — sampling leaves the field unchanged.
- **PRIVATE_TRACE** — a learner degrades only its own future observation at a
  recently sampled patch.
- **SHARED_TRACE** — that same local refractory modification is persistent in
  one shared field and changes later agents' evidence.

| condition | source coverage / 4 | source specialization | computation divergence | late unique cells | spatial separation | late prediction MSE | full 4/4 coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| NO_TRACE | 1.79 ± 0.71 | 0.99 ± 0.01 | 0.028 ± 0.010 | 19.5 ± 4.5 | 0.18 ± 0.06 | **0.04 ± 0.03** | 0/24 |
| PRIVATE_TRACE | 3.62 ± 0.48 | 0.83 ± 0.09 | **0.922 ± 0.087** | 54.7 ± 8.1 | 0.50 ± 0.04 | 1.10 ± 0.02 | 15/24 |
| SHARED_TRACE | **3.75 ± 0.43** | 0.83 ± 0.08 | 0.917 ± 0.091 | **57.5 ± 6.4** | **0.52 ± 0.03** | 1.10 ± 0.02 | **18/24** |

## What actually happened

Removing source labels did not remove the Experiment-0 phenomenon. Without any
trace, local prediction competence becomes self-reinforcing and most agents
remain in a narrow part of the continuous field.

A refractory trace breaks that lock-in strongly.

But the crucial ablation is PRIVATE_TRACE versus SHARED_TRACE.

Private self-avoidance already produces almost the entire effect:

~~~text
source coverage
private  3.62
shared   3.75

computation divergence
private  0.922
shared   0.917

full coverage
private  15 / 24
shared   18 / 24
~~~

The shared environment adds a modest population-level spreading effect, but it
is not the main cause of differentiation.

Worse, the trace works by degrading sampled evidence. The late predictive MSE
rises from roughly 0.04 without traces to 1.10 with either trace.

## Earned statement

> **A persistent local environmental modification can slightly increase
> population-wide coverage beyond private self-avoidance, but this refractory
> trace does so by making evidence worse. It does not earn a useful
> stigmergic-computation claim.**

This is a useful negative result.

The experiment replaced an explicit crowding cost with a physical consequence,
but the consequence was merely depletion. That is too weak an analogue of
SwarmWorld, where persistent artifacts can create new capabilities and future
opportunities rather than only making occupied territory unattractive.

## Why the next trace must be constructive

A better test is:

~~~text
signal passes through region x
        ↓
a bounded local operator at x changes
        ↓
the operator persists after the learner leaves
        ↓
future signals / future learners encounter
a differently filtered or mixed stream
~~~

Then a trace can carry **computation**, not just occupancy history.

That creates a falsifiable bridge to the older persistent-body / matrix thread:

~~~text
local sub-operator
+ temporal passage
+ local plasticity
        ↓
persistent field of learned filters
~~~

The next experiment should ask whether those operator traces can reduce
redundant sampling or expose different components of a mixed signal without
being handed source identities.

## SwarmWorld constraint

SwarmWorld's useful lesson here is not "stigmergy spreads agents." Its stronger
mechanism is that successful work becomes part of a persistent environment
later agents can encounter and reuse.

Experiment 1 implements only the persistence half, not the constructive-reuse
half.

See [SWARMWORLD_NOTE.md](../SWARMWORLD_NOTE.md).

CI receipt: run 11, commit
66893c0795537cf0b701e8ec6d185bd621a982c1. Six smoke tests passed.
