# BlackBoxLab — sampling policy can become anatomy

**Experiment 0: Sampling-policy organogenesis**

A continuous world contains several simultaneously available temporal regimes. A population of initially identical local learners does **not** receive assigned roles. Each unit has only a small online predictor, one observation choice at each step, a short private history, and a learned value for where it is worth looking next.

> **Can different sampling histories make initially identical learners diverge into different computations?**

The attacker is just as important:

> **If every learner optimizes only its own easy prediction, do they all collapse onto the same comfortable slice of reality?**

This grew directly out of the Child line. There the policy controlling what gets observed could become self-sealing. Here the observation policy itself becomes a developmental variable.

~~~text
continuous world
      ↓
sampling policy
      ↓
private experience stream
      ↓
local predictive plasticity
      ↓
different internal filter
      ↓
different future sampling value
      ↺
~~~

That feedback loop can break symmetry. But it may also create monoculture.

## First receipt

With identical predictor weights and identical sampling priors, stochastic
experience alone produced strong PRIVATE specialists, but the eight-organ
population covered only **2.417 ± 0.640 of four ecologies** and achieved full
4/4 coverage in **0/24** seeds.

Adding only the finite-territory crowding term produced **4/4 coverage in
24/24 seeds** and increased pairwise computation divergence from
**0.081 ± 0.033** to **0.182 ± 0.013**, with comparable prediction error.

See [the full Experiment 0 receipt](results/EXPERIMENT0.md).


## World

The toy world exposes four spatial "ecologies" at every physical timestep:

1. a fast modulated oscillator;
2. a slow modulated oscillator;
3. a persistent stochastic process;
4. a sparse burst-and-decay process.

All four exist simultaneously. An organ observes only the ecology it samples on that step.

Each organ is the **same algorithm** at birth: an online autoregressor using lags 1, 2, 4 and 8 over its own sampled stream. Its coefficients are not preassigned. Sampling different temporal statistics changes those coefficients, producing a measurable computational fingerprint.

## Three conditions

### YOKED

Every organ receives the same ecology.

This is the symmetry control. Identical data should produce near-identical computation.

### PRIVATE

Every organ learns a private sampling value from its own prediction success and occasionally explores.

There is no population-level diversity objective. This tests whether historical accidents alone are enough to produce persistent niches.

### ECOLOGY

Same learner, plus a small instantaneous crowding cost when several organs try to sample the same ecology on the same step.

The cost does not tell an organ *which role to become*. It only makes observation territory finite. This is the analogue of limited receptors, wiring, metabolic territory, or competition for an input stream.

If this condition differentiates strongly while PRIVATE collapses, the lesson is not "sampling magically creates organs." It is narrower:

> **sampling × plasticity supplies the positive feedback; finite observation ecology supplies the symmetry-breaking pressure.**

## What to measure

The experiment reports:

- **coverage** — how many of the four ecologies acquire at least one dominant organ;
- **specialization** — one minus normalized entropy of each organ's late sampling distribution;
- **computation divergence** — pairwise cosine distance between learned lag-weight vectors;
- **prediction error** — because diversity that cannot compute anything is not interesting;
- the dominant ecology and lag fingerprint of every organ.

The browser demo exposes the mechanism live. The Python receipt runs matched seeds and prints the aggregate comparison.

## Run

~~~bash
python experiments/sampling_policy_organogenesis.py
~~~

Open index.html directly, or publish the repository root with GitHub Pages.

No dependencies beyond NumPy for the Python receipt. The page is dependency-free.

## What would count

Interesting:

~~~text
same starting units
+ different sampled histories
→ stable different sampling niches
→ stable different lag computations
~~~

More interesting:

~~~text
PRIVATE partially differentiates
but often self-seals / collapses

ECOLOGY restores broad coverage
without assigning semantic roles
~~~

Dead / ordinary outcomes:

- every organ converges to the same easiest ecology;
- fingerprints remain effectively identical despite different samples;
- random fixed assignment produces the same result more cheaply;
- explicit mixture-of-experts routing dominates the whole mechanism;
- diversity appears only because the code secretly hard-wires roles.

## Why this matters to the repo line

Child reached the problem from epistemics:

~~~text
belief
→ decides what to observe
→ observations reinforce belief
→ memory preserves selected evidence
→ future learner inherits the resulting world
~~~

BlackBoxLab turns the same loop inward:

~~~text
local body
→ decides what to sample
→ sample statistics train local body
→ changed body values different samples
→ developmental trajectory diverges
~~~

In a brain-shaped interpretation, two initially similar pieces of tissue need not be assigned "vision" and "timing" in a lookup table. Persistent differences in what reaches them, what they can successfully predict, and which inputs remain available can gradually make them different machines.

That is the thing under test here. Not a brain model. Not cortical development. Not a new learning theorem.

Just:

> **sampling policy is allowed to become part of computation.**

## Related trail

- [Child](https://github.com/anttiluode/Child) — active observation, selective memory, provenance and self-sealing policies.
- [AlgoSchalgo](https://github.com/anttiluode/AlgoSchalgo) — ambiguity and active measurement.
- [Twensday](https://github.com/anttiluode/Twensday) — stateful local computation and experiment choice.
- [Monday](https://github.com/anttiluode/Monday) / [Tuesday](https://github.com/anttiluode/Tuesday) — temporal statistics as separating information.
