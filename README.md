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


## Experiment 1 — stigmergic continuous field

Experiment 0 used an explicit crowding term. Experiment 1 removes it and
also removes the four named action channels.

Four latent temporal processes are smoothly mixed across a 64-position ring.
Every learner starts at the same position with the same weights, values, and
learning rule. A learner chooses only a nearby position and receives one scalar
sample.

The three arms are:

~~~text
NO_TRACE
    sampling leaves no environmental memory

PRIVATE_TRACE
    sampling leaves a local refractory trace
    visible only to that same learner

SHARED_TRACE
    sampling leaves the trace in one shared field
    so later learners encounter changed evidence
~~~

No policy receives an occupancy count, source identity, or diversity reward.
Latent source labels are used only after the run to score whether different
learners settled into different parts of the mixed field.

The shared trace is deliberately primitive: repeated sampling makes a local
patch temporarily less faithful / predictable. It is the smallest test of
physical stigmergy, not a biological mechanism and not a reproduction of
SwarmWorld's artifact ecology.

### Experiment 1 receipt: mostly a negative result

Across 24 matched seeds:

| condition | source coverage / 4 | computation divergence | late prediction MSE | full 4/4 |
|---|---:|---:|---:|---:|
| NO_TRACE | 1.79 ± 0.71 | 0.028 ± 0.010 | **0.04 ± 0.03** | 0/24 |
| PRIVATE_TRACE | 3.62 ± 0.48 | **0.922 ± 0.087** | 1.10 ± 0.02 | 15/24 |
| SHARED_TRACE | **3.75 ± 0.43** | 0.917 ± 0.091 | 1.10 ± 0.02 | **18/24** |

The shared field adds only a modest spreading effect beyond private
self-avoidance, while both trace conditions destroy predictive quality. The
mechanism is therefore best understood as **depletion / refractory
self-avoidance**, not useful stigmergic computation.

See [the full Experiment 1 receipt](results/EXPERIMENT1.md).

Open [the live Experiment 1 field](stigmergy.html), and see
[SWARMWORLD_NOTE.md](SWARMWORLD_NOTE.md) for the neighboring paper and the
limits of the analogy.

The next stronger version, if this survives, is to replace the scalar
refractory trace with a **persistent local operator** so that what passed
through a region changes how future signals are mixed or filtered there.

## Datarium 1 — lineage microscope

The instanton-descended **Three Fates** field gives this repo a second kind of
substrate: a two-dimensional periodic phi^4 wave field whose local resource
`r(x,y)` is depleted by activity, recovers slowly, and gates growth.

The LOCAL BUDGET arm is interesting because it produces continuing domain
turnover rather than the extinction of NO BUDGET or the winner-take-all
fixation of GLOBAL BUDGET. But the old observer was not good enough for
genealogy: ordinary connected-component labels cut the periodic seam, one hard
threshold turns breathing boundaries into births/deaths, and tiny threshold
fragments count alongside substantial domains.

Datarium 1 changes the **observer**, not the physics:

~~~text
same local-budget field
        ↓
high / low threshold hysteresis
        ↓
4-connected components ON THE TORUS
        ↓
minimum area + positive-mass floor
        ↓
frame-to-frame overlap graph
        ↓
continuation / split / merge / birth / death
~~~

A one-to-one overlap keeps the same persistent ID. A split terminates the
parent ID and creates children that record that parent. A merge terminates all
parents and creates a new child. No trait is copied, no fitness is assigned,
and no genome exists.

The instrument also records simple physical phenotype coordinates — positive
mass, area, compactness and eccentricity — so a later Datarium can ask whether
sisters resemble one another more than unrelated domains **before** adding an
external heredity mechanism.

Open **[Datarium 1 live](datarium1.html)**. Click a measured domain to inspect
its identity and ancestry.

Python receipt:

~~~bash
python experiments/datarium1_lineage.py
~~~

The reusable tracker lives in `datarium/lineage.py`.

### Datarium 1 receipt

The corrected observer changes the picture substantially:

~~~text
late substantial domains        4.26 mean, 13 max
dust rejected                   1.33 components / measurement
frames containing dust          47.9%

individual identity
median lifetime                 1.60 time
p90 lifetime                    5.20
max lifetime                   24.40

genealogical web
median ancestry depth             21
max ancestry depth                120
max ancestral span             236.80
~~~

So the bodies are mostly short-lived, but explicit split/merge ancestry can be
long-lived. That is **not heredity**. It means the next test can finally ask
whether parent and child domains resemble one another more than unrelated
mass-matched domains.

See [the full Datarium 1 receipt](results/DATARIUM1.md).

The stopping line is explicit:

> **A lineage graph is not evolution. Datarium 1 earns only the right to ask
> whether physical phenotype persists through field-generated demographic
> events.**

## Datarium 2 — ecological thinkers

Datarium 1 established that the local-budget field has short-lived bodies but
long split/merge ancestry. Datarium 2 attaches a **tiny digital recurrent
controller to that measured lineage**.

Each active lineage body gets one bound scout. The scout senses local field
value and velocity, gradient, resource, local/global homeostatic error, body
mass, scout/body separation and one shared scalar signal. The 6-state recurrent
matrix produces five bounded actions:

~~~text
steer x
steer y
excite field locally
damp field locally
emit social signal
~~~

The field still owns all demographic events.

~~~text
birth   → random program
split   → copy + mutation
merge   → higher-scoring parent program wins
death   → field terminates the body
~~~

This is deliberately a **hybrid artificial-life model**. The matrix is not
claimed to arise from the field.

### First receipt

The short two-seed CI comparison is negative on adaptation:

| mode | stability | active domains | controller divergence | behavior divergence |
|---|---:|---:|---:|---:|
| NONE | 0.477 | 15.21 | 0.000 | 0.148 |
| HOMEOSTAT | 0.476 | 8.11 | 0.000 | 0.225 |
| RANDOM | 0.477 | 14.56 | 0.378 | 0.141 |
| EVOLVE | 0.477 | 14.49 | 0.413 | 0.167 |

EVOLVE produces inherited, behaviorally different programs but does **not**
yet improve the shared environment over RANDOM or NONE. HOMEOSTAT changes
demography strongly but also fails the fixed-energy stability objective.

See [the full Datarium 2 receipt](results/DATARIUM2.md).

Open **[Datarium 2 live](datarium2.html)**.

### Leave it running overnight

On Windows:

~~~text
run_datarium2_overnight.bat
~~~

Or directly:

~~~bash
python experiments/datarium2_thinkers.py --preset overnight --mode evolve --seed 7
~~~

The default overnight run uses a 96 x 96 field for 3000 simulated time units
and writes forensic controller/scout checkpoints every 100 time units under
`results/datarium2_overnight_seed7/`.

The overnight run is an observation experiment, not a promised optimizer.
Interesting evidence would be persistent separation from the RANDOM attacker
in recovery, environmental persistence, lineage success, relationships or
behavioral niches.

## Datarium 3 — when motion hardens into matter

The attached historical `train.html` had a powerful visual loop but an impure
emergence claim: five authored writing roles, type-specific forces and rewards,
a global fitness sort, crossover and mutation were already in the source.

Datarium 3 keeps only the local field idea and starts again:

~~~text
identical phase oscillators
        ↓ write / read
fast two-quadrature wave
        ↓
transient co-moving assemblies (observer only)
        ↓ local coherent wave-following reaction
precursor becomes slow oriented matter
        ↓
matter changes later wave propagation and motion
~~~

There are no particle types, roles, links, genomes, fitness scores,
reproduction rules or `Train` objects. The assembly observer is post-hoc and is
never visible to the physics.

### Four-seed receipt

| condition | late coherent fraction | largest measured assembly | maximum persistence |
|---|---:|---:|---:|
| wave only | 0.127 ± 0.022 | 3.91 ± 0.19 | 42.8 ± 5.8 samples |
| write-only material | 0.127 ± 0.022 | 3.91 ± 0.19 | 42.8 ± 5.8 |
| scalar material | 0.117 ± 0.015 | 3.87 ± 0.33 | 30.2 ± 2.9 |
| mean-field material | 0.111 ± 0.024 | 3.76 ± 0.39 | 27.5 ± 7.2 |
| **oriented material** | **0.530 ± 0.071** | **13.43 ± 4.44** | **94.0 ± 15.1** |

The decisive intervention snapshots the slow material, deletes every builder,
zeros the fast wave and releases the same naïve population. Intact material
produces **0.525 ± 0.072** fresh-population coherence versus **0.144 ± 0.052**
after erasure. Removing direction while preserving scalar material gives
0.123; shuffling intact local patches gives 0.392; cell-scrambling the same
histogram gives 0.211.

So history now survives outside the agents and has causal influence at a
slower material scale. The stopping line matters:

> **Datarium 3 has made infrastructure, not DNA. It does not copy a material
> pattern into a daughter pattern.**

Open **[[Datarium 3 live](datarium3.html)](https://anttiluode.github.io/BlackBoxLab/datarium3.html)** and press **REMOVE BUILDERS →
RELEASE STRANGERS** after the material has developed. See the
[full Datarium 3 receipt](results/DATARIUM3.md) and run:

~~~bash
python experiments/datarium3_layers.py --preset ci
~~~

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
