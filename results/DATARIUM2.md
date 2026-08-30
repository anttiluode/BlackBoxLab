# Datarium 2 receipt — Ecological Thinkers

Date: 2026-08-30

Datarium 2 adds the first deliberately **hybrid cognitive lineage** to the
Datarium field.

The field still owns:

- bounded domains;
- continuation;
- split;
- merge;
- rearrangement;
- birth;
- death.

A digital recurrent controller is attached to the measured lineage:

~~~text
10 local / shared sensors
        ↓
6 recurrent hidden units
        ↓
5 bounded actions

steer x
steer y
local excitation
local damping
scalar social signal
~~~

The scout remains bound to the current field body but also feels local field
gradients. Its actions modify the same field that determines future demographic
events.

This is **not** field-native intelligence. The controller matrix is an explicit
digital evolutionary wrapper.

## Historical scout connection

The old scout / instanton experiments already contained the useful physical
vocabulary:

~~~text
field → scout force
scout → local field feedback
memory traces
coupling
later: stress / frequency adaptation and inter-scout influence
~~~

Datarium 2 does not inherit the old claims of "cognition" from those scripts.
It takes only the bidirectional coupling idea and puts it behind explicit
baselines.

## Evolution rule

The initial implementation accidentally treated every split, merge and
rearrangement as a new cognitive generation. In the high-churn field this
produced approximately 280 digital generations in only 130 field-time units:
mutation drift disguised as evolution.

That was fixed.

Current semantics:

~~~text
BIRTH
    random controller
    generation 0

SPLIT
    parent controller
        ↓ copy
        ↓ mutation
    generation + 1

MERGE / REARRANGEMENT
    compare parent mean selection scores
        ↓
    fitter parent controller wins inheritance

DEATH
    field ends that body / controller segment
~~~

The selection score combines local homeostasis, the common global homeostasis
state, and an action-cost penalty.

RANDOM destroys heredity by assigning a fresh random controller to every new
field body.

## Four-arm attacker

All modes see the same field and perturbation schedule.

| mode | meaning |
|---|---|
| NONE | untouched local-budget field |
| HOMEOSTAT | hand-written field stabilizer |
| RANDOM | tiny matrices act, but heredity is destroyed |
| EVOLVE | split mutation + merge selection |

HOMEOSTAT is important: if an ordinary rule can stabilize the field and EVOLVE
cannot, the action channel is useful but evolution has not discovered the
solution.

## CI receipt

2 matched seeds, 64 x 64 field, t=130, common perturbation schedule.

| mode | stability | mean abs energy error | active domains | true split generations | controller divergence | behavior divergence |
|---|---:|---:|---:|---:|---:|---:|
| NONE | 0.477 ± 0.000 | 0.797 ± 0.000 | 15.21 ± 0.76 | 97.5 ± 4.5* | 0.000 | 0.148 ± 0.012 |
| HOMEOSTAT | 0.476 ± 0.000 | 0.800 ± 0.000 | 8.11 ± 0.62 | 60.0 ± 15.0* | 0.000 | 0.225 ± 0.009 |
| RANDOM | 0.477 ± 0.000 | 0.798 ± 0.000 | 14.56 ± 0.42 | 96.0 ± 3.0 | 0.378 ± 0.014 | 0.141 ± 0.017 |
| **EVOLVE** | **0.477 ± 0.000** | **0.797 ± 0.000** | **14.49 ± 0.54** | **21.5 ± 2.5** | **0.413 ± 0.005** | **0.167 ± 0.027** |

* The generation field is bookkeeping-only in modes without a heritable
controller. It counts physical split depth for comparison and is not a cognitive
generation.

Dedicated Datarium-2 matrix tests: **5 / 5 PASS**.

CI run: GitHub Actions run 33326749117, datarium2 job.

## What happened

The first useful result is architectural:

> **A short-lived field body can now carry a controller through a much longer
> measured lineage. Split events mutate it; merges select among competing
> programs; scouts can influence the demographic substrate.**

The first adaptive result is negative.

EVOLVE does not beat RANDOM or NONE on the environmental stability metric in
this short run. The values are effectively identical.

The hand-coded HOMEOSTAT also fails the chosen global stability metric and
strongly reduces the number of active domains. That tells us two things:

1. the action channel has enough leverage to alter demography;
2. the current definition of "stabilize the environment" is not yet a good
   ecological objective for this non-stationary resource field.

The matrices do differentiate:

~~~text
active controller probe divergence
RANDOM       0.378
EVOLVE       0.413

measured behavior divergence
RANDOM       0.141
EVOLVE       0.167
~~~

But diversity is not adaptation.

No intelligence claim is earned from those numbers.

## Why this is still worth an overnight run

The corrected EVOLVE process accumulates only about twenty split-generations
over the short CI horizon. Selection happens mainly when field lineages merge,
so it is intentionally much slower than a conventional genetic algorithm.

The overnight preset therefore exists as an **observation experiment**, not a
promised optimization run.

It checkpoints the active controller matrices and scout state so long-lived
lineages can be examined afterward.

~~~bash
python experiments/datarium2_thinkers.py --preset overnight --mode evolve --seed 7
~~~

Default overnight settings:

~~~text
grid                 96 x 96
field time            3000
warmup                  30
perturbation interval   120
checkpoint directory
    results/datarium2_overnight_seed7/
~~~

## What to look for overnight

The run becomes interesting only if one or more of these begin separating from
the RANDOM attacker:

- perturbation recovery;
- domain / controller lineage persistence;
- selection-score concentration in a subset of controller ancestry;
- repeated parent-child behavioral phenotype;
- stable contact partnerships;
- controller action niches such as movement-heavy, damping-heavy, excitation-
  heavy or signaling-heavy behavior;
- improved environmental persistence without collapsing the domain population.

A beautiful lineage tree is not enough.

## Next scientific correction

The current common-good metric uses a fixed energy target calibrated during
warmup. The local-budget resource field is not stationary, so this may be the
wrong homeostatic objective.

The next correction should be made **before tuning controller parameters**:

> define environmental health from perturbation recovery, resource health and
> population persistence rather than forcing the field toward one old energy
> value.

That gives the system something ecologically meaningful to stabilize.

## Live aquarium

Open [Datarium 2](../datarium2.html).

The browser is intentionally a visualization. The Python experiment and CI
receipts are authoritative.

## Earned statement

> **Datarium 2 now contains field-generated demographic lineages carrying tiny
> heritable recurrent controllers, bidirectionally coupled scouts, mutation,
> merge selection, perturbations, social signaling and explicit attackers. In
> the first short experiment the controllers diversify but do not improve the
> environment. Evolutionary intelligence remains an open experiment.**
