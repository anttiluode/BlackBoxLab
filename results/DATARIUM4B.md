# Datarium 4B receipt — the builders make the thing that leaves

Date: 2026-08-31

Datarium 4A proved a deliberately seeded mechanics bridge: if a diffuse closed
phase already exists, local soluble chemistry can polarize it, local boundary
stress can drive a solvent, and the whole phase can translate without an
object-level steering command.

Datarium 4B removes the explicit disk.

The question is narrower than "did trains make life?":

> Can Datarium-3-style builders write a local material state that nucleates a
> second movable compartment phase, after which the builders and fast wave can
> be removed and the phase still retains soluble state and moves under the same
> local body physics?

The first CI-sized answer is **yes for builder-written oriented material as a
phase nucleator**, with two important boundaries:

1. intact spatial organization matters more than the same local material state
   scrambled across space or reduced to a mean field;
2. full train/wave feedback is **not yet necessary**: WRITE ONLY and NO WAVE
   PRODUCTION still make smaller movable phases.

So this is a genuine removal of the authored body seed, but not yet a proof
that the train feedback loop specifically caused the body transition.

## Generative chain

~~~text
identical Datarium-3 builders
        ↓
fast local wave + motion
        ↓
slow material amount m(x,y)
+ slow director q(x,y)
        ↓
fixed pointwise catalyst
        ↓
soluble body precursor → diffuse phase phi
        ↓
builders deleted
fast wave zeroed
scaffold no longer acts
        ↓
source-free interface settling
        ↓
external soluble food field
        ↓
local chemistry → local stress → solvent → phase advection
~~~

There is no `Cell` class, body heading, center-of-mass force, parent ID,
fitness, reproduction instruction, loop detector, component ID, target shape,
or run-normalized threshold in the phase-creation physics.

The observer measures connected components and displacement only after the
local field updates.

## Why orientation is in the handoff

Datarium 3 already showed that scalar material amount was not the whole memory:
removing its direction nearly erased the replacement-population effect.

Datarium 4B therefore does **not** throw that earned result away. The local
second-phase catalyst requires both:

~~~text
absolute material amount m(x,y)
and
absolute local director magnitude |q(x,y)|
~~~

Each passes through a fixed pointwise sigmoid with absolute thresholds. The
gates are multiplied. Zero material or zero director gives zero catalytic
handoff.

No global mean, maximum, component shape or lineage variable is consulted.

## A shortcut caught during development

The first Datarium 4B draft appeared to move after builder removal, but the
NO STRESS and UNIFORM FOOD attackers moved almost the same amount.

That was not chemotaxis. The irregular newly formed phase was passively
coarsening and its center of mass shifted while the interface relaxed.

The corrected assay therefore inserts a **source-free settling interval**:

1. builders and scaffold influence stop;
2. the exact same phase equation runs with zero active stress and uniform food;
3. only after that interval is the directional food field placed;
4. ACTIVE, NO STRESS, PINNED and UNIFORM FOOD start from the settled intact
   phase.

After this correction, passive motion drops near zero while the active phase
still translates.

That attacker is part of the result, not an implementation footnote.

## First CI-sized receipt

This is not yet the long four-seed receipt.

~~~text
field                         40 x 40
builders                           52
matched seeds                        2
builder steps                      900
phase maturation steps             150
source-free settling steps         260
movement steps                     320
cargo steps                        260
~~~

The table reports means across the two matched seeds.

| scaffold arm | scaffold mean | director mean | final phase mass | largest thresholded component / thresholded phase | selective cargo retained | open-boundary cargo retained | movement toward food |
|---|---:|---:|---:|---:|---:|---:|---:|
| **INTACT** | **0.0028** | **0.0025** | **27.72** | **1.000** | **0.739** | 0.451 | **2.096** |
| SCRAMBLED local `(m,q1,q2)` triplets | 0.0028 | 0.0025 | 19.54 | **0.000** | 0.896 | 0.891 | 0.140 |
| MEAN FIELD scalar only | 0.0028 | 0.0000 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 |
| WRITE ONLY | 0.0022 | 0.0018 | 15.96 | 0.500 | 0.708 | 0.536 | 2.923 |
| NO WAVE PRODUCTION | 0.0011 | 0.0010 | 11.87 | 0.500 | 0.704 | 0.535 | 2.363 |
| ERASED | 0.0000 | 0.0000 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 |

A warning about the retention column: when there is no thresholded connected
body, as in SCRAMBLED, a high "retained" number can come from diffuse
low-amplitude phase spread rather than a meaningful compartment. That is why
the connected-phase and open-boundary controls must be read together rather
than treating retention as a standalone score.

## Exact intact-body movement attackers

These use the **same settled intact builder-made phase**:

| movement arm | displacement toward food |
|---|---:|
| **ACTIVE CHEMISTRY + STRESS** | **2.096** |
| NO STRESS | -0.171 |
| PINNED PHASE | ~0 |
| UNIFORM FOOD | -0.219 |

The small negative residuals are ordinary post-settling numerical/interface
drift. They are an order of magnitude below the active directional motion.

This closes the shortcut from the first draft: the intact phase's movement is
not just passive coarsening, and an external spatial asymmetry plus active
local stress are both load-bearing.

## What the geometry attackers say

### SCRAMBLED

SCRAMBLED uses one permutation for the complete local material triplet:

~~~text
(m, q1, q2)
~~~

so it preserves the scalar histogram, the director histogram and the local
relationship between amount and direction. It destroys only where those local
states sit relative to one another.

In this CI regime it still makes diffuse second-phase material, but **no cell
crosses the phase threshold into a connected measured body** after settling.

That is evidence that spatial arrangement contributes something beyond a bag
of local material values.

### MEAN FIELD

MEAN FIELD preserves only the global scalar material amount and deletes local
direction.

It makes no body phase.

This is the expected hard attacker for spatial addressability.

## The awkward useful result

WRITE ONLY and NO WAVE PRODUCTION still make smaller connected phases, and
those phases can move after the same source-free settling correction.

This prevents the stronger statement:

> "Only train-written material can make a body."

The current system has not earned that.

The more defensible statement is:

> **Builder-written oriented material can locally nucleate a second movable
> compartment phase without a pre-drawn body. Intact spatial organization
> produces a larger, fully connected thresholded phase than matched scrambled
> or mean-field material, but simplified builder histories can still produce
> smaller movable phases.**

That boundary matters for the next experiment. We should not tune the scalar
threshold merely to kill the surviving attackers.

Instead the next rung should ask for a function that the intact oriented
history can perform and the simplified bodies cannot.

## What comes next

The natural next layer is **internal structure carried by the body**.

Datarium 4B currently lets the scaffold *catalyse* the body phase, then the
scaffold stops acting. The moving phase therefore leaves mostly as a chemical
compartment, not as a portable copy of the oriented infrastructure that made
it.

The next gate should locally bind a fraction of the oriented material to the
new phase so that it advects with the body:

~~~text
builder-written q field
        ↓ local capture at phase formation
mobile internal fibre field
        ↓
changes permeability / conductivity / wave propagation
        ↓
body behavior depends on internal arrangement
~~~

Then the decisive assays become:

- intact internal fibres vs rotated/scrambled fibres inside the same body;
- same body chemistry, different internal transfer function;
- physical fission partitions those fibres;
- daughters are tested for inherited behavior without `copyGenome()`.

That is where "structure inside the cell" can begin to become computation and
later heredity rather than decoration.

## Run

~~~bash
python -m unittest tests.test_datarium4 tests.test_datarium4b -v
python experiments/datarium4b_builder_body.py --preset ci
python experiments/datarium4b_builder_body.py --preset receipt
~~~

Open the live assay:

https://anttiluode.github.io/BlackBoxLab/datarium4b.html

The live browser is qualitative and interactive. The Python experiment is the
authoritative receipt.

## Stopping line

> **Datarium 4B removes the explicit body seed. A Datarium-3-style oriented
> material history can nucleate a second phase that survives builder removal,
> retains soluble state through its interface, and moves under local chemistry
> and stress. Spatial arrangement matters. But full train/wave feedback is not
> yet necessary, and nothing here reproduces, inherits an internal program or
> constitutes a species.**
