# Datarium 5A receipt — field/substrate closure

Date: 2026-08-31

Datarium 4B removed the explicit body seed. Datarium 5A changes the question:
the body is held fixed so we can isolate what happens *inside* it.

The motivating loop is:

~~~text
fast internal field
        ↓
local interference / gradients
        ↓
fast activity trace
        ↓
slow oriented fibre tensor
        ↓
anisotropic propagation
        ↓
changed future field
~~~

This is deliberately a phenomenological field/material experiment. It is not a
model of entorhinal cortex, a neuron, an atom, or an ephaptic memory network.

## Why this gate exists

A simple "fractal-looking internal structure" would prove almost nothing.
Datarium 5 therefore does not optimize box-count dimension. It asks a causal
question first:

> Does fast activity write a slower morphology, and does the exact morphology
> measurably change the later field?

Only after that passes do we ask whether the morphology adds useful I/O
complexity.

## Body handoff

The body is still descended from Datarium 4B:

1. Datarium-3-style builders write slow oriented material;
2. the D4B local nucleation rule creates a diffuse body phase;
3. the phase is locally expanded for a fixed number of steps so that an
   internal-field assay has enough spatial area;
4. the body is then **pinned** for all Datarium 5 measurements.

There is no target body area and no center-directed growth rule. The expansion
is local reaction-diffusion, but the pinning is an explicit assay choice.

Two-seed CI-sized body summary:

~~~text
raw D4B phase mass        15.78
grown body mass          105.01
cells with phi > 0.5      93.0
~~~

The pinning is a stopping line: internal fibres do not yet travel with a moving
body.

## Internal field

Four local boundary patches are driven continuously at nearby frequencies.
Their waves superpose inside the body.

~~~text
port frequencies:
1.00, 1.07, 1.13, 1.21 × base
~~~

The word "moiré-like" is used only in the generic interference sense. No grid
cell or entorhinal mechanism is inserted.

The internal field has two quadratures. A fast local activity trace follows
field energy and a slower trace follows repeated activity. The slow trace
allows local fibre material to accumulate.

The director written at each point is the orientation of the local field
gradient tensor. Fibre amount and direction are therefore consequences of the
field history.

## The feedback

In FULL LOOP, the fibre director changes the directional second-derivative term
in the next field update.

So the code closes:

~~~text
field -> morphology -> field
~~~

rather than merely rendering fibre marks on top of an unchanged wave.

## Morphology attackers

After development, the morphology is frozen and the same probe inputs are
applied.

- **INTACT** — field-written amount + direction at their original addresses.
- **ISOTROPIC** — same fibre amount, direction deleted.
- **SCRAMBLED** — complete local fibre/director triplets are permuted *within
  the body support*.
- **ERASED** — no internal fibre.
- **WRITE ONLY** — morphology developed while fibre feedback to the field was
  disabled.

SCRAMBLED was corrected during development. An earlier draft shuffled over the
whole lattice and accidentally moved fibre into extracellular space. The final
attacker destroys address while keeping the local state inside the same body.

## CI-sized receipt

Means across two matched seeds:

| arm | fibre area | junction proxy | active dyadic spectral bands | fixed linear surrogate error | transfer effective rank | zone entropy | response-map separation |
|---|---:|---:|---:|---:|---:|---:|---:|
| **FULL INTACT** | **24.430** | **0.707** | 3.0 | 0.101 | 2.417 | 0.280 | 0.932 |
| FULL ISOTROPIC | 24.430 | 0.707 | 3.0 | 0.102 | 2.475 | 0.291 | 0.923 |
| FULL SCRAMBLED | 22.105 | 0.324 | 4.5 | 0.102 | 2.485 | 0.302 | 0.914 |
| FULL ERASED | 0 | 0 | 0 | 0.102 | 2.475 | 0.291 | 0.923 |
| WRITE ONLY INTACT | 21.010 | 0.709 | 3.0 | 0.101 | 2.411 | 0.280 | 0.932 |

The dyadic-band count is only a rough scale-span ruler. It is **not** a fractal
dimension and is not optimized.

## Direct causal field attacks

A fixed set of four probe patterns is run from zero fast state. The complete
final energy maps are concatenated into a response fingerprint.

Relative L2 changes:

~~~text
INTACT vs ERASED       0.1727 ± 0.0287
INTACT vs ISOTROPIC    0.1727 ± 0.0287
INTACT vs SCRAMBLED    0.1864 ± 0.0184
FULL-development fibre
vs WRITE-ONLY fibre    0.1694 ± 0.1038
~~~

This is the main positive result of Datarium 5A:

> **The field writes an internal oriented morphology, and changing or removing
> that morphology changes the later spatial field response by roughly 17–19%
> in this CI-sized regime.**

The exact spatial arrangement matters at least as much as simply deleting
direction.

The FULL-vs-WRITE-ONLY developmental difference is nonzero but variable. That
is evidence that closing the loop changes what is grown, but it is not yet a
strong claim of self-organized optimization.

## The important negative result

The fixed held-out I/O measures barely move.

The linear-surrogate error is about 0.10 in every arm. Effective rank,
functional-zone entropy and response-map separation also change only modestly.

So Datarium 5A has **not** earned:

- higher functional complexity;
- an internal neural organ;
- intelligence;
- a dendrite analogue;
- a benefit from fractal or multiscale morphology.

What it has earned is the prerequisite:

~~~text
signal history changes material
material changes later signal geometry
~~~

The structure is causal, but not yet demonstrably useful.

## Why the Aizenbud comparison remains useful

Aizenbud et al. do not equate branching with computation. Their FCI is based on
the difficulty of reproducing a neuron's I/O with a fixed-capacity temporal
network. In their morphology analysis, total dendritic area is much more
predictive of FCI than the number of bifurcations.

Datarium 5 therefore reports fibre area and a fixed-surrogate I/O measure
separately. The current result is exactly the sort of distinction we wanted:
we can make substantial internal structure without automatically making the
I/O more complex.

Our "fixed linear surrogate error" is only an inexpensive attacker. It is not
the published FCI.

## Next gate

The next material property should be functional rather than decorative.

The cleanest continuation is to let fibre amount and junction chemistry change
one or more local physical coefficients that were already on the table:

- conductivity / damping;
- local nonlinear susceptibility;
- permeability to an internal chemical;
- wave speed.

Then ask, before adding evolution:

1. does intact morphology beat isotropic / scrambled / erased in held-out I/O?
2. does increased internal area or compartment count predict the effect?
3. does a fixed-capacity nonlinear surrogate become measurably worse?
4. does the effect survive when total fibre amount is matched?
5. can the fibre field later be advected with a moving body?

Only then is "organ" the right word.

## Run

~~~bash
python -m unittest tests.test_datarium5 -v
python experiments/datarium5_internal_field.py --preset ci
python experiments/datarium5_internal_field.py --preset receipt
~~~

Live qualitative mirror:

https://anttiluode.github.io/BlackBoxLab/datarium5.html

## Stopping line

> **Datarium 5A earns field/substrate closure: activity writes slow oriented
> internal morphology and that morphology causally changes later field
> geometry. It does not yet earn increased functional complexity, an internal
> neural organ, heredity, fractality, or intelligence.**
