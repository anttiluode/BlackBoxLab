# Datarium 3 receipt — material memory above wave assemblies

Date: 2026-08-31

Datarium 3 asks one deliberately narrower question than “can trains become
life?”:

> Can transient wave-coupled assemblies create a slower spatial structure
> that remains causally effective after every builder has been removed?

The answer in the first four-seed receipt is **yes, for an oriented material
memory**. It is not yet yes for templating, reproduction, heredity, organisms,
open-ended evolution or intelligence.

## The supplied train: useful accident, impure claim

The attached `train.html` contains a genuinely useful loop:

~~~text
particle reads local field
        ↓
particle writes an extended oscillatory pattern
        ↓
field propagates and interferes
        ↓
field gradient changes later particle motion
~~~

But its source also contains five authored `WRITING_TYPES`, type-specific
writing equations, type-specific attraction rules, same-type attraction,
type-specific rewards, a global fitness sort every 300 frames, crossover and
mutation.

The visual trains may still have been surprising. The source cannot support
the stronger statement that undifferentiated field writers invented roles or
heredity.

Datarium 3 therefore keeps the local read/write/move loop and removes:

- named particle types;
- assigned roles;
- direct links or bonds;
- genomes and parent IDs;
- fitness and global sorting;
- reproduction;
- persistent private particle memory;
- any `Train` object in the dynamics.

Every particle has the same intrinsic frequency, speed and response law. At
initialization they differ only in position, heading and oscillator phase.

## Two memories, one local chemistry

### Fast memory: a phase-carrying wave

One real scalar does not retain the full local phase of an oscillation. The
fast medium is therefore represented by two quadratures,
`psi = wave_re + i wave_im`:

~~~text
d psi / dt = anisotropic_diffusion(psi, material)
             + i * carrier_frequency * psi
             + (resource_gain - damping - saturation) * psi
             + local_particle_emission
~~~

Particles emit their current phase into this field. They turn using the local
phase-sensitive gradient and their oscillator phase is entrained by the local
field. Interaction remains environment-mediated; no particle reads another
particle ID or state.

The Datarium local budget remains present:

~~~text
d resource / dt = recovery
                  - wave_activity_cost
                  + local_diffusion
~~~

### Slow memory: precursor becomes oriented matter

A separate precursor field can be converted into material. The conversion
rate is a product of strictly local quantities:

~~~text
soft density cooperativity
× phase coherence
× motion coherence
× wave amplitude gate
× positive agreement between motion flux and wave force
~~~

The last term closes the shortcut caught during development. In the first
draft, random traffic could write fibres and the fibre-alignment rule could
bootstrap itself while the wave became scenery. In the corrected model,
traffic alone cannot efficiently harden the medium: materialization requires
coherent mechanical agreement with a locally sustained wave.

The slow state has a scalar amount `m` and a 2-D nematic orientation
`(q1, q2) = m (cos 2 theta, sin 2 theta)`. It is deliberately generic. Depending
on the physical analogy, it can be read as a furrow, fibre, aligned polymer,
anisotropic viscosity or conductivity tensor.

The material changes later wave diffusion, wave persistence and particle
turning. No code detects a train before permitting this reaction.

## Post-hoc assembly observer

The physics never sees an assembly. The observer connects particles only when:

~~~text
torus distance <= 4 cells
heading cosine >= 0.50
component size >= 3
~~~

Measured component identities persist by conservative membership overlap. The
observer records component size, motion alignment, phase coherence,
elongation and persistence.

This does not prove that the resulting components are organisms. It gives the
word “train” a reproducible measurement rather than a visual impression.

## Build-arm receipt

Four matched seeds:

~~~text
field                         48 x 48
identical particles                 72
dt                                 0.04
builder steps                       4000
builder field time                   160
observation cadence             25 steps
~~~

| arm | late coherent fraction | late largest assembly | max persistence, samples | material spatial std | particle trace enrichment |
|---|---:|---:|---:|---:|---:|
| NO MEMORY | 0.127 ± 0.022 | 3.91 ± 0.19 | 42.8 ± 5.8 | 0 | 0 |
| WRITE ONLY | 0.127 ± 0.022 | 3.91 ± 0.19 | 42.8 ± 5.8 | 0.0108 | 1.05 |
| SCALAR | 0.117 ± 0.015 | 3.87 ± 0.33 | 30.2 ± 2.9 | 0.0115 | 1.07 |
| MEAN FIELD | 0.111 ± 0.024 | 3.76 ± 0.39 | 27.5 ± 7.2 | 0.0108 | 0.91 |
| **ORIENTED MATERIAL** | **0.530 ± 0.071** | **13.43 ± 4.44** | **94.0 ± 15.1** | **0.0275** | **2.66** |

`WRITE ONLY` is an exact intervention on feedback: it records the same local
history but the material cannot influence waves or particles. Its fast
dynamics match `NO MEMORY`, as they should.

The scalar and mean-field arms fail to improve assembly. The full effect needs
spatially addressed orientation, not merely another global number or a scalar
trail.

## Builder-removal receipt

At the end of each ORIENTED MATERIAL build:

1. snapshot the slow material;
2. delete every builder particle;
3. zero both fast wave quadratures and the wave envelope;
4. release the same naïve population from the same seed;
5. freeze the material so replacement particles cannot improve it during the
   assay;
6. run 1200 steps (48 field-time units).

| surviving material | fresh coherent fraction | largest fresh assembly | material enrichment | alignment to local director |
|---|---:|---:|---:|---:|
| **INTACT** | **0.525 ± 0.072** | **11.41 ± 1.56** | **2.05 ± 0.32** | **0.724 ± 0.012** |
| ISOTROPIC — same `m`, no direction | 0.123 ± 0.028 | 4.21 ± 0.47 | 0.97 ± 0.08 | 0 |
| PATCHWORK — intact local tiles, shuffled globally | 0.392 ± 0.035 | 7.89 ± 0.85 | 1.19 ± 0.06 | 0.466 ± 0.035 |
| CELL SCRAMBLE — same histogram, no texture | 0.211 ± 0.031 | 5.53 ± 1.31 | 0.99 ± 0.02 | 0.202 ± 0.018 |
| ROTATE 90° — same geometry, transverse directors | 0.290 ± 0.060 | 5.69 ± 0.94 | 0.77 ± 0.05 | 0.512 ± 0.042 |
| ERASED | 0.144 ± 0.052 | 4.43 ± 0.76 | 0 | 0 |

Three conclusions survive these controls:

1. **History survives outside the builders.** Intact material organizes a
   replacement population after all builder and fast-wave state is gone.
2. **Direction is load-bearing.** Preserving scalar material while deleting
   orientation is almost equivalent to erasure.
3. **Geometry exists at more than one scale.** Patchwork retains substantial
   function, so local fibre texture matters. Intact material still performs
   better, so longer connected arrangement matters too. A cell histogram is
   not enough.

The rotated arm is especially informative. A smooth correlated director field
can organize motion even when its direction is wrong, but rotation reduces
occupation of the material itself below the spatial mean. Generic ordering and
reuse of the written routes are distinct functions.

## Mechanism ablations

| condition | coherent fraction | largest assembly | material spatial std | wave RMS |
|---|---:|---:|---:|---:|
| **FULL** | **0.530 ± 0.071** | **13.43 ± 4.44** | **0.0275** | 0.515 |
| NO WAVE SENSING | 0.295 ± 0.063 | 6.18 ± 1.57 | 0.0062 | 0.241 |
| NO WAVE PRODUCTION | 0.185 ± 0.036 | 5.02 ± 1.23 | 0.0025 | 0.00055 |
| PHASE SHUFFLE EACH STEP | 0.589 ± 0.041 | 19.08 ± 3.96 | 0.0448 | 0.529 |
| UNIFORM RESOURCE | 0.478 ± 0.098 | 11.25 ± 3.11 | 0.0341 | 0.542 |

The wave is now load-bearing: removing sensing substantially weakens the
loop, and removing sustained production nearly eliminates structured material.

Two proposed ingredients are **not** load-bearing:

- Shuffling oscillator phases among particle positions on every step does not
  kill the effect; it strengthens it in this regime. Persistent individual
  oscillator identity is therefore not the inherited “letter.”
- Flattening the resource field has only a modest effect. The Three Fates
  local budget participates in the field dynamics, but its spatial niches are
  not the cause of this layer transition as currently parameterized.

Those are boundaries, not embarrassments. They tell the next experiment which
parts are mechanism and which parts are decoration.

## What emerged — and what did not

The earned ladder is:

~~~text
level 0   local resource + precursor
level 1   fast phase-carrying wave
level 2   transient measured particle assemblies
level 3   assembly-written oriented material infrastructure
level 4   replacement population organized by that infrastructure
~~~

The code contains levels 0 and 1 as affordances. It does not contain an object
for levels 2–4. Assemblies are post-hoc measurements; material is written by a
local reaction; replacement organization is an intervention result.

But the stronger ladder has **not** started:

~~~text
material pattern
      ↓ templates
daughter material pattern
      ↓ templates again
granddaughter pattern
~~~

Datarium 3 has made infrastructure, not DNA.

## The next non-cheating experiment

The next rung should ask whether structure can cause **new structure**, not
merely guide particles:

~~~text
parent material in chamber A
        ↓ wave only crosses the gap
naïve particles + precursor in chamber B
        ↓
candidate daughter material forms
        ↓ remove A and all particles
daughter alone must organize chamber C
~~~

The comparison is not pixel equality. It is whether daughter geometry carries
the same causal transfer function — wave routing, perturbation response and
ability to make another daughter — better than isotropic, patchwork,
phase-randomized and mass-matched controls.

No `copyGenome()`. No parent ID in the dynamics. No reward for looking like the
parent. If the cascade fails after one transfer, Datarium has built a road, not
a replicator.

## Run

~~~bash
python -m unittest tests.test_datarium3 -v
python experiments/datarium3_layers.py --preset ci
python experiments/datarium3_layers.py --preset receipt
~~~

Open [Datarium 3](../datarium3.html) for the live aquarium. The Python receipt
is authoritative.

## Neighboring physical results

- [Active wave-particle clusters](https://doi.org/10.1103/4cgg-hnyh) shows that
  particles coupled through self-generated waves can form stable bound
  clusters with collective modes.
- [Environmental memory boosts group formation of clueless individuals](https://doi.org/10.1038/s41467-023-43099-0)
  demonstrates group formation through paths physically opened in a passive
  environment.
- [Stigmergy in bacterial biofilms](https://pmc.ncbi.nlm.nih.gov/articles/PMC3984292/)
  reports aligned bacterial rafts ploughing furrows that later constrain other
  cells.
- [Multi-scale organization in communicating active matter](https://pmc.ncbi.nlm.nih.gov/articles/PMC9640622/)
  is a close precedent for local signal processing producing multiple
  collective dynamical scales.
- [SWARMWORLD_NOTE.md](../SWARMWORLD_NOTE.md) records the separate artifact-
  ecology analogy and its limits.

## Earned statement

> **Identical phase-emitting particles coupled only through local fields form
> transient measured assemblies. When coherent motion with the wave converts
> precursor into oriented material, a slower spatial organization appears.
> After every builder and all fast-wave state are removed, that material still
> organizes a matched naïve population, and its intact multiscale geometry
> outperforms isotropic, patchwork, cell-scrambled, rotated and erased controls.
> Datarium 3 has therefore produced a causally effective higher material layer,
> but not templating or heredity.**
