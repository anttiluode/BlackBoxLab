# Datarium 4A receipt — geography becomes body

Date: 2026-08-31

Datarium 3 earned a slower material geometry that remained causally effective
after its builders and fast wave were removed. Its stopping line was equally
important: the material was still fixed to the world.

Datarium 4A isolates the next missing physical prerequisite:

> Can a closed material phase retain an inside, acquire a local chemical
> polarity from its surroundings, and translate through the world without an
> object-level movement command?

The answer in this bridge experiment is **yes, from an explicitly seeded
compartment**. That qualifier is the main stopping line.

## Why the explicit seed is allowed here

This experiment is not the emergence gate yet. It is a mechanics gate.

If we immediately ask Datarium 3 builders to create a closed wall, make the wall
selectively permeable, give it internal chemistry, and make the whole result
mobile, then a moving shape would not tell us which rung actually worked.

Datarium 4A therefore starts with a smooth diffuse compartment and tests only
the body physics. The next integration must remove that seed and replace it
with builder-written material.

There is no `Cell` class, genome, fitness, controller, heading, steering
command or reproduction rule. The center of mass exists only in the
measurement code.

## Fields

The world contains five dynamical fields:

~~~text
phi(x,y)       diffuse material / inside phase
food(x,y)      soluble external resource
a(x,y)         internal soluble activator
u_x, u_y       overdamped solvent velocity
cargo(x,y)     passive tracer used only in the retention assay
~~~

The boundary is the diffuse interface of `phi`; no boundary polygon or object
ID is used by the physics.

### Local chemistry

A membrane patch converts locally encountered food into activator:

~~~text
uptake
  ~ interface(phi)
  × local food
  × local unsaturated activator
~~~

Activator diffuses, decays, and is lost rapidly outside the material phase.
With a spatially asymmetric food field, the inside therefore becomes
chemically asymmetric without receiving a source direction.

### Local force

Every grid cell computes only a local phase normal and local activator level:

~~~text
force
  = - active_stress
    × interface(phi)
    × activator
    × local_normal(phi)
~~~

That force enters a damped velocity field:

~~~text
du/dt = viscosity * laplacian(u)
        - drag * u
        + local_force
~~~

The material phase is advected by that velocity field and restored locally
toward a finite-width interface. A volume-restoring term prevents trivial
growth or collapse.

There is no rule of the form:

~~~text
body.position += velocity_toward_food
~~~

and no source vector is supplied to the phase.

## Rotational movement battery

The same body is tested against a replenished Gaussian food source placed at
eight angles around it. The metric projects the final measured displacement
onto the source direction. That source direction is used only by the observer.

~~~text
64 x 64 field
8 source angles
1500 steps
dt = 0.03
initial radius = 8
source radius = 18
~~~

| arm | movement toward source | sideways movement | chemical polarity | final phase mass / initial |
|---|---:|---:|---:|---:|
| **ACTIVE** | **17.715 ± 0.767** | **0.087 ± 0.020** | **0.3179 ± 0.0056** | 1.040 ± 0.020 |
| UNIFORM FOOD | ~0 | ~0 | 0.0002 | 1.045 |
| NO STRESS | ~0 | ~0 | 0.0782 ± 0.0002 | 0.995 |
| PINNED | ~0 | ~0 | 0.0738 | 1.000 |

Three parts of the causal chain separate:

1. **Asymmetric food creates asymmetric internal chemistry.** NO STRESS and
   PINNED still develop polarity.
2. **Chemistry is not movement by itself.** NO STRESS remains stationary.
3. **Local force is not sufficient when material cannot advect.** PINNED
   develops chemistry and solvent forcing but the boundary does not translate.
4. **No external asymmetry, no preferred movement.** UNIFORM FOOD removes the
   chemical polarity and translation.

The ACTIVE phase travels almost the full initial 18-cell source distance while
remaining almost collinear with the source direction over all eight
orientations.

## Selective-boundary retention assay

A separate fixed-boundary assay loads a passive cargo inside the same diffuse
compartment. Cargo evolves under conservative `div(D grad cargo)` diffusion.
Only the local diffusivity at the diffuse interface changes.

~~~text
selective membrane:
    D = D0 * (1 - 0.995 * interface(phi))

open control:
    D = D0
~~~

After 1600 steps:

| arm | initial cargo still inside | inside / outside concentration |
|---|---:|---:|
| **SELECTIVE MEMBRANE** | **0.794** | **30.58 x** |
| NO BARRIER | 0.478 | 11.12 x |

Total cargo is conserved numerically to floating-point precision in both arms.

This is deliberately not a claim about real lipid membranes. It establishes the
simulation primitive needed for an inside state to persist while the boundary
remains a field rather than an object.

## What is deliberately absent

The fast Datarium 3 wave is not present in Datarium 4A.

That is intentional. During Datarium 3 development an attacker caught a draft
where motion could write material without the wave doing causal work. The same
mistake would be easy here: a moving compartment plus decorative waves would
look more advanced while proving less.

The next integration should reintroduce the wave only after body mechanics is
stable, with conductivity/permeability coupling that can itself be ablated.

## Earned ladder

~~~text
Datarium 3
transient wave assemblies
        ↓
builder-written oriented material
        ↓
material organizes later strangers
        ↓
Datarium 4A
closed material phase retains an inside
        ↓
local chemistry creates local stress
        ↓
the material phase can leave its birthplace
~~~

What is still missing is the essential emergence link:

~~~text
Datarium 3 builders
        ↓
build a closed movable boundary
        ↓
that boundary retains chemistry
        ↓
that builder-made compartment moves
~~~

## Next gate — remove the seed

Datarium 4B should begin from Datarium 3-style builders and precursor, not a
pre-drawn `phi` disk.

The observer may detect candidate closed components after the fact, but the
physics may not receive a component ID. A candidate counts only if it:

1. closes through local materialization rather than an authored ring;
2. maintains a measurable inside/outside concentration difference;
3. remains connected while its material advects;
4. translates under local chemistry farther than mass-matched shuffled and
   pinned controls;
5. continues to work when the builders are removed.

Only after that gate should the inside receive slow structural chemistry:
fibres/catalysts that alter transport or wave propagation. Those structures can
then be tested for functional inheritance through physical fission.

## Run

~~~bash
python -m unittest tests.test_datarium4 -v
python experiments/datarium4_body.py --preset ci
python experiments/datarium4_body.py --preset receipt --write
~~~

Open the live bridge:

https://anttiluode.github.io/BlackBoxLab/datarium4.html

## Stopping line

> **Datarium 4A has made a movable compartment physics scaffold. The boundary
> can retain a soluble state and local chemical asymmetry can move the whole
> phase through a solvent without object-level steering. But the compartment is
> explicitly seeded. We have not yet shown that the train/material system builds
> the body, nor reproduction, heredity, speciation or an internal neural
> system.**
