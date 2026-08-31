# Order test — does the material record *when*, or only *how much*?

The gate that has to pass before "the field is DNA" means anything. It is missing
from both ladders: Sol's assay shows the material is causally active infrastructure,
which is not the same as the material carrying history.

Substrate-agnostic. `cell0.py` is one medium to run it against; the two test scripts
only need a world that exposes a material map, a director field, and a way to switch
the population's drive sign partway through.

## Why this test and not the erasure assay

Sunday measured that a conserved-budget, proportional-deposition mass rule is a
**contraction**: `mass* = F(program)`, corr 0.99998 from any starting point, no path
dependence. In a contracting medium the builder-removal assay passes *trivially* —
fresh scouts get organised by a structure that is a deterministic function of the
drive statistics, and nothing was inherited. Both tests here are designed so a
contraction fails them.

## The design

Two arms, and in each one a **single scalar summary of the material is the attacker**.
If mean material separates the arms as well as the map does, the world records a
level, not a history.

**`order.py` — matched marginals, different sequence.**
FIRST-SEEK climbs the field envelope for t < T/2 then descends; FIRST-AVOID reverses.
Same population, same genomes, same total time in each mode, same budget.

**`recency.py` — same ending, different past.**
Both arms spend their final 6000 steps avoiding; one spent 2000 steps seeking first.
Material half-life is ~2170 steps, so 14.7% of anything laid in that first phase
survives to the end. This is the one that matters: it rules out "the last thing that
happened."

## Measured (128², 2500 scouts, 6–8 worlds per arm, leave-one-out nearest centroid)

| test | features | LOO acc | max effect size |
|---|---|---|---|
| order | mean material only | **0.562** | 0.71 |
| order | spatial (radial spectrum + quantiles) | 1.000 | 2.00 |
| order | director field | 1.000 | 1.98 |
| recency | mean material only | **0.583** | 0.56 |
| recency | spatial | 1.000 | 1.80 |
| recency | director field | 1.000 | 1.86 |

The scalar sits at chance in both. The geometry and the fibre direction separate the
arms perfectly. So the medium records a 2000-step episode that ended 6000 steps and
~2.8 material half-lives ago, and the record is in the arrangement, not the amount.

That is not a contraction.

## What this does not show

Not inheritance. It shows the precondition: the world can hold a signature that
outlives its writers and is not reducible to a summary statistic. Whether a later
reader can transcribe it is the next gate, and it needs a reader.

Caveat worth keeping: 12–16 worlds against ~10 features, so LOO nearest centroid can
flatter. The internal control is that the same pipeline returns chance for the scalar.
