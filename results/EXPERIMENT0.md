# Experiment 0 receipt — sampling-policy organogenesis

Run on 24 matched seeds, 6,000 physical timesteps, eight initially identical
organs, four simultaneous temporal ecologies, and autoregressive lag support
1/2/4/8.

All organs start with:

- identical predictor weights;
- identical sampling values;
- identical learning rules.

Only stochastic experience breaks symmetry.

| condition | ecologies covered | specialization | computation divergence | late prediction MSE | full 4/4 coverage |
|---|---:|---:|---:|---:|---:|
| YOKED | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0/24 |
| PRIVATE | 2.417 ± 0.640 | 0.905 ± 0.003 | 0.081 ± 0.033 | 0.188 ± 0.064 | 0/24 |
| **ECOLOGY** | **4.000 ± 0.000** | **0.904 ± 0.003** | **0.182 ± 0.013** | **0.167 ± 0.023** | **24/24** |

## What happened

PRIVATE is already enough to create strong individual specialization. Once an
organ accidentally acquires competence on one temporal regime, that regime
becomes valuable to sample again, which trains the same computation further.

But that positive feedback does **not** create a complete population-level
division of labor. Across these seeds the private population covers only 2.42
of four ecologies on average and never covers all four.

ECOLOGY adds one thing:

~~~text
sampling utility
    =
private learned value
    -
0.30 × current occupancy of that ecology
~~~

No role names. No target allocation. No organ is told to become "fast", "slow",
"persistent", or "burst".

Under that finite-territory pressure all four ecologies acquire at least one
dominant organ in every seed. Learned lag-weight divergence increases from
0.081 to 0.182 while prediction error remains comparable.

## Earned statement

> **Sampling × local plasticity can turn stochastic differences in experience
> into stable computational specialization. In this toy, private
> self-reinforcement alone produces specialists but not reliable population
> coverage; a small finite-territory pressure stabilizes a complete division of
> observational labor without assigning semantic roles.**

## What is not earned

- brain organogenesis;
- spontaneous cortical areas;
- a new mixture-of-experts algorithm;
- specialization without stochastic symmetry breaking;
- a claim that crowding is the biological mechanism;
- superiority to fixed random assignment;
- superiority to explicit routing or ordinary MoE;
- useful specialization when all ecologies have the same temporal statistics;
- a result about high-dimensional sensory representations.

The YOKED MSE is nearly zero because ecology 0 is a deliberately easy
deterministic oscillator. It is a symmetry control, not a performance baseline.

## Next attackers

1. **FIXED RANDOM TERRITORY** — give each organ a fixed random ecology. If it
   achieves the same fingerprints and coverage more cheaply, learned sampling
   has not earned much.
2. **REMOVE ECOLOGY LABELS** — expose a continuous mixed field rather than four
   named channels and let sampling position itself become continuous.
3. **SWAP THE WORLD** — exchange temporal statistics between regions after
   specialization. Do organs migrate, relearn in place, or remain developmentally
   locked?
4. **COMMUNICATION** — let organs exchange summaries. Does useful integration
   appear, or does one successful sampling policy capture the population?
5. **BODY PLASTICITY** — allow lag support / memory timescale itself to grow or
   prune, not merely change coefficients.

CI receipt: experiment run 4 on commit 26d23f5d12e2bce6239cc473538fbd009af1e91f.
All three smoke tests passed.
