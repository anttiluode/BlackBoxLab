# Datarium 1 receipt — lineage microscope

Date: 2026-08-30

Datarium 1 does **not** add organisms to the Three Fates field. It upgrades the
observer so that later heredity questions have a defensible object to refer to.

The substrate is the supplied Three Fates **LOCAL BUDGET** arm:

~~~text
phi^4 wave field
+ periodic boundaries
+ local resource r
+ r recovery
+ activity cost proportional to phi^2
+ r-gated local growth
~~~

No genome, trait, fitness, reward, controller or reproduction rule was added.

## Observer corrections

The old observer used one hard threshold, ordinary connected-component labels,
and a leader centroid.

Datarium 1 replaces that with:

~~~text
high/low threshold hysteresis
        ↓
4-connected components on the TORUS
        ↓
minimum component area
+ minimum positive mass
        ↓
frame-to-frame overlap graph
        ↓
continuation / split / merge / birth / death
~~~

A one-to-one overlap keeps an ID. A split ends the old ID and creates children
with an explicit parent link. A merge ends the parent IDs and creates a new
child with all parents recorded.

## CI receipt

Frozen run:

~~~text
grid N                         96
field dt                       0.02
end time                     360.0
observer cadence               0.4
high threshold                0.30
low threshold                 0.24
minimum area                    20
minimum positive mass          8.0
minimum overlap               0.12
measurements                   900
~~~

Datarium unit tests:

~~~text
6 / 6 PASS
~~~

They explicitly test torus seam joining, seam-crossing identity, hysteresis,
dust rejection, split ancestry and merge ancestry.

### Domain population after the audit

~~~text
late accepted domains
    mean                        4.26
    max                        13

rejected threshold dust
    mean components/frame      1.33
    frames containing dust     47.9%
~~~

So the historical visual statement "10–20 domains" should not be treated as
"10–20 substantial organisms." A large fraction of frames contain small
threshold fragments, and the mass floor matters.

### The periodic-seam bug matters much more for genealogy than for jump counts

~~~text
periodic vs nonperiodic component count differed
    772 / 900 measurements

persistent continuations observed across a seam
    51
~~~

The earlier transfer counter changed only modestly under a torus-aware audit.
That did **not** mean the seam was harmless. A jump counter can survive a few
misclassified transitions; a genealogy cannot survive having an identity cut
every time it crosses the world boundary.

### Demographic events

~~~text
births          126
splits          103
deaths          113
merges          105
rearrangements   12
~~~

The local-budget field is therefore not a collection of stable particles. It
is a high-turnover domain population.

### Individual identity is short

~~~text
individual track lifetime, simulated time

median          1.60
p90             5.20
max            24.40

fraction surviving >= 2 time    0.425
fraction surviving >= 5 time    0.110
fraction surviving >=10 time    0.021
~~~

Only about two percent of individual track segments survive ten simulated time
units.

If "organism" means one unbroken threshold component, this is a weak organism
substrate.

### But ancestry persists much longer than bodies

When splits and merges are treated as explicit genealogical edges:

~~~text
tracks with recorded parents      394
median ancestry depth              21
maximum ancestry depth            120
maximum ancestral time span     236.80
~~~

This is the interesting Datarium-1 result.

It does **not** show heredity.

A thresholded field can repeatedly split and merge while transmitting no
stable physical phenotype at all. Deep ancestry can therefore be a demographic
fact without being an evolutionary fact.

What it earns is narrower:

> **The local-budget field contains a long-lived genealogical web even though
> its individual domain bodies are short-lived. It is now meaningful to ask
> whether any physical or computational phenotype is preserved along those
> parent-child edges.**

## Why this changes the Datarium picture

The candidate unit may not be:

~~~text
one stable blob
    ↓
lives a long time
    ↓
copies itself
~~~

It may instead be closer to:

~~~text
material organization
      ↓
temporary bounded body
      ↓
split / merge / re-form
      ↓
descendant material organization
      ↓
temporary bounded body
      ...
~~~

That distinction matters enormously.

If phenotype resemblance survives the changing body boundary, then identity may
live more naturally in a **material lineage** than in a persistent particle.

If phenotype resemblance does not survive, the deep graph is only threshold
genealogy and Datarium must add a genuine body-carried state before talking
about inheritance.

## Required next attacker

Do not add genes yet.

The next test should measure parent/child physical phenotype and compare it
against unrelated mass-matched domains:

~~~text
parent P
   ↓ split
child A, child B

compare:
    similarity(P,A)
    similarity(P,B)
    similarity(A,B)

against:
    unrelated domains
    matched for mass / age
~~~

Candidate phenotype coordinates can start with quantities already available
without changing the physics:

~~~text
mass
area
compactness
eccentricity
local oscillation spectrum
impulse / resonance response
lag / chirality response
~~~

The key metric is not raw variance. It is **excess parent-child resemblance
over an unrelated matched null**.

If that fails, there is no physical heredity in this substrate as written.

If it survives, Datarium 2 can ask whether the inherited phenotype has
computational consequences.

## Run

~~~bash
python experiments/datarium1_lineage.py
python -m unittest tests.test_datarium_lineage -v
~~~

Live microscope:

[datarium1.html](../datarium1.html)

CI run for the receipt: GitHub Actions run 33325408287, Datarium job PASS.

## Earned statement

> **Datarium 1 converts the local-budget instanton descendant from a pretty
> churn field into an auditable demographic substrate. Substantial individual
> domains are short-lived, but explicit split/merge ancestry can persist for
> hundreds of field-time units. Whether anything is inherited along that
> ancestry remains completely open.**
