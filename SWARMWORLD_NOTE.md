# SwarmWorld as a constraint on BlackBoxLab

Reference: Subhadeep Pal, Fiona Y. Wang, Markus J. Buehler,
*SwarmWorld: Stigmergic technological evolution in societies of language-model
agents*, arXiv:2608.26081 (2026).

SwarmWorld is not evidence for BlackBoxLab's neural analogy. It is a neighboring
systems result that sharpens the experiment.

The relevant structural ingredients are:

~~~text
initially homogeneous agents
+ local observation
+ no assigned roles
+ persistent shared world
+ actions that alter later agents' opportunities
        ↓
emergent behavioral differentiation
~~~

SwarmWorld's important distinction for this repo is that coordination need not
live only in messages. Persistent artifacts become part of the world later
agents physically observe. Their reported differentiation is post-hoc
behavioral differentiation rather than prompted role assignment.

BlackBoxLab Experiment 0 had a cruder mechanism:

~~~text
other agents are here
        ↓
explicit crowding penalty
        ↓
look elsewhere
~~~

Experiment 1 removes that population-level term. Instead:

~~~text
sample here
    ↓
leave a persistent local environmental trace
    ↓
later sampling at this location is physically different
    ↓
another learner changes its policy from the consequence
~~~

That is the minimal stigmergic step.

## Important difference

SwarmWorld agents build persistent technologies, programs, and cultural
lineages. Experiment 1 does nothing remotely as rich. Its trace is merely a
local refractory modification of an observation field.

This simplification is deliberate. Before adding language, artifacts, or
program inheritance, we want to isolate one question:

> Can a shared persistent trace change population-level sampling and
> computational differentiation beyond what each learner achieves through
> private self-avoidance?

The PRIVATE_TRACE arm is therefore essential.

## Why not call the trace a message?

Because no agent reads an agent ID, role label, occupancy count, or instruction.
The only communication channel is the changed world.

~~~text
A acts on location x
        ↓
world(x) changes
        ↓
B later samples x
        ↓
B receives different evidence
~~~

If that changes B's developmental trajectory, the coordination is
environment-mediated.

## Next step if Experiment 1 survives

The current trace changes signal quality. A stronger BlackBoxLab test would let
the trace itself become a local computational operator:

~~~text
signal passes through local sub-operator
        ↓
operator changes from passage / prediction error
        ↓
future signals are mixed / filtered differently
        ↓
different regions accumulate different effective computations
~~~

That would connect the stigmergic idea back to the older matrix-under-the-dot
and persistent-body thread without assuming the result in advance.
