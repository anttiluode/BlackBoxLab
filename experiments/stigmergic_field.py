"""Experiment 1: stigmergic sampling in a continuous mixed field.

This removes Experiment 0's four action labels and explicit crowding penalty.

A one-dimensional ring contains four latent temporal processes mixed smoothly
across space. Agents do not choose a source label; they choose where to move
and sample. Post-hoc evaluation can inspect which latent process dominated the
regions they actually sampled, but those labels are unavailable to the policy.

The key ablation is environmental persistence:

NO_TRACE
    sampling leaves the world unchanged.

PRIVATE_TRACE
    sampling leaves a local refractory trace visible only to the same agent.
    This tests individual self-avoidance / exploration.

SHARED_TRACE
    sampling leaves the same refractory trace in one shared field. Later
    agents therefore encounter a world changed by earlier sampling.

The trace is intentionally primitive. It is not a biological mechanism and it
is not SwarmWorld's artifact system. It is the smallest physical stigmergy
test: local action changes future local evidence without an explicit
population-level diversity objective.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


STEPS = 5000
AGENTS = 8
CELLS = 64
SOURCES = 4
LAGS = (1, 2, 4, 8)
SEEDS = 24

EPSILON = 0.045
LEARN_RATE = 0.035
VALUE_RATE = 0.035
TRACE_DECAY = 0.992
TRACE_WRITE = 0.20
TRACE_NOISE = 1.10
MOVE_RADIUS = 2


def circular_distance(a: np.ndarray, b: float, n: int) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, n - d)


def mixing_matrix(cells: int = CELLS) -> np.ndarray:
    """Smooth overlapping latent-source profiles on a ring.

    Source identities are used only to generate the world and score the result.
    Agents receive scalar samples, not these weights or source labels.
    """

    x = np.arange(cells, dtype=float)
    centers = np.linspace(0, cells, SOURCES, endpoint=False)
    sigma = cells / 7.0
    cols = []
    for center in centers:
        d = circular_distance(x, center, cells)
        cols.append(np.exp(-0.5 * (d / sigma) ** 2))
    a = np.column_stack(cols)
    a /= np.sqrt(np.sum(a * a, axis=1, keepdims=True)) + 1e-12
    return a


def make_sources(steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 20_000)
    t = np.arange(steps, dtype=float)

    fast = np.sin(0.33 * t + 0.20 * np.sin(0.012 * t))
    slow = np.sin(0.050 * t + 0.45 * np.sin(0.0025 * t))

    persistent = np.zeros(steps)
    for i in range(1, steps):
        persistent[i] = 0.93 * persistent[i - 1] + 0.25 * rng.normal()
    persistent = np.tanh(persistent)

    burst = np.zeros(steps)
    for i in range(1, steps):
        burst[i] = 0.84 * burst[i - 1]
        if rng.random() < 0.035:
            burst[i] += rng.choice((-1.0, 1.0)) * (0.8 + 0.8 * rng.random())

    s = np.column_stack((fast, slow, persistent, burst))
    s -= s.mean(axis=0, keepdims=True)
    s /= s.std(axis=0, keepdims=True) + 1e-12
    return s


def make_world(steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    a = mixing_matrix()
    s = make_sources(steps, seed)
    world = s @ a.T
    world /= world.std(axis=0, keepdims=True) + 1e-12
    return world, a


@dataclass
class Agent:
    rng: np.random.Generator
    position: int
    weights: np.ndarray
    values: np.ndarray
    visits: np.ndarray
    history: deque

    @classmethod
    def make(cls, seed: int) -> "Agent":
        # Every agent begins in the same place with identical learned state.
        return cls(
            rng=np.random.default_rng(seed),
            position=0,
            weights=np.zeros(len(LAGS) + 1, dtype=float),
            values=np.full(CELLS, -1.0, dtype=float),
            visits=np.zeros(CELLS, dtype=float),
            history=deque(maxlen=max(LAGS) + 1),
        )

    def features(self) -> np.ndarray | None:
        if len(self.history) < max(LAGS):
            return None
        h = list(self.history)
        return np.asarray([1.0] + [h[-lag] for lag in LAGS], dtype=float)

    def learn(self, observation: float) -> float:
        features = self.features()
        prediction = 0.0 if features is None else float(self.weights @ features)
        error = float(observation - prediction)

        if features is not None:
            scale = 1.0 + float(features @ features)
            self.weights += LEARN_RATE * error * features / scale
            self.weights[:] = np.clip(self.weights, -2.0, 2.0)

        self.history.append(float(observation))
        return error * error


def candidate_positions(position: int) -> np.ndarray:
    return np.asarray(
        [(position + d) % CELLS for d in range(-MOVE_RADIUS, MOVE_RADIUS + 1)],
        dtype=int,
    )


def choose_position(agent: Agent) -> int:
    candidates = candidate_positions(agent.position)
    if agent.rng.random() < EPSILON:
        return int(agent.rng.choice(candidates))

    utility = agent.values[candidates].copy()
    # Tiny action noise is the only symmetry breaker in the policy.
    utility += 1e-7 * agent.rng.normal(size=len(candidates))
    return int(candidates[int(np.argmax(utility))])


def write_trace(trace: np.ndarray, position: int) -> None:
    """Leave a smooth persistent local environmental modification."""

    x = np.arange(CELLS)
    d = circular_distance(x, position, CELLS)
    kernel = np.exp(-0.5 * (d / 1.5) ** 2)
    trace += TRACE_WRITE * kernel
    np.clip(trace, 0.0, 0.95, out=trace)


def computation_divergence(weights: np.ndarray) -> float:
    distances: list[float] = []
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            a, b = weights[i], weights[j]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom < 1e-12:
                distances.append(0.0)
            else:
                distances.append(1.0 - float(a @ b) / denom)
    return float(np.mean(distances)) if distances else 0.0


def source_specialization(labels: np.ndarray) -> float:
    counts = np.bincount(labels, minlength=SOURCES).astype(float)
    p = counts / max(float(counts.sum()), 1.0)
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log(p))) / np.log(SOURCES)
    return 1.0 - entropy


def mean_pairwise_ring_distance(positions: np.ndarray) -> float:
    # positions shape: time x agent
    vals = []
    for row in positions:
        for i in range(AGENTS):
            for j in range(i + 1, AGENTS):
                d = abs(int(row[i]) - int(row[j]))
                vals.append(min(d, CELLS - d) / (CELLS / 2))
    return float(np.mean(vals))


def run(mode: str, seed: int) -> dict[str, object]:
    if mode not in {"no_trace", "private_trace", "shared_trace"}:
        raise ValueError(mode)

    world, mix = make_world(STEPS, seed)
    dominant_source_at_cell = np.argmax(mix, axis=1)

    agents = [
        Agent.make(seed * 10_000 + 137 * i + 31)
        for i in range(AGENTS)
    ]

    shared_trace = np.zeros(CELLS, dtype=float)
    private_traces = np.zeros((AGENTS, CELLS), dtype=float)

    positions = np.zeros((STEPS, AGENTS), dtype=np.int16)
    errors = np.zeros((STEPS, AGENTS), dtype=float)
    trace_mass = np.zeros(STEPS, dtype=float)

    for t in range(STEPS):
        if mode == "shared_trace":
            shared_trace *= TRACE_DECAY
        elif mode == "private_trace":
            private_traces *= TRACE_DECAY

        # Staggered updates: earlier actions can change evidence encountered by
        # later agents on the same physical tick, as in a shared persistent world.
        for i, agent in enumerate(agents):
            pos = choose_position(agent)
            raw = float(world[t, pos])

            if mode == "shared_trace":
                local_trace = float(shared_trace[pos])
            elif mode == "private_trace":
                local_trace = float(private_traces[i, pos])
            else:
                local_trace = 0.0

            # A used patch becomes temporarily less faithful / less predictable.
            # This is the physical consequence of the trace; there is no
            # occupancy count or diversity bonus in the policy.
            noise = agent.rng.normal()
            observation = (1.0 - local_trace) * raw + (
                local_trace * TRACE_NOISE * noise
            )

            sq_error = agent.learn(observation)
            reward = -min(sq_error, 4.0)

            agent.visits[pos] += 1.0
            eta = min(VALUE_RATE, 1.0 / agent.visits[pos])
            agent.values[pos] = (
                (1.0 - eta) * agent.values[pos] + eta * reward
            )
            agent.position = pos

            if mode == "shared_trace":
                write_trace(shared_trace, pos)
            elif mode == "private_trace":
                write_trace(private_traces[i], pos)

            positions[t, i] = pos
            errors[t, i] = sq_error

        if mode == "shared_trace":
            trace_mass[t] = float(np.mean(shared_trace))
        elif mode == "private_trace":
            trace_mass[t] = float(np.mean(private_traces))

    late = positions[STEPS // 2 :]
    late_source = dominant_source_at_cell[late]

    dominant_sources = []
    specializations = []
    for i in range(AGENTS):
        labels = late_source[:, i]
        counts = np.bincount(labels, minlength=SOURCES)
        dominant_sources.append(int(np.argmax(counts)))
        specializations.append(source_specialization(labels))

    fingerprints = np.stack([agent.weights for agent in agents])

    visited_cells = np.unique(late)
    return {
        "mode": mode,
        "seed": seed,
        "source_coverage": len(set(dominant_sources)),
        "full_source_coverage": len(set(dominant_sources)) == SOURCES,
        "mean_source_specialization": float(np.mean(specializations)),
        "computation_divergence": computation_divergence(fingerprints),
        "late_prediction_mse": float(np.mean(errors[-1200:])),
        "late_unique_cells": int(len(visited_cells)),
        "population_spatial_separation": mean_pairwise_ring_distance(
            late[::10]
        ),
        "mean_trace_mass_late": float(np.mean(trace_mass[-1000:])),
        "dominant_sources": dominant_sources,
        "fingerprints": fingerprints.tolist(),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = (
        "source_coverage",
        "mean_source_specialization",
        "computation_divergence",
        "late_prediction_mse",
        "late_unique_cells",
        "population_spatial_separation",
        "mean_trace_mass_late",
    )
    out: dict[str, object] = {}
    for metric in metrics:
        vals = np.asarray([float(r[metric]) for r in rows])
        out[metric] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }
    out["full_source_coverage_fraction"] = float(
        np.mean([bool(r["full_source_coverage"]) for r in rows])
    )
    return out


def run_all() -> dict[str, object]:
    modes = ("no_trace", "private_trace", "shared_trace")
    rows = {
        mode: [run(mode, seed) for seed in range(SEEDS)]
        for mode in modes
    }
    return {
        "config": {
            "steps": STEPS,
            "agents": AGENTS,
            "cells": CELLS,
            "sources": SOURCES,
            "lags": LAGS,
            "seeds": SEEDS,
            "epsilon": EPSILON,
            "trace_decay": TRACE_DECAY,
            "trace_write": TRACE_WRITE,
            "trace_noise": TRACE_NOISE,
        },
        "summary": {mode: summarize(rows[mode]) for mode in modes},
        "per_seed": rows,
    }


def main() -> None:
    receipt = run_all()
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "stigmergic_field.json"
    out_path.write_text(json.dumps(receipt, indent=2) + "\n")

    print("BlackBoxLab — Experiment 1: stigmergic continuous field")
    print(
        f"{'mode':14s} {'cover':>11s} {'special':>11s} {'comp-div':>11s} "
        f"{'cells':>11s} {'spatial':>11s} {'MSE':>11s} {'full':>9s}"
    )
    for mode in ("no_trace", "private_trace", "shared_trace"):
        r = receipt["summary"][mode]
        print(
            f"{mode:14s} "
            f"{r['source_coverage']['mean']:5.2f}±"
            f"{r['source_coverage']['std']:.2f} "
            f"{r['mean_source_specialization']['mean']:5.2f}±"
            f"{r['mean_source_specialization']['std']:.2f} "
            f"{r['computation_divergence']['mean']:5.3f}±"
            f"{r['computation_divergence']['std']:.3f} "
            f"{r['late_unique_cells']['mean']:5.1f}±"
            f"{r['late_unique_cells']['std']:.1f} "
            f"{r['population_spatial_separation']['mean']:5.2f}±"
            f"{r['population_spatial_separation']['std']:.2f} "
            f"{r['late_prediction_mse']['mean']:5.2f}±"
            f"{r['late_prediction_mse']['std']:.2f} "
            f"{r['full_source_coverage_fraction']:8.3f}"
        )

    print("\nGuardrail:")
    print(
        "Agents never receive source labels or an occupancy penalty. "
        "PRIVATE_TRACE tests self-avoidance. SHARED_TRACE is the stigmergic "
        "condition because one agent's local action changes later agents' "
        "local evidence."
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
