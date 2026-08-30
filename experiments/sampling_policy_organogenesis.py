"""Experiment 0: sampling-policy organogenesis.

Initially identical online predictors share one physical world but can sample
only one of four simultaneously existing temporal ecologies at each step.

No semantic role is assigned to any learner. The experiment asks whether the
feedback

    sample -> learn -> become competent -> value that sample -> sample again

can make different sampling histories become different computations.

Three conditions are compared:
- yoked: every organ sees ecology 0;
- private: each organ follows only its own learned sampling value;
- ecology: private sampling plus a small instantaneous crowding cost.

The crowding term is deliberately explicit. It is the symmetry-breaking
attacker: if private sampling collapses but finite observation territory
differentiates, the earned claim is sampling x plasticity x resource ecology,
not magical spontaneous organ formation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


STEPS = 6000
ORGANS = 8
ECOLOGIES = 4
LAGS = (1, 2, 4, 8)
SEEDS = 24
EPSILON = 0.03
CROWDING = 0.30
LEARN_RATE = 0.035
VALUE_RATE = 0.04


def make_world(steps: int, seed: int) -> np.ndarray:
    """Four simultaneous temporal regimes, normalized independently."""

    rng = np.random.default_rng(seed + 10_000)
    t = np.arange(steps, dtype=float)

    fast = np.sin(0.34 * t + 0.18 * np.sin(0.013 * t))
    slow = np.sin(0.052 * t + 0.42 * np.sin(0.0027 * t))

    persistent = np.zeros(steps)
    for i in range(1, steps):
        persistent[i] = 0.94 * persistent[i - 1] + 0.22 * rng.normal()
    persistent = np.tanh(persistent)

    burst = np.zeros(steps)
    for i in range(1, steps):
        burst[i] = 0.86 * burst[i - 1]
        if rng.random() < 0.03:
            burst[i] += rng.choice((-1.0, 1.0)) * (
                0.8 + 0.7 * rng.random()
            )

    world = np.column_stack((fast, slow, persistent, burst))
    world -= world.mean(axis=0, keepdims=True)
    world /= world.std(axis=0, keepdims=True) + 1e-12
    return world


@dataclass
class Organ:
    rng: np.random.Generator
    weights: np.ndarray
    values: np.ndarray
    visits: np.ndarray
    history: deque

    @classmethod
    def make(cls, seed: int) -> "Organ":
        rng = np.random.default_rng(seed)
        return cls(
            rng=rng,
            weights=np.zeros(len(LAGS) + 1, dtype=float),
            values=np.full(ECOLOGIES, -1.0, dtype=float),
            visits=np.zeros(ECOLOGIES, dtype=float),
            history=deque(maxlen=max(LAGS) + 1),
        )

    def features(self) -> np.ndarray | None:
        if len(self.history) < max(LAGS):
            return None
        h = list(self.history)
        return np.asarray([1.0] + [h[-lag] for lag in LAGS], dtype=float)

    def observe_and_learn(self, value: float) -> float:
        features = self.features()
        prediction = 0.0 if features is None else float(self.weights @ features)
        error = float(value - prediction)

        if features is not None:
            scale = 1.0 + float(features @ features)
            self.weights += LEARN_RATE * error * features / scale
            self.weights[:] = np.clip(self.weights, -2.0, 2.0)

        self.history.append(float(value))
        return error * error


def choose_ecology(
    organ: Organ,
    mode: str,
    occupancy: np.ndarray,
) -> int:
    if mode == "yoked":
        return 0

    if organ.rng.random() < EPSILON:
        return int(organ.rng.integers(ECOLOGIES))

    utility = organ.values.copy()
    if mode == "ecology":
        utility -= CROWDING * occupancy
    elif mode != "private":
        raise ValueError(mode)

    utility += 1e-6 * organ.rng.normal(size=ECOLOGIES)
    return int(np.argmax(utility))


def normalized_specialization(counts: np.ndarray) -> float:
    probabilities = counts / max(float(counts.sum()), 1.0)
    nz = probabilities[probabilities > 0]
    entropy = -float(np.sum(nz * np.log(nz))) / np.log(ECOLOGIES)
    return 1.0 - entropy


def computation_divergence(weights: np.ndarray) -> float:
    distances = []
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            a, b = weights[i], weights[j]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom < 1e-12:
                distances.append(0.0)
            else:
                distances.append(1.0 - float(a @ b) / denom)
    return float(np.mean(distances)) if distances else 0.0


def run(mode: str, seed: int) -> dict[str, object]:
    world = make_world(STEPS, seed)
    organs = [
        Organ.make(seed * 1000 + 97 * i + 23)
        for i in range(ORGANS)
    ]

    choices = np.zeros((STEPS, ORGANS), dtype=np.int8)
    errors = np.zeros((STEPS, ORGANS), dtype=float)

    for t in range(STEPS):
        occupancy = np.zeros(ECOLOGIES, dtype=float)

        for i, organ in enumerate(organs):
            ecology = choose_ecology(organ, mode, occupancy)
            sq_error = organ.observe_and_learn(world[t, ecology])

            reward = -min(sq_error, 4.0)
            organ.visits[ecology] += 1.0
            eta = min(VALUE_RATE, 1.0 / organ.visits[ecology])
            organ.values[ecology] = (
                (1.0 - eta) * organ.values[ecology] + eta * reward
            )

            occupancy[ecology] += 1.0
            choices[t, i] = ecology
            errors[t, i] = sq_error

    late = choices[STEPS // 2 :]
    dominant = []
    specialization = []

    for i in range(ORGANS):
        counts = np.bincount(late[:, i], minlength=ECOLOGIES).astype(float)
        dominant.append(int(np.argmax(counts)))
        specialization.append(normalized_specialization(counts))

    fingerprints = np.stack([organ.weights for organ in organs])

    return {
        "mode": mode,
        "seed": seed,
        "coverage": len(set(dominant)),
        "specialization": float(np.mean(specialization)),
        "computation_divergence": computation_divergence(fingerprints),
        "late_prediction_mse": float(np.mean(errors[-1500:])),
        "dominant_ecologies": dominant,
        "fingerprints": fingerprints.tolist(),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = (
        "coverage",
        "specialization",
        "computation_divergence",
        "late_prediction_mse",
    )
    result: dict[str, object] = {}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows])
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    result["full_coverage_fraction"] = float(
        np.mean([int(row["coverage"]) == ECOLOGIES for row in rows])
    )
    return result


def run_all() -> dict[str, object]:
    modes = ("yoked", "private", "ecology")
    rows = {
        mode: [run(mode, seed) for seed in range(SEEDS)]
        for mode in modes
    }
    return {
        "config": {
            "steps": STEPS,
            "organs": ORGANS,
            "ecologies": ECOLOGIES,
            "lags": LAGS,
            "seeds": SEEDS,
            "epsilon": EPSILON,
            "crowding": CROWDING,
        },
        "summary": {
            mode: summarize(rows[mode])
            for mode in modes
        },
        "per_seed": rows,
    }


def main() -> None:
    receipt = run_all()
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sampling_policy_organogenesis.json"
    out_path.write_text(json.dumps(receipt, indent=2) + "\n")

    print("BlackBoxLab — sampling-policy organogenesis")
    print(
        f"{'mode':10s} {'coverage':>12s} {'specialize':>12s} "
        f"{'comp-div':>12s} {'MSE':>12s} {'full-cover':>12s}"
    )

    for mode in ("yoked", "private", "ecology"):
        row = receipt["summary"][mode]
        print(
            f"{mode:10s} "
            f"{row['coverage']['mean']:6.3f}±{row['coverage']['std']:.3f} "
            f"{row['specialization']['mean']:6.3f}±"
            f"{row['specialization']['std']:.3f} "
            f"{row['computation_divergence']['mean']:6.3f}±"
            f"{row['computation_divergence']['std']:.3f} "
            f"{row['late_prediction_mse']['mean']:6.3f}±"
            f"{row['late_prediction_mse']['std']:.3f} "
            f"{row['full_coverage_fraction']:12.3f}"
        )

    print("\nInterpretation guard:")
    print(
        "YOKED tests whether identical sampled history keeps computation alike. "
        "PRIVATE tests self-reinforcing sampling without a diversity objective. "
        "ECOLOGY adds only finite observation territory, not named roles."
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
