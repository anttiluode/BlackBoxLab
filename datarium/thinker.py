"""Small heritable recurrent controllers for Datarium 2.

The controller is deliberately tiny and inspectable:

    h[t+1] = tanh(A h[t] + B s[t] + b)
    a[t]   = tanh(C h[t+1] + d)

The field still owns demographic events. This module only defines the digital
controller inherited when the lineage microscope reports a split or merge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


SENSOR_NAMES = (
    "phi",
    "phi_velocity",
    "grad_x",
    "grad_y",
    "resource",
    "local_energy_error",
    "global_energy_error",
    "domain_mass",
    "scout_distance",
    "local_signal",
)

ACTION_NAMES = (
    "steer_x",
    "steer_y",
    "excite",
    "damp",
    "signal",
)


@dataclass
class TinyController:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    b: np.ndarray
    d: np.ndarray

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        hidden: int = 6,
        sensors: int = len(SENSOR_NAMES),
        actions: int = len(ACTION_NAMES),
        scale: float = 0.28,
    ) -> "TinyController":
        return cls(
            A=rng.normal(0.0, scale / np.sqrt(hidden), (hidden, hidden)),
            B=rng.normal(0.0, scale / np.sqrt(sensors), (hidden, sensors)),
            C=rng.normal(0.0, scale / np.sqrt(hidden), (actions, hidden)),
            b=rng.normal(0.0, scale * 0.10, hidden),
            d=rng.normal(0.0, scale * 0.10, actions),
        )

    @property
    def hidden_size(self) -> int:
        return int(self.A.shape[0])

    def step(
        self,
        sensors: np.ndarray,
        hidden_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(
            self.A @ hidden_state + self.B @ sensors + self.b
        )
        action = np.tanh(self.C @ hidden + self.d)
        return action, hidden

    def mutated(
        self,
        rng: np.random.Generator,
        sigma: float = 0.045,
    ) -> "TinyController":
        def m(x: np.ndarray) -> np.ndarray:
            return x + rng.normal(0.0, sigma, x.shape)

        return TinyController(
            A=m(self.A),
            B=m(self.B),
            C=m(self.C),
            b=m(self.b),
            d=m(self.d),
        )

    @staticmethod
    def blend(
        controllers: list["TinyController"],
        weights: np.ndarray,
    ) -> "TinyController":
        if not controllers:
            raise ValueError("need at least one controller")
        w = np.asarray(weights, dtype=float)
        w = np.maximum(w, 1e-9)
        w /= w.sum()

        def blend_attr(name: str) -> np.ndarray:
            return sum(
                weight * getattr(controller, name)
                for weight, controller in zip(w, controllers)
            )

        return TinyController(
            A=blend_attr("A"),
            B=blend_attr("B"),
            C=blend_attr("C"),
            b=blend_attr("b"),
            d=blend_attr("d"),
        )

    def flat(self) -> np.ndarray:
        return np.concatenate(
            [self.A.ravel(), self.B.ravel(), self.C.ravel(), self.b, self.d]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "C": self.C.tolist(),
            "b": self.b.tolist(),
            "d": self.d.tolist(),
        }


@dataclass
class Thinker:
    track_id: int
    controller: TinyController | None
    hidden: np.ndarray
    scout_pos: np.ndarray
    scout_vel: np.ndarray
    parents: tuple[int, ...]
    generation: int
    born_t: float
    selection_score: float = 0.0
    global_score_sum: float = 0.0
    local_score_sum: float = 0.0
    action_cost_sum: float = 0.0
    control_steps: int = 0
    descendants: int = 0
    contacts: int = 0
    action_abs_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(len(ACTION_NAMES), dtype=float)
    )
    sample_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=float)
    )

    def mean_actions(self) -> np.ndarray:
        if self.control_steps <= 0:
            return np.zeros(len(ACTION_NAMES), dtype=float)
        return self.action_abs_sum / self.control_steps

    def behavior_vector(self) -> np.ndarray:
        if self.control_steps <= 0:
            return np.zeros(9, dtype=float)
        return np.concatenate(
            [
                self.mean_actions(),
                self.sample_sum / self.control_steps,
            ]
        )

    def to_summary(self) -> dict[str, object]:
        return {
            "track_id": int(self.track_id),
            "parents": list(self.parents),
            "generation": int(self.generation),
            "born_t": float(self.born_t),
            "selection_score": float(self.selection_score),
            "control_steps": int(self.control_steps),
            "descendants": int(self.descendants),
            "contacts": int(self.contacts),
            "mean_abs_actions": {
                name: float(value)
                for name, value in zip(ACTION_NAMES, self.mean_actions())
            },
            "mean_sampled": {
                name: float(value)
                for name, value in zip(
                    ("resource", "abs_phi", "local_energy", "signal"),
                    self.sample_sum / max(self.control_steps, 1),
                )
            },
        }


def probe_behavior(
    controller: TinyController,
    seed: int = 0,
    probes: int = 64,
) -> np.ndarray:
    """Controller fingerprint in behavior space, not raw weight space."""

    rng = np.random.default_rng(seed)
    hidden = np.zeros(controller.hidden_size, dtype=float)
    actions = []
    for _ in range(probes):
        sensors = np.clip(rng.normal(0.0, 0.65, len(SENSOR_NAMES)), -1.5, 1.5)
        action, hidden = controller.step(sensors, hidden)
        actions.append(action)
    arr = np.asarray(actions)
    return np.concatenate((arr.mean(axis=0), arr.std(axis=0)))


def pairwise_behavior_divergence(
    controllers: list[TinyController],
) -> float:
    if len(controllers) < 2:
        return 0.0
    features = [probe_behavior(c, seed=991) for c in controllers]
    distances = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            a, b = features[i], features[j]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom < 1e-12:
                distances.append(0.0)
            else:
                distances.append(1.0 - float(a @ b) / denom)
    return float(np.mean(distances))
