"""Datarium 2 — Ecological Thinkers.

A tiny recurrent controller is attached to each *lineage segment* reported by
Datarium 1. The controller gets a scout that samples the local field, can move
on the torus, emits one scalar social signal, and exerts bounded local
excitation/damping on the same phi field.

Crucially:

- the field still owns birth / split / merge / death;
- a split copies the controller with mutation only in EVOLVE mode;
- a merge blends parent controllers using their measured homeostatic score;
- RANDOM gives every new track a fresh controller, killing heredity;
- HOMEOSTAT is a hand-coded stabilizer attacker;
- NONE is the untouched local-budget field.

This is therefore a deliberately hybrid artificial-life experiment. The
digital matrix is not claimed to be field-native. The question is narrower:

    can heritable local controllers coupled through a shared field improve
    persistence / recovery / stability, and do behavioral phenotypes diverge?

The uploaded historical scout code inspired the bidirectional coupling:
scouts feel the field and locally modify it. This implementation replaces its
hand-authored "cognition" labels with explicit evolutionary and baseline tests.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datarium.lineage import LineageTracker, torus_delta, torus_distance
from datarium.thinker import (
    ACTION_NAMES,
    SENSOR_NAMES,
    Thinker,
    TinyController,
    pairwise_behavior_divergence,
)


DT = 0.02
C_WAVE = 1.0
A_POT = 0.1
B_POT = 0.1
TAU_RECOVER = 100.0
BURN = 0.033
DAMPING = 0.01

HIGH_THRESHOLD = 0.30
LOW_THRESHOLD = 0.24
MIN_OVERLAP = 0.12

CONTROL_EVERY = 10
MUTATION_SIGMA = 0.05
SCOUT_MAX_SPEED = 1.0
SCOUT_BINDING = 0.060
SCOUT_STEER = 0.14
SCOUT_FIELD_SURF = 0.018
ACT_FORCE = 0.16
ACT_DAMP = 0.20
SIGNAL_DECAY = 0.94
SIGNAL_DIFFUSION = 0.055
SIGNAL_WRITE = 0.32

MODES = ("none", "homeostat", "random", "evolve")


def laplacian(field: np.ndarray) -> np.ndarray:
    return (
        np.roll(field, -1, 1)
        + np.roll(field, 1, 1)
        + np.roll(field, -1, 0)
        + np.roll(field, 1, 0)
        - 4.0 * field
    )


def circular_displacement(
    source: np.ndarray,
    target: np.ndarray,
    n: int,
) -> np.ndarray:
    """Vector pointing from source to target on an N x N torus."""

    dx = float(torus_delta(target[0], source[0], n))
    dy = float(torus_delta(target[1], source[1], n))
    return np.asarray([dx, dy], dtype=float)


def torus_gaussian(
    n: int,
    pos: np.ndarray,
    sigma: float,
) -> np.ndarray:
    y, x = np.ogrid[:n, :n]
    dx = torus_delta(x, float(pos[0]), n)
    dy = torus_delta(y, float(pos[1]), n)
    return np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))


def sample_bilinear(field: np.ndarray, pos: np.ndarray) -> float:
    n = field.shape[0]
    x = float(pos[0]) % n
    y = float(pos[1]) % n
    x0 = int(np.floor(x)) % n
    y0 = int(np.floor(y)) % n
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n
    fx = x - np.floor(x)
    fy = y - np.floor(y)

    return float(
        (1 - fx) * (1 - fy) * field[y0, x0]
        + fx * (1 - fy) * field[y0, x1]
        + (1 - fx) * fy * field[y1, x0]
        + fx * fy * field[y1, x1]
    )


def patch_mean(field: np.ndarray, pos: np.ndarray, radius: int = 2) -> float:
    n = field.shape[0]
    x = int(round(float(pos[0]))) % n
    y = int(round(float(pos[1]))) % n
    vals = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            vals.append(field[(y + dy) % n, (x + dx) % n])
    return float(np.mean(vals))


class ThinkerField:
    def __init__(self, n: int, seed: int):
        self.n = int(n)
        self.rng = np.random.default_rng(seed)
        y, x = np.ogrid[:n, :n]
        start = (n * 0.25, n * 0.5)
        radius = max(6.0, n / 9.6)
        distance = np.sqrt((x - start[0]) ** 2 + (y - start[1]) ** 2)
        phi = 1.5 / np.cosh(distance / radius)
        gx = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / 2.0

        self.phi = phi.astype(float)
        self.prev = (phi + 0.25 * gx * DT).astype(float)
        self.r = np.ones((n, n), dtype=float)
        self.signal = np.zeros((n, n), dtype=float)
        self.control_force = np.zeros((n, n), dtype=float)
        self.control_damping = np.zeros((n, n), dtype=float)
        self.t = 0.0

    @property
    def velocity(self) -> np.ndarray:
        return (self.phi - self.prev) / DT

    def energy_density(self) -> float:
        vel = self.velocity
        gy, gx = np.gradient(self.phi)
        return float(
            np.mean(
                0.5 * vel * vel
                + 0.5 * (gx * gx + gy * gy)
                + 0.025 * self.phi**4
            )
        )

    def step(self) -> None:
        velocity = self.velocity
        acc = (
            C_WAVE**2 * laplacian(self.phi)
            - B_POT * self.phi**3
            + A_POT * self.r * self.phi
            - DAMPING * velocity
            + self.control_force
            - self.control_damping * velocity
        )

        self.r += DT * (
            (1.0 - self.r) / TAU_RECOVER - BURN * self.phi**2
        )
        np.clip(self.r, 0.0, 1.0, out=self.r)

        new = 2.0 * self.phi - self.prev + DT**2 * acc
        self.prev, self.phi = self.phi, new

        self.signal *= SIGNAL_DECAY ** (1.0 / CONTROL_EVERY)
        self.signal += (
            SIGNAL_DIFFUSION / CONTROL_EVERY
        ) * laplacian(self.signal)
        np.clip(self.signal, -1.0, 1.0, out=self.signal)
        self.t += DT

    def pulse(self, x: float, y: float, amplitude: float) -> None:
        pos = np.asarray([x, y], dtype=float)
        self.phi += amplitude * torus_gaussian(
            self.n, pos, sigma=max(3.0, self.n / 14.0)
        )


def normalized_error(value: float, target: float) -> float:
    return float(
        np.tanh(np.log((value + 1e-9) / (target + 1e-9)))
    )


def stability_score(error: float) -> float:
    return float(np.exp(-abs(error)))


def hand_homeostat(
    sensors: np.ndarray,
) -> np.ndarray:
    """Boring attacker: regulate energy error, steer up local gradient."""

    action = np.zeros(len(ACTION_NAMES), dtype=float)
    grad = np.asarray([sensors[2], sensors[3]], dtype=float)
    gn = float(np.linalg.norm(grad))
    if gn > 1e-9:
        action[:2] = grad / gn

    local_error = float(sensors[5])
    global_error = float(sensors[6])

    # Positive error means too much field energy.
    action[2] = float(np.clip(-0.55 * local_error, -1.0, 1.0))
    action[3] = float(
        np.clip(0.75 * max(local_error, 0.0) + 0.25 * max(global_error, 0.0), 0.0, 1.0)
    )
    action[4] = float(np.clip(-global_error, -1.0, 1.0))
    return action


def controller_mean_score(thinker: Thinker) -> float:
    if thinker.control_steps <= 0:
        return 0.0
    return thinker.selection_score / thinker.control_steps


def create_thinker(
    *,
    track_id: int,
    parents: tuple[int, ...],
    domain_center: np.ndarray,
    t: float,
    mode: str,
    thinkers: dict[int, Thinker],
    rng: np.random.Generator,
) -> Thinker:
    parent_thinkers = [thinkers[p] for p in parents if p in thinkers]
    generation = (
        1 + max((p.generation for p in parent_thinkers), default=-1)
    )

    controller: TinyController | None
    hidden = np.zeros(6, dtype=float)
    scout_vel = np.zeros(2, dtype=float)

    if mode in ("none", "homeostat"):
        controller = None
    elif mode == "random" or not parent_thinkers:
        controller = TinyController.random(rng)
    elif len(parent_thinkers) == 1:
        parent = parent_thinkers[0]
        controller = parent.controller.mutated(
            rng, MUTATION_SIGMA
        ) if parent.controller is not None else TinyController.random(rng)
        hidden = parent.hidden.copy() + rng.normal(0.0, 0.01, parent.hidden.shape)
        scout_vel = parent.scout_vel.copy() * 0.5
        parent.descendants += 1
    else:
        controllers = [
            p.controller for p in parent_thinkers if p.controller is not None
        ]
        if controllers:
            raw_scores = np.asarray(
                [controller_mean_score(p) for p in parent_thinkers],
                dtype=float,
            )
            raw_scores -= np.max(raw_scores)
            weights = np.exp(2.0 * raw_scores)
            controller = TinyController.blend(
                [p.controller for p in parent_thinkers if p.controller is not None],
                weights[: len(controllers)],
            ).mutated(rng, MUTATION_SIGMA * 0.6)
            hidden = np.average(
                np.stack([p.hidden for p in parent_thinkers]),
                axis=0,
                weights=np.maximum(weights, 1e-6),
            )
            scout_vel = np.average(
                np.stack([p.scout_vel for p in parent_thinkers]),
                axis=0,
                weights=np.maximum(weights, 1e-6),
            ) * 0.5
        else:
            controller = TinyController.random(rng)
        for parent in parent_thinkers:
            parent.descendants += 1

    return Thinker(
        track_id=track_id,
        controller=controller,
        hidden=hidden,
        scout_pos=domain_center.astype(float).copy(),
        scout_vel=scout_vel,
        parents=parents,
        generation=max(generation, 0),
        born_t=float(t),
    )


def behavior_divergence(thinkers: Iterable[Thinker]) -> float:
    features = [
        t.behavior_vector()
        for t in thinkers
        if t.control_steps >= 8
    ]
    if len(features) < 2:
        return 0.0
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


def recovery_times(
    errors: list[float],
    pulse_indices: list[int],
    control_dt: float,
    threshold: float = 0.18,
    hold: int = 4,
) -> list[float]:
    out = []
    for start in pulse_indices:
        found = None
        for i in range(start, max(start, len(errors) - hold)):
            if all(abs(errors[j]) < threshold for j in range(i, min(i + hold, len(errors)))):
                found = (i - start) * control_dt
                break
        if found is not None:
            out.append(float(found))
    return out


def checkpoint(
    path: Path,
    *,
    mode: str,
    seed: int,
    field: ThinkerField,
    thinkers: dict[int, Thinker],
    active_ids: list[int],
    target_energy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "mode": mode,
        "seed": seed,
        "t": field.t,
        "target_energy": target_energy,
        "energy_density": field.energy_density(),
        "active_ids": active_ids,
        "thinkers": {
            str(tid): {
                **thinkers[tid].to_summary(),
                "controller": (
                    thinkers[tid].controller.to_dict()
                    if thinkers[tid].controller is not None
                    else None
                ),
                "hidden": thinkers[tid].hidden.tolist(),
                "scout_pos": thinkers[tid].scout_pos.tolist(),
            }
            for tid in active_ids
            if tid in thinkers
        },
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def run_mode(
    mode: str,
    seed: int,
    *,
    n: int,
    t_end: float,
    warmup: float,
    pulse_interval: float,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: float = 250.0,
) -> dict[str, object]:
    if mode not in MODES:
        raise ValueError(mode)

    rng = np.random.default_rng(seed + 40_000)
    field = ThinkerField(n, seed)
    tracker = LineageTracker(
        (n, n),
        high_threshold=HIGH_THRESHOLD,
        low_threshold=LOW_THRESHOLD,
        min_area=max(10, int(20 * (n / 96.0) ** 2)),
        min_mass=max(4.0, 8.0 * (n / 96.0) ** 2),
        min_overlap=MIN_OVERLAP,
    )
    thinkers: dict[int, Thinker] = {}

    warmup_energies: list[float] = []
    energy_errors: list[float] = []
    stability_scores: list[float] = []
    active_counts: list[int] = []
    generation_max: list[int] = []
    contact_edges: defaultdict[tuple[int, int], int] = defaultdict(int)

    target_energy = 0.0
    postwarmup_started = False
    pulse_indices: list[int] = []
    pulse_events: list[dict[str, float]] = []
    next_pulse = warmup + pulse_interval
    next_checkpoint = warmup + checkpoint_interval
    control_index = 0

    total_steps = int(round(t_end / DT))
    control_dt = CONTROL_EVERY * DT

    for step in range(total_steps):
        field.step()

        # Same perturbation schedule for every mode at a given seed.
        if field.t >= next_pulse and field.t < t_end - 1e-9:
            prng = np.random.default_rng(
                seed * 1_000_003 + int(round(next_pulse * 10))
            )
            x, y = prng.uniform(0, n, 2)
            amplitude = float(prng.choice((-1.0, 1.0)) * 0.65)
            field.pulse(x, y, amplitude)
            pulse_indices.append(control_index)
            pulse_events.append(
                {"t": float(field.t), "x": float(x), "y": float(y), "amplitude": amplitude}
            )
            next_pulse += pulse_interval

        if step % CONTROL_EVERY:
            continue

        energy = field.energy_density()
        if field.t < warmup:
            warmup_energies.append(energy)

        domains = tracker.update(field.phi, field.t)

        if not postwarmup_started and field.t >= warmup:
            target_energy = float(
                np.median(warmup_energies[-max(10, len(warmup_energies) // 2):])
            )
            postwarmup_started = True

        # Warm-up is observer-only. Controllers begin after target calibration.
        if not postwarmup_started:
            field.control_force.fill(0.0)
            field.control_damping.fill(0.0)
            continue

        active_ids: list[int] = []
        for domain in domains:
            active_ids.append(domain.track_id)
            if domain.track_id not in thinkers:
                tr = tracker.tracks[domain.track_id]
                thinker = create_thinker(
                    track_id=domain.track_id,
                    parents=tr.parents,
                    domain_center=np.asarray([domain.cx, domain.cy]),
                    t=field.t,
                    mode=mode,
                    thinkers=thinkers,
                    rng=rng,
                )
                thinkers[domain.track_id] = thinker

        global_error = normalized_error(energy, target_energy)
        global_score = stability_score(global_error)
        energy_errors.append(global_error)
        stability_scores.append(global_score)
        active_counts.append(len(domains))
        generation_max.append(
            max(
                (thinkers[d.track_id].generation for d in domains if d.track_id in thinkers),
                default=0,
            )
        )

        force_map = np.zeros((n, n), dtype=float)
        damping_map = np.zeros((n, n), dtype=float)
        signal_add = np.zeros((n, n), dtype=float)

        domain_by_id = {d.track_id: d for d in domains}
        active_thinkers = [
            thinkers[d.track_id]
            for d in domains
            if d.track_id in thinkers
        ]

        for thinker in active_thinkers:
            domain = domain_by_id[thinker.track_id]

            # Field-derived surfing + body binding + controller steering.
            phi_val = sample_bilinear(field.phi, thinker.scout_pos)
            vel_val = sample_bilinear(field.velocity, thinker.scout_pos)
            gx_field = (
                np.roll(field.phi, -1, 1) - np.roll(field.phi, 1, 1)
            ) / 2.0
            gy_field = (
                np.roll(field.phi, -1, 0) - np.roll(field.phi, 1, 0)
            ) / 2.0
            gx = sample_bilinear(gx_field, thinker.scout_pos)
            gy = sample_bilinear(gy_field, thinker.scout_pos)
            resource = sample_bilinear(field.r, thinker.scout_pos)
            local_energy = patch_mean(
                0.5 * field.velocity**2
                + 0.025 * field.phi**4,
                thinker.scout_pos,
                radius=2,
            )
            local_error = normalized_error(local_energy, target_energy)
            signal_here = sample_bilinear(field.signal, thinker.scout_pos)
            centroid = np.asarray([domain.cx, domain.cy], dtype=float)
            displacement = circular_displacement(
                thinker.scout_pos, centroid, n
            )
            scout_distance = float(np.linalg.norm(displacement))

            sensors = np.asarray(
                [
                    np.tanh(phi_val / 1.5),
                    np.tanh(vel_val / 1.5),
                    np.tanh(gx),
                    np.tanh(gy),
                    2.0 * resource - 1.0,
                    local_error,
                    global_error,
                    np.tanh(np.log1p(domain.mass) / 4.0 - 0.8),
                    np.tanh(scout_distance / 8.0),
                    np.clip(signal_here, -1.0, 1.0),
                ],
                dtype=float,
            )

            if mode == "none":
                action = np.zeros(len(ACTION_NAMES), dtype=float)
            elif mode == "homeostat":
                action = hand_homeostat(sensors)
            else:
                assert thinker.controller is not None
                action, thinker.hidden = thinker.controller.step(
                    sensors, thinker.hidden
                )

            field_grad = np.asarray([gx, gy], dtype=float)
            surf_force = (
                SCOUT_FIELD_SURF
                * abs(phi_val)
                * np.clip(field_grad, -2.0, 2.0)
            )
            thinker.scout_vel = (
                0.88 * thinker.scout_vel
                + SCOUT_BINDING * displacement
                + SCOUT_STEER * action[:2]
                + surf_force
            )
            speed = float(np.linalg.norm(thinker.scout_vel))
            if speed > SCOUT_MAX_SPEED:
                thinker.scout_vel *= SCOUT_MAX_SPEED / speed
            thinker.scout_pos = (
                thinker.scout_pos + thinker.scout_vel
            ) % n

            kernel = torus_gaussian(n, thinker.scout_pos, sigma=2.6)
            excite = float(action[2])
            damp = float(max(action[3], 0.0))
            emission = float(action[4])

            force_map += ACT_FORCE * excite * kernel
            damping_map += ACT_DAMP * damp * kernel
            signal_add += SIGNAL_WRITE * emission * kernel

            action_cost = float(
                0.35 * np.mean(action[:2] ** 2)
                + 0.35 * excite * excite
                + 0.20 * damp * damp
                + 0.10 * emission * emission
            )
            local_score = stability_score(local_error)
            thinker.selection_score += (
                0.60 * local_score
                + 0.40 * global_score
                - 0.08 * action_cost
            )
            thinker.global_score_sum += global_score
            thinker.local_score_sum += local_score
            thinker.action_cost_sum += action_cost
            thinker.control_steps += 1
            thinker.action_abs_sum += np.abs(action)
            thinker.sample_sum += np.asarray(
                [resource, abs(phi_val), local_energy, abs(signal_here)]
            )

        # Relationships are not preassigned. Count persistent close encounters.
        for i in range(len(active_thinkers)):
            for j in range(i + 1, len(active_thinkers)):
                a, b = active_thinkers[i], active_thinkers[j]
                if torus_distance(
                    tuple(a.scout_pos),
                    tuple(b.scout_pos),
                    (n, n),
                ) < 8.0:
                    a.contacts += 1
                    b.contacts += 1
                    key = tuple(sorted((a.track_id, b.track_id)))
                    contact_edges[key] += 1

        field.control_force = np.clip(force_map, -0.8, 0.8)
        field.control_damping = np.clip(damping_map, 0.0, 0.8)
        field.signal = np.clip(field.signal + signal_add, -1.0, 1.0)

        control_index += 1

        if checkpoint_dir is not None and field.t >= next_checkpoint:
            checkpoint(
                checkpoint_dir / f"{mode}_seed{seed}_t{field.t:07.1f}.json",
                mode=mode,
                seed=seed,
                field=field,
                thinkers=thinkers,
                active_ids=active_ids,
                target_energy=target_energy,
            )
            next_checkpoint += checkpoint_interval

    post_events = [
        e for e in tracker.events if float(e["t"]) >= warmup
    ]
    event_counts: defaultdict[str, int] = defaultdict(int)
    for event in post_events:
        event_counts[str(event["type"])] += 1

    active_ids = [d.track_id for d in tracker.previous]
    active_thinkers = [thinkers[i] for i in active_ids if i in thinkers]

    controllers = [
        t.controller
        for t in active_thinkers
        if t.controller is not None
    ]
    recovery = recovery_times(
        energy_errors, pulse_indices, control_dt
    )

    top_contacts = sorted(
        (
            {"a": int(a), "b": int(b), "count": int(count)}
            for (a, b), count in contact_edges.items()
        ),
        key=lambda row: row["count"],
        reverse=True,
    )[:20]

    long_thinkers = sorted(
        thinkers.values(),
        key=lambda t: (t.generation, t.control_steps),
        reverse=True,
    )[:20]

    result = {
        "mode": mode,
        "seed": seed,
        "config": {
            "n": n,
            "dt": DT,
            "t_end": t_end,
            "warmup": warmup,
            "control_every": CONTROL_EVERY,
            "pulse_interval": pulse_interval,
            "mutation_sigma": MUTATION_SIGMA,
            "sensors": SENSOR_NAMES,
            "actions": ACTION_NAMES,
        },
        "target_energy_density": target_energy,
        "global": {
            "mean_stability_score": float(np.mean(stability_scores))
            if stability_scores else 0.0,
            "mean_abs_energy_error": float(np.mean(np.abs(energy_errors)))
            if energy_errors else 0.0,
            "final_energy_density": field.energy_density(),
            "mean_active_domains": float(np.mean(active_counts))
            if active_counts else 0.0,
            "max_active_domains": int(max(active_counts, default=0)),
            "mean_recovery_time": float(np.mean(recovery))
            if recovery else None,
            "recovered_pulses": len(recovery),
            "pulses": len(pulse_events),
        },
        "demography": {
            "postwarmup_events": dict(event_counts),
            "tracker_genealogy": tracker.summary()["genealogy"],
            "thinkers_created": len(thinkers),
            "active_thinkers": len(active_thinkers),
            "max_controller_generation": int(
                max((t.generation for t in thinkers.values()), default=0)
            ),
        },
        "diversity": {
            "active_controller_probe_divergence": pairwise_behavior_divergence(
                [c for c in controllers if c is not None]
            ),
            "all_behavior_divergence": behavior_divergence(thinkers.values()),
        },
        "relationships": {
            "edge_count": len(contact_edges),
            "top_contacts": top_contacts,
        },
        "pulse_events": pulse_events,
        "top_thinkers": [t.to_summary() for t in long_thinkers],
    }

    if checkpoint_dir is not None:
        checkpoint(
            checkpoint_dir / f"{mode}_seed{seed}_FINAL.json",
            mode=mode,
            seed=seed,
            field=field,
            thinkers=thinkers,
            active_ids=active_ids,
            target_energy=target_energy,
        )

    return result


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = {
        "stability": [r["global"]["mean_stability_score"] for r in rows],
        "energy_error": [r["global"]["mean_abs_energy_error"] for r in rows],
        "active_domains": [r["global"]["mean_active_domains"] for r in rows],
        "controller_generation": [
            r["demography"]["max_controller_generation"] for r in rows
        ],
        "controller_divergence": [
            r["diversity"]["active_controller_probe_divergence"] for r in rows
        ],
        "behavior_divergence": [
            r["diversity"]["all_behavior_divergence"] for r in rows
        ],
    }

    out: dict[str, object] = {}
    for name, values in metrics.items():
        arr = np.asarray(values, dtype=float)
        out[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    recoveries = [
        r["global"]["mean_recovery_time"]
        for r in rows
        if r["global"]["mean_recovery_time"] is not None
    ]
    out["recovery_time"] = {
        "mean": float(np.mean(recoveries)) if recoveries else None,
        "std": float(np.std(recoveries)) if recoveries else None,
        "n": len(recoveries),
    }
    return out


def run_suite(
    *,
    n: int,
    t_end: float,
    warmup: float,
    pulse_interval: float,
    seeds: list[int],
) -> dict[str, object]:
    rows = {
        mode: [
            run_mode(
                mode,
                seed,
                n=n,
                t_end=t_end,
                warmup=warmup,
                pulse_interval=pulse_interval,
            )
            for seed in seeds
        ]
        for mode in MODES
    }
    return {
        "experiment": "Datarium 2 — Ecological Thinkers",
        "claim_boundary": (
            "Hybrid artificial life: field owns demography; a digital recurrent "
            "matrix is inherited along measured field lineages."
        ),
        "config": {
            "n": n,
            "t_end": t_end,
            "warmup": warmup,
            "pulse_interval": pulse_interval,
            "seeds": seeds,
        },
        "summary": {
            mode: summarize(rows[mode])
            for mode in MODES
        },
        "per_seed": rows,
    }


def print_suite(receipt: dict[str, object]) -> None:
    print("DATARIUM 2 — ECOLOGICAL THINKERS")
    print("=" * 86)
    print(
        f"{'mode':11s} {'stability':>14s} {'|E error|':>14s} "
        f"{'domains':>13s} {'ctrl-gen':>12s} {'ctrl-div':>12s} "
        f"{'beh-div':>11s} {'recovery':>11s}"
    )
    for mode in MODES:
        r = receipt["summary"][mode]
        recovery = r["recovery_time"]["mean"]
        rec = "—" if recovery is None else f"{recovery:.2f}"
        print(
            f"{mode:11s} "
            f"{r['stability']['mean']:6.3f}±{r['stability']['std']:.3f} "
            f"{r['energy_error']['mean']:6.3f}±{r['energy_error']['std']:.3f} "
            f"{r['active_domains']['mean']:6.2f}±{r['active_domains']['std']:.2f} "
            f"{r['controller_generation']['mean']:6.1f}±{r['controller_generation']['std']:.1f} "
            f"{r['controller_divergence']['mean']:6.3f}±{r['controller_divergence']['std']:.3f} "
            f"{r['behavior_divergence']['mean']:6.3f}±{r['behavior_divergence']['std']:.3f} "
            f"{rec:>11s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument(
        "--preset", choices=("ci", "overnight"), default="ci"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--t-end", type=float)
    parser.add_argument("--grid", type=int)
    parser.add_argument("--warmup", type=float)
    parser.add_argument("--pulse-interval", type=float)
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()

    if args.preset == "overnight":
        n = args.grid or 96
        t_end = args.t_end or 3000.0
        warmup = args.warmup or 30.0
        pulse_interval = args.pulse_interval or 120.0
        checkpoint_dir = args.checkpoint_dir or (
            ROOT / "results" / f"datarium2_overnight_seed{args.seed}"
        )
        mode = args.mode or "evolve"
        result = run_mode(
            mode,
            args.seed,
            n=n,
            t_end=t_end,
            warmup=warmup,
            pulse_interval=pulse_interval,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=250.0,
        )
        out = ROOT / "results" / f"datarium2_{mode}_overnight_seed{args.seed}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["global"], indent=2))
        print(json.dumps(result["demography"], indent=2))
        print(json.dumps(result["diversity"], indent=2))
        print(f"checkpoints: {checkpoint_dir}")
        print(f"final receipt: {out}")
        return

    n = args.grid or 64
    t_end = args.t_end or 130.0
    warmup = args.warmup or 16.0
    pulse_interval = args.pulse_interval or 34.0

    if args.mode:
        result = run_mode(
            args.mode,
            args.seed,
            n=n,
            t_end=t_end,
            warmup=warmup,
            pulse_interval=pulse_interval,
        )
        out = ROOT / "results" / f"datarium2_{args.mode}_seed{args.seed}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"wrote {out}")
        return

    receipt = run_suite(
        n=n,
        t_end=t_end,
        warmup=warmup,
        pulse_interval=pulse_interval,
        seeds=[0, 1],
    )
    out = ROOT / "results" / "datarium2.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print_suite(receipt)
    print(
        "\nGuardrail: EVOLVE earns nothing merely by having inherited matrices. "
        "It must beat RANDOM / NONE on ecological outcomes, while HOMEOSTAT "
        "shows whether the action channel can stabilize the field at all."
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
