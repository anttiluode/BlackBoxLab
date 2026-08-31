"""Datarium 4A — geography becomes body.

This is a deliberately narrow bridge experiment. Datarium 3 earned a slow
material geometry but that geometry was nailed to the lattice. Datarium 4A
asks whether a diffuse closed material phase can act as a movable compartment
under strictly local chemistry and local forces.

Important stopping line:
- the initial compartment is explicitly seeded;
- there is no reproduction, heredity, cell class, fitness or controller;
- the center of mass is measured only after the physics step and is never fed
  back into the dynamics.

The experiment isolates two prerequisites for later emergence:
1. a boundary can retain a passive cargo;
2. an asymmetric soluble food field can create asymmetric internal chemistry,
   local boundary stress and whole-compartment translation through an
   overdamped solvent field.

The next integration should replace the explicit seed with material written by
Datarium 3 builders, then ask whether builder-made closed boundaries can enter
the same body-physics regime.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def lap(a: np.ndarray) -> np.ndarray:
    return (
        np.roll(a, 1, axis=0)
        + np.roll(a, -1, axis=0)
        + np.roll(a, 1, axis=1)
        + np.roll(a, -1, axis=1)
        - 4.0 * a
    )


def grad(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) * 0.5
    gy = (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) * 0.5
    return gx, gy


def div_diffusion(c: np.ndarray, diffusivity: np.ndarray) -> np.ndarray:
    """Conservative div(D grad c) on the periodic grid."""
    dr = 0.5 * (diffusivity + np.roll(diffusivity, -1, axis=1))
    dl = 0.5 * (diffusivity + np.roll(diffusivity, 1, axis=1))
    dd = 0.5 * (diffusivity + np.roll(diffusivity, -1, axis=0))
    du = 0.5 * (diffusivity + np.roll(diffusivity, 1, axis=0))
    flux_r = dr * (np.roll(c, -1, axis=1) - c)
    flux_l = dl * (c - np.roll(c, 1, axis=1))
    flux_d = dd * (np.roll(c, -1, axis=0) - c)
    flux_u = du * (c - np.roll(c, 1, axis=0))
    return flux_r - flux_l + flux_d - flux_u


@dataclass(frozen=True)
class Config:
    n: int = 64
    dt: float = 0.03
    radius: float = 8.0
    interface_width: float = 0.9
    source_radius: float = 18.0
    source_sigma: float = 6.0
    movement_steps: int = 1500
    retention_steps: int = 1600
    active_stress: float = 10.0
    interface_diffusion: float = 0.40
    interface_reaction: float = 2.6
    volume_gain: float = 20.0
    advection_gain: float = 1.6
    solvent_viscosity: float = 0.9
    solvent_drag: float = 2.0
    food_diffusion: float = 0.65
    food_recovery: float = 0.08
    uptake_rate: float = 0.9
    food_cost: float = 0.16
    activator_diffusion: float = 0.12
    activator_gain: float = 0.70
    activator_decay: float = 0.06
    outside_decay: float = 0.55
    cargo_diffusion: float = 0.50
    membrane_barrier: float = 0.995


class BodyWorld:
    """Local field physics only; no object-level locomotion rule."""

    def __init__(
        self,
        config: Config,
        source_angle: float = 0.0,
        *,
        active_stress: float | None = None,
        uniform_food: bool = False,
        pinned: bool = False,
    ) -> None:
        self.cfg = config
        self.n = config.n
        self.yy, self.xx = np.mgrid[0 : self.n, 0 : self.n]
        self.cx0 = self.n / 2.0
        self.cy0 = self.n / 2.0
        self.source_angle = float(source_angle)
        self.source_unit = np.asarray(
            [np.cos(self.source_angle), np.sin(self.source_angle)], dtype=float
        )
        self.source_x = self.cx0 + config.source_radius * self.source_unit[0]
        self.source_y = self.cy0 + config.source_radius * self.source_unit[1]

        r = np.hypot(self.xx - self.cx0, self.yy - self.cy0)
        self.phi = 1.0 / (
            1.0 + np.exp((r - config.radius) / config.interface_width)
        )
        self.initial_mass = float(np.sum(self.phi))

        self.activator = np.zeros_like(self.phi)
        self.ux = np.zeros_like(self.phi)
        self.uy = np.zeros_like(self.phi)

        if uniform_food:
            self.food_target = np.full_like(self.phi, 0.22)
        else:
            dx = self.xx - self.source_x
            dy = self.yy - self.source_y
            self.food_target = 0.03 + 0.97 * np.exp(
                -(dx * dx + dy * dy) / (2.0 * config.source_sigma**2)
            )
        self.food = self.food_target.copy()

        self.active_stress = (
            config.active_stress if active_stress is None else float(active_stress)
        )
        self.pinned = bool(pinned)
        self.time = 0.0

    def interface(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gx, gy = grad(self.phi)
        magnitude = np.hypot(gx, gy) + 1e-12
        nx = gx / magnitude
        ny = gy / magnitude
        boundary = 4.0 * self.phi * (1.0 - self.phi)
        return boundary, nx, ny, magnitude

    def step(self) -> None:
        c = self.cfg
        dt = c.dt
        boundary, nx, ny, _ = self.interface()

        uptake = c.uptake_rate * boundary * self.food * (1.0 - self.activator)
        self.food += dt * (
            c.food_diffusion * lap(self.food)
            + c.food_recovery * (self.food_target - self.food)
            - c.food_cost * uptake
        )
        np.clip(self.food, 0.0, 1.2, out=self.food)

        self.activator += dt * (
            c.activator_diffusion * lap(self.activator)
            + c.activator_gain * uptake
            - c.activator_decay * self.activator
            - c.outside_decay * (1.0 - self.phi) * self.activator
        )
        np.clip(self.activator, 0.0, 2.0, out=self.activator)

        # Local active normal stress. There is no center-of-mass force,
        # direction-to-source vector, body heading or steering command.
        fx = -self.active_stress * boundary * self.activator * nx
        fy = -self.active_stress * boundary * self.activator * ny

        self.ux += dt * (
            c.solvent_viscosity * lap(self.ux) - c.solvent_drag * self.ux + fx
        )
        self.uy += dt * (
            c.solvent_viscosity * lap(self.uy) - c.solvent_drag * self.uy + fy
        )

        if not self.pinned:
            phix, phiy = grad(self.phi)
            advection = self.ux * phix + self.uy * phiy
            relative_volume_error = (
                self.initial_mass - float(np.sum(self.phi))
            ) / self.initial_mass
            interface_restore = (
                c.interface_reaction
                * self.phi
                * (1.0 - self.phi)
                * (self.phi - 0.5)
            )
            volume_restore = (
                c.volume_gain
                * relative_volume_error
                * self.phi
                * (1.0 - self.phi)
            )
            self.phi += dt * (
                -c.advection_gain * advection
                + c.interface_diffusion * lap(self.phi)
                + interface_restore
                + volume_restore
            )
            np.clip(self.phi, 0.0, 1.0, out=self.phi)

        self.time += dt

    def center(self) -> np.ndarray:
        mass = float(np.sum(self.phi)) + 1e-12
        return np.asarray(
            [
                float(np.sum(self.phi * self.xx) / mass),
                float(np.sum(self.phi * self.yy) / mass),
            ]
        )

    def chemistry_polarity(self) -> float:
        boundary, _, _, _ = self.interface()
        dx = self.xx - self.cx0
        dy = self.yy - self.cy0
        signed = dx * self.source_unit[0] + dy * self.source_unit[1]
        front = signed >= 0.0
        back = ~front
        front_mean = float(np.sum(self.activator * boundary * front)) / (
            float(np.sum(boundary * front)) + 1e-12
        )
        back_mean = float(np.sum(self.activator * boundary * back)) / (
            float(np.sum(boundary * back)) + 1e-12
        )
        return front_mean - back_mean

    def movement_summary(self) -> dict[str, float]:
        center = self.center()
        displacement = center - np.asarray([self.cx0, self.cy0])
        toward = float(displacement @ self.source_unit)
        perpendicular = float(
            abs(
                self.source_unit[0] * displacement[1]
                - self.source_unit[1] * displacement[0]
            )
        )
        return {
            "toward_source": toward,
            "perpendicular": perpendicular,
            "distance": float(np.linalg.norm(displacement)),
            "chemistry_polarity": self.chemistry_polarity(),
            "mass_ratio": float(np.sum(self.phi)) / self.initial_mass,
        }


def run_movement_arm(
    arm: str,
    source_angle: float,
    config: Config,
) -> dict[str, float | str]:
    kwargs: dict[str, object] = {}
    if arm == "active":
        pass
    elif arm == "uniform_food":
        kwargs["uniform_food"] = True
    elif arm == "no_stress":
        kwargs["active_stress"] = 0.0
    elif arm == "pinned":
        kwargs["pinned"] = True
    else:
        raise ValueError(arm)

    world = BodyWorld(config, source_angle, **kwargs)
    for _ in range(config.movement_steps):
        world.step()
    return {"arm": arm, "angle": source_angle, **world.movement_summary()}


def retention_trial(config: Config, barrier: float) -> dict[str, float]:
    n = config.n
    yy, xx = np.mgrid[0:n, 0:n]
    center = n / 2.0
    r = np.hypot(xx - center, yy - center)
    phi = 1.0 / (
        1.0 + np.exp((r - config.radius) / config.interface_width)
    )
    cargo = phi.copy()
    initial_inside = float(np.sum(cargo * phi))

    for _ in range(config.retention_steps):
        boundary = 4.0 * phi * (1.0 - phi)
        diffusivity = config.cargo_diffusion * (1.0 - barrier * boundary)
        diffusivity = np.maximum(diffusivity, config.cargo_diffusion * 0.001)
        cargo += config.dt * div_diffusion(cargo, diffusivity)
        np.maximum(cargo, 0.0, out=cargo)

    inside = float(np.sum(cargo * phi))
    outside = float(np.sum(cargo * (1.0 - phi)))
    inside_weight = float(np.sum(phi))
    outside_weight = float(np.sum(1.0 - phi))
    inside_mean = inside / (inside_weight + 1e-12)
    outside_mean = outside / (outside_weight + 1e-12)
    return {
        "barrier": barrier,
        "retained_inside_fraction": inside / (initial_inside + 1e-12),
        "inside_outside_concentration_ratio": inside_mean
        / (outside_mean + 1e-12),
        "cargo_conservation": float(np.sum(cargo))
        / (float(np.sum(phi)) + 1e-12),
    }


def aggregate(rows: Iterable[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    rows = list(rows)
    metrics = (
        "toward_source",
        "perpendicular",
        "distance",
        "chemistry_polarity",
        "mass_ratio",
    )
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return result


def run_receipt(config: Config, angles: int) -> dict[str, object]:
    source_angles = [2.0 * np.pi * i / angles + 0.11 for i in range(angles)]
    arms = ("active", "uniform_food", "no_stress", "pinned")
    movement = {
        arm: [run_movement_arm(arm, angle, config) for angle in source_angles]
        for arm in arms
    }
    selective = retention_trial(config, config.membrane_barrier)
    open_membrane = retention_trial(config, 0.0)
    return {
        "config": {**config.__dict__, "angles": angles},
        "movement_summary": {
            arm: aggregate(rows)
            for arm, rows in movement.items()
        },
        "movement_rows": movement,
        "retention": {
            "selective_membrane": selective,
            "no_barrier": open_membrane,
        },
    }


def preset(name: str) -> tuple[Config, int]:
    if name == "ci":
        return (
            Config(
                n=48,
                radius=6.0,
                source_radius=13.0,
                source_sigma=5.0,
                movement_steps=700,
                retention_steps=700,
            ),
            4,
        )
    if name == "receipt":
        return Config(), 8
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("ci", "receipt"), default="ci")
    parser.add_argument(
        "--write",
        action="store_true",
        help="write results/datarium4.json",
    )
    args = parser.parse_args()

    config, angles = preset(args.preset)
    receipt = run_receipt(config, angles)

    print("BlackBoxLab — Datarium 4A: geography becomes body")
    print(
        f"{'arm':14s} {'toward source':>18s} {'sideways':>18s} "
        f"{'chem polarity':>18s} {'mass ratio':>18s}"
    )
    for arm, row in receipt["movement_summary"].items():
        print(
            f"{arm:14s} "
            f"{row['toward_source']['mean']:8.3f}±{row['toward_source']['std']:.3f} "
            f"{row['perpendicular']['mean']:8.3f}±{row['perpendicular']['std']:.3f} "
            f"{row['chemistry_polarity']['mean']:8.4f}±"
            f"{row['chemistry_polarity']['std']:.4f} "
            f"{row['mass_ratio']['mean']:8.3f}±{row['mass_ratio']['std']:.3f}"
        )

    print("\nretention")
    for name, row in receipt["retention"].items():
        print(
            f"{name:20s} retained={row['retained_inside_fraction']:.3f} "
            f"inside/out={row['inside_outside_concentration_ratio']:.2f}x"
        )

    print(
        "\nStopping line: this validates movable compartment physics from an "
        "explicitly seeded boundary. It does not yet show a builder-made cell, "
        "reproduction or heredity."
    )

    if args.write:
        out = Path(__file__).resolve().parents[1] / "results" / "datarium4.json"
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
