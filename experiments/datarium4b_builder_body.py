"""Datarium 4B — remove the explicit body seed.

Datarium 4A proved only a mechanics bridge: if a diffuse closed phase already
exists, local chemistry and local stress can move it. Datarium 4B asks whether
Datarium-3-style builders can *create the phase that enters that body physics*
without a pre-drawn disk.

The deliberately narrow generative chain is:

    identical wave-writing builders
            ->
    slow builder-written material scaffold
            ->
    local catalytic conversion of soluble body precursor
            ->
    diffuse bulk phase with an interface
            ->
    builders + fast wave removed
            ->
    local chemistry / stress from Datarium 4A
            ->
    measured body translation

There is still no Cell object, genome, parent ID, body heading, steering rule,
fitness score, or reproduction. The scaffold-to-phase conversion is local and
uses a fixed material threshold; it does not detect a loop, component, center,
or desired shape.

Important stopping line: this is a coacervate-like "builder catalyses a second
phase" route, not yet proof that builders literally weave a membrane ring.
The body boundary is the interface of the second phase.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datarium.lineage import periodic_components
from experiments.datarium3_layers import Config as D3Config
from experiments.datarium3_layers import MaterialWorld
from experiments.datarium4_body import BodyWorld
from experiments.datarium4_body import Config as BodyConfig
from experiments.datarium4_body import div_diffusion, lap


@dataclass(frozen=True)
class Config:
    n: int = 48
    agents: int = 72
    builder_steps: int = 2600
    phase_steps: int = 220
    move_steps: int = 900
    cargo_steps: int = 700

    # Local scaffold -> phase chemistry. No run-normalization or component ID.
    scaffold_half: float = 0.028
    scaffold_sharpness: float = 0.010
    phase_gain: float = 0.95
    phase_diffusion: float = 0.18
    phase_sharpen: float = 1.45
    phase_threshold: float = 0.50

    source_radius: float = 13.0
    source_sigma: float = 5.0
    body_active_stress: float = 10.0
    body_volume_gain: float = 12.0
    membrane_barrier: float = 0.995
    cargo_diffusion: float = 0.50


def _d3_config(config: Config, *, wave_source: float = 1.0) -> D3Config:
    return D3Config(
        n=config.n,
        agents=config.agents,
        build_steps=config.builder_steps,
        probe_steps=1,
        wave_source=wave_source,
    )


def build_scaffold(
    config: Config,
    seed: int,
    mode: str,
) -> MaterialWorld:
    """Run the Datarium 3 builder physics and return its slow material."""
    if mode == "full":
        world = MaterialWorld(_d3_config(config), seed=seed)
        feedback = "tensor"
    elif mode == "write_only":
        world = MaterialWorld(_d3_config(config), seed=seed)
        feedback = "write_only"
    elif mode == "no_wave_production":
        world = MaterialWorld(
            _d3_config(config, wave_source=0.0),
            seed=seed,
        )
        feedback = "tensor"
    else:
        raise ValueError(mode)

    for _ in range(config.builder_steps):
        world.step(feedback)
    return world


def scaffold_variant(
    matrix: np.ndarray,
    variant: str,
    seed: int,
) -> np.ndarray:
    """Apply post-build attackers without changing the material histogram."""
    matrix = np.asarray(matrix, dtype=float)
    if variant == "intact":
        return matrix.copy()
    if variant == "scrambled":
        rng = np.random.default_rng(seed)
        return matrix.ravel()[rng.permutation(matrix.size)].reshape(matrix.shape)
    if variant == "mean_field":
        return np.full_like(matrix, float(np.mean(matrix)))
    if variant == "erased":
        return np.zeros_like(matrix)
    raise ValueError(variant)


def phase_from_scaffold(
    scaffold: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Local fixed-law conversion from builder scaffold into a second phase.

    The catalyst is a sigmoid of *local absolute scaffold amount*. There is no
    division by run mean/max and no component detection. The subsequent
    diffusion + bistable sharpening is also local.
    """
    scaffold = np.asarray(scaffold, dtype=float)
    if scaffold.shape != (config.n, config.n):
        raise ValueError(scaffold.shape)

    width = max(config.scaffold_sharpness, 1e-9)
    raw_catalyst = 1.0 / (
        1.0 + np.exp(-(scaffold - config.scaffold_half) / width)
    )
    # Exactly zero written scaffold must make exactly zero second phase.  The
    # subtraction removes the sigmoid's nonzero baseline without consulting
    # any run statistic.
    baseline = 1.0 / (1.0 + np.exp(config.scaffold_half / width))
    catalyst = np.clip(
        (raw_catalyst - baseline) / (1.0 - baseline),
        0.0,
        1.0,
    )
    phi = np.zeros_like(scaffold)
    precursor = np.ones_like(scaffold)
    dt = 0.035

    for _ in range(config.phase_steps):
        production = (
            config.phase_gain
            * catalyst
            * precursor
            * (1.0 - phi)
        )
        # Local phase maturation. The cubic term sharpens a diffuse mixed
        # region into low/high phase without naming a body or boundary.
        sharpening = (
            config.phase_sharpen
            * phi
            * (1.0 - phi)
            * (phi - 0.42)
        )
        phi += dt * (
            production
            + config.phase_diffusion * lap(phi)
            + sharpening
        )
        precursor -= dt * production
        np.clip(phi, 0.0, 1.0, out=phi)
        np.clip(precursor, 0.0, 1.0, out=precursor)

    return phi


def phase_metrics(phi: np.ndarray, threshold: float) -> dict[str, float | int]:
    mask = phi >= threshold
    components = periodic_components(mask)
    sizes = [len(component) for component in components]
    largest = max(sizes, default=0)

    # Boundary/interior ratio is a compactness attacker: scattered pixels have
    # high exposed perimeter relative to area; bulk droplets are lower.
    exposed = np.zeros_like(mask, dtype=float)
    if np.any(mask):
        neighbors = (
            np.roll(mask, 1, axis=0).astype(int)
            + np.roll(mask, -1, axis=0).astype(int)
            + np.roll(mask, 1, axis=1).astype(int)
            + np.roll(mask, -1, axis=1).astype(int)
        )
        exposed[mask] = 4 - neighbors[mask]

    return {
        "phase_mean": float(np.mean(phi)),
        "phase_mass": float(np.sum(phi)),
        "above_threshold_fraction": float(np.mean(mask)),
        "component_count": len(components),
        "largest_component_cells": largest,
        "largest_component_fraction_of_phase": (
            largest / max(int(np.sum(mask)), 1)
        ),
        "exposed_edges_per_phase_cell": (
            float(np.sum(exposed)) / max(float(np.sum(mask)), 1.0)
        ),
    }


def cargo_retention(
    phi: np.ndarray,
    config: Config,
    *,
    barrier: float,
) -> dict[str, float]:
    """Passive cargo assay on the builder-made phase, no object ID in physics."""
    cargo = np.clip(phi.copy(), 0.0, 1.0)
    initial_inside = float(np.sum(cargo * phi))
    initial_total = float(np.sum(cargo))

    for _ in range(config.cargo_steps):
        boundary = 4.0 * phi * (1.0 - phi)
        diffusivity = config.cargo_diffusion * (1.0 - barrier * boundary)
        diffusivity = np.maximum(
            diffusivity,
            config.cargo_diffusion * 0.001,
        )
        cargo += 0.03 * div_diffusion(cargo, diffusivity)
        np.maximum(cargo, 0.0, out=cargo)

    inside = float(np.sum(cargo * phi))
    outside = float(np.sum(cargo * (1.0 - phi)))
    inside_weight = float(np.sum(phi))
    outside_weight = float(np.sum(1.0 - phi))
    return {
        "retained_inside_fraction": inside / (initial_inside + 1e-12),
        "inside_outside_ratio": (
            inside / (inside_weight + 1e-12)
        ) / (
            outside / (outside_weight + 1e-12) + 1e-12
        ),
        "cargo_conservation": float(np.sum(cargo)) / (initial_total + 1e-12),
    }


def body_config(config: Config) -> BodyConfig:
    return BodyConfig(
        n=config.n,
        radius=6.0,  # unused when initial_phi is supplied
        source_radius=config.source_radius,
        source_sigma=config.source_sigma,
        movement_steps=config.move_steps,
        retention_steps=config.cargo_steps,
        active_stress=config.body_active_stress,
        volume_gain=config.body_volume_gain,
        membrane_barrier=config.membrane_barrier,
        cargo_diffusion=config.cargo_diffusion,
    )


def move_phase(
    phi: np.ndarray,
    config: Config,
    source_angle: float,
    *,
    active_stress: float | None = None,
    pinned: bool = False,
    uniform_food: bool = False,
) -> dict[str, float]:
    world = BodyWorld(
        body_config(config),
        source_angle=source_angle,
        initial_phi=phi,
        active_stress=active_stress,
        pinned=pinned,
        uniform_food=uniform_food,
    )
    for _ in range(config.move_steps):
        world.step()
    return world.movement_summary()


def one_seed(config: Config, seed: int) -> dict[str, object]:
    full = build_scaffold(config, seed, "full")
    write_only = build_scaffold(config, seed, "write_only")
    no_wave = build_scaffold(config, seed, "no_wave_production")

    scaffolds = {
        "intact": full.matrix.copy(),
        "scrambled": scaffold_variant(full.matrix, "scrambled", seed + 1000),
        "mean_field": scaffold_variant(full.matrix, "mean_field", seed + 1000),
        "write_only": write_only.matrix.copy(),
        "no_wave_production": no_wave.matrix.copy(),
        "erased": np.zeros_like(full.matrix),
    }

    phases = {
        name: phase_from_scaffold(scaffold, config)
        for name, scaffold in scaffolds.items()
    }
    angle = 0.29 + (seed % 8) * np.pi / 4.0

    rows: dict[str, object] = {}
    for name, phi in phases.items():
        metrics = phase_metrics(phi, config.phase_threshold)
        if float(metrics["phase_mass"]) > 1e-6:
            retention = cargo_retention(
                phi,
                config,
                barrier=config.membrane_barrier,
            )
            open_retention = cargo_retention(phi, config, barrier=0.0)
        else:
            retention = {
                "retained_inside_fraction": 0.0,
                "inside_outside_ratio": 0.0,
                "cargo_conservation": 1.0,
            }
            open_retention = retention.copy()

        # Movement is meaningful only for a nontrivial phase. Keep the failure
        # explicit instead of silently inserting a fallback disk.
        if float(metrics["phase_mass"]) >= 5.0:
            movement = move_phase(phi, config, angle)
        else:
            movement = {
                "toward_source": 0.0,
                "perpendicular": 0.0,
                "distance": 0.0,
                "chemistry_polarity": 0.0,
                "mass_ratio": 0.0,
            }

        rows[name] = {
            "scaffold_mean": float(np.mean(scaffolds[name])),
            "scaffold_std": float(np.std(scaffolds[name])),
            **metrics,
            "retention": retention,
            "open_retention": open_retention,
            "movement": movement,
        }

    # Mechanism attackers on the exact intact builder-made phase.
    intact_phi = phases["intact"]
    rows["intact_no_stress"] = {
        "movement": move_phase(
            intact_phi,
            config,
            angle,
            active_stress=0.0,
        )
    }
    rows["intact_pinned"] = {
        "movement": move_phase(
            intact_phi,
            config,
            angle,
            pinned=True,
        )
    }
    rows["intact_uniform_food"] = {
        "movement": move_phase(
            intact_phi,
            config,
            angle,
            uniform_food=True,
        )
    }

    return {
        "seed": seed,
        "angle": angle,
        "rows": rows,
    }


def _aggregate_scalar(values: Iterable[float]) -> dict[str, float]:
    a = np.asarray(list(values), dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std())}


def summarize(receipts: list[dict[str, object]]) -> dict[str, object]:
    names = (
        "intact",
        "scrambled",
        "mean_field",
        "write_only",
        "no_wave_production",
        "erased",
    )
    summary: dict[str, object] = {}
    for name in names:
        rows = [receipt["rows"][name] for receipt in receipts]
        summary[name] = {
            "scaffold_mean": _aggregate_scalar(
                float(row["scaffold_mean"]) for row in rows
            ),
            "scaffold_std": _aggregate_scalar(
                float(row["scaffold_std"]) for row in rows
            ),
            "phase_mass": _aggregate_scalar(
                float(row["phase_mass"]) for row in rows
            ),
            "largest_component_fraction_of_phase": _aggregate_scalar(
                float(row["largest_component_fraction_of_phase"])
                for row in rows
            ),
            "exposed_edges_per_phase_cell": _aggregate_scalar(
                float(row["exposed_edges_per_phase_cell"])
                for row in rows
            ),
            "retained_inside_fraction": _aggregate_scalar(
                float(row["retention"]["retained_inside_fraction"])
                for row in rows
            ),
            "open_retained_inside_fraction": _aggregate_scalar(
                float(row["open_retention"]["retained_inside_fraction"])
                for row in rows
            ),
            "toward_source": _aggregate_scalar(
                float(row["movement"]["toward_source"])
                for row in rows
            ),
        }

    for attacker in (
        "intact_no_stress",
        "intact_pinned",
        "intact_uniform_food",
    ):
        summary[attacker] = {
            "toward_source": _aggregate_scalar(
                float(receipt["rows"][attacker]["movement"]["toward_source"])
                for receipt in receipts
            )
        }
    return summary


def run_suite(config: Config, seeds: Iterable[int]) -> dict[str, object]:
    rows = [one_seed(config, int(seed)) for seed in seeds]
    return {
        "config": asdict(config),
        "summary": summarize(rows),
        "per_seed": rows,
    }


def preset(name: str) -> tuple[Config, tuple[int, ...]]:
    if name == "smoke":
        return (
            Config(
                n=32,
                agents=36,
                builder_steps=300,
                phase_steps=100,
                move_steps=180,
                cargo_steps=140,
                source_radius=8.0,
                source_sigma=3.5,
            ),
            (0,),
        )
    if name == "ci":
        return (
            Config(
                n=40,
                agents=52,
                builder_steps=900,
                phase_steps=150,
                move_steps=320,
                cargo_steps=260,
                source_radius=10.0,
                source_sigma=4.2,
            ),
            (0, 1),
        )
    if name == "receipt":
        return Config(), (0, 1, 2, 3)
    raise ValueError(name)


def print_receipt(receipt: dict[str, object]) -> None:
    print("BlackBoxLab — Datarium 4B: remove the explicit body seed")
    print(
        f"{'arm':20s} {'scaf mean':>10s} {'phase':>10s} "
        f"{'largest':>10s} {'edges':>10s} {'retain':>10s} {'move':>10s}"
    )
    for name in (
        "intact",
        "scrambled",
        "mean_field",
        "write_only",
        "no_wave_production",
        "erased",
    ):
        row = receipt["summary"][name]
        print(
            f"{name:20s} "
            f"{row['scaffold_mean']['mean']:10.4f} "
            f"{row['phase_mass']['mean']:10.2f} "
            f"{row['largest_component_fraction_of_phase']['mean']:10.3f} "
            f"{row['exposed_edges_per_phase_cell']['mean']:10.3f} "
            f"{row['retained_inside_fraction']['mean']:10.3f} "
            f"{row['toward_source']['mean']:10.3f}"
        )
    print("\nmovement mechanism attackers")
    for name in (
        "intact_no_stress",
        "intact_pinned",
        "intact_uniform_food",
    ):
        value = receipt["summary"][name]["toward_source"]["mean"]
        print(f"{name:20s} {value:10.4f}")
    print(
        "\nStopping line: builders locally catalyse a movable second phase; "
        "this is not yet a woven membrane, reproduction, heredity or a cell "
        "lineage."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=("smoke", "ci", "receipt"),
        default="ci",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "datarium4b.json",
    )
    args = parser.parse_args()

    config, seeds = preset(args.preset)
    receipt = run_suite(config, seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print_receipt(receipt)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
