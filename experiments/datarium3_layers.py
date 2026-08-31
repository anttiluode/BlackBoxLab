"""Datarium 3 — can coherent motion write a higher causal layer?

This experiment keeps the useful kernel of the supplied ``train.html`` and
removes its authored ontology.  There are no writing types, roles, genomes,
fitness values, reproduction rules, links, or Train objects.  Identical
oscillating particles only:

* emit into a fast local-budget wave field;
* turn using local gradients of recent wave energy;
* consume no globally assigned reward;
* convert a separate precursor into slow oriented material when local motion
  and emission are simultaneously dense and coherent.

The oriented material is a small 2-D nematic tensor.  It can alter wave memory
and guide later motion, like a deliberately abstract mixture of a furrow,
fibre, viscosity change and anisotropic conductivity.  It is chemistry-level
affordance, not a named higher object.

The decisive intervention removes all builders, zeros the fast field and
releases matched naive particles into intact, isotropic, scrambled, rotated,
or erased material.  If intact geometry changes the new population more than
the controls, history has survived outside the agents and acquired downward
causal influence.  That still is not heredity or open-ended evolution.
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

from datarium.layers import CoherentAssemblyTracker
from datarium.lineage import periodic_components


ARMS = ("no_memory", "write_only", "scalar", "mean_field", "tensor")
PROBES = (
    "intact",
    "isotropic",
    "patchwork",
    "scrambled",
    "rotated",
    "erased",
)
ABLATIONS = (
    "full",
    "no_wave_sensing",
    "no_wave_production",
    "phase_shuffle_each_step",
    "uniform_resource",
)


@dataclass(frozen=True)
class Config:
    n: int = 48
    agents: int = 72
    dt: float = 0.04
    build_steps: int = 3000
    probe_steps: int = 900
    observe_every: int = 25

    wave_diffusion: float = 1.20
    wave_rotation: float = 2.00
    wave_damping: float = 0.200
    wave_source: float = 1.00
    wave_resource_gain: float = 0.10
    wave_nonlinearity: float = 0.040
    wave_memory: float = 0.970
    wave_memory_material_gain: float = 0.020
    wave_speed_material_gain: float = 0.16
    wave_tensor_gain: float = 0.10

    resource_recovery: float = 100.0
    resource_burn: float = 0.012
    resource_diffusion: float = 0.020

    material_rate: float = 0.120
    material_decay: float = 0.004
    precursor_diffusion: float = 0.016
    cooperative_half_density: float = 0.16
    polymerization_wave_half: float = 0.003
    polymerization_flow_power: float = 4.0

    wave_attraction: float = 0.0
    resonant_wave_attraction: float = 2.0
    density_repulsion: float = 5.0
    tensor_guidance: float = 40.0
    phase_lock: float = 2.00
    oscillator_frequency: float = 2.0
    particle_speed: float = 0.80
    rotational_noise: float = 0.040

    local_smoothing_steps: int = 4
    local_smoothing_rate: float = 0.16
    envelope_diffusion: float = 0.040
    matrix_threshold: float = 0.20


def laplacian(field: np.ndarray) -> np.ndarray:
    """Periodic five-point Laplacian over the final two axes."""

    return (
        np.roll(field, 1, axis=-2)
        + np.roll(field, -1, axis=-2)
        + np.roll(field, 1, axis=-1)
        + np.roll(field, -1, axis=-1)
        - 4.0 * field
    )


def sample_nearest(field: np.ndarray, positions: np.ndarray) -> np.ndarray:
    n = field.shape[0]
    x = np.floor(positions[:, 0]).astype(int) % n
    y = np.floor(positions[:, 1]).astype(int) % n
    return field[y, x]


def _central_gradients(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / 2.0
    gy = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / 2.0
    return gx, gy


class MaterialWorld:
    """Fast wave, local resource, identical particles and slow material."""

    def __init__(self, config: Config, seed: int, *, quiet_field: bool = False):
        self.config = config
        self.rng = np.random.default_rng(seed)
        n, count = config.n, config.agents

        # Two real channels are the quadratures of one fast oscillatory field.
        # Keeping phase in the medium lets a particle respond differently to
        # an in-phase and an anti-phase wave without reading another particle.
        self.wave_re = np.zeros((n, n), dtype=float)
        self.wave_im = np.zeros((n, n), dtype=float)
        if not quiet_field:
            # A weak descendant of Antti's moving instanton supplies an initial
            # asymmetry.  It is not detected or tracked by the dynamics.
            y, x = np.ogrid[:n, :n]
            cx, cy = n * 0.25, n * 0.50
            radius = max(4.0, n / 10.0)
            distance = np.hypot(x - cx, y - cy)
            amplitude = 0.35 / np.cosh(distance / radius)
            phase_ramp = 0.22 * (x - cx)
            self.wave_re[:] = amplitude * np.cos(phase_ramp)
            self.wave_im[:] = amplitude * np.sin(phase_ramp)

        self.resource = np.ones((n, n), dtype=float)
        self.precursor = np.ones((n, n), dtype=float)
        self.matrix = np.zeros((n, n), dtype=float)
        self.q1 = np.zeros((n, n), dtype=float)
        self.q2 = np.zeros((n, n), dtype=float)
        self.envelope = np.zeros((n, n), dtype=float)

        self.positions = self.rng.uniform(0.0, n, (count, 2))
        self.headings = self.rng.uniform(0.0, 2.0 * np.pi, count)
        self.phases = self.rng.uniform(0.0, 2.0 * np.pi, count)
        # Every particle has the same intrinsic parameters.  These arrays are
        # deliberately absent: type, role, genome, fitness, energy, parent.
        self.frequencies = np.full(
            count, config.oscillator_frequency, dtype=float
        )

        self.t = 0.0
        self.step_index = 0
        self.last_density = np.zeros((n, n), dtype=float)
        self.last_polymerization = np.zeros((n, n), dtype=float)

    def _local_maps(self, *, shuffle_phases: bool = False) -> np.ndarray:
        config = self.config
        n = config.n
        x = np.floor(self.positions[:, 0]).astype(int) % n
        y = np.floor(self.positions[:, 1]).astype(int) % n
        phases = self.phases
        if shuffle_phases:
            phases = phases[self.rng.permutation(config.agents)]

        values = (
            np.ones(config.agents),
            np.sin(phases),
            np.cos(phases),
            np.cos(self.headings),
            np.sin(self.headings),
            np.cos(2.0 * self.headings),
            np.sin(2.0 * self.headings),
        )
        fields = np.zeros((len(values), n, n), dtype=float)
        for channel, value in enumerate(values):
            np.add.at(fields[channel], (y, x), value)

        for _ in range(config.local_smoothing_steps):
            fields += config.local_smoothing_rate * laplacian(fields)
        return fields

    def polymerization_map(self, local_maps: np.ndarray) -> np.ndarray:
        density, sin_phase, cos_phase, hx, hy, _, _ = local_maps
        phase_coherence = np.hypot(sin_phase, cos_phase) / (density + 1e-9)
        motion_coherence = np.hypot(hx, hy) / (density + 1e-9)
        joint_coherence = np.clip(
            phase_coherence * motion_coherence, 0.0, 1.0
        )

        half = self.config.cooperative_half_density
        density4 = density**4
        cooperative = density4 / (density4 + half**4 + 1e-12)

        # Materialization is not permitted from traffic alone.  It requires
        # local wave energy and positive mechanical agreement between the
        # population's motion flux and the gradient of that energy.  This is
        # a local reaction term: no measured assembly identity enters it.
        re_x, re_y = _central_gradients(self.wave_re)
        im_x, im_y = _central_gradients(self.wave_im)
        # The collective phasor selects the force that this local population
        # actually experiences from the two wave quadratures.
        force_x = cos_phase * re_x + sin_phase * im_x
        force_y = cos_phase * re_y + sin_phase * im_y
        force_magnitude = np.hypot(force_x, force_y)
        forward_flow = np.maximum(hx * force_x + hy * force_y, 0.0)
        flow_alignment = np.clip(
            forward_flow / (density * force_magnitude + 1e-9),
            0.0,
            1.0,
        )
        wave_gate = self.envelope / (
            self.envelope + self.config.polymerization_wave_half
        )
        return (
            self.config.material_rate
            * self.precursor
            * cooperative
            * joint_coherence**4
            * wave_gate
            * flow_alignment**self.config.polymerization_flow_power
        )

    def _write_material(self, local_maps: np.ndarray) -> None:
        config = self.config
        dt = config.dt
        density, _, _, _, _, cos2, sin2 = local_maps
        polymerization = self.polymerization_map(local_maps)
        decay = config.material_decay * self.matrix

        self.matrix += dt * (polymerization - decay)
        self.precursor += dt * (
            config.precursor_diffusion * laplacian(self.precursor)
            - polymerization
            + decay
        )
        self.q1 += dt * (
            polymerization * cos2 / (density + 1e-9)
            - config.material_decay * self.q1
        )
        self.q2 += dt * (
            polymerization * sin2 / (density + 1e-9)
            - config.material_decay * self.q2
        )

        np.clip(self.matrix, 0.0, 1.0, out=self.matrix)
        np.clip(self.precursor, 0.0, 1.0, out=self.precursor)
        magnitude = np.hypot(self.q1, self.q2)
        scale = np.minimum(1.0, self.matrix / (magnitude + 1e-12))
        self.q1 *= scale
        self.q2 *= scale
        self.last_polymerization = polymerization

    def feedback_fields(
        self,
        mode: str,
    ) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
        if mode in ("no_memory", "write_only", "erased"):
            return 0.0, 0.0, 0.0
        if mode in ("scalar", "isotropic"):
            return self.matrix, 0.0, 0.0
        if mode == "mean_field":
            return float(np.mean(self.matrix)), 0.0, 0.0
        if mode in (
            "tensor",
            "intact",
            "patchwork",
            "scrambled",
            "rotated",
        ):
            return self.matrix, self.q1, self.q2
        raise ValueError(f"unknown feedback mode {mode!r}")

    def step(
        self,
        mode: str,
        *,
        write_material: bool | None = None,
        freeze_material: bool = False,
        shuffle_phases: bool = False,
        uniform_resource: bool = False,
    ) -> None:
        config = self.config
        dt = config.dt
        if write_material is None:
            write_material = mode != "no_memory"

        local_maps = self._local_maps(shuffle_phases=shuffle_phases)
        self.last_density = local_maps[0]
        if write_material and not freeze_material:
            self._write_material(local_maps)
        else:
            self.last_polymerization.fill(0.0)

        material, q1, q2 = self.feedback_fields(mode)

        def material_wave_operator(field: np.ndarray) -> np.ndarray:
            xx = (
                np.roll(field, -1, axis=1)
                - 2.0 * field
                + np.roll(field, 1, axis=1)
            )
            yy = (
                np.roll(field, -1, axis=0)
                - 2.0 * field
                + np.roll(field, 1, axis=0)
            )
            xy = (
                np.roll(np.roll(field, -1, axis=0), -1, axis=1)
                - np.roll(np.roll(field, -1, axis=0), 1, axis=1)
                - np.roll(np.roll(field, 1, axis=0), -1, axis=1)
                + np.roll(np.roll(field, 1, axis=0), 1, axis=1)
            ) / 4.0
            isotropic = (
                1.0 + config.wave_speed_material_gain * material
            ) * (xx + yy)
            anisotropic = config.wave_tensor_gain * (
                q1 * (xx - yy) + 2.0 * q2 * xy
            )
            return config.wave_diffusion * (isotropic + anisotropic)

        magnitude2 = self.wave_re**2 + self.wave_im**2
        damping = config.wave_damping * (1.0 - 0.70 * material)
        growth = (
            config.wave_resource_gain * self.resource
            - damping
            - config.wave_nonlinearity * magnitude2
        )
        delta_re = (
            material_wave_operator(self.wave_re)
            - config.wave_rotation * self.wave_im
            + growth * self.wave_re
            + config.wave_source * local_maps[2]
        )
        delta_im = (
            material_wave_operator(self.wave_im)
            + config.wave_rotation * self.wave_re
            + growth * self.wave_im
            + config.wave_source * local_maps[1]
        )
        self.wave_re += dt * delta_re
        self.wave_im += dt * delta_im
        magnitude2 = self.wave_re**2 + self.wave_im**2

        self.resource += dt * (
            (1.0 - self.resource) / config.resource_recovery
            - config.resource_burn * magnitude2
            + config.resource_diffusion * laplacian(self.resource)
        )
        np.clip(self.resource, 0.0, 1.0, out=self.resource)
        if uniform_resource:
            self.resource.fill(float(np.mean(self.resource)))

        raw_envelope = magnitude2
        memory = np.minimum(
            0.997,
            config.wave_memory
            + config.wave_memory_material_gain * material,
        )
        self.envelope = (
            memory * self.envelope
            + (1.0 - memory) * raw_envelope
            + config.envelope_diffusion * laplacian(self.envelope)
        )
        np.maximum(self.envelope, 0.0, out=self.envelope)

        env_x, env_y = _central_gradients(self.envelope)
        re_x, re_y = _central_gradients(self.wave_re)
        im_x, im_y = _central_gradients(self.wave_im)
        density_x, density_y = _central_gradients(local_maps[0])
        positions = self.positions
        resonant_x = (
            np.cos(self.phases) * sample_nearest(re_x, positions)
            + np.sin(self.phases) * sample_nearest(im_x, positions)
        )
        resonant_y = (
            np.cos(self.phases) * sample_nearest(re_y, positions)
            + np.sin(self.phases) * sample_nearest(im_y, positions)
        )
        force_x = (
            config.wave_attraction * sample_nearest(env_x, positions)
            + config.resonant_wave_attraction * resonant_x
            - config.density_repulsion * sample_nearest(density_x, positions)
        )
        force_y = (
            config.wave_attraction * sample_nearest(env_y, positions)
            + config.resonant_wave_attraction * resonant_y
            - config.density_repulsion * sample_nearest(density_y, positions)
        )
        hx = np.cos(self.headings)
        hy = np.sin(self.headings)
        torque = hx * force_y - hy * force_x

        if mode in (
            "tensor",
            "intact",
            "patchwork",
            "scrambled",
            "rotated",
        ):
            local_q1 = sample_nearest(np.asarray(q1), positions)
            local_q2 = sample_nearest(np.asarray(q2), positions)
            torque += config.tensor_guidance * (
                local_q2 * np.cos(2.0 * self.headings)
                - local_q1 * np.sin(2.0 * self.headings)
            )

        self.headings += (
            dt * torque
            + np.sqrt(dt)
            * config.rotational_noise
            * self.rng.normal(size=config.agents)
        )
        local_re = sample_nearest(self.wave_re, positions)
        local_im = sample_nearest(self.wave_im, positions)
        self.phases += dt * (
            self.frequencies
            + config.phase_lock
            * (
                -np.sin(self.phases) * local_re
                + np.cos(self.phases) * local_im
            )
        )
        self.positions = (
            self.positions
            + dt
            * config.particle_speed
            * np.column_stack(
                (np.cos(self.headings), np.sin(self.headings))
            )
        ) % config.n

        self.t += dt
        self.step_index += 1

    def reset_fast_state_and_agents(self, seed: int) -> None:
        """Remove builders and release a matched naive population."""

        self.rng = np.random.default_rng(seed)
        self.wave_re.fill(0.0)
        self.wave_im.fill(0.0)
        self.envelope.fill(0.0)
        self.resource.fill(1.0)
        self.positions = self.rng.uniform(
            0.0, self.config.n, (self.config.agents, 2)
        )
        self.headings = self.rng.uniform(
            0.0, 2.0 * np.pi, self.config.agents
        )
        self.phases = self.rng.uniform(
            0.0, 2.0 * np.pi, self.config.agents
        )
        self.frequencies.fill(self.config.oscillator_frequency)
        self.t = 0.0
        self.step_index = 0

    def copy_material_from(self, other: "MaterialWorld") -> None:
        self.matrix[:] = other.matrix
        self.q1[:] = other.q1
        self.q2[:] = other.q2
        self.precursor[:] = other.precursor


def matrix_metrics(world: MaterialWorld) -> dict[str, float | int]:
    config = world.config
    mask = world.matrix >= config.matrix_threshold
    components = periodic_components(mask)
    largest = max((len(component) for component in components), default=0)
    orientation_magnitude = np.hypot(world.q1, world.q2)
    weighted_order = float(
        np.sum(orientation_magnitude) / (np.sum(world.matrix) + 1e-12)
    )
    sampled = sample_nearest(world.matrix, world.positions)
    return {
        "mean": float(np.mean(world.matrix)),
        "std": float(np.std(world.matrix)),
        "max": float(np.max(world.matrix)),
        "above_threshold_fraction": float(np.mean(mask)),
        "component_count": len(components),
        "largest_component_fraction": largest / (config.n * config.n),
        "local_orientation_order": weighted_order,
        "agent_sample_enrichment": float(
            np.mean(sampled) / (np.mean(world.matrix) + 1e-12)
        )
        if np.mean(world.matrix) > 1e-10
        else 0.0,
        "precursor_plus_matrix_mean": float(
            np.mean(world.precursor + world.matrix)
        ),
    }


def _observe_write_association(
    world: MaterialWorld,
    assemblies: Iterable,
) -> tuple[float | None, float | None]:
    members = {member for assembly in assemblies for member in assembly.members}
    polymerization = sample_nearest(
        world.last_polymerization, world.positions
    )
    coherent = [polymerization[i] for i in members]
    solo = [
        polymerization[i]
        for i in range(world.config.agents)
        if i not in members
    ]
    return (
        float(np.mean(coherent)) if coherent else None,
        float(np.mean(solo)) if solo else None,
    )


def run_arm(
    mode: str,
    seed: int,
    config: Config,
    *,
    return_world: bool = False,
    shuffle_phases: bool = False,
    uniform_resource: bool = False,
) -> dict[str, object] | tuple[dict[str, object], MaterialWorld]:
    if mode not in ARMS:
        raise ValueError(mode)

    world = MaterialWorld(config, seed)
    tracker = CoherentAssemblyTracker(config.agents, config.n)
    coherent_writes: list[float] = []
    solo_writes: list[float] = []

    for step in range(config.build_steps):
        world.step(
            mode,
            shuffle_phases=shuffle_phases,
            uniform_resource=uniform_resource,
        )
        if step % config.observe_every:
            continue
        assemblies = tracker.update(
            world.positions, world.headings, world.phases
        )
        coherent, solo = _observe_write_association(world, assemblies)
        if coherent is not None:
            coherent_writes.append(coherent)
        if solo is not None:
            solo_writes.append(solo)

    assembly = tracker.summary()
    coherent_write = float(np.mean(coherent_writes)) if coherent_writes else 0.0
    solo_write = float(np.mean(solo_writes)) if solo_writes else 0.0
    result: dict[str, object] = {
        "mode": mode,
        "seed": seed,
        "assembly": assembly,
        "material": matrix_metrics(world),
        "coupling_audit": {
            "polymerization_at_coherent_agents": coherent_write,
            "polymerization_at_other_agents": solo_write,
            "coherent_to_other_ratio": coherent_write / (solo_write + 1e-12),
        },
        "world": {
            "resource_mean": float(np.mean(world.resource)),
            "field_rms": float(
                np.sqrt(np.mean(world.wave_re**2 + world.wave_im**2))
            ),
            "field_max_abs": float(
                np.max(np.hypot(world.wave_re, world.wave_im))
            ),
        },
        "interventions": {
            "phase_shuffle_each_step": bool(shuffle_phases),
            "uniform_resource_each_step": bool(uniform_resource),
        },
    }
    if return_world:
        return result, world
    return result


def _material_variant(
    builder: MaterialWorld,
    variant: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = builder.matrix.copy()
    q1 = builder.q1.copy()
    q2 = builder.q2.copy()
    if variant == "intact":
        return matrix, q1, q2
    if variant == "isotropic":
        return matrix, np.zeros_like(q1), np.zeros_like(q2)
    if variant == "patchwork":
        rng = np.random.default_rng(seed)
        n = matrix.shape[0]
        target = max(4, n // 8)
        divisors = [d for d in range(4, n + 1) if n % d == 0]
        tile = min(divisors, key=lambda d: abs(d - target))
        side = n // tile
        permutation = rng.permutation(side * side)

        def rearrange(field: np.ndarray) -> np.ndarray:
            out = np.empty_like(field)
            for destination, source in enumerate(permutation):
                dy, dx = divmod(destination, side)
                sy, sx = divmod(int(source), side)
                out[
                    dy * tile : (dy + 1) * tile,
                    dx * tile : (dx + 1) * tile,
                ] = field[
                    sy * tile : (sy + 1) * tile,
                    sx * tile : (sx + 1) * tile,
                ]
            return out

        return rearrange(matrix), rearrange(q1), rearrange(q2)
    if variant == "rotated":
        return matrix, -q1, -q2
    if variant == "erased":
        return np.zeros_like(matrix), np.zeros_like(q1), np.zeros_like(q2)
    if variant == "scrambled":
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(matrix.size)
        return (
            matrix.ravel()[permutation].reshape(matrix.shape),
            q1.ravel()[permutation].reshape(q1.shape),
            q2.ravel()[permutation].reshape(q2.shape),
        )
    raise ValueError(variant)


def run_probe(
    builder: MaterialWorld,
    variant: str,
    seed: int,
) -> dict[str, object]:
    if variant not in PROBES:
        raise ValueError(variant)
    config = builder.config
    world = MaterialWorld(config, seed, quiet_field=True)
    world.reset_fast_state_and_agents(seed)
    matrix, q1, q2 = _material_variant(builder, variant, seed + 991)
    world.matrix[:] = matrix
    world.q1[:] = q1
    world.q2[:] = q2

    mode = {
        "intact": "intact",
        "isotropic": "isotropic",
        "patchwork": "patchwork",
        "scrambled": "scrambled",
        "rotated": "rotated",
        "erased": "erased",
    }[variant]
    tracker = CoherentAssemblyTracker(config.agents, config.n)
    material_samples: list[float] = []
    nematic_alignment: list[float] = []

    for step in range(config.probe_steps):
        world.step(
            mode,
            write_material=False,
            freeze_material=True,
        )
        if step % config.observe_every:
            continue
        tracker.update(world.positions, world.headings, world.phases)
        material_samples.append(
            float(np.mean(sample_nearest(world.matrix, world.positions)))
        )
        local_q1 = sample_nearest(world.q1, world.positions)
        local_q2 = sample_nearest(world.q2, world.positions)
        local_magnitude = np.hypot(local_q1, local_q2)
        alignment = (
            local_q1 * np.cos(2.0 * world.headings)
            + local_q2 * np.sin(2.0 * world.headings)
        ) / (local_magnitude + 1e-12)
        valid = local_magnitude > 1e-4
        if np.any(valid):
            nematic_alignment.append(float(np.mean(alignment[valid])))

    summary = tracker.summary()
    return {
        "variant": variant,
        "seed": seed,
        "assembly": summary,
        "fresh_agent_material_enrichment": float(
            np.mean(material_samples) / (np.mean(world.matrix) + 1e-12)
        )
        if np.mean(world.matrix) > 1e-10
        else 0.0,
        "fresh_agent_nematic_alignment": float(np.mean(nematic_alignment))
        if nematic_alignment
        else 0.0,
        "builder_agents_removed": True,
        "fast_field_zeroed": True,
        "material_frozen": True,
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)) if len(array) else 0.0,
        "std": float(np.std(array)) if len(array) else 0.0,
    }


def summarize_arms(rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for mode in ARMS:
        selected = [row for row in rows if row["mode"] == mode]
        out[mode] = {
            "coherent_fraction": _mean_std(
                [
                    float(row["assembly"]["late_mean_coherent_fraction"])
                    for row in selected
                ]
            ),
            "largest_assembly": _mean_std(
                [
                    float(row["assembly"]["late_mean_largest_size"])
                    for row in selected
                ]
            ),
            "assembly_lifetime": _mean_std(
                [
                    float(row["assembly"]["max_lifetime_samples"])
                    for row in selected
                ]
            ),
            "matrix_spatial_std": _mean_std(
                [float(row["material"]["std"]) for row in selected]
            ),
            "agent_trace_enrichment": _mean_std(
                [
                    float(row["material"]["agent_sample_enrichment"])
                    for row in selected
                ]
            ),
        }
    return out


def summarize_probes(rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for variant in PROBES:
        selected = [row for row in rows if row["variant"] == variant]
        out[variant] = {
            "coherent_fraction": _mean_std(
                [
                    float(row["assembly"]["late_mean_coherent_fraction"])
                    for row in selected
                ]
            ),
            "largest_assembly": _mean_std(
                [
                    float(row["assembly"]["late_mean_largest_size"])
                    for row in selected
                ]
            ),
            "material_enrichment": _mean_std(
                [float(row["fresh_agent_material_enrichment"]) for row in selected]
            ),
            "nematic_alignment": _mean_std(
                [float(row["fresh_agent_nematic_alignment"]) for row in selected]
            ),
        }
    return out


def summarize_ablations(rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in ABLATIONS:
        selected = [row for row in rows if row["ablation"] == name]
        out[name] = {
            "coherent_fraction": _mean_std(
                [
                    float(row["assembly"]["late_mean_coherent_fraction"])
                    for row in selected
                ]
            ),
            "largest_assembly": _mean_std(
                [
                    float(row["assembly"]["late_mean_largest_size"])
                    for row in selected
                ]
            ),
            "matrix_spatial_std": _mean_std(
                [float(row["material"]["std"]) for row in selected]
            ),
            "field_rms": _mean_std(
                [float(row["world"]["field_rms"]) for row in selected]
            ),
        }
    return out


def preset(name: str) -> tuple[Config, tuple[int, ...]]:
    if name == "smoke":
        return Config(n=40, agents=48, build_steps=450, probe_steps=220), (1,)
    if name == "ci":
        return Config(), (1, 2)
    if name == "receipt":
        return Config(build_steps=4000, probe_steps=1200), (1, 2, 3, 4)
    raise ValueError(name)


def run_suite(config: Config, seeds: Iterable[int]) -> dict[str, object]:
    arm_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []

    for seed in seeds:
        builder: MaterialWorld | None = None
        for mode in ARMS:
            if mode == "tensor":
                row, builder = run_arm(
                    mode, seed, config, return_world=True
                )
            else:
                row = run_arm(mode, seed, config)
            arm_rows.append(row)
            if mode == "tensor":
                full_row = dict(row)
                full_row["ablation"] = "full"
                ablation_rows.append(full_row)

        assert builder is not None
        for variant in PROBES:
            probe_rows.append(
                run_probe(builder, variant, seed=seed + 100_000)
            )

        ablation_specs = (
            (
                "no_wave_sensing",
                replace(
                    config,
                    resonant_wave_attraction=0.0,
                    wave_attraction=0.0,
                    phase_lock=0.0,
                ),
                {},
            ),
            (
                "no_wave_production",
                replace(config, wave_source=0.0),
                {},
            ),
            (
                "phase_shuffle_each_step",
                config,
                {"shuffle_phases": True},
            ),
            (
                "uniform_resource",
                config,
                {"uniform_resource": True},
            ),
        )
        for name, ablation_config, kwargs in ablation_specs:
            row = run_arm("tensor", seed, ablation_config, **kwargs)
            row["ablation"] = name
            ablation_rows.append(row)

    return {
        "experiment": "Datarium 3 — material memory above wave assemblies",
        "claim_boundary": (
            "The experiment can establish transient measured assemblies, "
            "history-dependent material geometry, and downward causal "
            "influence on replacement particles. It does not implement or "
            "claim genomes, reproduction, heredity, organisms, open-ended "
            "evolution, or intelligence."
        ),
        "anti_cheating": {
            "authored_particle_types": False,
            "role_labels_in_dynamics": False,
            "train_object_in_dynamics": False,
            "links_or_bonds": False,
            "fitness_or_global_sort": False,
            "reproduction_or_genome_copy": False,
            "assembly_detector_visible_to_physics": False,
            "persistent_agent_memory": False,
            "only_initial_particle_differences": (
                "position, heading and oscillator phase"
            ),
        },
        "config": asdict(config),
        "arms": list(ARMS),
        "probe_variants": list(PROBES),
        "ablations": list(ABLATIONS),
        "arm_summary": summarize_arms(arm_rows),
        "probe_summary": summarize_probes(probe_rows),
        "ablation_summary": summarize_ablations(ablation_rows),
        "arm_rows": arm_rows,
        "probe_rows": probe_rows,
        "ablation_rows": ablation_rows,
    }


def print_receipt(receipt: dict[str, object]) -> None:
    print("DATARIUM 3 — MATERIAL MEMORY ABOVE WAVE ASSEMBLIES")
    print("=" * 76)
    print("No types, roles, links, fitness, genomes, reproduction, or Train object")
    print()
    print("BUILD ARMS")
    print("mode          coherent   largest   life(samples)   matrix std   reuse")
    for mode in ARMS:
        row = receipt["arm_summary"][mode]
        print(
            f"{mode:12s}  "
            f"{row['coherent_fraction']['mean']:8.3f}  "
            f"{row['largest_assembly']['mean']:8.2f}  "
            f"{row['assembly_lifetime']['mean']:13.1f}  "
            f"{row['matrix_spatial_std']['mean']:10.3f}  "
            f"{row['agent_trace_enrichment']['mean']:6.2f}"
        )
    print()
    print("BUILDER-REMOVAL PROBE")
    print("variant       coherent   largest   material reuse   director align")
    for variant in PROBES:
        row = receipt["probe_summary"][variant]
        print(
            f"{variant:12s}  "
            f"{row['coherent_fraction']['mean']:8.3f}  "
            f"{row['largest_assembly']['mean']:8.2f}  "
            f"{row['material_enrichment']['mean']:14.2f}  "
            f"{row['nematic_alignment']['mean']:14.3f}"
        )
    print()
    print("CAUSAL ABLATIONS")
    print("condition                 coherent   largest   matrix std   field rms")
    for name in ABLATIONS:
        row = receipt["ablation_summary"][name]
        print(
            f"{name:24s}  "
            f"{row['coherent_fraction']['mean']:8.3f}  "
            f"{row['largest_assembly']['mean']:8.2f}  "
            f"{row['matrix_spatial_std']['mean']:10.3f}  "
            f"{row['field_rms']['mean']:9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset", choices=("smoke", "ci", "receipt"), default="ci"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "datarium3.json")
    args = parser.parse_args()

    config, seeds = preset(args.preset)
    receipt = run_suite(config, seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print_receipt(receipt)
    print()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
