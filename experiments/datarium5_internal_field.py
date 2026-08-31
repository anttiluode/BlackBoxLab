"""Datarium 5A — the field grows an organ.

This gate follows Datarium 4B but changes the question. 4B earned a body phase
without a pre-drawn disk. Datarium 5A holds that builder-nucleated body fixed
and asks whether fast internal activity can create a slow *portable-in-
principle* internal morphology that changes later activity.

The key causal loop is deliberately local:

    boundary drives -> two-quadrature internal field
                   -> interference / spatial gradients
                   -> fast activity trace
                   -> slow oriented fibre tensor
                   -> anisotropic field propagation
                   -> changed future interference

This is inspired by the general ephaptic idea "field -> excitable substrate ->
field", not a model of cortex or entorhinal grid cells.

Important stopping lines:
- the D4B body is locally expanded for assay size and then pinned;
- internal fibres are not yet advected with a moving body;
- no neuron, synapse, genome, role, reward or neural-network controller exists;
- the fixed-surrogate score below is FCI-inspired, not the Aizenbud FCI.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.datarium4_body import lap
from experiments.datarium4b_builder_body import (
    Config as D4BConfig,
    build_scaffold,
    phase_from_scaffold,
)


@dataclass(frozen=True)
class Config:
    n: int = 40
    agents: int = 52
    builder_steps: int = 900
    phase_steps: int = 150
    body_growth_steps: int = 600

    develop_steps: int = 750
    probe_steps: int = 75
    probe_tail: int = 20
    probe_trials: int = 36

    dt: float = 0.025
    wave_diffusion: float = 0.50
    wave_frequency: float = 1.8
    wave_damping: float = 0.16
    outside_damping: float = 2.8
    wave_saturation: float = 0.45
    fibre_feedback: float = 0.80

    fast_tau: float = 0.7
    slow_tau: float = 3.0
    fibre_gain: float = 1.20
    fibre_half: float = 0.002
    fibre_decay: float = 0.006
    director_gain: float = 1.50
    director_decay: float = 0.010
    fibre_diffusion: float = 0.025

    drive_gain: float = 0.80
    port_sigma: float = 1.5
    body_growth_diffusion: float = 0.24
    body_growth_gain: float = 1.40
    body_growth_bias: float = 0.05

    fibre_threshold: float = 0.18
    ridge: float = 1e-4


def _d4_config(config: Config) -> D4BConfig:
    return D4BConfig(
        n=config.n,
        agents=config.agents,
        builder_steps=config.builder_steps,
        phase_steps=config.phase_steps,
        settle_steps=1,
        move_steps=1,
        cargo_steps=1,
        source_radius=max(8.0, config.n * 0.25),
        source_sigma=max(3.5, config.n * 0.10),
    )


def grow_body(phi: np.ndarray, config: Config) -> np.ndarray:
    """Local phase growth from the D4B nucleus; no target area or center."""
    phi = np.asarray(phi, dtype=float).copy()
    dt = 0.035
    for _ in range(config.body_growth_steps):
        reaction = (
            config.body_growth_gain
            * phi
            * (1.0 - phi)
            * (phi - config.body_growth_bias)
        )
        phi += dt * (
            config.body_growth_diffusion * lap(phi) + reaction
        )
        np.clip(phi, 0.0, 1.0, out=phi)
    return phi


def builder_body(config: Config, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    d4 = _d4_config(config)
    world = build_scaffold(d4, seed, "full")
    phi0 = phase_from_scaffold(world.matrix, d4, world.q1, world.q2)
    phi = grow_body(phi0, config)
    metrics = {
        "raw_phase_mass": float(np.sum(phi0)),
        "body_mass": float(np.sum(phi)),
        "body_area_gt_half": float(np.sum(phi >= 0.5)),
        "builder_material_mean": float(np.mean(world.matrix)),
        "builder_director_mean": float(np.mean(np.hypot(world.q1, world.q2))),
    }
    return phi, metrics


def grad(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) * 0.5
    gy = (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) * 0.5
    return gx, gy


def second_derivatives(
    a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dxx = np.roll(a, -1, axis=1) - 2.0 * a + np.roll(a, 1, axis=1)
    dyy = np.roll(a, -1, axis=0) - 2.0 * a + np.roll(a, 1, axis=0)
    dxy = (
        np.roll(np.roll(a, -1, axis=0), -1, axis=1)
        - np.roll(np.roll(a, -1, axis=0), 1, axis=1)
        - np.roll(np.roll(a, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(a, 1, axis=0), 1, axis=1)
    ) * 0.25
    return dxx, dyy, dxy


def weighted_center(phi: np.ndarray) -> tuple[float, float, float]:
    yy, xx = np.mgrid[: phi.shape[0], : phi.shape[1]]
    mass = float(np.sum(phi)) + 1e-12
    return (
        float(np.sum(phi * xx) / mass),
        float(np.sum(phi * yy) / mass),
        mass,
    )


def port_masks(
    phi: np.ndarray,
    config: Config,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Observer-defined stimulation/readout sites for the I/O assay."""
    n = phi.shape[0]
    yy, xx = np.mgrid[:n, :n]
    cx, cy, mass = weighted_center(phi)
    radius = max(3.0, np.sqrt(mass / np.pi) * 0.65)
    drive_masks: list[np.ndarray] = []
    read_masks: list[np.ndarray] = []
    for angle in np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False):
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        mask = np.exp(-d2 / (2.0 * config.port_sigma**2)) * phi
        # Drives are physical local patches: unit local amplitude, not a
        # globally conserved injection divided by patch area.
        mask /= float(np.max(mask)) + 1e-12
        drive_masks.append(mask)

        ra = angle + np.pi / 4.0
        rx = cx + 0.72 * radius * np.cos(ra)
        ry = cy + 0.72 * radius * np.sin(ra)
        rd2 = (xx - rx) ** 2 + (yy - ry) ** 2
        rmask = np.exp(-rd2 / (2.0 * config.port_sigma**2)) * phi
        rmask /= float(np.sum(rmask)) + 1e-12
        read_masks.append(rmask)
    return drive_masks, read_masks


class InternalField:
    """Two-way field <-> slow oriented internal material."""

    def __init__(
        self,
        phi: np.ndarray,
        config: Config,
        *,
        fibre: np.ndarray | None = None,
        q1: np.ndarray | None = None,
        q2: np.ndarray | None = None,
        feedback: bool = True,
        plastic: bool = True,
    ) -> None:
        self.phi = np.asarray(phi, dtype=float)
        self.cfg = config
        self.re = np.zeros_like(self.phi)
        self.im = np.zeros_like(self.phi)
        self.fast = np.zeros_like(self.phi)
        self.slow = np.zeros_like(self.phi)
        self.fibre = (
            np.zeros_like(self.phi)
            if fibre is None
            else np.asarray(fibre, dtype=float).copy()
        )
        self.q1 = (
            np.zeros_like(self.phi)
            if q1 is None
            else np.asarray(q1, dtype=float).copy()
        )
        self.q2 = (
            np.zeros_like(self.phi)
            if q2 is None
            else np.asarray(q2, dtype=float).copy()
        )
        self.feedback = feedback
        self.plastic = plastic
        self.drives, self.reads = port_masks(self.phi, config)
        self.t = 0.0
        self.port_omega = np.asarray([1.00, 1.07, 1.13, 1.21]) * 2.0

    def reset_fast(self) -> None:
        self.re.fill(0.0)
        self.im.fill(0.0)
        self.fast.fill(0.0)
        self.slow.fill(0.0)
        self.t = 0.0

    def _aniso(self, a: np.ndarray) -> np.ndarray:
        if not self.feedback:
            return np.zeros_like(a)
        dxx, dyy, dxy = second_derivatives(a)
        return self.q1 * (dxx - dyy) + 2.0 * self.q2 * dxy

    def step(self, amplitudes: np.ndarray) -> None:
        c = self.cfg
        dt = c.dt
        amplitudes = np.asarray(amplitudes, dtype=float)

        source = np.zeros_like(self.phi)
        for i, mask in enumerate(self.drives):
            source += (
                c.drive_gain
                * amplitudes[i]
                * np.sin(self.port_omega[i] * self.t + 0.37 * i)
                * mask
            )

        mag2 = self.re * self.re + self.im * self.im
        damping = c.wave_damping + c.outside_damping * (1.0 - self.phi)
        ar = self._aniso(self.re)
        ai = self._aniso(self.im)

        new_re = self.re + dt * (
            c.wave_diffusion * lap(self.re)
            + c.fibre_feedback * ar
            - c.wave_frequency * self.im
            - damping * self.re
            - c.wave_saturation * mag2 * self.re
            + source
        )
        new_im = self.im + dt * (
            c.wave_diffusion * lap(self.im)
            + c.fibre_feedback * ai
            + c.wave_frequency * self.re
            - damping * self.im
            - c.wave_saturation * mag2 * self.im
        )
        self.re = new_re
        self.im = new_im

        activity = self.phi * (self.re * self.re + self.im * self.im)
        self.fast += dt * (activity - self.fast) / c.fast_tau
        self.slow += dt * (self.fast - self.slow) / c.slow_tau

        if self.plastic:
            rx, ry = grad(self.re)
            ix, iy = grad(self.im)
            gxx = rx * rx + ix * ix
            gyy = ry * ry + iy * iy
            gxy = rx * ry + ix * iy
            gsum = gxx + gyy + 1e-12
            d1 = (gxx - gyy) / gsum
            d2 = 2.0 * gxy / gsum

            gate = self.slow / (self.slow + c.fibre_half)
            growth = c.fibre_gain * self.phi * gate * (1.0 - self.fibre)
            self.fibre += dt * (
                growth
                - c.fibre_decay * self.fibre
                + c.fibre_diffusion * lap(self.fibre)
            )
            np.clip(self.fibre, 0.0, 1.0, out=self.fibre)

            self.q1 += dt * (
                c.director_gain * growth * d1 - c.director_decay * self.q1
            )
            self.q2 += dt * (
                c.director_gain * growth * d2 - c.director_decay * self.q2
            )
            norm = np.hypot(self.q1, self.q2)
            scale = np.minimum(1.0, self.fibre / (norm + 1e-12))
            self.q1 *= scale
            self.q2 *= scale

        self.t += dt

    def readout(self) -> np.ndarray:
        energy = self.re * self.re + self.im * self.im
        return np.asarray(
            [float(np.sum(mask * energy)) for mask in self.reads],
            dtype=float,
        )


def develop(
    phi: np.ndarray,
    config: Config,
    *,
    feedback: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world = InternalField(phi, config, feedback=feedback, plastic=True)
    # Four slightly different frequencies coexist continuously. Their
    # interference is not named or segmented; only local field gradients can
    # write the slow tensor.
    amplitudes = np.ones(4, dtype=float)
    for _ in range(config.develop_steps):
        world.step(amplitudes)
    return world.fibre.copy(), world.q1.copy(), world.q2.copy()


def morphology_variant(
    fibre: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    variant: str,
    seed: int,
    phi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if variant == "intact":
        return fibre.copy(), q1.copy(), q2.copy()
    if variant == "isotropic":
        return fibre.copy(), np.zeros_like(q1), np.zeros_like(q2)
    if variant == "scrambled":
        rng = np.random.default_rng(seed)
        if phi is None:
            support = np.ones(fibre.shape, dtype=bool)
        else:
            support = np.asarray(phi) >= 0.25
        idx = np.flatnonzero(support.ravel())
        perm = rng.permutation(idx)
        sf = fibre.copy().ravel()
        s1 = q1.copy().ravel()
        s2 = q2.copy().ravel()
        # Preserve the complete local tensor histogram *inside the same body*
        # so the attacker destroys arrangement rather than moving material
        # into extracellular space.
        sf[idx] = fibre.ravel()[perm]
        s1[idx] = q1.ravel()[perm]
        s2[idx] = q2.ravel()[perm]
        return sf.reshape(fibre.shape), s1.reshape(q1.shape), s2.reshape(q2.shape)
    if variant == "erased":
        z = np.zeros_like(fibre)
        return z.copy(), z.copy(), z.copy()
    raise ValueError(variant)


def morphology_metrics(
    phi: np.ndarray,
    fibre: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    config: Config,
) -> dict[str, float]:
    mask = (fibre >= config.fibre_threshold) & (phi >= 0.25)
    neighbors = (
        np.roll(mask, 1, axis=0).astype(int)
        + np.roll(mask, -1, axis=0).astype(int)
        + np.roll(mask, 1, axis=1).astype(int)
        + np.roll(mask, -1, axis=1).astype(int)
    )
    junction = mask & (neighbors >= 3)

    # Dyadic spectral occupancy: a ruler for scale span, not an objective.
    z = fibre * phi
    z = z - float(np.mean(z))
    power = np.abs(np.fft.fftshift(np.fft.fft2(z))) ** 2
    n = z.shape[0]
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - n / 2.0, yy - n / 2.0)
    band_power: list[float] = []
    lo = 1.0
    while lo < n / 2.0:
        hi = min(2.0 * lo, n / 2.0)
        band = (rr >= lo) & (rr < hi)
        band_power.append(float(np.sum(power[band])))
        lo *= 2.0
    total_band = sum(band_power) + 1e-12
    active_octaves = sum(p / total_band > 0.04 for p in band_power)

    return {
        "fibre_area": float(np.sum(fibre * phi)),
        "fibre_mean_inside": float(np.sum(fibre * phi) / (np.sum(phi) + 1e-12)),
        "director_mean_inside": float(
            np.sum(np.hypot(q1, q2) * phi) / (np.sum(phi) + 1e-12)
        ),
        "thresholded_fibre_cells": float(np.sum(mask)),
        "junction_cells": float(np.sum(junction)),
        "junction_fraction": float(np.sum(junction) / max(np.sum(mask), 1)),
        "active_spectral_octaves": float(active_octaves),
    }


def _single_response_map(
    phi: np.ndarray,
    config: Config,
    fibre: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    input_index: int,
) -> np.ndarray:
    world = InternalField(
        phi,
        config,
        fibre=fibre,
        q1=q1,
        q2=q2,
        feedback=True,
        plastic=False,
    )
    amps = np.zeros(4)
    amps[input_index] = 1.0
    acc = np.zeros_like(phi)
    count = 0
    for step in range(config.probe_steps):
        world.step(amps)
        if step >= config.probe_steps - config.probe_tail:
            acc += world.re * world.re + world.im * world.im
            count += 1
    return acc / max(count, 1)


def functional_zoning(
    phi: np.ndarray,
    config: Config,
    fibre: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> dict[str, float]:
    maps = np.stack(
        [
            _single_response_map(phi, config, fibre, q1, q2, i)
            for i in range(4)
        ],
        axis=0,
    )
    preference = np.argmax(maps, axis=0)
    strength = np.max(maps, axis=0) * phi
    masses = np.asarray(
        [float(np.sum(strength[preference == i])) for i in range(4)]
    )
    p = masses / (float(np.sum(masses)) + 1e-12)
    entropy = -float(np.sum(p * np.log(p + 1e-12))) / np.log(4.0)

    # How different are the four spatial response maps?
    flat = maps.reshape(4, -1)
    flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)
    similarity = flat @ flat.T
    off = similarity[~np.eye(4, dtype=bool)]
    separation = 1.0 - float(np.mean(off))
    return {
        "zone_entropy": entropy,
        "response_map_separation": separation,
    }


def response_fingerprint(
    phi: np.ndarray,
    config: Config,
    fibre: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    """Deterministic field response used only for causal morphology attacks."""
    patterns = (
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 1.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0, 1.0]),
    )
    pieces: list[np.ndarray] = []
    for pattern in patterns:
        world = InternalField(
            phi,
            config,
            fibre=fibre,
            q1=q1,
            q2=q2,
            feedback=True,
            plastic=False,
        )
        for _ in range(config.probe_steps):
            world.step(pattern)
        energy = (world.re * world.re + world.im * world.im) * phi
        pieces.append(energy.ravel())
    return np.concatenate(pieces)


def relative_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def probe_io(
    phi: np.ndarray,
    config: Config,
    fibre: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    patterns = rng.integers(0, 2, size=(config.probe_trials, 4)).astype(float)
    # Avoid the uninformative all-zero trial.
    zero = np.sum(patterns, axis=1) == 0
    patterns[zero, rng.integers(0, 4, size=int(np.sum(zero)))] = 1.0

    outputs: list[np.ndarray] = []
    for pattern in patterns:
        world = InternalField(
            phi,
            config,
            fibre=fibre,
            q1=q1,
            q2=q2,
            feedback=True,
            plastic=False,
        )
        tail: list[np.ndarray] = []
        for step in range(config.probe_steps):
            world.step(pattern)
            if step >= config.probe_steps - config.probe_tail:
                tail.append(world.readout())
        outputs.append(np.mean(np.stack(tail), axis=0))

    y = np.stack(outputs)
    x = np.concatenate([np.ones((len(patterns), 1)), patterns], axis=1)
    split = max(8, int(0.72 * len(patterns)))
    xt, xv = x[:split], x[split:]
    yt, yv = y[:split], y[split:]
    gram = xt.T @ xt + config.ridge * np.eye(xt.shape[1])
    weights = np.linalg.solve(gram, xt.T @ yt)
    pred = xv @ weights
    rmse = float(np.sqrt(np.mean((pred - yv) ** 2)))
    scale = float(np.std(yv)) + 1e-12
    surrogate_error = rmse / scale

    # Single-input transfer matrix and effective rank.
    transfer = []
    for i in range(4):
        pattern = np.zeros(4)
        pattern[i] = 1.0
        world = InternalField(
            phi,
            config,
            fibre=fibre,
            q1=q1,
            q2=q2,
            feedback=True,
            plastic=False,
        )
        tail = []
        for step in range(config.probe_steps):
            world.step(pattern)
            if step >= config.probe_steps - config.probe_tail:
                tail.append(world.readout())
        transfer.append(np.mean(np.stack(tail), axis=0))
    transfer = np.stack(transfer)
    s = np.linalg.svd(transfer, compute_uv=False)
    ps = s / (float(np.sum(s)) + 1e-12)
    effective_rank = float(np.exp(-np.sum(ps * np.log(ps + 1e-12))))

    zoning = functional_zoning(phi, config, fibre, q1, q2)
    return {
        "fixed_linear_surrogate_error": surrogate_error,
        "transfer_effective_rank": effective_rank,
        **zoning,
    }


def run_seed(config: Config, seed: int) -> dict[str, object]:
    phi, body = builder_body(config, seed)

    full = develop(phi, config, feedback=True)
    write_only = develop(phi, config, feedback=False)

    variants: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "full_intact": morphology_variant(
            *full, "intact", seed + 100, phi
        ),
        "full_isotropic": morphology_variant(
            *full, "isotropic", seed + 100, phi
        ),
        "full_scrambled": morphology_variant(
            *full, "scrambled", seed + 100, phi
        ),
        "full_erased": morphology_variant(
            *full, "erased", seed + 100, phi
        ),
        "write_only_intact": morphology_variant(
            *write_only, "intact", seed + 200, phi
        ),
    }

    rows: dict[str, object] = {}
    fingerprints: dict[str, np.ndarray] = {}
    for name, (fibre, q1, q2) in variants.items():
        rows[name] = {
            "morphology": morphology_metrics(
                phi, fibre, q1, q2, config
            ),
            "function": probe_io(
                phi, config, fibre, q1, q2, seed + 1000
            ),
        }
        fingerprints[name] = response_fingerprint(
            phi, config, fibre, q1, q2
        )

    causal = {
        "intact_vs_erased": relative_delta(
            fingerprints["full_intact"],
            fingerprints["full_erased"],
        ),
        "intact_vs_isotropic": relative_delta(
            fingerprints["full_intact"],
            fingerprints["full_isotropic"],
        ),
        "intact_vs_scrambled": relative_delta(
            fingerprints["full_intact"],
            fingerprints["full_scrambled"],
        ),
        "full_vs_write_only_morphology": float(
            np.linalg.norm(full[0] - write_only[0])
            / (np.linalg.norm(write_only[0]) + 1e-12)
        ),
    }

    return {
        "seed": seed,
        "body": body,
        "rows": rows,
        "causal": causal,
    }


def _agg(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(a)), "std": float(np.std(a))}


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    names = (
        "full_intact",
        "full_isotropic",
        "full_scrambled",
        "full_erased",
        "write_only_intact",
    )
    result: dict[str, object] = {}
    for name in names:
        result[name] = {}
        for section in ("morphology", "function"):
            keys = rows[0]["rows"][name][section].keys()
            result[name][section] = {
                key: _agg(
                    [
                        float(row["rows"][name][section][key])
                        for row in rows
                    ]
                )
                for key in keys
            }
    result["body"] = {
        key: _agg([float(row["body"][key]) for row in rows])
        for key in rows[0]["body"].keys()
    }
    result["causal"] = {
        key: _agg([float(row["causal"][key]) for row in rows])
        for key in rows[0]["causal"].keys()
    }
    return result


def run_suite(config: Config, seeds: tuple[int, ...]) -> dict[str, object]:
    per_seed = [run_seed(config, seed) for seed in seeds]
    return {
        "config": asdict(config),
        "summary": summarize(per_seed),
        "per_seed": per_seed,
    }


def preset(name: str) -> tuple[Config, tuple[int, ...]]:
    if name == "smoke":
        return (
            Config(
                n=32,
                agents=36,
                builder_steps=300,
                phase_steps=100,
                body_growth_steps=220,
                develop_steps=220,
                probe_steps=32,
                probe_tail=8,
                probe_trials=16,
            ),
            (0,),
        )
    if name == "ci":
        return (
            Config(
                n=36,
                agents=44,
                builder_steps=650,
                phase_steps=125,
                body_growth_steps=420,
                develop_steps=520,
                probe_steps=45,
                probe_tail=10,
                probe_trials=20,
            ),
            (0, 1),
        )
    if name == "receipt":
        return Config(), (0, 1, 2, 3)
    raise ValueError(name)


def print_receipt(receipt: dict[str, object]) -> None:
    print("BlackBoxLab — Datarium 5A: the field grows an organ")
    body = receipt["summary"]["body"]
    print(
        "body: "
        f"raw phase={body['raw_phase_mass']['mean']:.2f}, "
        f"grown mass={body['body_mass']['mean']:.2f}, "
        f"area>0.5={body['body_area_gt_half']['mean']:.1f}"
    )
    print()
    print(
        f"{'arm':22s} {'fib area':>9s} {'junction':>9s} {'oct':>5s} "
        f"{'surrogate':>10s} {'rank':>7s} {'zones':>7s} {'maps':>7s}"
    )
    for name in (
        "full_intact",
        "full_isotropic",
        "full_scrambled",
        "full_erased",
        "write_only_intact",
    ):
        m = receipt["summary"][name]["morphology"]
        f = receipt["summary"][name]["function"]
        print(
            f"{name:22s} "
            f"{m['fibre_area']['mean']:9.3f} "
            f"{m['junction_fraction']['mean']:9.3f} "
            f"{m['active_spectral_octaves']['mean']:5.2f} "
            f"{f['fixed_linear_surrogate_error']['mean']:10.3f} "
            f"{f['transfer_effective_rank']['mean']:7.3f} "
            f"{f['zone_entropy']['mean']:7.3f} "
            f"{f['response_map_separation']['mean']:7.3f}"
        )
    causal = receipt["summary"]["causal"]
    print("\ncausal morphology attacks")
    for key in (
        "intact_vs_erased",
        "intact_vs_isotropic",
        "intact_vs_scrambled",
        "full_vs_write_only_morphology",
    ):
        print(f"{key:32s} {causal[key]['mean']:.4f} ± {causal[key]['std']:.4f}")
    print(
        "\nStopping line: this asks whether field-written internal morphology "
        "changes a fixed body's field and I/O. It is not a neuron, FCI, "
        "moving organ, heredity or intelligence claim."
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
        default=ROOT / "results" / "datarium5.json",
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
