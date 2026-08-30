"""Datarium lineage microscope.

The tracker is intentionally separate from the field equations. It observes a
scalar field and makes only measurement claims:

- connected components are labelled on a torus, matching np.roll physics;
- high/low hysteresis prevents breathing boundaries from flickering IDs;
- minimum area and positive-mass floors reject threshold dust;
- frame-to-frame identity is based on component overlap, not centroid jumps;
- split, merge, birth and death are explicit lineage events.

No inherited trait, fitness function, genome or controller lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


NEIGHBOURS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class Domain:
    pixels: np.ndarray
    area: int
    mass: float
    cx: float
    cy: float
    perimeter: int
    compactness: float
    eccentricity: float
    track_id: int = -1

    def phenotype(self) -> dict[str, float]:
        return {
            "mass": float(self.mass),
            "area": float(self.area),
            "compactness": float(self.compactness),
            "eccentricity": float(self.eccentricity),
            "cx": float(self.cx),
            "cy": float(self.cy),
        }


@dataclass
class Track:
    track_id: int
    parents: tuple[int, ...]
    born_frame: int
    born_t: float
    last_frame: int
    last_t: float
    ended_frame: int | None = None
    ended_t: float | None = None
    samples: list[dict[str, float]] = field(default_factory=list)

    @property
    def lifetime_frames(self) -> int:
        return self.last_frame - self.born_frame + 1


def torus_delta(value: np.ndarray | float, center: float, period: int):
    return (np.asarray(value) - center + period / 2.0) % period - period / 2.0


def torus_distance(
    a: tuple[float, float],
    b: tuple[float, float],
    shape: tuple[int, int],
) -> float:
    h, w = shape
    dx = float(torus_delta(a[0], b[0], w))
    dy = float(torus_delta(a[1], b[1], h))
    return float(np.hypot(dx, dy))


def _circular_mean(values: np.ndarray, weights: np.ndarray, period: int) -> float:
    angles = 2.0 * np.pi * values / period
    z = np.sum(weights * np.exp(1j * angles))
    if abs(z) < 1e-12:
        return float(values.mean() % period)
    return float((np.angle(z) % (2.0 * np.pi)) * period / (2.0 * np.pi))


def periodic_components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components with wrap-around in both axes."""

    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")

    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []

    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        y0 = int(y0)
        x0 = int(x0)
        if seen[y0, x0]:
            continue

        stack = [(y0, x0)]
        seen[y0, x0] = True
        pixels: list[int] = []

        while stack:
            y, x = stack.pop()
            pixels.append(y * w + x)
            for dy, dx in NEIGHBOURS4:
                yy = (y + dy) % h
                xx = (x + dx) % w
                if mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))

        components.append(np.asarray(pixels, dtype=np.int32))

    return components


def nonperiodic_components(mask: np.ndarray) -> list[np.ndarray]:
    """Reference labeller used only for seam audits."""

    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    ys, xs = np.nonzero(mask)

    for y0, x0 in zip(ys, xs):
        y0 = int(y0)
        x0 = int(x0)
        if seen[y0, x0]:
            continue

        stack = [(y0, x0)]
        seen[y0, x0] = True
        pixels: list[int] = []
        while stack:
            y, x = stack.pop()
            pixels.append(y * w + x)
            for dy, dx in NEIGHBOURS4:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))

        components.append(np.asarray(pixels, dtype=np.int32))

    return components


def describe_component(
    pixels: np.ndarray,
    positive_field: np.ndarray,
) -> Domain:
    h, w = positive_field.shape
    ys = pixels // w
    xs = pixels % w
    weights = positive_field.ravel()[pixels].astype(float) + 1e-12

    mass = float(weights.sum())
    area = int(len(pixels))
    cx = _circular_mean(xs.astype(float), weights, w)
    cy = _circular_mean(ys.astype(float), weights, h)

    dx = torus_delta(xs.astype(float), cx, w)
    dy = torus_delta(ys.astype(float), cy, h)

    if area >= 2 and float(np.sum(weights)) > 0:
        sw = float(np.sum(weights))
        mx = float(np.sum(weights * dx) / sw)
        my = float(np.sum(weights * dy) / sw)
        ux = dx - mx
        uy = dy - my
        cxx = float(np.sum(weights * ux * ux) / sw)
        cyy = float(np.sum(weights * uy * uy) / sw)
        cxy = float(np.sum(weights * ux * uy) / sw)
        eig = np.linalg.eigvalsh(np.asarray([[cxx, cxy], [cxy, cyy]]))
        small = max(float(eig[0]), 0.0)
        large = max(float(eig[1]), 1e-12)
        eccentricity = float(np.sqrt(max(0.0, 1.0 - small / large)))
    else:
        eccentricity = 0.0

    member = np.zeros((h, w), dtype=bool)
    member.ravel()[pixels] = True
    perimeter = 0
    for dy0, dx0 in NEIGHBOURS4:
        neighbour = np.roll(np.roll(member, dy0, axis=0), dx0, axis=1)
        perimeter += int(np.sum(member & ~neighbour))

    compactness = (
        float(4.0 * np.pi * area / (perimeter * perimeter))
        if perimeter > 0
        else 1.0
    )

    return Domain(
        pixels=pixels,
        area=area,
        mass=mass,
        cx=cx,
        cy=cy,
        perimeter=perimeter,
        compactness=compactness,
        eccentricity=eccentricity,
    )


class LineageTracker:
    """Conservative identity tracker for thresholded field domains."""

    def __init__(
        self,
        shape: tuple[int, int],
        high_threshold: float = 0.30,
        low_threshold: float = 0.24,
        min_area: int = 20,
        min_mass: float = 8.0,
        min_overlap: float = 0.12,
    ):
        if low_threshold >= high_threshold:
            raise ValueError("low_threshold must be below high_threshold")
        self.shape = tuple(shape)
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.min_area = int(min_area)
        self.min_mass = float(min_mass)
        self.min_overlap = float(min_overlap)

        self.frame = 0
        self.next_id = 1
        self.previous_support = np.zeros(self.shape, dtype=bool)
        self.previous: list[Domain] = []
        self.tracks: dict[int, Track] = {}
        self.events: list[dict[str, object]] = []
        self.seam_continuations = 0
        self.raw_periodic_count = 0
        self.raw_nonperiodic_count = 0

    def _detect(self, phi: np.ndarray) -> list[Domain]:
        if phi.shape != self.shape:
            raise ValueError(f"expected field shape {self.shape}, got {phi.shape}")

        raw_support = (phi >= self.high_threshold) | (
            self.previous_support & (phi >= self.low_threshold)
        )
        positive = np.maximum(phi, 0.0)

        periodic = periodic_components(raw_support)
        nonperiodic = nonperiodic_components(raw_support)
        self.raw_periodic_count = len(periodic)
        self.raw_nonperiodic_count = len(nonperiodic)

        kept: list[Domain] = []
        stable_support = np.zeros(self.shape, dtype=bool)
        for pixels in periodic:
            domain = describe_component(pixels, positive)
            if domain.area < self.min_area or domain.mass < self.min_mass:
                continue
            kept.append(domain)
            stable_support.ravel()[pixels] = True

        # Dust never earns hysteresis. Only accepted domains carry the low
        # threshold into the next measurement.
        self.previous_support = stable_support
        return kept

    def _new_track(
        self,
        parents: Iterable[int],
        t: float,
    ) -> int:
        track_id = self.next_id
        self.next_id += 1
        parent_tuple = tuple(sorted(int(p) for p in parents))
        self.tracks[track_id] = Track(
            track_id=track_id,
            parents=parent_tuple,
            born_frame=self.frame,
            born_t=float(t),
            last_frame=self.frame,
            last_t=float(t),
        )
        return track_id

    def _end_track(self, track_id: int, t: float) -> None:
        tr = self.tracks.get(track_id)
        if tr is None or tr.ended_frame is not None:
            return
        tr.ended_frame = self.frame
        tr.ended_t = float(t)

    def _overlap_graph(
        self,
        current: list[Domain],
    ) -> tuple[list[list[int]], list[list[int]]]:
        old_to_new = [[] for _ in self.previous]
        new_to_old = [[] for _ in current]
        if not self.previous or not current:
            return old_to_new, new_to_old

        old_labels = np.zeros(self.shape, dtype=np.int32)
        new_labels = np.zeros(self.shape, dtype=np.int32)
        for i, d in enumerate(self.previous, start=1):
            old_labels.ravel()[d.pixels] = i
        for j, d in enumerate(current, start=1):
            new_labels.ravel()[d.pixels] = j

        mask = (old_labels > 0) & (new_labels > 0)
        if not np.any(mask):
            return old_to_new, new_to_old

        stride = len(current) + 1
        keys = old_labels[mask].astype(np.int64) * stride + new_labels[mask]
        uniq, counts = np.unique(keys, return_counts=True)

        for key, count in zip(uniq, counts):
            oi = int(key // stride) - 1
            nj = int(key % stride) - 1
            if oi < 0 or nj < 0:
                continue
            denom = min(self.previous[oi].area, current[nj].area)
            score = float(count) / max(denom, 1)
            if score >= self.min_overlap:
                old_to_new[oi].append(nj)
                new_to_old[nj].append(oi)

        return old_to_new, new_to_old

    def update(self, phi: np.ndarray, t: float) -> list[Domain]:
        current = self._detect(phi)
        old_to_new, new_to_old = self._overlap_graph(current)

        assigned_new: set[int] = set()
        visited_old: set[int] = set()

        # Connected components of the bipartite overlap graph distinguish
        # continuation from split/merge/rearrangement without centroid guesses.
        for old0 in range(len(self.previous)):
            if old0 in visited_old or not old_to_new[old0]:
                continue

            old_block: set[int] = set()
            new_block: set[int] = set()
            queue: list[tuple[str, int]] = [("old", old0)]

            while queue:
                side, idx = queue.pop()
                if side == "old":
                    if idx in old_block:
                        continue
                    old_block.add(idx)
                    visited_old.add(idx)
                    for new_idx in old_to_new[idx]:
                        queue.append(("new", new_idx))
                else:
                    if idx in new_block:
                        continue
                    new_block.add(idx)
                    for old_idx in new_to_old[idx]:
                        queue.append(("old", old_idx))

            parent_ids = tuple(
                sorted(self.previous[i].track_id for i in old_block)
            )

            if len(old_block) == 1 and len(new_block) == 1:
                oi = next(iter(old_block))
                nj = next(iter(new_block))
                track_id = self.previous[oi].track_id
                current[nj].track_id = track_id
                assigned_new.add(nj)

                old = self.previous[oi]
                new = current[nj]
                raw_dx = abs(new.cx - old.cx)
                raw_dy = abs(new.cy - old.cy)
                h, w = self.shape
                if raw_dx > w / 2.0 or raw_dy > h / 2.0:
                    self.seam_continuations += 1
                continue

            if len(old_block) == 1 and len(new_block) > 1:
                event_type = "split"
            elif len(old_block) > 1 and len(new_block) == 1:
                event_type = "merge"
            else:
                event_type = "rearrangement"

            child_ids: list[int] = []
            for nj in sorted(new_block):
                track_id = self._new_track(parent_ids, t)
                current[nj].track_id = track_id
                assigned_new.add(nj)
                child_ids.append(track_id)

            for parent_id in parent_ids:
                self._end_track(parent_id, t)

            self.events.append(
                {
                    "frame": self.frame,
                    "t": float(t),
                    "type": event_type,
                    "parents": list(parent_ids),
                    "children": child_ids,
                }
            )

        # Old domains with no accepted overlap are deaths.
        for oi, old in enumerate(self.previous):
            if old_to_new[oi]:
                continue
            self._end_track(old.track_id, t)
            self.events.append(
                {
                    "frame": self.frame,
                    "t": float(t),
                    "type": "death",
                    "parents": [old.track_id],
                    "children": [],
                }
            )

        # New domains with no accepted parent are births.
        for nj, domain in enumerate(current):
            if nj in assigned_new:
                continue
            track_id = self._new_track((), t)
            domain.track_id = track_id
            self.events.append(
                {
                    "frame": self.frame,
                    "t": float(t),
                    "type": "birth",
                    "parents": [],
                    "children": [track_id],
                }
            )

        for domain in current:
            tr = self.tracks[domain.track_id]
            tr.last_frame = self.frame
            tr.last_t = float(t)
            sample = domain.phenotype()
            sample["t"] = float(t)
            tr.samples.append(sample)

        self.previous = current
        self.frame += 1
        return current

    def summary(self) -> dict[str, object]:
        event_counts: dict[str, int] = {}
        for event in self.events:
            name = str(event["type"])
            event_counts[name] = event_counts.get(name, 0) + 1

        lifetimes = np.asarray(
            [track.lifetime_frames for track in self.tracks.values()],
            dtype=float,
        )
        if len(lifetimes):
            lifetime = {
                "median_frames": float(np.median(lifetimes)),
                "p90_frames": float(np.quantile(lifetimes, 0.90)),
                "max_frames": int(np.max(lifetimes)),
            }
        else:
            lifetime = {
                "median_frames": 0.0,
                "p90_frames": 0.0,
                "max_frames": 0,
            }

        depth_cache: dict[int, int] = {}
        root_birth_cache: dict[int, float] = {}

        def lineage_depth(track_id: int) -> int:
            if track_id in depth_cache:
                return depth_cache[track_id]
            tr = self.tracks[track_id]
            if not tr.parents:
                value = 1
            else:
                known = [p for p in tr.parents if p in self.tracks]
                value = 1 + max((lineage_depth(p) for p in known), default=0)
            depth_cache[track_id] = value
            return value

        def earliest_root_birth(track_id: int) -> float:
            if track_id in root_birth_cache:
                return root_birth_cache[track_id]
            tr = self.tracks[track_id]
            known = [p for p in tr.parents if p in self.tracks]
            if not known:
                value = tr.born_t
            else:
                value = min(earliest_root_birth(p) for p in known)
            root_birth_cache[track_id] = value
            return value

        depths = [lineage_depth(tid) for tid in self.tracks]
        ancestral_spans = [
            self.tracks[tid].last_t - earliest_root_birth(tid)
            for tid in self.tracks
        ]

        genealogy = {
            "tracks_with_parents": int(
                sum(bool(tr.parents) for tr in self.tracks.values())
            ),
            "max_depth": int(max(depths, default=0)),
            "median_depth": float(np.median(depths)) if depths else 0.0,
            "max_ancestral_span": float(max(ancestral_spans, default=0.0)),
            "median_ancestral_span": (
                float(np.median(ancestral_spans))
                if ancestral_spans
                else 0.0
            ),
        }

        return {
            "tracks_created": len(self.tracks),
            "active_domains": len(self.previous),
            "event_counts": event_counts,
            "lifetimes": lifetime,
            "genealogy": genealogy,
            "seam_continuations": int(self.seam_continuations),
            "raw_periodic_count_last": int(self.raw_periodic_count),
            "raw_nonperiodic_count_last": int(self.raw_nonperiodic_count),
        }


def synthetic_torus_blob(
    shape: tuple[int, int],
    center: tuple[float, float],
    sigma: float = 3.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Small helper for tracker tests and demonstrations."""

    h, w = shape
    y, x = np.ogrid[:h, :w]
    cx, cy = center
    dx = torus_delta(x, cx, w)
    dy = torus_delta(y, cy, h)
    return amplitude * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
