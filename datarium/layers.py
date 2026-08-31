"""Post-hoc instruments for Datarium 3.

Nothing in this module is visible to the simulated particles or fields.  It
only asks whether local wave-coupled particles happen to form persistent,
co-moving assemblies.  The dynamics never receives an assembly identifier,
size, score, or phenotype.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def torus_delta(values: np.ndarray, period: float) -> np.ndarray:
    return (np.asarray(values) + period / 2.0) % period - period / 2.0


@dataclass
class Assembly:
    """One measured co-moving component at one observation time."""

    members: tuple[int, ...]
    alignment: float
    phase_coherence: float
    elongation: float
    cx: float
    cy: float
    track_id: int = -1

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class AssemblyTrack:
    track_id: int
    born_sample: int
    last_sample: int
    samples: list[dict[str, float]] = field(default_factory=list)
    ended_sample: int | None = None

    @property
    def lifetime_samples(self) -> int:
        return self.last_sample - self.born_sample + 1


def _unwrap_component(
    positions: np.ndarray,
    members: tuple[int, ...],
    period: float,
) -> np.ndarray:
    points = np.asarray(positions[list(members)], dtype=float)
    if len(points) == 0:
        return points
    reference = points[0]
    return reference + torus_delta(points - reference, period)


def describe_assembly(
    positions: np.ndarray,
    headings: np.ndarray,
    phases: np.ndarray,
    members: tuple[int, ...],
    period: float,
) -> Assembly:
    ids = np.asarray(members, dtype=int)
    heading_order = np.mean(np.exp(1j * headings[ids]))
    phase_order = np.mean(np.exp(1j * phases[ids]))

    unwrapped = _unwrap_component(positions, members, period)
    center = np.mean(unwrapped, axis=0) % period
    if len(unwrapped) >= 3:
        covariance = np.cov(unwrapped.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        # A finite width floor stops three nearly collinear points from
        # producing a meaningless million-to-one aspect ratio.
        small = max(float(eigenvalues[0]), 0.0625)
        large = max(float(eigenvalues[1]), small)
        elongation = float(np.sqrt(large / small))
    else:
        elongation = 1.0

    return Assembly(
        members=tuple(sorted(int(i) for i in members)),
        alignment=float(abs(heading_order)),
        phase_coherence=float(abs(phase_order)),
        elongation=min(elongation, 50.0),
        cx=float(center[0]),
        cy=float(center[1]),
    )


class CoherentAssemblyTracker:
    """Track co-moving particle components by membership overlap.

    Edges use only proximity and velocity-direction agreement.  Phase
    coherence and elongation are measured phenotypes, not membership rules.
    This keeps the observer from defining the wave coherence it hopes to see.
    """

    def __init__(
        self,
        agent_count: int,
        period: float,
        link_radius: float = 4.0,
        min_heading_cosine: float = 0.50,
        min_size: int = 3,
        min_membership_overlap: float = 0.25,
    ):
        self.agent_count = int(agent_count)
        self.period = float(period)
        self.link_radius = float(link_radius)
        self.min_heading_cosine = float(min_heading_cosine)
        self.min_size = int(min_size)
        self.min_membership_overlap = float(min_membership_overlap)

        self.sample = 0
        self.next_id = 1
        self.previous: list[Assembly] = []
        self.tracks: dict[int, AssemblyTrack] = {}
        self.history: list[dict[str, float]] = []

    def _components(
        self,
        positions: np.ndarray,
        headings: np.ndarray,
    ) -> list[tuple[int, ...]]:
        displacement = positions[:, None, :] - positions[None, :, :]
        displacement = torus_delta(displacement, self.period)
        distance = np.linalg.norm(displacement, axis=2)
        agreement = np.cos(headings[:, None] - headings[None, :])
        adjacency = (
            (distance <= self.link_radius)
            & (agreement >= self.min_heading_cosine)
        )
        np.fill_diagonal(adjacency, False)

        seen = np.zeros(self.agent_count, dtype=bool)
        components: list[tuple[int, ...]] = []
        for start in range(self.agent_count):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            members: list[int] = []
            while stack:
                node = stack.pop()
                members.append(node)
                for neighbour in np.flatnonzero(adjacency[node]):
                    neighbour = int(neighbour)
                    if not seen[neighbour]:
                        seen[neighbour] = True
                        stack.append(neighbour)
            if len(members) >= self.min_size:
                components.append(tuple(sorted(members)))
        return components

    @staticmethod
    def _jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
        aa, bb = set(a), set(b)
        return len(aa & bb) / max(len(aa | bb), 1)

    def update(
        self,
        positions: np.ndarray,
        headings: np.ndarray,
        phases: np.ndarray,
    ) -> list[Assembly]:
        components = self._components(positions, headings)
        current = [
            describe_assembly(
                positions,
                headings,
                phases,
                members,
                self.period,
            )
            for members in components
        ]

        candidates: list[tuple[float, int, int]] = []
        for old_index, old in enumerate(self.previous):
            for new_index, new in enumerate(current):
                score = self._jaccard(old.members, new.members)
                if score >= self.min_membership_overlap:
                    candidates.append((score, old_index, new_index))

        used_old: set[int] = set()
        used_new: set[int] = set()
        for _, old_index, new_index in sorted(candidates, reverse=True):
            if old_index in used_old or new_index in used_new:
                continue
            current[new_index].track_id = self.previous[old_index].track_id
            used_old.add(old_index)
            used_new.add(new_index)

        for old_index, old in enumerate(self.previous):
            if old_index not in used_old:
                self.tracks[old.track_id].ended_sample = self.sample

        for new_index, assembly in enumerate(current):
            if new_index not in used_new:
                assembly.track_id = self.next_id
                self.tracks[self.next_id] = AssemblyTrack(
                    track_id=self.next_id,
                    born_sample=self.sample,
                    last_sample=self.sample,
                )
                self.next_id += 1

            track = self.tracks[assembly.track_id]
            track.last_sample = self.sample
            track.samples.append(
                {
                    "size": float(assembly.size),
                    "alignment": assembly.alignment,
                    "phase_coherence": assembly.phase_coherence,
                    "elongation": assembly.elongation,
                }
            )

        coherent_members = {
            member for assembly in current for member in assembly.members
        }
        self.history.append(
            {
                "assembly_count": float(len(current)),
                "coherent_fraction": len(coherent_members)
                / max(self.agent_count, 1),
                "largest_size": float(
                    max((assembly.size for assembly in current), default=0)
                ),
                "mean_alignment": float(
                    np.mean([a.alignment for a in current])
                )
                if current
                else 0.0,
                "mean_phase_coherence": float(
                    np.mean([a.phase_coherence for a in current])
                )
                if current
                else 0.0,
                "mean_elongation": float(
                    np.mean([a.elongation for a in current])
                )
                if current
                else 0.0,
            }
        )
        self.previous = current
        self.sample += 1
        return current

    def summary(self, late_fraction: float = 0.40) -> dict[str, float | int]:
        start = int(len(self.history) * (1.0 - late_fraction))
        late = self.history[start:] or self.history

        def mean(name: str) -> float:
            return float(np.mean([row[name] for row in late])) if late else 0.0

        lifetimes = np.asarray(
            [track.lifetime_samples for track in self.tracks.values()],
            dtype=float,
        )
        return {
            "tracks_created": len(self.tracks),
            "late_mean_assembly_count": mean("assembly_count"),
            "late_mean_coherent_fraction": mean("coherent_fraction"),
            "late_mean_largest_size": mean("largest_size"),
            "late_mean_alignment": mean("mean_alignment"),
            "late_mean_phase_coherence": mean("mean_phase_coherence"),
            "late_mean_elongation": mean("mean_elongation"),
            "max_lifetime_samples": int(np.max(lifetimes))
            if len(lifetimes)
            else 0,
            "p90_lifetime_samples": float(np.quantile(lifetimes, 0.90))
            if len(lifetimes)
            else 0.0,
        }
