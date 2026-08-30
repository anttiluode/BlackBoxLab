"""Datarium 1 — lineage microscope on the Three Fates local-budget field.

This copies the LOCAL BUDGET arm of the supplied three_fates.py equations,
then replaces the old instantaneous scipy labels / leader-centroid heuristic
with the Datarium lineage instrument.

Nothing is inherited. Nothing is rewarded. Nothing is called a species.
Datarium 1 asks only whether thresholded domains possess auditable identity
long enough for later heredity experiments to make sense.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datarium.lineage import LineageTracker


N = 96
DT = 0.02
DX = 1.0
C_WAVE = 1.0
A_POT = 0.1
B_POT = 0.1
AMPLITUDE = 1.5
RADIUS = 10.0
VELOCITY = 0.25

TAU_RECOVER = 100.0
BURN = 0.033
DAMPING = 0.01

T_END = 360.0
MEASURE_EVERY = 20  # field steps = 0.4 simulated time units

HIGH_THRESHOLD = 0.30
LOW_THRESHOLD = 0.24
MIN_AREA = 20
MIN_MASS = 8.0
MIN_OVERLAP = 0.12


def laplacian(field: np.ndarray) -> np.ndarray:
    lx = (
        np.roll(field, -1, 1)
        - 2.0 * field
        + np.roll(field, 1, 1)
    ) / DX**2
    ly = (
        np.roll(field, -1, 0)
        - 2.0 * field
        + np.roll(field, 1, 0)
    ) / DX**2
    return lx + ly


class LocalBudgetField:
    """Exact local-budget mechanism from the supplied Three Fates script."""

    def __init__(self):
        y, x = np.ogrid[:N, :N]
        start = (N * 0.25, N * 0.5)
        distance = np.sqrt(
            (x - start[0]) ** 2 + (y - start[1]) ** 2
        )
        phi = AMPLITUDE / np.cosh(distance / RADIUS)
        gx = (
            np.roll(phi, -1, 1) - np.roll(phi, 1, 1)
        ) / (2.0 * DX)

        self.phi = phi
        self.prev = phi + VELOCITY * gx * DT
        self.r = np.ones((N, N), dtype=float)
        self.t = 0.0

    def step(self):
        acc = (
            C_WAVE**2 * laplacian(self.phi)
            - B_POT * self.phi**3
            + A_POT * self.r * self.phi
            - DAMPING * (self.phi - self.prev) / DT
        )

        self.r += DT * (
            (1.0 - self.r) / TAU_RECOVER
            - BURN * self.phi**2
        )
        np.clip(self.r, 0.0, 1.0, out=self.r)

        new = 2.0 * self.phi - self.prev + DT**2 * acc
        self.prev, self.phi = self.phi, new
        self.t += DT


def _track_receipt(tracker: LineageTracker) -> list[dict[str, object]]:
    longest = sorted(
        tracker.tracks.values(),
        key=lambda tr: tr.lifetime_frames,
        reverse=True,
    )[:16]

    out: list[dict[str, object]] = []
    for tr in longest:
        last = tr.samples[-1] if tr.samples else {}
        out.append(
            {
                "track_id": tr.track_id,
                "parents": list(tr.parents),
                "born_t": tr.born_t,
                "last_t": tr.last_t,
                "lifetime_frames": tr.lifetime_frames,
                "last_phenotype": last,
            }
        )
    return out


def run() -> dict[str, object]:
    field = LocalBudgetField()
    tracker = LineageTracker(
        (N, N),
        high_threshold=HIGH_THRESHOLD,
        low_threshold=LOW_THRESHOLD,
        min_area=MIN_AREA,
        min_mass=MIN_MASS,
        min_overlap=MIN_OVERLAP,
    )

    domain_counts: list[int] = []
    raw_counts: list[int] = []
    dust_counts: list[int] = []
    resource_means: list[float] = []
    seam_label_mismatch_frames = 0

    total_steps = int(round(T_END / DT))
    for step in range(total_steps):
        field.step()
        if step % MEASURE_EVERY:
            continue

        domains = tracker.update(field.phi, field.t)
        domain_counts.append(len(domains))
        raw_counts.append(tracker.raw_periodic_count)
        dust_counts.append(max(tracker.raw_periodic_count - len(domains), 0))
        resource_means.append(float(field.r.mean()))

        if tracker.raw_periodic_count != tracker.raw_nonperiodic_count:
            seam_label_mismatch_frames += 1

    summary = tracker.summary()
    event_counts = Counter(str(event["type"]) for event in tracker.events)

    lifetimes = np.asarray(
        [tr.lifetime_frames for tr in tracker.tracks.values()],
        dtype=float,
    )
    measurement_dt = MEASURE_EVERY * DT

    def fraction_at_least(frames: int) -> float:
        if not len(lifetimes):
            return 0.0
        return float(np.mean(lifetimes >= frames))

    late_start = len(domain_counts) // 2
    receipt = {
        "experiment": "Datarium 1 — lineage microscope",
        "claim_boundary": (
            "This is an identity instrument. It does not implement heredity, "
            "fitness, traits, genomes, or evolution."
        ),
        "source_substrate": (
            "Three Fates LOCAL BUDGET phi^4 wave arm: local resource recovers "
            "and is burned by phi^2 activity."
        ),
        "config": {
            "N": N,
            "dt": DT,
            "t_end": T_END,
            "measure_every_steps": MEASURE_EVERY,
            "measurement_dt": measurement_dt,
            "high_threshold": HIGH_THRESHOLD,
            "low_threshold": LOW_THRESHOLD,
            "min_area": MIN_AREA,
            "min_mass": MIN_MASS,
            "min_overlap": MIN_OVERLAP,
        },
        "tracker": summary,
        "event_counts": dict(event_counts),
        "measurements": len(domain_counts),
        "late_domain_count": {
            "mean": float(np.mean(domain_counts[late_start:])),
            "median": float(np.median(domain_counts[late_start:])),
            "max": int(np.max(domain_counts[late_start:])),
        },
        "raw_domain_count": {
            "mean": float(np.mean(raw_counts)),
            "max": int(np.max(raw_counts)),
        },
        "dust_removed_per_measurement": {
            "mean": float(np.mean(dust_counts)),
            "max": int(np.max(dust_counts)),
            "fraction_frames_with_dust": float(
                np.mean(np.asarray(dust_counts) > 0)
            ),
        },
        "seam_audit": {
            "frames_periodic_vs_nonperiodic_count_differ": int(
                seam_label_mismatch_frames
            ),
            "fraction_measurements": float(
                seam_label_mismatch_frames / max(len(domain_counts), 1)
            ),
            "continuations_across_seam": int(
                tracker.seam_continuations
            ),
        },
        "lifetime": {
            "median_simulated_time": float(
                np.median(lifetimes) * measurement_dt
            )
            if len(lifetimes)
            else 0.0,
            "p90_simulated_time": float(
                np.quantile(lifetimes, 0.90) * measurement_dt
            )
            if len(lifetimes)
            else 0.0,
            "max_simulated_time": float(
                np.max(lifetimes) * measurement_dt
            )
            if len(lifetimes)
            else 0.0,
            "fraction_ge_2_time": fraction_at_least(
                int(np.ceil(2.0 / measurement_dt))
            ),
            "fraction_ge_5_time": fraction_at_least(
                int(np.ceil(5.0 / measurement_dt))
            ),
            "fraction_ge_10_time": fraction_at_least(
                int(np.ceil(10.0 / measurement_dt))
            ),
        },
        "resource": {
            "final_mean": float(resource_means[-1]),
            "late_mean": float(np.mean(resource_means[late_start:])),
        },
        "longest_tracks": _track_receipt(tracker),
        "events_tail": tracker.events[-60:],
    }
    return receipt


def main() -> None:
    receipt = run()
    out = ROOT / "results" / "datarium1.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")

    print("DATARIUM 1 — LINEAGE MICROSCOPE")
    print("=" * 68)
    print(
        "observer: torus components + threshold hysteresis + mass floor + "
        "overlap lineage"
    )
    print("NO genes, traits, reward, or inherited bookkeeping")
    print()
    print(
        f"late accepted domains: "
        f"{receipt['late_domain_count']['mean']:.2f} mean, "
        f"{receipt['late_domain_count']['max']} max"
    )
    print(
        f"dust removed: "
        f"{receipt['dust_removed_per_measurement']['mean']:.2f} components/"
        f"measurement; present on "
        f"{100*receipt['dust_removed_per_measurement']['fraction_frames_with_dust']:.1f}% "
        "of frames"
    )
    print(
        f"seam: periodic/nonperiodic labels differ on "
        f"{receipt['seam_audit']['frames_periodic_vs_nonperiodic_count_differ']} "
        f"/ {receipt['measurements']} measurements; "
        f"{receipt['seam_audit']['continuations_across_seam']} tracked seam continuations"
    )
    print("events:", receipt["event_counts"])
    print(
        "lifetime simulated time: "
        f"median={receipt['lifetime']['median_simulated_time']:.2f}, "
        f"p90={receipt['lifetime']['p90_simulated_time']:.2f}, "
        f"max={receipt['lifetime']['max_simulated_time']:.2f}"
    )
    print(
        "survival fractions >=2/5/10 time: "
        f"{receipt['lifetime']['fraction_ge_2_time']:.3f} / "
        f"{receipt['lifetime']['fraction_ge_5_time']:.3f} / "
        f"{receipt['lifetime']['fraction_ge_10_time']:.3f}"
    )
    genealogy = receipt["tracker"]["genealogy"]
    print(
        "genealogy: "
        f"max depth={genealogy['max_depth']}  "
        f"median depth={genealogy['median_depth']:.1f}  "
        f"max ancestral span={genealogy['max_ancestral_span']:.2f}  "
        f"tracks with parents={genealogy['tracks_with_parents']}"
    )
    print(
        f"resource late mean={receipt['resource']['late_mean']:.3f}"
    )
    print()
    print(
        "Interpretation guardrail: a split/merge graph is now measurable. "
        "Whether its identities persist long enough to support physical "
        "heredity is the RESULT, not an assumption."
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
