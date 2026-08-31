"""
CELL 0 -- can the train build a wall that closes?

Medium:
  psi   fast Klein-Gordon field, dpsi2/dt2 = c^2 lap(psi) - V(x) psi - gam dpsi/dt + sources
  M     slow material map, V = V0 * M.  High M = high cutoff frequency = opaque wall.
  R     resource, regrows, is the only source of material.

Scouts:
  read the local field envelope E, move on grad E, emit at their own frequency,
  deposit material out of a private store which they refill from R.

Conserved: sum(M) + sum(R) + sum(store) is exactly constant except for regrowth,
which is accounted separately.  Printed as a check.

NO fitness function. NO agent-agent forces. NO reproduction in stage 1.
Question: does material end up in CLOSED walls more than in a surrogate with the
same power spectrum and the same value histogram?
"""
import numpy as np


def lap(a):
    return (np.roll(a, 1, 0) + np.roll(a, -1, 0)
            + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4.0 * a)


class World:
    def __init__(self, N=128, n_scouts=8000, seed=0, deposit=True,
                 V0=0.8, gam=0.02, dt=0.4, c=1.0,
                 dep_rate=0.040, uptake=0.0040, metab=0.0010,
                 decay=0.0008, regrow=0.0025, R0=0.08, cap=0.25,
                 emit=0.010, env_a=0.02, thresh=0.35, g_lo=-1.0, g_hi=1.0):
        self.rng = np.random.default_rng(seed)
        self.N, self.dt, self.c, self.V0, self.gam = N, dt, c, V0, gam
        self.deposit = deposit
        self.dep_rate, self.uptake, self.metab = dep_rate, uptake, metab
        self.decay, self.regrow, self.cap, self.R0 = decay, regrow, cap, R0
        self.emit, self.env_a, self.thresh = emit, env_a, thresh

        self.psi = np.zeros((N, N), np.float64)
        self.old = np.zeros((N, N), np.float64)
        self.E = np.zeros((N, N), np.float64)
        self.M = np.zeros((N, N), np.float64)
        self.Mx = np.zeros((N, N), np.float64)
        self.My = np.zeros((N, N), np.float64)
        self.R = np.ones((N, N), np.float64) * R0
        self.t = 0.0

        n = n_scouts
        self.x = self.rng.uniform(0, N, n)
        self.y = self.rng.uniform(0, N, n)
        self.vx = self.rng.normal(0, .05, n)
        self.vy = self.rng.normal(0, .05, n)
        self.store = np.full(n, 0.05)
        # genome: w deposit gain, g gradient gain (signed), f emission freq
        self.w = self.rng.uniform(0.5, 1.5, n)
        self.g = self.rng.uniform(g_lo, g_hi, n) * 6.0
        self.f = self.rng.uniform(0.40, 0.60, n)
        self.ph = self.rng.uniform(0, 2 * np.pi, n)
        self.alive = np.ones(n, bool)
        self.regrown = 0.0
        self.burned = 0.0

        # seed the field with broadband noise so nothing is hand-placed
        ky, kx = np.meshgrid(np.fft.fftfreq(N) * N, np.fft.fftfreq(N) * N, indexing='ij')
        self.gk = np.exp(-0.5 * (6.0 ** 2) * (2 * np.pi / N) ** 2 * (kx ** 2 + ky ** 2))
        self.Eb = np.zeros((N, N))
        self.psi += self.rng.normal(0, 1e-3, (N, N))
        self.old = self.psi.copy()

    def total(self):
        return self.M.sum() + self.R.sum() + self.store[self.alive].sum()

    def step(self):
        N, dt = self.N, self.dt
        ix = np.mod(self.x.astype(np.int32), N)
        iy = np.mod(self.y.astype(np.int32), N)
        al = self.alive

        # ---- sources: each live scout drives the field at its own frequency
        src = np.zeros((N, N))
        amp = self.emit * np.sin(self.f[al] * self.t + self.ph[al])
        np.add.at(src, (ix[al], iy[al]), amp)

        # ---- wave update (leapfrog, variable mass term)
        V = self.V0 * self.M
        new = (2 * self.psi - self.old
               + dt * dt * (self.c ** 2 * lap(self.psi) - V * self.psi + src)
               - self.gam * dt * (self.psi - self.old))
        self.old, self.psi = self.psi, new
        self.E += self.env_a * (np.abs(self.psi) - self.E)
        if int(self.t / dt) % 5 == 0:
            self.Eb = np.fft.ifft2(np.fft.fft2(self.E) * self.gk).real

        # ---- material turnover: M decays back into R, R regrows toward 1
        d = self.decay * self.M * dt
        self.M -= d
        self.R += d
        f = 1.0 - self.decay * dt            # director decays with its own mass
        self.Mx *= f
        self.My *= f
        g = self.regrow * (self.R0 - self.R) * dt
        self.R += g
        self.regrown += g.sum()

        # ---- scouts
        Ex = (np.roll(self.E, -1, 0) - np.roll(self.E, 1, 0)) * 0.5
        Ey = (np.roll(self.E, -1, 1) - np.roll(self.E, 1, 1)) * 0.5
        gx, gy = Ex[ix, iy], Ey[ix, iy]
        sc = 1.0 / (np.abs(gx) + np.abs(gy) + 1e-6)
        self.vx += self.g * gx * sc * dt * 0.2
        self.vy += self.g * gy * sc * dt * 0.2
        self.vx *= 0.96
        self.vy *= 0.96
        sp = np.hypot(self.vx, self.vy)
        too = sp > 0.6
        self.vx[too] *= 0.6 / sp[too]
        self.vy[too] *= 0.6 / sp[too]
        self.vx += self.rng.normal(0, .006, len(self.vx))
        self.vy += self.rng.normal(0, .006, len(self.vy))
        self.x = np.mod(self.x + self.vx * dt * 4, N)
        self.y = np.mod(self.y + self.vy * dt * 4, N)

        # ---- eat
        Rl = self.R[ix, iy]
        take = np.where(al, np.minimum(self.uptake * dt * Rl,
                                       np.maximum(self.cap - self.store, 0)), 0.0)
        np.add.at(self.R, (ix, iy), -take)
        self.store += take
        b = self.metab * dt * al
        self.store -= b
        self.burned += b.sum()

        # ---- deposit: gated by the local field envelope only
        if self.deposit:
            El = self.E[ix, iy]
            drive = np.clip(El / (self.Eb[ix, iy] + 1e-12) - 1.0, 0.0, 3.0)
            room = np.maximum(1.0 - self.M[ix, iy], 0.0)   # occupancy limit
            put = np.where(al, np.minimum(self.dep_rate * self.w * drive * room * dt,
                                          0.05 * np.maximum(self.store, 0)), 0.0)
            self.store -= put
            np.add.at(self.M, (ix, iy), put)
            sp2 = np.hypot(self.vx, self.vy) + 1e-9      # fibre laid along travel
            np.add.at(self.Mx, (ix, iy), put * self.vx / sp2)
            np.add.at(self.My, (ix, iy), put * self.vy / sp2)
            np.clip(self.M, 0, 1.0, out=self.M)

        dead = self.alive & (self.store <= 0)
        if dead.any():                      # bodies decompose back into resource
            np.add.at(self.R, (ix[dead], iy[dead]), np.maximum(self.store[dead], 0))
        self.alive &= ~dead
        self.store[~self.alive] = 0.0
        self.t += dt

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self


# ---------- measurement ----------

def torus_label(mask):
    """Connected components of a boolean mask on a torus. Returns labels, sizes."""
    N = mask.shape[0]
    parent = np.arange(N * N)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    idx = np.arange(N * N).reshape(N, N)
    for ax in (0, 1):
        nb = np.roll(idx, -1, ax)
        both = mask & np.roll(mask, -1, ax)
        for a, b in zip(idx[both], nb[both]):
            union(int(a), int(b))
    lab = np.full(N * N, -1)
    flat = mask.ravel()
    roots = {}
    for i in np.nonzero(flat)[0]:
        r = find(int(i))
        if r not in roots:
            roots[r] = len(roots)
        lab[i] = roots[r]
    lab = lab.reshape(N, N)
    k = len(roots)
    sizes = np.bincount(lab[lab >= 0], minlength=k) if k else np.array([])
    return lab, sizes


def voids(M, q=0.80, min_area=12):
    """Enclosed background regions = candidate cell interiors."""
    thr = np.quantile(M, q)
    wall = M > thr
    lab, sizes = torus_label(~wall)
    if len(sizes) == 0:
        return 0, 0.0, lab, wall
    big = int(np.argmax(sizes))
    enc = [i for i in range(len(sizes)) if i != big and sizes[i] >= min_area]
    return len(enc), float(sum(sizes[i] for i in enc)), lab, wall


def surrogate(M, rng):
    """Same power spectrum, same value histogram, geometry destroyed."""
    F = np.fft.fft2(M)
    ph = rng.uniform(0, 2 * np.pi, M.shape)
    ph = (ph - ph[::-1, ::-1]) / 2.0          # antisymmetric -> real output
    S = np.fft.ifft2(np.abs(F) * np.exp(1j * ph)).real
    # rank-remap onto the exact original value distribution
    out = np.empty_like(S)
    out.ravel()[np.argsort(S.ravel())] = np.sort(M.ravel())
    return out
