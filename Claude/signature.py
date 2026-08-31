"""
SIGNATURE TEST -- the gate that has to pass before "the field is DNA" means anything.

Two populations with MATCHED MARGINALS: identical |g| distribution, identical deposit
gain, opposite sign of g.  SEEKERS climb the field envelope, AVOIDERS descend it.
Same deposition rule, same physics, same budget.

Q1  Does the material they leave behind distinguish them at all?
Q2  If yes, does the SPATIAL ARRANGEMENT carry anything one scalar summary does not?
    (Sunday's resource attacker, transplanted.)

Q2 is the one that matters.  If a single number -- mean material -- separates the two
populations as well as the whole map does, then the world records a level, not a
history, and there is nothing for a later reader to transcribe.
"""
import numpy as np
from cell0 import World, voids

CFG = dict(n_scouts=2500, V0=8.0, R0=0.60, regrow=0.0040,
           metab=0.0006, uptake=0.02, dep_rate=0.08)
T = 4000
NW = 8


def radial(M, nb=6):
    F = np.abs(np.fft.fft2(M - M.mean())) ** 2
    N = M.shape[0]
    ky, kx = np.meshgrid(np.fft.fftfreq(N) * N, np.fft.fftfreq(N) * N, indexing='ij')
    k = np.hypot(kx, ky).ravel()
    F = F.ravel()
    edges = np.geomspace(1, N / 2, nb + 1)
    p = np.array([F[(k >= edges[i]) & (k < edges[i + 1])].mean() for i in range(nb)])
    return p / (p.sum() + 1e-30)                 # shape only, total power divided out


def features(w):
    M, S = w.M, np.hypot(w.Mx, w.My) / (w.M + 1e-9)
    mw = (S * M).sum() / (M.sum() + 1e-12)
    # director autocorrelation at lag 4, along x and y
    nx, ny = w.Mx / (M + 1e-9), w.My / (M + 1e-9)
    ac = np.mean([(nx * np.roll(nx, L, a) + ny * np.roll(ny, L, a)).mean()
                  for a in (0, 1) for L in (2, 6)])
    return dict(
        scalar=np.array([M.mean()]),
        spatial=np.concatenate([radial(M), np.quantile(M, [.5, .9, .99])]),
        director=np.array([mw, ac, S.std()]),
        all=np.concatenate([radial(M), np.quantile(M, [.5, .9, .99]),
                            [mw, ac, S.std(), M.mean()]]),
    )


def loo(X, y):
    """Leave-one-out nearest centroid on z-scored features."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ok = 0
    for i in range(len(X)):
        m = np.ones(len(X), bool)
        m[i] = False
        c0 = X[m & (y == 0)].mean(0)
        c1 = X[m & (y == 1)].mean(0)
        ok += int((np.linalg.norm(X[i] - c1) < np.linalg.norm(X[i] - c0)) == y[i])
    return ok / len(X)


rows, lab = [], []
for cls, (lo, hi) in enumerate([(0.0, 1.0), (-1.0, 0.0)]):     # seekers / avoiders
    for s in range(NW):
        w = World(seed=100 * cls + s, g_lo=lo, g_hi=hi, **CFG).run(T)
        rows.append(features(w))
        lab.append(cls)
        print('.', end='', flush=True)
print()
y = np.array(lab)
print('\n%-10s %8s   %s' % ('features', 'LOO acc', 'class means'))
for key in ('scalar', 'spatial', 'director', 'all'):
    X = np.array([r[key] for r in rows])
    a = loo(X, y)
    m0, m1 = X[y == 0].mean(0), X[y == 1].mean(0)
    sd = X.std(0) + 1e-12
    print('%-10s %8.3f   max |d| = %.2f' % (key, a, np.max(np.abs(m0 - m1) / sd)))
