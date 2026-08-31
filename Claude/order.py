"""
ORDER TEST -- does the material record WHEN, or only HOW MUCH?

Sunday's design, transplanted.  Two arms with exactly matched marginals:

  FIRST-SEEK   every scout climbs the envelope for t < T/2, descends it after
  FIRST-AVOID  the reverse

Same population, same genomes, same total time in each mode, same budget, same
physics.  The ONLY difference is the order.

If a classifier can tell the final material apart, the world records sequence --
which is the property a contraction destroys, and the precondition for anything
that deserves the word inheritance.  If it cannot, the material is a function of
drive statistics and there is no history in it to transcribe.

Control arm MIXED (half seekers, half avoiders throughout) gives the scale of
seed-to-seed variation the effect has to clear.
"""
import numpy as np
from cell0 import World
from signature import features, loo, radial, CFG

T, HALF, NW = 4000, 2000, 8


def phased(seed, seek_first):
    w = World(seed=seed, g_lo=0.0, g_hi=1.0, **CFG)        # magnitudes only
    w.g = np.abs(w.g) * (1.0 if seek_first else -1.0)
    w.run(HALF)
    w.g = -w.g                                             # flip; nothing else changes
    w.run(T - HALF)
    return w


def mixed(seed):
    return World(seed=seed, g_lo=-1.0, g_hi=1.0, **CFG).run(T)


rows, y, feats = [], [], []
for cls, sf in enumerate([True, False]):
    for s in range(NW):
        w = phased(1000 * cls + s, sf)
        feats.append(features(w)); y.append(cls)
        print('.', end='', flush=True)
print()
y = np.array(y)

print('\n%-10s %8s %10s' % ('features', 'LOO acc', 'max |d|'))
for key in ('scalar', 'spatial', 'director', 'all'):
    X = np.array([f[key] for f in feats])
    m0, m1, sd = X[y == 0].mean(0), X[y == 1].mean(0), X.std(0) + 1e-12
    print('%-10s %8.3f %10.2f' % (key, loo(X, y), np.max(np.abs(m0 - m1) / sd)))

# how much does the material vary between seeds within one arm, for scale
ctrl = [features(mixed(9000 + s)) for s in range(4)]
Xc = np.array([f['all'] for f in ctrl])
Xa = np.array([f['all'] for f in feats])[y == 0]
print('\nwithin-arm spread (mixed control) %.3f' % np.mean(np.std(Xc, 0) / (Xa.std(0) + 1e-12)))
