"""
Same ending, different past.  Both arms spend their last 6000 steps AVOIDING.
One of them spent 2000 steps SEEKING first.  Material half-life is ~2170 steps,
so only ~15% of anything laid in that first phase is still there at the end.

If a classifier can still separate them, the mark is not undecayed material.
It is structure the medium locked in.
"""
import numpy as np
from cell0 import World
from signature import features, loo, CFG

P1, P2, NW = 2000, 6000, 6

def arm(seed, seek_first):
    w = World(seed=seed, g_lo=0.0, g_hi=1.0, **CFG)
    w.g = np.abs(w.g) * (-1.0 if not seek_first else 1.0)
    w.run(P1)
    w.g = -np.abs(w.g)          # both arms avoid from here on
    w.run(P2)
    return w

feats, y = [], []
for cls, sf in enumerate([True, False]):        # 0 = seek-then-avoid, 1 = avoid throughout
    for s in range(NW):
        feats.append(features(arm(2000*cls+s, sf))); y.append(cls); print('.', end='', flush=True)
print()
y=np.array(y)
print('\nsurviving fraction of phase-1 material: %.1f%%' % (100*np.exp(-0.0008*0.4*P2)))
print('%-10s %8s %10s' % ('features','LOO acc','max |d|'))
for key in ('scalar','spatial','director','all'):
    X=np.array([f[key] for f in feats]); m0,m1,sd=X[y==0].mean(0),X[y==1].mean(0),X.std(0)+1e-12
    print('%-10s %8.3f %10.2f' % (key, loo(X,y), np.max(np.abs(m0-m1)/sd)))
