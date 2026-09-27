import os, sys, time
os.environ["XDG_CACHE_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tinygrad_rsvd import randomized_svd_tg

def make_matrix(m, n, seed=0):
    rng = np.random.default_rng(seed)
    r = min(m, n, 120)
    U,_ = np.linalg.qr(rng.standard_normal((m,r)))
    V,_ = np.linalg.qr(rng.standard_normal((n,r)))
    s = 1000.0*(0.85**np.arange(r))
    return (U*s)@V.T + 0.1*rng.standard_normal((m,n))

for (m,n) in [(4000,1000),(15000,2000)]:
    A = make_matrix(m,n)
    ref = np.linalg.svd(A, compute_uv=False)[:50]
    ts=[]
    for i in range(4):
        t0=time.perf_counter(); U,S,Vh = randomized_svd_tg(A,k=50,p=10,q_iter=2,device="METAL"); dt=time.perf_counter()-t0
        if i>=1: ts.append(dt)
    rec = np.linalg.norm(A - (U[:, :50]*S[:50])@Vh[:50,:])/np.linalg.norm(A)
    se = float(np.max(np.abs(S[:50]-ref)/np.abs(ref)))
    print(f"tg_hybrid {m}x{n}: t={min(ts)*1e3:.2f} ms  recon={rec:.4f} sv_relerr={se:.2e}")
