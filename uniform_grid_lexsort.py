#!/usr/bin/env python3
"""
Uniform grid with lexsort + np.unique offsets (vectorized binning),
3x3 neighbor scan. NOTE: on CPU the pure-Python neighbor loop is still
slower than numpy-vectorized brute force (~0.4x) - the SoA+SIMD advantage
only materializes once the inner loop is JIT-compiled (numba njit / Cython).
This is the honest profile result; next step is the numba kernel.
"""
import numpy as np, time

def make(n,seed=0):
    return np.random.default_rng(seed).random((n,2))

def brute(pos,r):
    n=len(pos); r2=r*r; out=[[] for _ in range(n)]
    for i in range(n):
        d=pos-pos[i]; idx=np.where((d*d).sum(1)<=r2)[0]
        out[i]=[int(j) for j in idx if j!=i]
    return out

def uniform_grid(pos,r):
    n=len(pos)
    cell=np.floor(pos/r).astype(np.int64)
    order=np.lexsort((cell[:,1],cell[:,0]))
    key=(cell[:,0]*100000+cell[:,1])[order]
    uniq,starts,counts=np.unique(key,return_index=True,return_counts=True)
    lookup={int(k):(int(s),int(c)) for k,s,c in zip(uniq,starts,counts)}
    r2=r*r; out=[[] for _ in range(n)]
    for i in range(n):
        cx,cy=cell[i]; cand=[]
        for dx in(-1,0,1):
            for dy in(-1,0,1):
                slot=lookup.get(int((cx+dx)*100000+(cy+dy)))
                if slot:
                    s,c=slot; cand.extend(order[s:s+c].tolist())
        for j in cand:
            if j==i: continue
            d=pos[i]-pos[j]
            if (d*d).sum()<=r2: out[i].append(int(j))
    return out

def main():
    r=0.05
    print(f"[uniform grid + lexsort] r={r}")
    print(f"{'N':>7} {'brute(s)':>11} {'grid(s)':>11} {'speedup':>9} {'agree':>7}")
    for n in [500,1000,2000,4000]:
        pos=make(n)
        t0=time.perf_counter(); bf=brute(pos,r); tb=time.perf_counter()-t0
        t0=time.perf_counter(); ug=uniform_grid(pos,r); tg=time.perf_counter()-t0
        agree=all(sorted(bf[i])==sorted(ug[i]) for i in range(n))
        print(f"{n:>7} {tb:>11.5f} {tg:>11.5f} {tb/tg:>8.1f}x {str(agree):>7}")
    print("[note] pure-Python grid loses to numpy brute on CPU; numba/Cython inner loop next")

if __name__=='__main__': main()
