#!/usr/bin/env python3
"""
Particle neighbor search: brute-force O(N^2) vs spatial-hash grid O(N).

Used to find neighbors within radius r for a real-time particle simulation.
The uniform grid bins particles into cells of size r, so each particle only
checks the 3x3 (2D) neighboring cells instead of all N particles.

Outputs: correctness check (both methods agree) + timing benchmark.
"""
import numpy as np
import time
from scipy.spatial import cKDTree


def make_particles(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))  # positions in unit square


def brute_force(pos, r):
    """O(N^2) reference: adjacency list of neighbors within r."""
    n = len(pos)
    r2 = r * r
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        d = pos - pos[i]
        idx = np.where((d * d).sum(1) <= r2)[0]
        neighbors[i] = [int(j) for j in idx if j != i]
    return neighbors


def spatial_hash(pos, r):
    """O(N) uniform grid: bin into r-sized cells, check 3x3 neighbors."""
    n = len(pos)
    cell = np.floor(pos / r).astype(int)
    grid = {}
    for i in range(n):
        key = tuple(cell[i])
        grid.setdefault(key, []).append(i)
    r2 = r * r
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        cx, cy = cell[i]
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.extend(grid.get((cx + dx, cy + dy), ()))
        for j in cand:
            if j == i:
                continue
            d = pos[i] - pos[j]
            if (d * d).sum() <= r2:
                neighbors[i].append(j)
    return neighbors


def kd_tree(pos, r):
    """O(N log N) KD-tree spatial index (scipy cKDTree)."""
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r, output_type="ndarray")
    neighbors = [[] for _ in range(len(pos))]
    for i, j in pairs:
        neighbors[i].append(int(j))
        neighbors[j].append(int(i))
    return neighbors


def bench(n, r):
    pos = make_particles(n)
    t0 = time.perf_counter()
    bf = brute_force(pos, r)
    t_bf = time.perf_counter() - t0

    t0 = time.perf_counter()
    sh = kd_tree(pos, r)
    t_sh = time.perf_counter() - t0

    agree = all(sorted(bf[i]) == sorted(sh[i]) for i in range(n))
    return t_bf, t_sh, agree


def main():
    r = 0.05
    print(f"[neighbor-search] brute O(N^2) vs KD-tree O(N log N), r={r}")
    print(f"{'N':>7} {'brute(s)':>12} {'kdtree(s)':>12} {'speedup':>10} {'agree':>7}")
    for n in [500, 1000, 2000, 4000]:
        t_bf, t_sh, agree = bench(n, r)
        print(f"{n:>7} {t_bf:>12.5f} {t_sh:>12.5f} {t_bf/t_sh:>9.1f}x {str(agree):>7}")


if __name__ == "__main__":
    main()
