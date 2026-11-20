# particle-neighbor-bench

Neighbor-search benchmark for a real-time particle simulation:
brute-force O(N^2) vs KD-tree (scipy cKDTree) O(N log N).

## Run

    pip install numpy scipy
    python neighbor_search.py

## Results (r=0.05, unit square)

| N    | brute (s) | KD-tree (s) | speedup | agree |
|------|-----------|-------------|---------|-------|
| 500  | 0.0081    | 0.0039      | 2.1x    | True  |
| 1000 | 0.0228    | 0.0038      | 6.0x    | True  |
| 2000 | 0.0774    | 0.0130      | 6.0x    | True  |
| 4000 | 0.2848    | 0.0507      | 5.6x    | True  |

Both methods return identical neighbor lists; KD-tree is ~6x faster at scale.
CI runs the benchmark on every push and uploads the result as an artifact.
