# flowsom-rs

Rust SOM training for [FlowSOM](https://github.com/saeyslab/FlowSOM_Python), via PyO3.

Implements the batch-learning SOM from [Otsuka et al. 2025](https://doi.org/10.1002/cyto.a.24918) with rayon parallelism. Unlike FlowSOM's online SOM, the batch algorithm processes all events simultaneously each epoch, so **results are deterministic regardless of input event order**.

## Install

```bash
pip install flowsom-rs
```

Requires Python ≥ 3.10 and numpy ≥ 1.24.

## Quick start

```python
import numpy as np
import flowsom_rs
from scipy.spatial.distance import pdist, squareform

data = np.random.randn(500_000, 7)

# SOM grid setup (same format FlowSOM_Python uses)
grid = [(x, y) for x in range(10) for y in range(10)]
nhbrdist = squareform(pdist(grid, metric="chebyshev"))
radii = (np.quantile(nhbrdist, 0.67), 0.0)
codes = data[np.random.choice(len(data), 100, replace=False)]

# Train
codes, bmu_idx, bmu_dist = flowsom_rs.train_batch_som(
    data, codes, nhbrdist, radii, rlen=10
)
```

## Functions

**`train_batch_som(data, codes, nhbrdist, radii, rlen, n_threads=None)`**
Batch-learning SOM. Parallel BMU search + accumulation. Deterministic.
Returns `(codes, bmu_indices, bmu_distances)`.

**`train_online_som(data, codes, nhbrdist, alphas, radii, rlen, seed)`**
Sequential Kohonen SOM, same algorithm as FlowSOM_Python's `SOMEstimator`.

**`train_replicas_som(data, codes, nhbrdist, alphas, radii, rlen, num_replicas=10, seed=42, n_threads=None)`**
Parallel replicas merged via median, same approach as FlowSOM_Python's `BatchSOMEstimator`.

**`map_data_to_codes(data, codes)`**
Parallel nearest-code assignment. Returns `(indices, distances)`.

## Benchmarks

10×10 grid, 10 epochs, Apple Silicon:

| Events | Numba Online | **Rust Batch** | Speedup |
|--------|-------------|----------------|---------|
| 50K    | 364 ms      | **44 ms**      | 5.9×    |
| 100K   | 491 ms      | **85 ms**      | 6.4×    |
| 500K   | 2,709 ms    | **391 ms**     | 6.9×    |

Thread scaling at 500K events: 1→2→4→8 threads gives 1.0→2.0→3.3→4.6× speedup.

## Build from source

```bash
pip install maturin
maturin develop --release
```

## License

MIT
