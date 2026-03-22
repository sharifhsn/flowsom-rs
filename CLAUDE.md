# flowsom-rs

PyO3 extension module implementing batch-learning SOM (Otsuka et al. 2025) with rayon parallelism, for use with [FlowSOM_Python](https://github.com/saeyslab/FlowSOM_Python).

## Build

```bash
uv venv --python 3.12 .venv
uv pip install maturin numpy scipy
.venv/bin/maturin develop --release    # build + install into venv
cargo test                              # run Rust tests
```

## Project layout

```
src/
  lib.rs             # PyO3 module: train_batch_som, train_online_som, train_replicas_som, map_data_to_codes
  som.rs             # All three SOM algorithms + map_data_to_codes
  distance.rs        # squared_euclidean
  neighborhood.rs    # gaussian neighborhood function
```

## Key dependencies

- `pyo3 0.28` — Python bindings. Uses `Bound<'py, T>` API (not the removed GIL refs). GIL release is `py.detach(|| { ... })`, NOT `py.allow_threads`.
- `numpy 0.28` — PyO3 numpy interop. Accept arrays as `PyReadonlyArray2<'_, f64>`, return as `Bound<'py, PyArray2<f64>>` via `IntoPyArray::into_pyarray`.
- `ndarray 0.17` with `rayon` feature — `axis_iter(Axis(0)).into_par_iter()` for parallel row iteration.
- `rayon 1.11` — work-stealing parallelism. Custom thread pools via `ThreadPoolBuilder::new().num_threads(n).build()`.
- `rand 0.9` + `rand_chacha 0.9` — deterministic RNG for the online SOM. `rng.random_range(0..n)` (not `gen_range`).

## Three SOM backends

1. **`train_batch_som`** — True BL-FlowSOM. Deterministic regardless of input order. Precomputes sparse neighborhood weights per epoch to avoid redundant `exp()` calls. Both BMU search and accumulation parallelized via rayon fold+reduce.

2. **`train_online_som`** — Sequential Kohonen SOM matching FlowSOM_Python's `SOMEstimator`. Uses hard threshold on Chebyshev distance (not Gaussian). NOT order-invariant.

3. **`train_replicas_som`** — Parallel replicas: split data into batches, run independent online SOMs, merge via element-wise median. Matches FlowSOM_Python's `BatchSOMEstimator` approach.

## PyPI publishing

Published at https://pypi.org/project/flowsom-rs/

### Release process

1. Bump version in both `Cargo.toml` and `pyproject.toml`
2. Commit, tag (`git tag v0.x.y`), push tag
3. Create GitHub release — triggers `.github/workflows/release.yml`
4. Workflow: `cargo test` → build wheels (Linux x86_64, macOS x86_64/aarch64, Windows x86_64) + sdist → `uv publish` with API token

### Infrastructure

- **GitHub secret**: `PYPI_API_TOKEN` on `sharifhsn/flowsom-rs`
- **GitHub environment**: `pypi` (required by the publish job)
- **Wheel builder**: `PyO3/maturin-action@v1`
- **Publisher**: `uv publish` via `astral-sh/setup-uv@v6`
- **No Linux aarch64 wheel** — cross-compilation fails in the manylinux QEMU container (Python interpreter not found). Those users build from the sdist.

## Relationship to FlowSOM_Python

This is a standalone package. The eventual plan is to integrate it into FlowSOM_Python as an optional backend (`RustSOMEstimator`) following the existing `PyFlowSOM_SOMEstimator` pattern — lazy import in `fit()`, same constructor signature, same output attributes. That integration hasn't been done yet; the package needs stress testing first.

A numpy optimization PR was submitted separately: https://github.com/saeyslab/FlowSOM_Python/pull/28

## Benchmarks (Apple Silicon, 10x10 grid, 10 epochs)

| Events | Numba Online | Rust Batch | Speedup |
|--------|-------------|------------|---------|
| 50K    | 364 ms      | 44 ms      | 5.9x   |
| 100K   | 491 ms      | 85 ms      | 6.4x   |
| 500K   | 2,709 ms    | 391 ms     | 6.9x   |
