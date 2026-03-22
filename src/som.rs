use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::distance::squared_euclidean;
use crate::neighborhood::gaussian;

// ---------------------------------------------------------------------------
// Precomputed sparse neighborhood — shared by batch SOM
// ---------------------------------------------------------------------------

/// For each potential BMU, store the list of (node_index, h_weight) pairs
/// where h > threshold. Precomputed once per epoch to avoid millions of
/// redundant exp() calls.
type NeighborList = Vec<Vec<(usize, f64)>>;

fn precompute_neighbors(nhbrdist: &ArrayView2<f64>, sigma: f64) -> NeighborList {
    let n_nodes = nhbrdist.nrows();
    (0..n_nodes)
        .map(|bmu| {
            (0..n_nodes)
                .filter_map(|j| {
                    let h = gaussian(nhbrdist[[bmu, j]], sigma);
                    if h > 1e-6 {
                        Some((j, h))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Batch-learning SOM (BL-FlowSOM, Otsuka et al. 2025)
// ---------------------------------------------------------------------------

/// Train a batch-learning SOM. Deterministic regardless of input order.
///
/// Both BMU search (Phase 1) and accumulation (Phase 2) are parallelized.
/// Neighborhood weights are precomputed once per epoch (sparse), avoiding
/// redundant exp() calls across events.
pub fn train_batch_som(
    data: &ArrayView2<f64>,
    codes: &mut Array2<f64>,
    nhbrdist: &ArrayView2<f64>,
    sigma_start: f64,
    sigma_end: f64,
    epochs: usize,
) -> (Vec<usize>, Vec<f64>) {
    let n_nodes = codes.nrows();
    let n_features = codes.ncols();

    if epochs == 0 {
        return map_data_to_codes(data, &codes.view());
    }

    let mut last_bmus: Vec<(usize, f64)> = Vec::new();

    for epoch in 0..epochs {
        let sigma = if epochs > 1 {
            sigma_start + (sigma_end - sigma_start) * (epoch as f64) / ((epochs - 1) as f64)
        } else {
            sigma_start
        };

        // Precompute sparse neighbor list for this epoch's sigma.
        // O(n_nodes^2) exp() calls — vs O(n_events * n_nodes) without this.
        let neighbors = precompute_neighbors(nhbrdist, sigma);

        // Phase 1: Parallel BMU assignment
        let bmus: Vec<(usize, f64)> = data
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|event| find_bmu(&event, codes, n_nodes))
            .collect();

        // Phase 2: Parallel accumulation via fold + reduce
        let (numerator, denominator) = bmus
            .par_iter()
            .enumerate()
            .fold(
                || {
                    (
                        Array2::<f64>::zeros((n_nodes, n_features)),
                        Array1::<f64>::zeros(n_nodes),
                    )
                },
                |(mut num, mut den), (i, &(bmu, _))| {
                    let event = data.row(i);
                    // Only iterate over nodes with meaningful neighborhood weight
                    for &(j, h) in &neighbors[bmu] {
                        num.row_mut(j).scaled_add(h, &event);
                        den[j] += h;
                    }
                    (num, den)
                },
            )
            .reduce(
                || {
                    (
                        Array2::<f64>::zeros((n_nodes, n_features)),
                        Array1::<f64>::zeros(n_nodes),
                    )
                },
                |(mut a_num, mut a_den), (b_num, b_den)| {
                    a_num += &b_num;
                    a_den += &b_den;
                    (a_num, a_den)
                },
            );

        // Phase 3: Update weights
        for j in 0..n_nodes {
            if denominator[j] > 0.0 {
                let inv = 1.0 / denominator[j];
                for f in 0..n_features {
                    codes[[j, f]] = numerator[[j, f]] * inv;
                }
            }
        }

        last_bmus = bmus;
    }

    // Return BMU info from final epoch with euclidean distances
    let mut bmu_indices = Vec::with_capacity(last_bmus.len());
    let mut bmu_distances = Vec::with_capacity(last_bmus.len());
    for (idx, sq_dist) in last_bmus {
        bmu_indices.push(idx);
        bmu_distances.push(sq_dist.sqrt());
    }

    (bmu_indices, bmu_distances)
}

// ---------------------------------------------------------------------------
// Online SOM (sequential, matches FlowSOM_Python's algorithm)
// ---------------------------------------------------------------------------

/// Train a sequential online SOM. This is the classic Kohonen algorithm
/// where each event updates the weights immediately.
///
/// NOT order-invariant — results depend on the random event sampling order.
fn online_som_inner(
    data: &ArrayView2<f64>,
    codes: &mut Array2<f64>,
    nhbrdist: &ArrayView2<f64>,
    alpha_start: f64,
    alpha_end: f64,
    radius_start: f64,
    radius_end: f64,
    rlen: usize,
    rng: &mut ChaCha8Rng,
) {
    let n = data.nrows();
    let n_nodes = codes.nrows();
    let n_features = codes.ncols();
    let niter = rlen * n;

    let mut threshold = radius_start;
    let threshold_step = (radius_start - radius_end) / niter as f64;

    for k in 0..niter {
        let i = rng.random_range(0..n);

        // Find BMU
        let mut best_dist = f64::MAX;
        let mut best_node = 0;
        for j in 0..n_nodes {
            let dist = squared_euclidean(&data.row(i), &codes.row(j));
            if dist < best_dist {
                best_dist = dist;
                best_node = j;
            }
        }

        // Update neighborhood (hard threshold, matching Python)
        let alpha = alpha_start - (alpha_start - alpha_end) * k as f64 / niter as f64;
        let effective_threshold = if threshold < 1.0 { 0.5 } else { threshold };

        for j in 0..n_nodes {
            if nhbrdist[[j, best_node]] > effective_threshold {
                continue;
            }
            for f in 0..n_features {
                let diff = data[[i, f]] - codes[[j, f]];
                codes[[j, f]] += diff * alpha;
            }
        }

        threshold -= threshold_step;
    }
}

/// Train an online SOM (single-threaded, exposed for benchmarking).
pub fn train_online_som(
    data: &ArrayView2<f64>,
    codes: &mut Array2<f64>,
    nhbrdist: &ArrayView2<f64>,
    alpha_start: f64,
    alpha_end: f64,
    radius_start: f64,
    radius_end: f64,
    rlen: usize,
    seed: u64,
) -> (Vec<usize>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    online_som_inner(
        data,
        codes,
        nhbrdist,
        alpha_start,
        alpha_end,
        radius_start,
        radius_end,
        rlen,
        &mut rng,
    );
    map_data_to_codes(data, &codes.view())
}

// ---------------------------------------------------------------------------
// Parallel-replicas SOM (the Numba batch approach, but with rayon)
// ---------------------------------------------------------------------------

/// Compute element-wise median across N code matrices.
fn median_codes(batch_codes: &[Array2<f64>]) -> Array2<f64> {
    let num = batch_codes.len();
    let n_nodes = batch_codes[0].nrows();
    let n_features = batch_codes[0].ncols();
    let mut result = Array2::zeros((n_nodes, n_features));
    let mut values = vec![0.0f64; num];

    for i in 0..n_nodes {
        for j in 0..n_features {
            for (k, codes) in batch_codes.iter().enumerate() {
                values[k] = codes[[i, j]];
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            result[[i, j]] = if num % 2 == 0 {
                (values[num / 2 - 1] + values[num / 2]) / 2.0
            } else {
                values[num / 2]
            };
        }
    }
    result
}

/// Train using parallel replicas: split data into batches, run independent
/// online SOMs in parallel via rayon, merge via element-wise median.
///
/// This mirrors FlowSOM_Python's `SOM_Batch` approach but with proper
/// multi-threading (no GIL contention). Faster than true batch SOM but
/// NOT input-order invariant.
pub fn train_replicas_som(
    data: &ArrayView2<f64>,
    codes: &mut Array2<f64>,
    nhbrdist: &ArrayView2<f64>,
    alpha_start: f64,
    alpha_end: f64,
    radius_start: f64,
    radius_end: f64,
    rlen: usize,
    num_replicas: usize,
    seed: u64,
) -> (Vec<usize>, Vec<f64>) {
    let n = data.nrows();

    // Split data into interleaved batches (matching Numba approach)
    let batches: Vec<Array2<f64>> = (0..num_replicas)
        .map(|r| {
            let rows: Vec<usize> = (r..n).step_by(num_replicas).collect();
            let n_rows = rows.len();
            let n_features = data.ncols();
            let mut batch = Array2::zeros((n_rows, n_features));
            for (out_i, &data_i) in rows.iter().enumerate() {
                batch.row_mut(out_i).assign(&data.row(data_i));
            }
            batch
        })
        .collect();

    // Serial init phase: run one pass of online SOM on full data
    // (matches the Numba code's Phase 0 initialization)
    let mut rng_init = ChaCha8Rng::seed_from_u64(seed);
    let init_rlen = 1; // one pass for initialization
    online_som_inner(
        data,
        codes,
        nhbrdist,
        alpha_start,
        alpha_end,
        radius_start,
        radius_end,
        init_rlen,
        &mut rng_init,
    );

    // Remaining epochs: parallel replicas + median merge
    let remaining_rlen = if rlen > 1 { rlen / 2 } else { 1 };

    for epoch in 0..remaining_rlen {
        // Decay radius and alpha across epochs
        let epoch_frac = epoch as f64 / remaining_rlen as f64;
        let r_start = radius_start * (1.0 - epoch_frac);
        let a_start = alpha_start - (alpha_start - alpha_end) * epoch_frac;

        // Run independent online SOMs in parallel
        let batch_codes: Vec<Array2<f64>> = (0..num_replicas)
            .into_par_iter()
            .map(|r| {
                let mut local_codes = codes.clone();
                let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(epoch as u64 * 1000 + r as u64));
                online_som_inner(
                    &batches[r].view(),
                    &mut local_codes,
                    nhbrdist,
                    a_start,
                    alpha_end,
                    r_start,
                    radius_end,
                    1, // one pass per replica per epoch
                    &mut rng,
                );
                local_codes
            })
            .collect();

        // Merge via element-wise median
        *codes = median_codes(&batch_codes);
    }

    map_data_to_codes(data, &codes.view())
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Find the best matching unit for a single event.
#[inline]
fn find_bmu(
    event: &ndarray::ArrayView1<f64>,
    codes: &Array2<f64>,
    n_nodes: usize,
) -> (usize, f64) {
    let mut best_dist = f64::MAX;
    let mut best_node = 0;
    for j in 0..n_nodes {
        let node = codes.row(j);
        let dist = squared_euclidean(event, &node);
        if dist < best_dist {
            best_dist = dist;
            best_node = j;
        }
    }
    (best_node, best_dist)
}

/// Map data points to their nearest code (BMU). Parallel via rayon.
pub fn map_data_to_codes(
    data: &ArrayView2<f64>,
    codes: &ArrayView2<f64>,
) -> (Vec<usize>, Vec<f64>) {
    let n_nodes = codes.nrows();

    data.axis_iter(Axis(0))
        .into_par_iter()
        .map(|event| {
            let (node, sq_dist) = find_bmu(&event, &codes.into_owned(), n_nodes);
            (node, sq_dist.sqrt())
        })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn simple_grid_nhbrdist(xdim: usize, ydim: usize) -> Array2<f64> {
        let n = xdim * ydim;
        let mut dist = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let (ix, iy) = (i % xdim, i / xdim);
            for j in 0..n {
                let (jx, jy) = (j % xdim, j / xdim);
                let dx = (ix as f64 - jx as f64).abs();
                let dy = (iy as f64 - jy as f64).abs();
                dist[[i, j]] = dx.max(dy);
            }
        }
        dist
    }

    #[test]
    fn test_map_data_to_codes() {
        let codes = array![[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]];
        let data = array![[1.0, 1.0], [9.0, 9.0], [21.0, 21.0], [0.0, 0.0]];
        let (indices, distances) = map_data_to_codes(&data.view(), &codes.view());
        assert_eq!(indices, vec![0, 1, 2, 0]);
        assert!(distances[0] > 0.0);
        assert!(distances[3] < 1e-10);
    }

    #[test]
    fn test_batch_som_converges() {
        let mut data_vec = Vec::new();
        for _ in 0..100 {
            data_vec.extend_from_slice(&[0.0, 0.0]);
        }
        for _ in 0..100 {
            data_vec.extend_from_slice(&[10.0, 10.0]);
        }
        let data = Array2::from_shape_vec((200, 2), data_vec).unwrap();
        let mut codes = array![[5.0, 5.0], [5.1, 5.1]];
        let nhbrdist = simple_grid_nhbrdist(2, 1);

        let (bmu_indices, _) =
            train_batch_som(&data.view(), &mut codes, &nhbrdist.view(), 1.0, 0.1, 20);

        let c0 = codes.row(0);
        let c1 = codes.row(1);
        let near_zero =
            (c0[0].abs() < 2.0 && c0[1].abs() < 2.0) || (c1[0].abs() < 2.0 && c1[1].abs() < 2.0);
        let near_ten = ((c0[0] - 10.0).abs() < 2.0 && (c0[1] - 10.0).abs() < 2.0)
            || ((c1[0] - 10.0).abs() < 2.0 && (c1[1] - 10.0).abs() < 2.0);
        assert!(near_zero, "Expected code near [0,0], got {:?}", codes);
        assert!(near_ten, "Expected code near [10,10], got {:?}", codes);
        assert_eq!(bmu_indices.len(), 200);
    }

    #[test]
    fn test_batch_som_deterministic() {
        let data = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0],];
        let nhbrdist = simple_grid_nhbrdist(2, 1);
        let mut codes1 = array![[2.0, 2.0], [8.0, 8.0]];
        let mut codes2 = codes1.clone();

        let (idx1, dist1) =
            train_batch_som(&data.view(), &mut codes1, &nhbrdist.view(), 1.0, 0.1, 10);
        let (idx2, dist2) =
            train_batch_som(&data.view(), &mut codes2, &nhbrdist.view(), 1.0, 0.1, 10);

        assert_eq!(codes1, codes2);
        assert_eq!(idx1, idx2);
        for (d1, d2) in dist1.iter().zip(dist2.iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_batch_som_order_invariant() {
        let data_ordered = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0],];
        let data_shuffled = array![[10.0, 10.0], [0.0, 0.0], [11.0, 11.0], [1.0, 1.0],];
        let nhbrdist = simple_grid_nhbrdist(2, 1);
        let init_codes = array![[2.0, 2.0], [8.0, 8.0]];

        let mut codes1 = init_codes.clone();
        let mut codes2 = init_codes.clone();
        train_batch_som(&data_ordered.view(), &mut codes1, &nhbrdist.view(), 1.0, 0.1, 10);
        train_batch_som(
            &data_shuffled.view(),
            &mut codes2,
            &nhbrdist.view(),
            1.0,
            0.1,
            10,
        );

        for i in 0..codes1.nrows() {
            for j in 0..codes1.ncols() {
                assert!(
                    (codes1[[i, j]] - codes2[[i, j]]).abs() < 1e-10,
                    "Codes differ at [{},{}]: {} vs {}",
                    i, j, codes1[[i, j]], codes2[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_online_som_converges() {
        let mut data_vec = Vec::new();
        for _ in 0..100 {
            data_vec.extend_from_slice(&[0.0, 0.0]);
        }
        for _ in 0..100 {
            data_vec.extend_from_slice(&[10.0, 10.0]);
        }
        let data = Array2::from_shape_vec((200, 2), data_vec).unwrap();
        let mut codes = array![[5.0, 5.0], [5.1, 5.1]];
        let nhbrdist = simple_grid_nhbrdist(2, 1);

        let (bmu_indices, _) = train_online_som(
            &data.view(),
            &mut codes,
            &nhbrdist.view(),
            0.05, 0.01, 1.0, 0.0, 10, 42,
        );

        assert_eq!(bmu_indices.len(), 200);
        // Should separate the two clusters
        let c0 = codes.row(0);
        let c1 = codes.row(1);
        let dist_between = ((c0[0] - c1[0]).powi(2) + (c0[1] - c1[1]).powi(2)).sqrt();
        assert!(dist_between > 5.0, "Codes should separate, distance: {}", dist_between);
    }

    #[test]
    fn test_replicas_som_converges() {
        let mut data_vec = Vec::new();
        for _ in 0..100 {
            data_vec.extend_from_slice(&[0.0, 0.0]);
        }
        for _ in 0..100 {
            data_vec.extend_from_slice(&[10.0, 10.0]);
        }
        let data = Array2::from_shape_vec((200, 2), data_vec).unwrap();
        let mut codes = array![[5.0, 5.0], [5.1, 5.1]];
        let nhbrdist = simple_grid_nhbrdist(2, 1);

        let (bmu_indices, _) = train_replicas_som(
            &data.view(),
            &mut codes,
            &nhbrdist.view(),
            0.05, 0.01, 1.0, 0.0, 10, 4, 42,
        );

        assert_eq!(bmu_indices.len(), 200);
        let c0 = codes.row(0);
        let c1 = codes.row(1);
        let dist_between = ((c0[0] - c1[0]).powi(2) + (c0[1] - c1[1]).powi(2)).sqrt();
        assert!(dist_between > 3.0, "Codes should separate, distance: {}", dist_between);
    }

    #[test]
    fn test_precompute_neighbors_sparsity() {
        let nhbrdist = simple_grid_nhbrdist(10, 10);
        // Large sigma: most nodes are neighbors
        let neighbors_wide = precompute_neighbors(&nhbrdist.view(), 5.0);
        // Small sigma: few neighbors
        let neighbors_narrow = precompute_neighbors(&nhbrdist.view(), 0.5);

        let avg_wide: f64 =
            neighbors_wide.iter().map(|n| n.len() as f64).sum::<f64>() / 100.0;
        let avg_narrow: f64 =
            neighbors_narrow.iter().map(|n| n.len() as f64).sum::<f64>() / 100.0;

        assert!(
            avg_wide > avg_narrow,
            "Wide sigma should have more neighbors: {} vs {}",
            avg_wide, avg_narrow
        );
        assert!(avg_narrow < 20.0, "Narrow sigma should be sparse: {}", avg_narrow);
    }
}
