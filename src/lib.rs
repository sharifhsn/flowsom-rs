use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

mod distance;
mod neighborhood;
mod som;

/// Train a batch-learning SOM (BL-FlowSOM, Otsuka et al. 2025).
///
/// Deterministic regardless of input event order. Both BMU search and
/// accumulation are parallelized via rayon with precomputed sparse
/// neighborhoods.
///
/// Parameters
/// ----------
/// data : numpy.ndarray, shape (n_events, n_features), float64
/// codes : numpy.ndarray, shape (n_nodes, n_features), float64
/// nhbrdist : numpy.ndarray, shape (n_nodes, n_nodes), float64
/// radii : tuple[float, float]
///     (sigma_start, sigma_end) for Gaussian neighborhood decay.
/// rlen : int
///     Number of training epochs.
/// n_threads : int or None
///     Rayon thread count. None = all cores.
///
/// Returns
/// -------
/// (codes, bmu_indices, bmu_distances)
#[pyfunction]
#[pyo3(signature = (data, codes, nhbrdist, radii, rlen, n_threads=None))]
fn train_batch_som<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'_, f64>,
    codes: PyReadonlyArray2<'_, f64>,
    nhbrdist: PyReadonlyArray2<'_, f64>,
    radii: (f64, f64),
    rlen: usize,
    n_threads: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let data_arr = data.as_array().to_owned();
    let mut codes_arr = codes.as_array().to_owned();
    let nhbrdist_arr = nhbrdist.as_array().to_owned();
    let (sigma_start, sigma_end) = radii;

    let pool = n_threads
        .map(|t| rayon::ThreadPoolBuilder::new().num_threads(t).build())
        .transpose()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let (bmu_indices, bmu_distances) = py.detach(|| {
        if let Some(ref pool) = pool {
            pool.install(|| {
                som::train_batch_som(
                    &data_arr.view(),
                    &mut codes_arr,
                    &nhbrdist_arr.view(),
                    sigma_start,
                    sigma_end,
                    rlen,
                )
            })
        } else {
            som::train_batch_som(
                &data_arr.view(),
                &mut codes_arr,
                &nhbrdist_arr.view(),
                sigma_start,
                sigma_end,
                rlen,
            )
        }
    });

    Ok((
        codes_arr.into_pyarray(py),
        Array1::from(bmu_indices).into_pyarray(py),
        Array1::from(bmu_distances).into_pyarray(py),
    ))
}

/// Train an online SOM (sequential Kohonen algorithm in Rust).
///
/// Parameters
/// ----------
/// data : numpy.ndarray, shape (n_events, n_features), float64
/// codes : numpy.ndarray, shape (n_nodes, n_features), float64
/// nhbrdist : numpy.ndarray, shape (n_nodes, n_nodes), float64
/// alphas : tuple[float, float]
///     (alpha_start, alpha_end) for learning rate decay.
/// radii : tuple[float, float]
///     (radius_start, radius_end) for neighborhood threshold decay.
/// rlen : int
///     Number of passes over the data.
/// seed : int
///     Random seed for event sampling.
///
/// Returns
/// -------
/// (codes, bmu_indices, bmu_distances)
#[pyfunction]
#[pyo3(signature = (data, codes, nhbrdist, alphas, radii, rlen, seed))]
fn train_online_som<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'_, f64>,
    codes: PyReadonlyArray2<'_, f64>,
    nhbrdist: PyReadonlyArray2<'_, f64>,
    alphas: (f64, f64),
    radii: (f64, f64),
    rlen: usize,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let data_arr = data.as_array().to_owned();
    let mut codes_arr = codes.as_array().to_owned();
    let nhbrdist_arr = nhbrdist.as_array().to_owned();

    let (bmu_indices, bmu_distances) = py.detach(|| {
        som::train_online_som(
            &data_arr.view(),
            &mut codes_arr,
            &nhbrdist_arr.view(),
            alphas.0,
            alphas.1,
            radii.0,
            radii.1,
            rlen,
            seed,
        )
    });

    Ok((
        codes_arr.into_pyarray(py),
        Array1::from(bmu_indices).into_pyarray(py),
        Array1::from(bmu_distances).into_pyarray(py),
    ))
}

/// Train using parallel replicas: split data, run independent online SOMs
/// in parallel via rayon, merge via median. Same approach as FlowSOM_Python's
/// BatchSOMEstimator but with proper multi-threading.
///
/// Parameters
/// ----------
/// data : numpy.ndarray, shape (n_events, n_features), float64
/// codes : numpy.ndarray, shape (n_nodes, n_features), float64
/// nhbrdist : numpy.ndarray, shape (n_nodes, n_nodes), float64
/// alphas : tuple[float, float]
/// radii : tuple[float, float]
/// rlen : int
/// num_replicas : int
///     Number of parallel replicas (default: 10).
/// seed : int
/// n_threads : int or None
///
/// Returns
/// -------
/// (codes, bmu_indices, bmu_distances)
#[pyfunction]
#[pyo3(signature = (data, codes, nhbrdist, alphas, radii, rlen, num_replicas=10, seed=42, n_threads=None))]
fn train_replicas_som<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'_, f64>,
    codes: PyReadonlyArray2<'_, f64>,
    nhbrdist: PyReadonlyArray2<'_, f64>,
    alphas: (f64, f64),
    radii: (f64, f64),
    rlen: usize,
    num_replicas: usize,
    seed: u64,
    n_threads: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let data_arr = data.as_array().to_owned();
    let mut codes_arr = codes.as_array().to_owned();
    let nhbrdist_arr = nhbrdist.as_array().to_owned();

    let pool = n_threads
        .map(|t| rayon::ThreadPoolBuilder::new().num_threads(t).build())
        .transpose()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let (bmu_indices, bmu_distances) = py.detach(|| {
        if let Some(ref pool) = pool {
            pool.install(|| {
                som::train_replicas_som(
                    &data_arr.view(),
                    &mut codes_arr,
                    &nhbrdist_arr.view(),
                    alphas.0,
                    alphas.1,
                    radii.0,
                    radii.1,
                    rlen,
                    num_replicas,
                    seed,
                )
            })
        } else {
            som::train_replicas_som(
                &data_arr.view(),
                &mut codes_arr,
                &nhbrdist_arr.view(),
                alphas.0,
                alphas.1,
                radii.0,
                radii.1,
                rlen,
                num_replicas,
                seed,
            )
        }
    });

    Ok((
        codes_arr.into_pyarray(py),
        Array1::from(bmu_indices).into_pyarray(py),
        Array1::from(bmu_distances).into_pyarray(py),
    ))
}

/// Map data points to nearest code (BMU). Parallel via rayon.
#[pyfunction]
#[pyo3(signature = (data, codes))]
fn map_data_to_codes<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'_, f64>,
    codes: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f64>>)> {
    let data_arr = data.as_array().to_owned();
    let codes_arr = codes.as_array().to_owned();

    let (indices, distances) =
        py.detach(|| som::map_data_to_codes(&data_arr.view(), &codes_arr.view()));

    Ok((
        Array1::from(indices).into_pyarray(py),
        Array1::from(distances).into_pyarray(py),
    ))
}

#[pymodule]
fn flowsom_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_batch_som, m)?)?;
    m.add_function(wrap_pyfunction!(train_online_som, m)?)?;
    m.add_function(wrap_pyfunction!(train_replicas_som, m)?)?;
    m.add_function(wrap_pyfunction!(map_data_to_codes, m)?)?;
    Ok(())
}
