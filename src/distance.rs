use ndarray::ArrayView1;

/// Squared Euclidean distance between two 1D array views.
/// Avoids sqrt for performance during BMU search.
#[inline]
pub fn squared_euclidean(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_squared_euclidean() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];
        let dist = squared_euclidean(&a.view(), &b.view());
        assert!((dist - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_squared_euclidean_same() {
        let a = array![1.0, 2.0, 3.0];
        let dist = squared_euclidean(&a.view(), &a.view());
        assert!((dist - 0.0).abs() < 1e-10);
    }
}
