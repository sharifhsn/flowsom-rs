/// Gaussian neighborhood coefficient.
///
/// Given the distance between two nodes on the SOM grid (from the
/// precomputed `nhbrdist` matrix), returns the Gaussian neighborhood
/// weight: `h = exp(-dist^2 / (2 * sigma^2))`.
///
/// When sigma is near zero, returns 1.0 for the node itself (dist ≈ 0)
/// and 0.0 for all others, avoiding division by zero.
#[inline]
pub fn gaussian(dist: f64, sigma: f64) -> f64 {
    const EPSILON: f64 = 1e-10;
    if sigma < EPSILON {
        if dist < EPSILON {
            1.0
        } else {
            0.0
        }
    } else {
        (-dist * dist / (2.0 * sigma * sigma)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_self() {
        // Distance 0 should give h = 1.0
        assert!((gaussian(0.0, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_decay() {
        // Farther nodes should have smaller h
        let h_near = gaussian(1.0, 2.0);
        let h_far = gaussian(3.0, 2.0);
        assert!(h_near > h_far);
        assert!(h_near > 0.0);
        assert!(h_far > 0.0);
    }

    #[test]
    fn test_gaussian_zero_sigma() {
        // When sigma ≈ 0, only self gets weight
        assert!((gaussian(0.0, 0.0) - 1.0).abs() < 1e-10);
        assert!((gaussian(1.0, 0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_known_value() {
        // h = exp(-1^2 / (2 * 1^2)) = exp(-0.5) ≈ 0.6065
        let h = gaussian(1.0, 1.0);
        assert!((h - (-0.5_f64).exp()).abs() < 1e-10);
    }
}
