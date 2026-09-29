//! Synthetic series shared by the unit tests.

const SEASONAL: [f64; 12] = [
    1.05, 0.80, 0.95, 1.00, 0.98, 0.97, 1.02, 1.00, 0.96, 1.01, 0.96, 1.30,
];

/// Level 100 growing 0.5% a period, known multiplicative seasonality
/// (December +30%, February −20%…), no noise.
pub(crate) fn seasonal_growth(n: usize) -> Vec<f64> {
    noisy_seasonal_growth(n, 0.0)
}

/// The same with deterministic multiplicative noise (a linear congruential
/// generator), uniform in ±`noise`/2.
pub(crate) fn noisy_seasonal_growth(n: usize, noise: f64) -> Vec<f64> {
    let mut seed: u64 = 42;
    (0..n)
        .map(|t| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            100.0 * 1.005f64.powi(t as i32) * SEASONAL[t % 12] * (1.0 + noise * u)
        })
        .collect()
}
