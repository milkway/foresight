//! Exact Gaussian likelihood and forecasts of a stationary ARMA process by
//! the innovations algorithm (Brockwell & Davis, *Time Series: Theory and
//! Methods*, §5.2–5.3 and §8.7).

use crate::linalg::solve;

/// X(t) = φ₁X(t−1) + … + φₚX(t−p) + Z(t) + θ₁Z(t−1) + … + θ_qZ(t−q).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Arma {
    pub phi: Vec<f64>,
    pub theta: Vec<f64>,
}

/// Coefficients θ(n, j) and mean squared errors v(n) of the one-step
/// predictors, in units of the innovation variance.
pub(crate) struct Innovations {
    /// `theta[n][j]` = θ(n, j), j = 1..=min(n, width).
    theta: Vec<Vec<f64>>,
    pub v: Vec<f64>,
}

/// What the likelihood needs from one pass over the data.
pub(crate) struct Filtered {
    /// One-step prediction errors x(t) − x̂(t).
    pub errors: Vec<f64>,
    /// Σ errors² / v.
    pub sum_squares: f64,
    /// Σ ln v.
    pub log_det: f64,
}

impl Arma {
    fn p(&self) -> usize {
        self.phi.len()
    }

    fn q(&self) -> usize {
        self.theta.len()
    }

    /// max(p, q).
    fn m(&self) -> usize {
        self.p().max(self.q())
    }

    /// θ with θ₀ = 1 and zeros past q.
    fn theta_at(&self, j: usize) -> f64 {
        match j {
            0 => 1.0,
            j if j <= self.q() => self.theta[j - 1],
            _ => 0.0,
        }
    }

    /// Coefficients ψ₀..ψₙ of the MA(∞) representation.
    pub fn psi(&self, n: usize) -> Vec<f64> {
        let mut psi = vec![0.0; n + 1];
        psi[0] = 1.0;
        for j in 1..=n {
            let ar: f64 = (1..=j.min(self.p()))
                .map(|k| self.phi[k - 1] * psi[j - k])
                .sum();
            psi[j] = self.theta_at(j) + ar;
        }
        psi
    }

    /// Autocovariances γ(0)..γ(max_lag) for unit innovation variance.
    fn acvf(&self, max_lag: usize) -> Option<Vec<f64>> {
        let (p, q) = (self.p(), self.q());
        let psi = self.psi(q);
        // γ(k) − Σ φⱼ γ(|k − j|) = Σ_{j ≥ k} θⱼ ψ(j − k)
        let rhs = |k: usize| -> f64 { (k..=q).map(|j| self.theta_at(j) * psi[j - k]).sum() };
        let mut a = vec![vec![0.0; p + 1]; p + 1];
        let mut b = vec![0.0; p + 1];
        for k in 0..=p {
            a[k][k] += 1.0;
            for j in 1..=p {
                a[k][k.abs_diff(j)] -= self.phi[j - 1];
            }
            b[k] = rhs(k);
        }
        let mut gamma = solve(a, b)?;
        for k in p + 1..=max_lag {
            let ar: f64 = (1..=p).map(|j| self.phi[j - 1] * gamma[k - j]).sum();
            gamma.push(ar + rhs(k));
        }
        gamma.truncate(max_lag + 1);
        Some(gamma)
    }

    /// Innovations coefficients for predictors 1..=`last`.
    pub fn innovations(&self, last: usize) -> Option<Innovations> {
        let (p, q, m) = (self.p(), self.q(), self.m());
        let gamma = self.acvf(2 * m + p)?;
        let ma: Vec<f64> = (0..=q)
            .map(|k| {
                (0..=q - k)
                    .map(|r| self.theta_at(r) * self.theta_at(r + k))
                    .sum()
            })
            .collect();
        // covariance of the transformed process, indices from 1
        let kappa = |i: usize, j: usize| -> f64 {
            let (lo, hi) = (i.min(j), i.max(j));
            let k = hi - lo;
            if hi <= m {
                gamma[k]
            } else if lo <= m {
                if hi <= 2 * m {
                    let ar: f64 = (1..=p)
                        .map(|r| self.phi[r - 1] * gamma[r.abs_diff(k)])
                        .sum();
                    gamma[k] - ar
                } else {
                    0.0
                }
            } else if k <= q {
                ma[k]
            } else {
                0.0
            }
        };
        let mut theta: Vec<Vec<f64>> = Vec::with_capacity(last + 1);
        let mut v = Vec::with_capacity(last + 1);
        theta.push(vec![0.0; 1]);
        v.push(kappa(1, 1));
        if v[0].is_nan() || v[0] <= 0.0 {
            return None;
        }
        for n in 1..=last {
            let first = n.saturating_sub(m);
            let mut row = vec![0.0; n.min(m) + 1];
            for k in first..n {
                let mut s = kappa(n + 1, k + 1);
                for j in first..k {
                    s -= theta[k][k - j] * row[n - j] * v[j];
                }
                row[n - k] = s / v[k];
            }
            let explained: f64 = (first..n).map(|j| row[n - j] * row[n - j] * v[j]).sum();
            let vn = kappa(n + 1, n + 1) - explained;
            if !vn.is_finite() || vn <= 0.0 {
                return None;
            }
            // past max(p, q) the covariances depend only on the lag, so the
            // coefficients settle; once they have, the rest is a copy
            let settled = n > 2 * m
                && (vn - v[n - 1]).abs() < 1e-14
                && row
                    .iter()
                    .zip(&theta[n - 1])
                    .all(|(a, b)| (a - b).abs() < 1e-14);
            theta.push(row);
            v.push(vn);
            if settled {
                for _ in n..last {
                    theta.push(theta[n].clone());
                    v.push(vn);
                }
                break;
            }
        }
        Some(Innovations { theta, v })
    }

    /// One-step predictions of `x` and the pieces of the likelihood.
    pub fn filter(&self, x: &[f64], inn: &Innovations) -> Filtered {
        let (p, q, m) = (self.p(), self.q(), self.m());
        let mut errors = Vec::with_capacity(x.len());
        let (mut sum_squares, mut log_det) = (0.0, 0.0);
        for n in 0..x.len() {
            // predictor of x[n] from x[..n]
            let predicted: f64 = if n < m {
                (1..=n).map(|j| inn.theta[n][j] * errors[n - j]).sum()
            } else {
                let ar: f64 = (1..=p).map(|i| self.phi[i - 1] * x[n - i]).sum();
                let ma: f64 = (1..=q).map(|j| inn.theta[n][j] * errors[n - j]).sum();
                ar + ma
            };
            let e = x[n] - predicted;
            sum_squares += e * e / inn.v[n];
            log_det += inn.v[n].ln();
            errors.push(e);
        }
        Filtered {
            errors,
            sum_squares,
            log_det,
        }
    }

    /// Forecasts of the `h` values after `x`, given its one-step errors.
    /// Needs `x.len() ≥ max(p, q)` and innovations up to `x.len() + h − 1`.
    pub fn forecast(&self, x: &[f64], errors: &[f64], inn: &Innovations, h: usize) -> Vec<f64> {
        let (p, q, n) = (self.p(), self.q(), x.len());
        let mut out: Vec<f64> = Vec::with_capacity(h);
        for k in 1..=h {
            let ar: f64 = (1..=p)
                .map(|i| {
                    let value = if i < k {
                        out[k - 1 - i]
                    } else {
                        x[n + k - 1 - i]
                    };
                    self.phi[i - 1] * value
                })
                .sum();
            let ma: f64 = (k..=q)
                .map(|j| inn.theta[n + k - 1][j] * errors[n + k - 1 - j])
                .sum();
            out.push(ar + ma);
        }
        out
    }
}

/// Coefficients of a stationary autoregression from unconstrained numbers:
/// each goes through tanh to a partial autocorrelation in (−1, 1), and the
/// Durbin-Levinson recursion turns those into coefficients (Jones, 1980).
pub(crate) fn stationary(u: &[f64]) -> Vec<f64> {
    let mut phi: Vec<f64> = Vec::with_capacity(u.len());
    for (k, x) in u.iter().enumerate() {
        let r = x.tanh();
        let previous = phi.clone();
        for j in 0..k {
            phi[j] = previous[j] - r * previous[k - 1 - j];
        }
        phi.push(r);
    }
    phi
}

/// True when 1 − φ₁z − … − φₚzᵖ has every root farther from the origin than
/// `margin` (1 for plain stationarity), by the step-down recursion on the
/// coefficients scaled by marginʲ.
pub(crate) fn roots_outside(phi: &[f64], margin: f64) -> bool {
    let mut c: Vec<f64> = phi
        .iter()
        .enumerate()
        .map(|(j, v)| v * margin.powi(j as i32 + 1))
        .collect();
    while let Some(r) = c.pop() {
        if r.is_nan() || r.abs() >= 1.0 {
            return false;
        }
        let k = c.len();
        let previous = c.clone();
        for j in 0..k {
            c[j] = (previous[j] + r * previous[k - 1 - j]) / (1.0 - r * r);
        }
    }
    true
}

/// Product of the polynomials 1 + a₁z + … and 1 + b₁zˢ + b₂z²ˢ + …, as the
/// coefficients of z, z², … of the result.
pub(crate) fn expand(a: &[f64], b: &[f64], s: usize) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() * s + 1];
    out[0] = 1.0;
    for (i, v) in a.iter().enumerate() {
        out[i + 1] = *v;
    }
    let base = out.clone();
    for (j, w) in b.iter().enumerate() {
        let shift = (j + 1) * s;
        for (i, v) in base.iter().enumerate().take(a.len() + 1) {
            out[i + shift] += v * w;
        }
    }
    out.remove(0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autocovariances_of_known_processes() {
        // AR(1), φ = 0.5: γ(k) = φᵏ / (1 − φ²)
        let ar = Arma {
            phi: vec![0.5],
            theta: vec![],
        };
        let g = ar.acvf(3).unwrap();
        for (k, v) in g.iter().enumerate() {
            assert!((v - 0.5f64.powi(k as i32) / 0.75).abs() < 1e-12);
        }
        // MA(1), θ = 0.4: γ(0) = 1 + θ², γ(1) = θ, γ(2) = 0
        let ma = Arma {
            phi: vec![],
            theta: vec![0.4],
        };
        let g = ma.acvf(2).unwrap();
        assert!((g[0] - 1.16).abs() < 1e-12 && (g[1] - 0.4).abs() < 1e-12 && g[2].abs() < 1e-12);
    }

    #[test]
    fn prediction_errors_of_an_ar1_are_the_residuals() {
        let arma = Arma {
            phi: vec![0.6],
            theta: vec![],
        };
        let x = [1.0, 0.5, -0.2, 0.3, 0.9];
        let inn = arma.innovations(x.len() + 2).unwrap();
        // v(0) is the variance of the process, then the innovation variance
        assert!((inn.v[0] - 1.0 / (1.0 - 0.36)).abs() < 1e-12);
        assert!((inn.v[1] - 1.0).abs() < 1e-12);
        let f = arma.filter(&x, &inn);
        assert!((f.errors[0] - 1.0).abs() < 1e-12);
        for t in 1..x.len() {
            assert!((f.errors[t] - (x[t] - 0.6 * x[t - 1])).abs() < 1e-12);
        }
        let p = arma.forecast(&x, &f.errors, &inn, 2);
        assert!((p[0] - 0.54).abs() < 1e-12 && (p[1] - 0.324).abs() < 1e-12);
    }

    #[test]
    fn ma1_predictors_converge_to_the_invertible_form() {
        let arma = Arma {
            phi: vec![],
            theta: vec![0.5],
        };
        let inn = arma.innovations(60).unwrap();
        assert!((inn.v[0] - 1.25).abs() < 1e-12);
        assert!((inn.v[60] - 1.0).abs() < 1e-9);
        assert!((inn.theta[60][1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn unconstrained_numbers_always_give_stationary_coefficients() {
        for u in [
            vec![3.0, -2.0, 1.5],
            vec![-4.0],
            vec![0.1, 0.2, 0.3, 5.0],
            vec![],
        ] {
            assert!(roots_outside(&stationary(&u), 1.0), "{u:?}");
        }
        assert_eq!(stationary(&[0.0, 0.0]), vec![0.0, 0.0]);
        assert!(!roots_outside(&[1.0], 1.0));
        assert!(!roots_outside(&[0.5, 0.6], 1.0));
        assert!(roots_outside(&[0.98], 1.0) && !roots_outside(&[0.995], 1.01));
    }

    #[test]
    fn seasonal_polynomials_are_multiplied_out() {
        // (1 + 0.5z)(1 + 0.3z⁴) = 1 + 0.5z + 0.3z⁴ + 0.15z⁵
        assert_eq!(expand(&[0.5], &[0.3], 4), vec![0.5, 0.0, 0.0, 0.3, 0.15]);
        assert_eq!(expand(&[], &[], 12), Vec::<f64>::new());
        assert_eq!(expand(&[0.2, 0.1], &[], 12), vec![0.2, 0.1]);
    }
}
