//! Backtest of two public revenue series of the state of Piauí, Brazil.
//!
//! ```text
//! cargo run -p foresight --example piaui
//! ```

use foresight::{models, Backtest, Series};

fn main() {
    let data = include_str!("../data/piaui_revenue.csv");
    let rows: Vec<Vec<f64>> = data
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("month"))
        .map(|l| l.split(',').skip(1).map(|v| v.parse().unwrap()).collect())
        .collect();

    for (column, name) in [(0, "ICMS"), (1, "FPE")] {
        let values: Vec<f64> = rows.iter().map(|r| r[column]).collect();
        // the first observation is March 2017
        let y = Series::monthly(&values, 2);
        let report = Backtest::default()
            .run(y, &models::thorough())
            .expect("series long enough");

        println!(
            "\n{name}: {} origins, {} months ahead",
            report.origins, report.horizon
        );
        let mut ranking: Vec<_> = report.candidates.iter().collect();
        ranking.sort_by(|a, b| a.score.total_cmp(&b.score));
        for c in ranking {
            let h = |i: usize| c.horizons[i].mape.unwrap_or(f64::NAN);
            println!(
                "  {:<45} MAPE {:5.2}%   h=1 {:5.2}%   h=12 {:5.2}%",
                c.name,
                c.score,
                h(0),
                h(11)
            );
        }
        let best = report.chosen();
        let half = best.cumulative(6).expect("six months");
        let i = half.interval(0.80).expect("80% interval");
        println!(
            "  next 6 months with {}: {:.2} bi (80%: {:.2} to {:.2})",
            best.name,
            half.mean / 1e9,
            i.lower / 1e9,
            i.upper / 1e9
        );
    }
}
