//! Forecasts compared with the R package `forecast` 9.0.2 on public data.
//! The reference values come from `tests/r/reference.R`.

use foresight::models::{Drift, SeasonalNaive, Theta};
use foresight::{Model, Series};

fn column(i: usize) -> Vec<f64> {
    include_str!("../data/piaui_revenue.csv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("month"))
        .map(|l| l.split(',').nth(i + 1).unwrap().parse().unwrap())
        .collect()
}

fn close(name: &str, ours: &[f64], reference: &[f64], tolerance: f64) {
    assert_eq!(ours.len(), reference.len());
    for (h, (a, b)) in ours.iter().zip(reference).enumerate() {
        assert!(
            (a / b - 1.0).abs() < tolerance,
            "{name} h={}: {a} × {b}",
            h + 1
        );
    }
}

struct Reference {
    theta: [f64; 12],
    snaive: [f64; 12],
    drift: [f64; 12],
}

const ICMS: Reference = Reference {
    theta: [
        821180569.452,
        826589211.539,
        825297914.497,
        854821247.833,
        842379893.905,
        875039597.321,
        875547670.792,
        758722396.695,
        705075455.027,
        771713771.372,
        729860110.323,
        812225243.518,
    ],
    snaive: [
        725046345.29,
        773297361.32,
        747978074.80,
        751243471.44,
        771989639.97,
        767244468.11,
        791797624.93,
        720099765.32,
        679758370.35,
        774001802.81,
        704801950.30,
        785725856.19,
    ],
    drift: [
        790441292.432,
        795156728.674,
        799872164.916,
        804587601.158,
        809303037.399,
        814018473.641,
        818733909.883,
        823449346.125,
        828164782.367,
        832880218.609,
        837595654.851,
        842311091.093,
    ],
};

const FPE: Reference = Reference {
    theta: [
        673005978.597,
        807295251.993,
        658367288.813,
        717159016.694,
        905840242.167,
        1021426816.045,
        964076441.901,
        1267218459.982,
        813672292.609,
        853565729.946,
        997969347.093,
        911315990.981,
    ],
    snaive: [
        521384984.85,
        685235336.58,
        553829994.03,
        585488783.66,
        790424639.46,
        892435659.23,
        829385876.87,
        1044022025.38,
        631613479.02,
        744050636.75,
        900471019.35,
        949640302.02,
    ],
    drift: [
        955919073.134,
        962197844.247,
        968476615.361,
        974755386.475,
        981034157.588,
        987312928.702,
        993591699.816,
        999870470.930,
        1006149242.043,
        1012428013.157,
        1018706784.271,
        1024985555.384,
    ],
};

#[test]
fn benchmarks_match_r_exactly() {
    for (i, name, r) in [(0, "icms", &ICMS), (1, "fpe", &FPE)] {
        let v = column(i);
        let y = Series::monthly(&v, 2);
        let snaive = SeasonalNaive::new().forecast(y, 12).unwrap();
        close(&format!("{name} snaive"), &snaive, &r.snaive, 1e-11);
        let drift = Drift.forecast(y, 12).unwrap();
        close(&format!("{name} drift"), &drift, &r.drift, 1e-11);
    }
}

/// R estimates the smoothing parameter and the initial level with a general
/// optimiser and we use a closed form plus a line search, so the optimum is
/// the same only up to the optimiser's precision.
#[test]
fn theta_matches_r_within_the_precision_of_the_optimiser() {
    for (i, name, r) in [(0, "icms", &ICMS), (1, "fpe", &FPE)] {
        let v = column(i);
        let theta = Theta.forecast(Series::monthly(&v, 2), 12).unwrap();
        close(&format!("{name} theta"), &theta, &r.theta, 1e-3);
    }
}

// ---------------------------------------------------------------- ARIMA

fn air_passengers() -> Vec<f64> {
    include_str!("../data/air_passengers.csv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("month"))
        .map(|l| l.split(',').nth(1).unwrap().parse().unwrap())
        .collect()
}

/// `Arima(y, c(0,1,1), c(0,1,1), lambda = 0, method = "ML")` and
/// `Arima(y, c(1,1,1), c(1,0,0), include.drift = TRUE, lambda = 0, method = "ML")`.
struct ArimaReference {
    /// ma1, sma1
    airline: [f64; 2],
    airline_log_likelihood: f64,
    airline_forecast: [f64; 12],
    /// ar1, ma1, sar1, drift
    mixed: [f64; 4],
    mixed_log_likelihood: f64,
    mixed_forecast: [f64; 12],
    /// `ndiffs(log(y))`, `nsdiffs(log(y))`, `BoxCox.lambda(y, "guerrero")`
    ndiffs: usize,
    nsdiffs: usize,
    guerrero: f64,
}

const AIR_ARIMA: ArimaReference = ArimaReference {
    airline: [-0.401827772272, -0.556945173790],
    airline_log_likelihood: 244.699530597,
    airline_forecast: [
        450.422367481,
        425.717200073,
        479.006845102,
        492.404454072,
        509.054949568,
        583.344938745,
        670.010755214,
        667.077616515,
        558.189353219,
        497.207787559,
        429.871976407,
        477.242566997,
    ],
    mixed: [
        0.73523427456737,
        -0.99999782402667,
        0.90689485144783,
        0.00974902950165,
    ],
    mixed_log_likelihood: 242.448600433,
    mixed_forecast: [
        451.206565262,
        431.080101367,
        463.420496218,
        509.066667861,
        522.999291038,
        588.497161586,
        676.998916227,
        663.026363330,
        566.299597243,
        519.568160399,
        447.183950722,
        491.361772829,
    ],
    ndiffs: 1,
    nsdiffs: 1,
    guerrero: -0.294715585559,
};

const ICMS_ARIMA: ArimaReference = ArimaReference {
    airline: [-0.578421753186, -0.999763443280],
    airline_log_likelihood: 84.6365661441,
    airline_forecast: [
        824087651.468,
        837605807.036,
        845097568.136,
        877176648.511,
        870651143.943,
        907612610.305,
        913286854.737,
        800548628.426,
        748701843.264,
        827298101.671,
        785068656.707,
        877638898.883,
    ],
    mixed: [
        0.48383770191791,
        -0.99999910823202,
        0.25608848394180,
        0.00856771840443,
    ],
    mixed_log_likelihood: 98.821747781,
    mixed_forecast: [
        782941216.435,
        798552851.179,
        795630977.591,
        801024935.582,
        811502478.963,
        815262883.613,
        827052472.882,
        812320289.324,
        805517396.433,
        838067575.348,
        823433868.959,
        852087487.492,
    ],
    ndiffs: 1,
    nsdiffs: 0,
    guerrero: 1.42600128229,
};

const FPE_ARIMA: ArimaReference = ArimaReference {
    airline: [-0.502273154813, -0.608616764629],
    airline_log_likelihood: 101.375486236,
    airline_forecast: [
        628811760.531,
        788007680.938,
        646993227.283,
        695072130.647,
        908738420.583,
        1013328981.886,
        950705731.575,
        1242079466.258,
        777066144.958,
        857946290.234,
        1029724467.485,
        1042460166.335,
    ],
    mixed: [
        -0.26331594148068,
        -0.28564719758111,
        0.89457071468830,
        0.00932160964173,
    ],
    mixed_log_likelihood: 96.8433302328,
    mixed_forecast: [
        605804412.253,
        778267385.323,
        643074349.628,
        676764261.759,
        885980341.219,
        988598770.154,
        926784158.249,
        1139780431.121,
        727780403.109,
        843484124.713,
        1001461652.909,
        1051274762.031,
    ],
    ndiffs: 1,
    nsdiffs: 1,
    guerrero: 0.0760382781277,
};

fn arima_cases() -> Vec<(&'static str, Vec<f64>, usize, &'static ArimaReference)> {
    vec![
        ("air", air_passengers(), 0, &AIR_ARIMA),
        ("icms", column(0), 2, &ICMS_ARIMA),
        ("fpe", column(1), 2, &FPE_ARIMA),
    ]
}

fn value(params: &[(String, f64)], name: &str) -> f64 {
    params.iter().find(|(k, _)| k == name).unwrap().1
}

/// R handles the differences with a diffuse prior and we difference the data,
/// which is the same likelihood in the limit; the optimisers differ too.
#[test]
fn arima_by_maximum_likelihood_matches_r() {
    use foresight::models::Arima;
    use foresight::Transformed;
    for (name, v, first, r) in arima_cases() {
        let y = Series::monthly(&v, first);

        let fit = Transformed::log(Arima::airline()).fit(y).unwrap();
        let p = fit.params();
        for (k, coefficient) in ["ma1", "sma1"].iter().enumerate() {
            assert!(
                (value(&p, coefficient) - r.airline[k]).abs() < 2e-3,
                "{name} airline {coefficient}: {} × {}",
                value(&p, coefficient),
                r.airline[k]
            );
        }
        assert!((value(&p, "log_likelihood") - r.airline_log_likelihood).abs() < 0.02);
        close(
            &format!("{name} airline"),
            &fit.forecast(12),
            &r.airline_forecast,
            1e-4,
        );

        let fit = Transformed::log(Arima::new(1, 1, 1).seasonal(1, 0, 0).constant(true))
            .fit(y)
            .unwrap();
        let p = fit.params();
        for (k, coefficient) in ["ar1", "ma1", "sar1", "drift"].iter().enumerate() {
            assert!(
                (value(&p, coefficient) - r.mixed[k]).abs() < 2e-3,
                "{name} mixed {coefficient}: {} × {}",
                value(&p, coefficient),
                r.mixed[k]
            );
        }
        assert!((value(&p, "log_likelihood") - r.mixed_log_likelihood).abs() < 0.02);
        close(
            &format!("{name} mixed"),
            &fit.forecast(12),
            &r.mixed_forecast,
            1e-4,
        );
    }
}

#[test]
fn differences_and_box_cox_match_r() {
    use foresight::diagnostics::{ndiffs, nsdiffs};
    use foresight::BoxCox;
    for (name, v, first, r) in arima_cases() {
        let log: Vec<f64> = v.iter().map(|x| x.ln()).collect();
        assert_eq!(ndiffs(&log, 2), r.ndiffs, "{name} ndiffs");
        // R measures the seasonal strength on an STL decomposition and we on
        // a classical one; the decision is the same on these series
        assert_eq!(nsdiffs(&log, 12), r.nsdiffs, "{name} nsdiffs");
        let lambda = BoxCox::guerrero(Series::monthly(&v, first))
            .unwrap()
            .lambda();
        assert!((lambda - r.guerrero).abs() < 1e-3, "{name} λ {lambda}");
    }
}

/// `auto.arima(y, lambda = 0)`: the same orders as R's stepwise search on the
/// airline passengers and on ICMS. (On FPE the two searches part ways and end
/// within one unit of AICc of each other.)
#[test]
fn automatic_orders_match_r() {
    use foresight::models::AutoArima;
    for (name, v, first, order, seasonal) in [
        ("air", air_passengers(), 0, (0, 1, 1), (0, 1, 1)),
        ("icms", column(0), 2, (0, 1, 1), (0, 0, 2)),
    ] {
        let log: Vec<f64> = v.iter().map(|x| x.ln()).collect();
        let fit = AutoArima::new()
            .select(Series::monthly(&log, first))
            .unwrap();
        assert_eq!(fit.order(), order, "{name}");
        assert_eq!(fit.seasonal_order(), seasonal, "{name}");
    }
}
