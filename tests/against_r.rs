//! Forecasts compared with the R packages `forecast` 9.0.2 and `prophet` 1.1.7
//! on public data. The reference values come from the scripts in `tests/r/`.

// the reference values are pasted as R prints them
#![allow(clippy::excessive_precision)]

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

// ---------------------------------------------------------------- Prophet

/// `prophet` 1.1.7 with its default priors, 25 changepoints and a seasonal
/// pattern of period 12 and 5 harmonics (0 for the last case); see
/// `tests/r/reference_prophet.R`.
const PROPHET: [(&str, [f64; 12]); 5] = [
    (
        "air",
        [
            465.159342120,
            457.700024268,
            494.497714684,
            490.689199594,
            497.069043822,
            536.158398487,
            577.436885865,
            576.439204633,
            529.375636835,
            492.773831354,
            460.610964699,
            488.831699906,
        ],
    ),
    (
        "airlog",
        [
            6.12936633602,
            6.11052001620,
            6.25623281675,
            6.22828613082,
            6.24183384877,
            6.36743747478,
            6.48711484227,
            6.48070409913,
            6.35141161822,
            6.21622225669,
            6.08751465413,
            6.20430269438,
        ],
    ),
    (
        "icms",
        [
            799991648.283,
            801449164.061,
            814319484.717,
            822632193.810,
            834703548.778,
            848279450.041,
            860422337.330,
            788887340.768,
            771602741.356,
            813459106.402,
            803839761.526,
            847079142.629,
        ],
    ),
    (
        "fpelog",
        [
            20.3150460091,
            20.3906146702,
            20.3018838256,
            20.2848513429,
            20.6304825552,
            20.6502040028,
            20.7049261944,
            20.8688811773,
            20.5207288971,
            20.5000313753,
            20.7582784957,
            20.5883568188,
        ],
    ),
    (
        "airflat",
        [
            480.471108596,
            483.335113735,
            486.199118873,
            489.063124012,
            491.927129151,
            494.791134290,
            497.655139429,
            500.519144568,
            503.383149706,
            506.247154845,
            509.111159984,
            511.975165123,
        ],
    ),
];

/// Prophet finds its estimate with a general optimiser on a function that has
/// kinks; we solve the same problem exactly. The forecasts differ by the
/// precision of that optimiser.
#[test]
fn prophet_matches_r() {
    use foresight::models::Prophet;
    let air = air_passengers();
    let log = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| x.ln()).collect() };
    for (name, reference) in &PROPHET {
        let (values, order) = match *name {
            "air" => (air.clone(), 5),
            "airlog" => (log(&air), 5),
            "icms" => (column(0), 5),
            "fpelog" => (log(&column(1)), 5),
            _ => (air.clone(), 0),
        };
        let forecast = Prophet::new()
            .fourier_order(order)
            .forecast(Series::new(&values, 12), 12)
            .unwrap();
        close(&format!("prophet {name}"), &forecast, reference, 5e-3);
    }
}

// ---------------------------------------------------------------- ETS

/// `ets(y, model, damped)`: series, model, log-likelihood and AICc; see
/// `tests/r/reference_ets.R`.
const ETS: [(&str, &str, f64, f64); 24] = [
    ("air", "ANN", -863.893377599, 1733.95818377),
    ("air", "AAN", -863.647302495, 1737.7293876),
    ("air", "AAdN", -863.796068067, 1740.20527482),
    ("air", "AAA", -765.935847512, 1570.72883788),
    ("air", "MAM", -682.403619058, 1403.66438097),
    ("air", "MAdM", -679.583216237, 1400.63843247),
    ("air", "MNM", -715.657196898, 1465.0643938),
    ("air", "MNA", -775.396153368, 1584.54230674),
    ("icms", "ANN", -2246.04992001, 4498.32206225),
    ("icms", "AAN", -2245.95553203, 4502.4771018),
    ("icms", "AAdN", -2245.02595847, 4502.85191695),
    ("icms", "AAA", -2211.70111867, 4463.91287563),
    ("icms", "MAM", -2229.11315395, 4498.7369462),
    ("icms", "MAdM", -2224.98062858, 4493.31609586),
    ("icms", "MNM", -2231.77342806, 4498.54685612),
    ("icms", "MNA", -2231.4187523, 4497.8375046),
    ("fpe", "ANN", -2335.69328148, 4677.60878518),
    ("fpe", "AAN", -2333.22397189, 4677.01398151),
    ("fpe", "AAdN", -2334.54709054, 4681.89418108),
    ("fpe", "AAA", -2227.54957112, 4495.60978053),
    ("fpe", "MAM", -2205.55417425, 4451.6189868),
    ("fpe", "MAdM", -2207.13093611, 4457.61671093),
    ("fpe", "MNM", -2219.48334587, 4473.96669174),
    ("fpe", "MNA", -2245.25864152, 4525.51728304),
];

fn ets_series(name: &str) -> Vec<f64> {
    match name {
        "air" => air_passengers(),
        "icms" => column(0),
        _ => column(1),
    }
}

/// R searches parameters and initial states together with a simplex method
/// that often stops short of the maximum, so its likelihood is a floor, not a
/// target: ours is never lower, and equal where R does get there.
#[test]
fn ets_likelihood_is_never_below_r() {
    use foresight::models::Ets;
    let mut equal = 0;
    for (name, code, log_likelihood, aicc) in ETS {
        let values = ets_series(name);
        let fit = Ets::from_code(code)
            .unwrap()
            .estimate(Series::new(&values, 12))
            .unwrap();
        assert!(
            fit.log_likelihood > log_likelihood - 1e-3,
            "{name} {code}: {} × {log_likelihood}",
            fit.log_likelihood
        );
        // the same likelihood gives the same criterion: the parameters are
        // counted alike
        if (fit.log_likelihood - log_likelihood).abs() < 5e-3 {
            assert!((fit.aicc - aicc).abs() < 1e-2, "{name} {code}");
            equal += 1;
        }
    }
    assert!(equal >= 2, "{equal}");
}

/// Simple exponential smoothing, where both reach the maximum.
#[test]
fn simple_exponential_smoothing_matches_r() {
    use foresight::models::Ets;
    for (name, reference) in [
        ("air", 431.995792636_f64),
        ("icms", 762311852.346_f64),
        ("fpe", 848669474.797_f64),
    ] {
        let values = ets_series(name);
        let forecast = Ets::from_code("ANN")
            .unwrap()
            .forecast(Series::new(&values, 12), 12)
            .unwrap();
        close(&format!("ets {name}"), &forecast, &[reference; 12], 1e-3);
    }
}

/// `ets(y)`: the same model on ICMS and FPE. On the airline passengers R
/// settles for ETS(M,Ad,M) because its search stops short on ETS(M,A,M).
#[test]
fn automatic_ets_matches_r() {
    use foresight::models::AutoEts;
    for (name, code) in [("icms", "AAA"), ("fpe", "MAM")] {
        let values = ets_series(name);
        let fit = AutoEts::new().select(Series::new(&values, 12)).unwrap();
        assert_eq!(fit.model().code(), code, "{name}");
    }
}

// ---------------------------------------------------------------- STL

const PICKED: [usize; 9] = [1, 2, 3, 50, 51, 52, 142, 143, 144];

fn picked(values: &[f64], positions: &[usize]) -> Vec<f64> {
    positions.iter().map(|i| values[i - 1]).collect()
}

fn exactly(name: &str, ours: &[f64], reference: &[f64]) {
    assert_eq!(ours.len(), reference.len());
    for (a, b) in ours.iter().zip(reference) {
        assert!(
            (a - b).abs() < 1e-10 * b.abs().max(1.0),
            "{name}: {a} × {b}"
        );
    }
}

/// `stl()` on the log of the airline passengers; see
/// `tests/r/reference_stl.R`. The same algorithm gives the same numbers.
#[test]
fn stl_matches_r() {
    use foresight::decompose::{SeasonalWindow, Stl};
    let air: Vec<f64> = air_passengers().iter().map(|x| x.ln()).collect();
    let cases = [
        (
            "periodic",
            Stl::new(12, SeasonalWindow::Periodic),
            [
                -0.091640424146522,
                -0.114028283920271,
                0.015865851044779,
                -0.114028283920271,
                0.015865851044779,
                -0.014027589973823,
                -0.070248358783457,
                -0.213527739592963,
                -0.100636250654005,
            ],
            [
                4.8293886539529,
                4.8303681398962,
                4.8313476258395,
                5.3817013546060,
                5.3922411311998,
                5.3984785236674,
                6.1875938244424,
                6.1963074920020,
                6.2047523564265,
            ],
        ),
        (
            "span13",
            Stl::new(12, SeasonalWindow::Span(13)),
            [
                -0.0904060780043153,
                -0.0836637751334351,
                0.0482334330149413,
                -0.0974759863024456,
                0.0394707343252156,
                -0.0039276791372576,
                -0.0704283807905346,
                -0.2151459072716038,
                -0.1115730324867191,
            ],
            [
                4.8162448382157,
                4.8199604288353,
                4.8236760194550,
                5.3760638278799,
                5.3855194397741,
                5.3949750516684,
                6.1839137015929,
                6.1919356466679,
                6.1999575917430,
            ],
        ),
        (
            "degree1",
            Stl::new(12, SeasonalWindow::Span(11))
                .degrees(1, 1, 1)
                .trend_window(21),
            [
                -0.0911574879099253,
                -0.0215561352295905,
                0.0841225137629235,
                -0.0932180134549157,
                0.0443587852853652,
                -0.0004079228215351,
                -0.0616371229335772,
                -0.2140886482110486,
                -0.1246185253352176,
            ],
            [
                4.8043772647416,
                4.8097339370475,
                4.8150906093534,
                5.3760442794580,
                5.3852536389086,
                5.3944629983592,
                6.1780595183650,
                6.1850210484115,
                6.1919825784580,
            ],
        ),
    ];
    for (name, stl, seasonal, trend) in cases {
        let d = stl.decompose(&air).unwrap();
        exactly(
            &format!("{name} seasonal"),
            &picked(&d.seasonal[0], &PICKED),
            &seasonal,
        );
        exactly(&format!("{name} trend"), &picked(&d.trend, &PICKED), &trend);
    }
    // robust, 5 rounds of one pass: first and last seasonal values
    let d = Stl::new(12, SeasonalWindow::Span(7))
        .robust(true)
        .iterations(1, 5)
        .decompose(&air)
        .unwrap();
    exactly(
        "robust",
        &[d.seasonal[0][0], d.seasonal[0][143]],
        &[-0.0708530269055, -0.1165670131205],
    );
}

/// `mstl()` of the package forecast, with one and with two seasonal periods.
#[test]
fn mstl_matches_r() {
    use foresight::decompose::Mstl;
    let air: Vec<f64> = air_passengers().iter().map(|x| x.ln()).collect();
    let d = Mstl::new(&[12]).decompose(&air).unwrap();
    exactly(
        "mstl seasonal",
        &picked(&d.seasonal[0], &PICKED),
        &[
            -0.0906711738360057,
            -0.0787778402369833,
            0.0518556568069013,
            -0.0956238190693830,
            0.0422890316389607,
            -0.0020400905349825,
            -0.0699806937282843,
            -0.2152399450163227,
            -0.1127019025044743,
        ],
    );
    exactly(
        "mstl trend",
        &picked(&d.trend, &PICKED),
        &[
            4.8154594385253,
            4.8192879728100,
            4.8231165070947,
            5.3759590706955,
            5.3853869671768,
            5.3948148636581,
            6.1831917970209,
            6.1910721898301,
            6.1989525826393,
        ],
    );

    let weekly = [5.0, 0.0, -2.0, -3.0, 0.0, 1.0, -1.0];
    let two: Vec<f64> = (0..420)
        .map(|t| {
            let x = t as f64;
            100.0
                + 0.05 * x
                + weekly[t % 7]
                + 8.0 * (2.0 * std::f64::consts::PI * x / 30.0).sin()
                + 2.0 * (x * 1.7).sin() * (x * 0.3).cos()
        })
        .collect();
    let d = Mstl::new(&[7, 30]).decompose(&two).unwrap();
    let positions = [1, 2, 3, 200, 201, 202, 418, 419, 420];
    exactly(
        "two seasonal 7",
        &picked(&d.seasonal[0], &positions),
        &[
            5.033962062787970,
            0.035176510741036,
            -2.129322803268722,
            -3.086314754228369,
            0.069767979458528,
            1.032998673942375,
            -0.124792166606373,
            0.938282183929101,
            -0.898954997659186,
        ],
    );
    exactly(
        "two seasonal 30",
        &picked(&d.seasonal[1], &positions),
        &[
            -0.090911163758398,
            1.743338626385907,
            3.241352290708424,
            -5.955326834161487,
            -6.918130532421581,
            -7.605873830646563,
            -4.733437975352143,
            -3.132438409506112,
            -1.598225429194311,
        ],
    );
    exactly(
        "two trend",
        &picked(&d.trend, &positions),
        &[
            100.12665833943,
            100.17046946140,
            100.21428058337,
            109.95009753047,
            110.00013113773,
            110.05016474498,
            120.97907207921,
            121.03602309996,
            121.09297412071,
        ],
    );
}

/// `stlf()` with the naive and the drift methods on what is left.
#[test]
fn forecasts_by_decomposition_match_r() {
    use foresight::models::{Decomposed, Naive};
    let icms = column(0);
    let series = Series::new(&icms, 12);
    exactly(
        "stlf naive",
        &Decomposed::new(Naive).forecast(series, 12).unwrap(),
        &[
            788061202.72154,
            795059525.55471,
            789233907.08247,
            786475266.23672,
            800670860.26393,
            809264287.47215,
            808734039.80359,
            737205954.47647,
            696410778.83617,
            750153769.60324,
            735018340.48927,
            785725856.19000,
        ],
    );
    exactly(
        "stlf drift",
        &Decomposed::new(Drift).forecast(series, 12).unwrap(),
        &[
            792191829.46341,
            803320779.03844,
            801625787.30807,
            802997773.20418,
            821323993.97325,
            834048047.92334,
            837648426.99665,
            770250968.41139,
            733586419.51296,
            791460037.02190,
            780455234.64979,
            835293377.09239,
        ],
    );
}

// ---------------------------------------------------------------- regression

fn revenue_column(i: usize) -> Vec<f64> {
    include_str!("../data/piaui_revenue.csv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("month"))
        .map(|l| l.split(',').nth(i).unwrap().parse().unwrap())
        .collect()
}

/// `Arima(..., xreg = ...)`: regression with ARIMA errors; see
/// `tests/r/reference_regression.R`.
#[test]
fn regression_with_arima_errors_matches_r() {
    use foresight::models::Arima;
    use foresight::{Fitted, Regressors};
    let index = revenue_column(3);
    let log = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| x.ln()).collect() };

    // a price index as regressor, known over the horizon
    let icms = log(&revenue_column(1));
    let fit = Arima::airline()
        .with_regressors(Regressors::new().with("ipca", log(&index)))
        .estimate(Series::new(&icms[..100], 12))
        .unwrap();
    assert!((fit.log_likelihood - 71.3269751088).abs() < 0.01);
    assert!((fit.regression[0].1 - 3.628961167101).abs() < 0.01);
    close(
        "index",
        &fit.forecast(12),
        &[
            20.4424516263,
            20.4434249567,
            20.4622874662,
            20.4867207767,
            20.4656206241,
            20.5001900609,
            20.5029053067,
            20.3654429375,
            20.3058916659,
            20.4129127608,
            20.3751970640,
            20.4812495021,
        ],
        1e-5,
    );

    // harmonics instead of seasonal terms, with drift
    let air = log(&air_passengers());
    let fit = Arima::new(1, 1, 1)
        .constant(true)
        .with_regressors(Regressors::fourier(12.0, 3, 156))
        .estimate(Series::new(&air, 12))
        .unwrap();
    assert!((fit.log_likelihood - 220.745055931).abs() < 0.01);
    assert!((fit.constant.unwrap() - 0.00967443136167).abs() < 1e-5);
    close(
        "fourier",
        &fit.forecast(12),
        &[
            6.11148655654,
            6.15834008721,
            6.21237653144,
            6.23937234485,
            6.26018036148,
            6.35199276944,
            6.48370469463,
            6.50120081342,
            6.35382780459,
            6.18866955419,
            6.14237100333,
            6.18258040658,
        ],
        1e-5,
    );

    // a regression in levels with autoregressive errors
    let fpe: Vec<f64> = revenue_column(2).iter().map(|x| x / 1e6).collect();
    let fit = Arima::new(1, 0, 0)
        .seasonal(1, 0, 0)
        .with_regressors(Regressors::new().with("ipca", index))
        .estimate(Series::new(&fpe[..100], 12))
        .unwrap();
    assert!((fit.log_likelihood - (-518.794753258)).abs() < 0.01);
    assert!((fit.regression[0].1 / 779.684429653289 - 1.0).abs() < 1e-4);
    close(
        "level",
        &fit.forecast(12),
        &[
            587.333383110,
            716.756829324,
            609.583369813,
            635.668780286,
            773.862676935,
            852.013532527,
            786.746648268,
            996.754336707,
            704.292578608,
            719.312598386,
            886.254521839,
            901.118519974,
        ],
        1e-5,
    );
}
