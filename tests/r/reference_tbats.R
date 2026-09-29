# Reference values for the TBATS tests in tests/against_r.rs, from the R
# package forecast. Only the numbers it prints are used.
#
#   Rscript tests/r/reference_tbats.R data/piaui_revenue.csv data/air_passengers.csv
args <- commandArgs(TRUE)
if (length(args) > 2) .libPaths(c(args[3], .libPaths()))
suppressPackageStartupMessages(library(forecast))
d <- read.csv(args[1], comment.char = "#")
a <- read.csv(args[2], comment.char = "#")
num <- function(x) paste(format(as.numeric(x), digits = 10), collapse = ",")
show <- function(nm, f) {
  cat(nm, "lambda", num(f$lambda), "damping", num(f$damping.parameter), "harmonics", num(f$k.vector),
      "p", length(f$ar.coefficients), "q", length(f$ma.coefficients),
      "likelihood", num(f$likelihood), "aic", num(f$AIC), "\n")
}
series <- list(air = ts(a$passengers, frequency = 12),
               icms = ts(d$icms, frequency = 12),
               fpe = ts(d$fpe, frequency = 12))
for (nm in names(series)) {
  y <- series[[nm]]
  show(paste0(nm, "_auto"), tbats(y, use.parallel = FALSE))
  show(paste0(nm, "_plain"), tbats(y, use.box.cox = FALSE, use.trend = TRUE, use.damped.trend = FALSE,
                                   use.arma.errors = FALSE, use.parallel = FALSE))
  show(paste0(nm, "_boxcox"), tbats(y, use.box.cox = TRUE, use.trend = TRUE, use.damped.trend = FALSE,
                                    use.arma.errors = FALSE, use.parallel = FALSE))
}
