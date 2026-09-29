# Reference values for the ETS tests in tests/against_r.rs, from the R package
# forecast. Only the numbers it prints are used.
#
#   Rscript tests/r/reference_ets.R data/piaui_revenue.csv data/air_passengers.csv
suppressPackageStartupMessages(library(forecast))
args <- commandArgs(TRUE)
d <- read.csv(args[1], comment.char = "#")
a <- read.csv(args[2], comment.char = "#")
num <- function(x) paste(format(as.numeric(x), digits = 12), collapse = ",")
series <- list(air = ts(a$passengers, frequency = 12),
               icms = ts(d$icms, frequency = 12),
               fpe = ts(d$fpe, frequency = 12))
for (nm in names(series)) {
  y <- series[[nm]]
  for (code in c("ANN", "AAN", "AAdN", "AAA", "MAM", "MAdM", "MNM", "MNA")) {
    f <- ets(y, model = gsub("d", "", code), damped = grepl("d", code))
    cat(nm, code, "loglik", num(f$loglik), "aicc", num(f$aicc),
        "forecast", num(forecast(f, h = 12)$mean), "\n")
  }
  g <- ets(y)
  cat(nm, "auto", g$method, "aicc", num(g$aicc), "\n")
}
