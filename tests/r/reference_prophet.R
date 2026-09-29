# Reference values for the Prophet tests in tests/against_r.rs, from the R
# package prophet. Only the numbers it prints are used.
#
#   Rscript tests/r/reference_prophet.R data/piaui_revenue.csv data/air_passengers.csv
#
# Observations are laid on consecutive days and the seasonal period is given in
# those units, so the design is the one of an equally spaced series.
suppressPackageStartupMessages(library(prophet))
args <- commandArgs(TRUE)
d <- read.csv(args[1], comment.char = "#")
a <- read.csv(args[2], comment.char = "#")
num <- function(x) paste(format(as.numeric(x), digits = 12), collapse = ",")
cat("prophet", as.character(packageVersion("prophet")), "\n")
run <- function(nm, y, period, order, h = 12) {
  n <- length(y)
  df <- data.frame(ds = as.Date("2000-01-01") + 0:(n - 1), y = as.numeric(y))
  m <- prophet(yearly.seasonality = FALSE, weekly.seasonality = FALSE,
               daily.seasonality = FALSE, uncertainty.samples = 0)
  if (order > 0) m <- add_seasonality(m, name = "cycle", period = period, fourier.order = order)
  m <- fit.prophet(m, df)
  f <- predict(m, data.frame(ds = as.Date("2000-01-01") + 0:(n + h - 1)))
  cat(nm, "forecast", num(tail(f$yhat, h)), "\n")
}
run("air", a$passengers, 12, 5)
run("airlog", log(a$passengers), 12, 5)
run("icms", d$icms, 12, 5)
run("fpelog", log(d$fpe), 12, 5)
run("airflat", a$passengers, 12, 0)
