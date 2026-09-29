# Reference values for tests/against_r.rs, from the R package forecast.
# Only the numbers it prints are used; none of its code is.
#
#   Rscript tests/r/reference.R data/piaui_revenue.csv
suppressPackageStartupMessages(library(forecast))
d <- read.csv(commandArgs(TRUE)[1], comment.char = "#")
cat("forecast", as.character(packageVersion("forecast")), "\n")
show <- function(col, name, f) {
  cat(col, name, paste(format(as.numeric(f$mean), digits = 12), collapse = ","), "\n")
}
for (col in c("icms", "fpe")) {
  y <- ts(d[[col]], frequency = 12, start = c(2017, 3))
  show(col, "theta", thetaf(y, h = 12))
  show(col, "snaive", snaive(y, h = 12))
  show(col, "drift", rwf(y, h = 12, drift = TRUE))
}

# ARIMA, differences and Box-Cox
num <- function(x) paste(format(as.numeric(x), digits = 12), collapse = ",")
series <- list(air = AirPassengers,
               icms = ts(d$icms, frequency = 12, start = c(2017, 3)),
               fpe = ts(d$fpe, frequency = 12, start = c(2017, 3)))
for (nm in names(series)) {
  y <- series[[nm]]
  f <- Arima(y, order = c(0, 1, 1), seasonal = c(0, 1, 1), lambda = 0, method = "ML")
  cat(nm, "airline", num(coef(f)), "loglik", num(f$loglik), "\n")
  cat(nm, "airline forecast", num(forecast(f, h = 12)$mean), "\n")
  g <- Arima(y, order = c(1, 1, 1), seasonal = c(1, 0, 0), lambda = 0, method = "ML",
             include.drift = TRUE)
  cat(nm, "mixed", num(coef(g)), "loglik", num(g$loglik), "\n")
  cat(nm, "mixed forecast", num(forecast(g, h = 12)$mean), "\n")
  cat(nm, "auto", paste(arimaorder(auto.arima(y, lambda = 0)), collapse = ","), "\n")
  cat(nm, "ndiffs", ndiffs(log(y)), "nsdiffs", nsdiffs(log(y)),
      "guerrero", num(BoxCox.lambda(y, method = "guerrero")), "\n")
}
