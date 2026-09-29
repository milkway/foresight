# Reference values for the tests of regression with ARIMA errors in
# tests/against_r.rs, from the R package forecast. Only the numbers are used.
#
#   Rscript tests/r/reference_regression.R data/piaui_revenue.csv data/air_passengers.csv
args <- commandArgs(TRUE)
if (length(args) > 2) .libPaths(c(args[3], .libPaths()))
suppressPackageStartupMessages(library(forecast))
d <- read.csv(args[1], comment.char = "#")
a <- read.csv(args[2], comment.char = "#")
num <- function(x) paste(format(as.numeric(x), digits = 12), collapse = ",")

# 1. a price index as regressor: fitted on the first 100 months, forecast for
#    the last 12 with the index that was observed
icms <- ts(log(d$icms[1:100]), frequency = 12)
x <- log(d$ipca_index)
f <- Arima(icms, order = c(0, 1, 1), seasonal = c(0, 1, 1), xreg = cbind(ipca = x[1:100]), method = "ML")
cat("index coef", num(coef(f)), "loglik", num(f$loglik), "\n")
cat("index forecast", num(forecast(f, xreg = cbind(ipca = x[101:112]))$mean), "\n")

# 2. harmonics instead of seasonal terms, with drift
air <- ts(log(a$passengers), frequency = 12)
g <- Arima(air, order = c(1, 1, 1), xreg = fourier(air, K = 3), include.drift = TRUE, method = "ML")
cat("fourier arma", num(coef(g)[1:2]), "drift", num(coef(g)["drift"]), "loglik", num(g$loglik), "\n")
cat("fourier forecast", num(forecast(g, xreg = fourier(air, K = 3, h = 12))$mean), "\n")

# 3. a level regression with autoregressive errors
h <- Arima(ts(d$fpe[1:100] / 1e6, frequency = 12), order = c(1, 0, 0), seasonal = c(1, 0, 0),
           xreg = cbind(ipca = d$ipca_index[1:100]), method = "ML")
cat("level coef", num(coef(h)), "loglik", num(h$loglik), "\n")
cat("level forecast", num(forecast(h, xreg = cbind(ipca = d$ipca_index[101:112]))$mean), "\n")
