# Reference values for the tests of intermittent demand and of outliers in
# tests/against_r.rs, from the R package forecast. Only the numbers are used.
#
#   Rscript tests/r/reference_croston.R data/piaui_revenue.csv data/air_passengers.csv
args <- commandArgs(TRUE)
if (length(args) > 2) .libPaths(c(args[3], .libPaths()))
suppressPackageStartupMessages(library(forecast))
d <- read.csv(args[1], comment.char = "#")
a <- read.csv(args[2], comment.char = "#")
num <- function(x) paste(format(as.numeric(x), digits = 12), collapse = ",")
# demand on one period in three or so, from a formula both sides can compute
t <- 0:59
demand <- ifelse((t * 7) %% 10 < 3, 1 + (t * 3) %% 5, 0)
cat("demand", num(demand), "\n")
for (alpha in c(0.1, 0.3)) {
  cat("croston", alpha, num(croston(demand, h = 1, alpha = alpha)$mean), "\n")
}
# outliers put into real series
air <- log(a$passengers); air[c(30, 100)] <- air[c(30, 100)] + c(0.8, -0.7)
icms <- d$icms; icms[c(40, 90)] <- icms[c(40, 90)] * c(2, 0.4)
fpe <- d$fpe
for (nm in c("air", "icms", "fpe")) {
  o <- tsoutliers(ts(get(nm), frequency = 12))
  cat(nm, "outliers", num(o$index), "replacements", num(o$replacements), "\n")
}
