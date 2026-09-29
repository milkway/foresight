# Reference values for the STL tests in tests/against_r.rs, from stl() in R
# and mstl()/stlf() in the package forecast. Only the numbers are used.
#
#   Rscript tests/r/reference_stl.R data/piaui_revenue.csv data/air_passengers.csv
args <- commandArgs(TRUE)
if (length(args) > 2) .libPaths(c(args[3], .libPaths()))
suppressPackageStartupMessages(library(forecast))
d <- read.csv(args[1], comment.char = "#")
a <- read.csv(args[2], comment.char = "#")
num <- function(x) paste(format(as.numeric(x), digits = 14), collapse = ",")
pick <- c(1, 2, 3, 50, 51, 52, 142, 143, 144)
air <- ts(log(a$passengers), frequency = 12)
show <- function(nm, f) {
  cat(nm, "seasonal", num(f$time.series[pick, "seasonal"]), "\n")
  cat(nm, "trend", num(f$time.series[pick, "trend"]), "\n")
}
show("periodic", stl(air, s.window = "periodic"))
show("span13", stl(air, s.window = 13))
show("robust7", stl(air, s.window = 7, robust = TRUE))
show("degree1", stl(air, s.window = 11, s.degree = 1, t.window = 21))
m <- mstl(air)
cat("mstl seasonal", num(m[pick, "Seasonal12"]), "\n")
cat("mstl trend", num(m[pick, "Trend"]), "\n")
# two seasonal periods, from a formula both sides can compute
t <- 0:419
two <- 100 + 0.05 * t + c(5, 0, -2, -3, 0, 1, -1)[t %% 7 + 1] +
  8 * sin(2 * pi * t / 30) + 2 * sin(t * 1.7) * cos(t * 0.3)
m2 <- mstl(msts(two, seasonal.periods = c(7, 30)))
pick2 <- c(1, 2, 3, 200, 201, 202, 418, 419, 420)
cat("two seasonal7", num(m2[pick2, "Seasonal7"]), "\n")
cat("two seasonal30", num(m2[pick2, "Seasonal30"]), "\n")
cat("two trend", num(m2[pick2, "Trend"]), "\n")
icms <- ts(d$icms, frequency = 12)
cat("stlf naive", num(stlf(icms, h = 12, method = "naive")$mean), "\n")
cat("stlf drift", num(stlf(icms, h = 12, method = "rwdrift")$mean), "\n")
cat("stlf airdrift", num(stlf(air, h = 12, method = "rwdrift")$mean), "\n")
