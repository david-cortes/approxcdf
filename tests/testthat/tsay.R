library("testthat")
library("mvtnorm")
context("BVCDF from Tsay")

cor2d <- function(x) {
    return(matrix(c(1,x,x,1), nrow=2, ncol=2))
}
test_that("CDF 2D Very Fast", {
    set.seed(123)
    for (i in 1:1000) {
        x <- runif(2, min=-5, max=5)
        x1 <- x[1L]
        x2 <- x[2L]
        rho <- runif(1, min=-0.99, max=0.99)
        cdf1 <- .Call(R_norm_cdf_2d_vfast, x1, x2, rho)
        cdf2 <- as.numeric(pmvnorm(upper=x, corr=cor2d(rho)))
        if (cdf1 == 0 || cdf2 == 0) {
            expect_lt(max(cdf1, cdf2), 1e-8)
            next
        }
        diff <- abs(cdf1 - cdf2)
        expect_lt(diff, 1e-3)
    }
})
