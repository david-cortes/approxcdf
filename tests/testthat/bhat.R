library("testthat")
library("mvtnorm")
context("Test Bhar's CDF for 4D")

cdf4d_genz <- function(x, S) {
    return(as.numeric(pmvnorm(upper=x, corr=S, algorithm=GenzBretz(abseps=1e-8))))
}

cdf4d_miwa <- function(x, S) {
    return(as.numeric(pmvnorm(upper=x, corr=S, algorithm=Miwa())))
}

test_that("CDF 4D Bhat", {
    set.seed(123)
    for (i in 1:1000) {
        S <- matrix(rnorm(16), nrow=4, ncol=4)
        S <- crossprod(S)
        if (det(S) <= 0) {
            S <- S + 1e-5 * diag(4)
        }
        std <- 1 / sqrt(diag(S))
        S <- diag(std) %*% S %*% diag(std)

        x <- rnorm(4)

        cdf1 <- .Call(R_norm_cdf_4d_bhat, x, S)
        cdf2 <- cdf4d_genz(x, S)
        cdf3 <- cdf4d_miwa(x, S)
        diff <- min(abs(cdf1 - cdf2), abs(cdf1 - cdf3))

        # expect_lt(diff, 0.015) # <- for simple ordering (author's)
        if (det(S) >= 1e-1) {
            expect_lt(diff, 0.003) # <- for GGE or TG orders, plus regularization
        }
        else if (det(S) >= 1e-3) {
            expect_lt(diff, 0.020) # <- for GGE or TG orders, plus regularization
        }
        if (diff > 1e-3) {
            if (max(abs(x)) > 4 || det(S) <= 0.05) next
            rdiff <- min(max(cdf1/cdf2, cdf2/cdf1), max(cdf1/cdf3, cdf3/cdf1))
            expect_lt(rdiff, 1.04)
        }
    }
})
