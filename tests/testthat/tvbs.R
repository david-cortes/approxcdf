library("testthat")
library("mvtnorm")
context("Test TVBS for high-dimensional N")

gen_cov <- function(n, min_det=1e-2) {
    S <- rnorm(n*n)
    S <- matrix(S, nrow=n)
    S <- crossprod(S)
    while (1) {
        S <- S + 1e-1*diag(n)
        std <- 1 / sqrt(diag(S))
        S <- diag(std) %*% S %*% diag(std)
        if (det(S) >= min_det) break
    }
    return(S)
}

test_that("TVBS for high-dimensions is close to Genz's MVNDST", {
    set.seed(123)
    for (i in 1:500) {
        n <- as.integer(runif(1, 5, 17))
        x <- rnorm(n, sd=0.1)
        S <- gen_cov(n, 0.5)

        p1 <- pmvn(x, S, is_standardized=TRUE)
        p2 <- as.numeric(pmvnorm(upper=x, corr=S))

        diff <- abs(p1-p2)
        expect_lt(diff, 1e-3)
    }
})
