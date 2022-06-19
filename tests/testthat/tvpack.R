library("testthat")
library("mvtnorm")
context("Adaptations from TVPACK")

cor2d <- function(x) {
    return(matrix(c(1,x,x,1), nrow=2, ncol=2))
}

cdf2d_tvpack <- function(x1, x2, rho) {
    return(as.numeric(pmvnorm(upper=c(x1,x2), corr=cor2d(rho), algorithm=TVPACK())))
}

cor3d <- function(r12, r13, r23) {
    return(matrix(c(1, r12, r13,  r12, 1, r23,  r13, r23, 1), nrow=3, ncol=3, byrow=TRUE))
}
cdf3d_tvpack <- function(x, R) {
    return(as.numeric(pmvnorm(upper=c(x), corr=R,
                              algorithm=TVPACK(abseps=1e-14))))
}

test_that("CDF 2D Exact", {
    set.seed(123)
    for (i in 1:5000) {
        x <- runif(2, min=-5, max=5)
        x1 <- x[1L]
        x2 <- x[2L]
        rho <- runif(1, min=-0.99, max=0.99)
        cdf1 <- .Call(R_norm_cdf_2d, x1, x2, rho)
        cdf2 <- cdf2d_tvpack(x1, x2, rho)
        diff <- abs(cdf1 - cdf2)
        expect_lt(diff, 1e-15)
    }
})

### Note: the default tolerance for TVPACK() is smaller than what's used in this package
test_that("CDF 3D Exact", {
    set.seed(123)
    for (i in 1:50000) {
        x <- runif(3, min=-5, max=5)
        x1 <- x[1L]
        x2 <- x[2L]
        x3 <- x[3L]
        rho <- runif(3, min=-0.99, max=0.99)
        R <- cor3d(rho[1], rho[2], rho[3])
        if (det(R) <= 0) next
        cdf1 <- .Call(R_norm_cdf_3d, x1, x2, x3, rho[1], rho[2], rho[3])
        cdf2 <- cdf3d_tvpack(x, R)
        diff <- abs(cdf1 - cdf2)
        expect_lt(diff, 1e-15)
    }
})
