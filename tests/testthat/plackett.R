library("testthat")
library("mvtnorm")
context("Test Plackett's CDF for 4D")

cdf4d_genz <- function(x, S) {
    return(as.numeric(pmvnorm(upper=x, corr=S, algorithm=GenzBretz(abseps=1e-8))))
}

cdf4d_miwa <- function(x, S) {
    return(as.numeric(pmvnorm(upper=x, corr=S, algorithm=Miwa())))
}

test_that("Singular CDF4", {
    set.seed(123)
    for (i in 1:5000) {
        S <- matrix(rnorm(16), nrow=4, ncol=4)
        S <- crossprod(S)
        coefs <- rnorm(3)
        Slast <- matrix(S[1:3,1:3] %*% coefs, ncol=1)
        S[1:3,4] <- Slast
        S[4,1:3] <- Slast
        eps <- 1e-8
        Sorig <- S
        while (TRUE) {
            diag(Sorig) <- diag(Sorig) + eps
            std <- 1 / sqrt(diag(Sorig))
            S <- diag(std) %*% Sorig %*% diag(std)
            eps <- eps * 1.2
            if (det(S) >= 1e-9) break
        }
        if (det(S) >= 1e-3) next


        x <- rnorm(4)
        cdf1 <- .Call(R_singular_cdf4, x, S)
        cdf2 <- cdf4d_genz(x, S)
        diff <- abs(cdf1 - cdf2)
        expect_lt(diff, 0.04)

        ### The matrix is not really singular, but 'pmvnorm' will throw an error
        ### when the determinant is too low, and when using Miwa, might even
        ### throw negative numbers, so make a double check.
        ### This other function will use the function for singular matrix
        ### when appropriate, and make a correction otherwise.
        x <- abs(rnorm(4))
        cdf11 <- .Call(R_singular_cdf4, x, S)
        cdf12 <- cdf4d_pg(x, S)
        cdf2 <- cdf4d_genz(x, S)
        diff <- min(abs(cdf11 - cdf2), abs(cdf12 - cdf2))
        expect_lt(diff, 0.0075)
    }
})

## This one is very imprecise, don't expect much from it
test_that("CDF 4D Plackett-Gassmann", {
    set.seed(123)
    for (i in 1:1000) {
        S <- matrix(rnorm(16), nrow=4, ncol=4)
        S <- crossprod(S)
        if (det(S) <= 1e-1) {
            S <- S + 1e-5 * diag(4)
        }
        std <- 1 / sqrt(diag(S))
        S <- diag(std) %*% S %*% diag(std)
        if (det(S) <= 1e-2) next


        x <- rnorm(4)

        cdf1 <- .Call(R_norm_cdf_4d_pg, x, S)
        cdf2 <- cdf4d_genz(x, S)
        cdf3 <- cdf4d_miwa(x, S)
        diff <- min(abs(cdf1 - cdf2), abs(cdf1 - cdf3))

        expect_lt(diff, 0.04)
    }
})

test_that("CDF 4D Independent variables", {
    set.seed(123)
    for (i in 1:1000) {
        S <- matrix(rnorm(16), nrow=4, ncol=4)
        S <- crossprod(S)
        if (det(S) <= 1e-1) {
            S <- S + 1e-5 * diag(4)
        }
        std <- 1 / sqrt(diag(S))
        S <- diag(std) %*% S %*% diag(std)

        x <- rnorm(4)

        S1 <- S
        S1[1, ] <- 0
        S1[, 1] <- 0
        S1[1,1] <- 1

        S2 <- S
        S2[2, ] <- 0
        S2[, 2] <- 0
        S2[2,2] <- 1

        S3 <- S
        S3[3, ] <- 0
        S3[, 3] <- 0
        S3[3,3] <- 1

        S4 <- S
        S4[4, ] <- 0
        S4[, 4] <- 0
        S4[4,4] <- 1

        cdf11 <- .Call(R_norm_cdf_4d_pg, x, S1)
        cdf12 <- .Call(R_norm_cdf_4d_pg, x, S2)
        cdf13 <- .Call(R_norm_cdf_4d_pg, x, S3)
        cdf14 <- .Call(R_norm_cdf_4d_pg, x, S4)

        cdf21 <- cdf4d_genz(x, S1)
        cdf22 <- cdf4d_genz(x, S2)
        cdf23 <- cdf4d_genz(x, S3)
        cdf24 <- cdf4d_genz(x, S4)

        diff1 <- abs(cdf11 - cdf21)
        diff2 <- abs(cdf12 - cdf22)
        diff3 <- abs(cdf13 - cdf23)
        diff4 <- abs(cdf14 - cdf24)

        tol <- 1e-3
        expect_lt(diff1, tol)
        expect_lt(diff2, tol)
        expect_lt(diff3, tol)
        expect_lt(diff4, tol)
    }
})

### Genz's doesn't handle indefinite matrices
# test_that("CDF 4D Perfect correlations", {
#     set.seed(123)
#     for (i in 1:1000) {
#         S <- matrix(rnorm(16), nrow=4, ncol=4)
#         S <- crossprod(S)
#         if (det(S) <= 1e-1) {
#             S <- S + 1e-5 * diag(4)
#         }
#         std <- 1 / sqrt(diag(S))
#         S <- diag(std) %*% S %*% diag(std)

#         x <- rnorm(4)

#         S12p <- S
#         S12n <- S
#         S12p[1,2] <- +1
#         S12n[1,2] <- -1
#         S12p[2,1] <- +1
#         S12n[2,1] <- -1

#         S13p <- S
#         S13n <- S
#         S13p[1,3] <- +1
#         S13n[1,3] <- -1
#         S13p[3,1] <- +1
#         S13n[3,1] <- -1

#         S14p <- S
#         S14n <- S
#         S14p[1,4] <- +1
#         S14n[1,4] <- -1
#         S14p[4,1] <- +1
#         S14n[4,1] <- -1

#         S23p <- S
#         S23n <- S
#         S23p[2,3] <- +1
#         S23n[2,3] <- -1
#         S23p[3,2] <- +1
#         S23n[3,2] <- -1

#         S24p <- S
#         S24n <- S
#         S24p[2,4] <- +1
#         S24n[2,4] <- -1
#         S24p[4,2] <- +1
#         S24n[4,2] <- -1

#         S34p <- S
#         S34n <- S
#         S34p[3,4] <- +1
#         S34n[3,4] <- -1
#         S34p[4,3] <- +1
#         S34n[4,3] <- -1

#         cdf112p <- cdf4d_pg(x, S12p)
#         cdf112n <- cdf4d_pg(x, S12n)
#         cdf113p <- cdf4d_pg(x, S13p)
#         cdf113n <- cdf4d_pg(x, S13n)
#         cdf114p <- cdf4d_pg(x, S14p)
#         cdf114n <- cdf4d_pg(x, S14n)
#         cdf123p <- cdf4d_pg(x, S23p)
#         cdf123n <- cdf4d_pg(x, S23n)
#         cdf124p <- cdf4d_pg(x, S24p)
#         cdf124n <- cdf4d_pg(x, S24n)
#         cdf134p <- cdf4d_pg(x, S34p)
#         cdf134n <- cdf4d_pg(x, S34n)

#         cdf212p <- cdf4d_genz(x, S12p)
#         cdf212n <- cdf4d_genz(x, S12n)
#         cdf213p <- cdf4d_genz(x, S13p)
#         cdf213n <- cdf4d_genz(x, S13n)
#         cdf214p <- cdf4d_genz(x, S14p)
#         cdf214n <- cdf4d_genz(x, S14n)
#         cdf223p <- cdf4d_genz(x, S23p)
#         cdf223n <- cdf4d_genz(x, S23n)
#         cdf224p <- cdf4d_genz(x, S24p)
#         cdf224n <- cdf4d_genz(x, S24n)
#         cdf234p <- cdf4d_genz(x, S34p)
#         cdf234n <- cdf4d_genz(x, S34n)

#         diff12p <- abs(cdf112p - cdf212p)
#         diff13p <- abs(cdf113p - cdf213p)
#         diff14p <- abs(cdf114p - cdf214p)
#         diff23p <- abs(cdf123p - cdf223p)
#         diff24p <- abs(cdf124p - cdf224p)
#         diff34p <- abs(cdf134p - cdf234p)

#         diff12n <- abs(cdf112n - cdf212n)
#         diff13n <- abs(cdf113n - cdf213n)
#         diff14n <- abs(cdf114n - cdf214n)
#         diff23n <- abs(cdf123n - cdf223n)
#         diff24n <- abs(cdf124n - cdf224n)
#         diff34n <- abs(cdf134n - cdf234n)

#         tol <- 1e-3

#         expect_lt(diff12p, tol)
#         expect_lt(diff13p, tol)
#         expect_lt(diff14p, tol)
#         expect_lt(diff23p, tol)
#         expect_lt(diff24p, tol)
#         expect_lt(diff34p, tol)

#         expect_lt(diff12n, tol)
#         expect_lt(diff13n, tol)
#         expect_lt(diff14n, tol)
#         expect_lt(diff23n, tol)
#         expect_lt(diff24n, tol)
#         expect_lt(diff34n, tol)
#     }
# })
