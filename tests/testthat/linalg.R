library("testthat")
context("Test hard-coded linear algebra")

test_that("Determinant of 4x4 matrices", {
    set.seed(123)
    for (i in 1:1000) {
        S <- matrix(rnorm(16), nrow=4, ncol=4)
        S <- crossprod(S)
        if (det(S) <= 0) {
            S <- S + 1e-5 * diag(4)
        }
        std <- 1 / sqrt(diag(S))
        S <- S <- diag(std) %*% S %*% diag(std)

        det1 <- .Call(R_determinant_4by4, S)
        det2 <- det(S)
        diff <- abs(det1 - det2)
        expect_lt(diff, 1e-15)
    }
})

test_that("Inverse of 4x4 matrices", {
    set.seed(123)
    for (i in 1:1000) {
        S <- matrix(rnorm(16), nrow=4, ncol=4)
        S <- crossprod(S)
        if (det(S) <= 0) {
            S <- S + 1e-5 * diag(4)
        }
        std <- 1 / sqrt(diag(S))
        S <- S <- diag(std) %*% S %*% diag(std)
        if (det(S) <= 0.05) next

        li1 <- .Call(R_inverse_4by4_loweronly, S)
        i2 <- solve(S)
        i3 <- i2
        i3[3:4,3:4] <- li1
        resid <- as.numeric(S %*% i3 - diag(4))
        resid <- resid %*% resid
        expect_lt(resid, 1e-8)
    }
})
