library("testthat")
context("Test LDLT factorization")

test_that("Special LDLT factorization", {
    set.seed(123)
    for (n in 5:10) {
        for (i in 1:50) {
            S <- matrix(rnorm(n*n), nrow=n, ncol=n)
            S <- crossprod(S)
            if (det(S) <= 0) {
                S <- S + 1e-8*diag(n)
            }

            factorized <- .Call(R_factorize_ldl_2by2blocks, S)
            L <- factorized[[1]]
            D <- factorized[[2]]
            Sf <- L %*% D %*% t(L)
            diff <- max(abs(S - Sf))
            # expect_lt(diff, 1e-13) # <- without regularization
            expect_lt(diff, 1e-4) # <- with regularization

            expect_equal(L[upper.tri(L)], numeric(n*(n-1)/2))
            expect_equal(diag(L), rep(1, n))
            for (idx in seq(2, n, 2)) {
                expect_equal(L[idx, idx-1], 0)
            }

            Dzero <- D
            for (idx in seq(1, n, 2)) {
                rc <- seq(idx, min(idx+1, n), 1)
                block <- D[rc, rc]
                Dzero[rc, rc] <- Dzero[rc, rc] - block
                if (!is.matrix(block)) {
                    expect_gt(abs(block), 0)
                } else {
                    expect_gt(norm(block, type="F"), 0)
                }
            }
            expect_equal(max(abs(Dzero)), 0)
        }
    }
})
