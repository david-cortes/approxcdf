library("testthat")
context("One-dimensional functions")

test_that("CDF 1D", {
    set.seed(123)
    for (i in 1:1000) {
        x <- runif(1, min=-35, max=8)
        cdf1 <- pnorm(x)
        cdf2 <- .Call(R_norm_cdf_1d, x)
        expect_equal(cdf1, cdf2)
    }
})

test_that("PDF 1D", {
    set.seed(123)
    for (i in 1:1000) {
        x <- runif(1, min=-35, max=35)
        pdf1 <- dnorm(x)
        pdf2 <- .Call(R_norm_pdf_1d, x)
        expect_equal(pdf1, pdf2)
    }
})
