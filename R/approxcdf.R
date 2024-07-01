#' @useDynLib approxcdf, .registration=TRUE
NULL

#' @export
#' @title Cumulative Distribution Function for Multivariate Normal Distribution
#' @description Computes an approximation to the CDF (cumulative distribution function)
#' of the multivariate normal (MVN) distribution, defined as the probability that each
#' variable will take a value lower than a desired threshold - e.g. if there are three
#' variables, then \eqn{\text{CDF}(q_1, q_2, q_3, \mu, \Sigma) = P(x_1 < q_1, x_2 < q_2, x_3 < q_3)} given
#' that \eqn{x_1, x_2, x_3} are random variables following a MVN distribution parameterized by
#' \eqn{\mu} (variable means) and \eqn{\Sigma} (covariance).
#' 
#' This could be seen as a much faster version of the function `pmvnorm` in the `mvtnorm` package, but
#' less precise and supporting only the `upper` bounds argument.
#' @details Note that this function does not handle cases in which one or more of the bounds is/are infinite.
#' 
#' The method used for the calculation will depend on the dimensionality (number of variables)
#' of the problem:
#' \itemize{
#' \item For \eqn{n \geq 5}, it will use the TVBS method (two-variate bivariate screening), which makes
#' iterative approximations by calculating the CDF for two variables at a time and then conditioning the
#' remaining variables after assuming that the earlier variables already took values below their thresholds.
#' Note that the implementation here differs from the author's in a few ways: (a) instead of sorting the
#' variables by the bounds/thresholds, it sorts them iteratively by applying univariate conditioning (this
#' is referred to as the "GGE order"in the references), (b) it does not truncate the thresholds (author
#' truncates thresholds to \eqn{-6} if they are lower than that), (c) given that it does not apply
#' truncation, it instead applies regularization to matrices with low determinants that are problematic to
#' invert, (d) while the author applies the same approximation technique until reducing the problem down to
#' a 1D CDF, the implementation here instead uses more exact methods for 2D and 3D CDF subproblems, adapted
#' from Genz's TVPACK library.
#' \item For \eqn{n = 4}, it will use Bhat's method, but given that there's less than 5 variables, it
#' differs a bit from TVBS since it uses fewer variables at a time. If the determinant of the correlation
#' matrix is too low, it will instead switch to Plackett's original method. Bhat's method as implemented
#' here again differs from the original in the same ways as TVBS, and Plackett's method as implemented
#' here also differs from the original in that (a) it applies regularization to correlation matrices with
#' low determinant, (c) it uses more Gaussian-Legendre points for evaluation (author's number was meant
#' for a paper-and-pen calculation).
#' \item For \eqn{n = 3}, it will use Plackett-Drezner's method. The implementation here differs from the
#' author's original in that it uses more Gauss-Legendre points.
#' \item For \eqn{n = 2}, it will use Drezner's method, again with more Gaussian-Legendre points.
#' }
#' 
#' Depending on the BLAS and LAPACK libraries being used by your R setup, it is recommended for better
#' speed to limit the number of BLAS threads to 1 if the library doesn't automatically determine when to
#' use multi-threading (MKL does for example, while not all OpenBLAS versions will), which can be done
#' through the package `RhpcBLASctl`.
#' @references \itemize{
#' \item Bhat, Chandra R.
#' "New matrix-based methods for the analytic evaluation of the multivariate cumulative normal distribution function."
#' Transportation Research Part B: Methodological 109 (2018): 238-256.
#' \item Plackett, Robin L. "A reduction formula for normal multivariate integrals." Biometrika 41.3/4 (1954): 351-360.
#' \item Gassmann, H. I.
#' "Multivariate normal probabilities: implementing an old idea of Plackett's."
#' Journal of Computational and Graphical Statistics 12.3 (2003): 731-752.
#' \item Drezner, Zvi, and George O. Wesolowsky.
#' "On the computation of the bivariate normal integral."
#' Journal of Statistical Computation and Simulation 35.1-2 (1990): 101-107.
#' \item Drezner, Zvi. "Computation of the trivariate normal integral." Mathematics of Computation 62.205 (1994): 289-294.
#' \item Genz, Alan.
#' "Numerical computation of rectangular bivariate and trivariate normal and t probabilities."
#' Statistics and Computing 14.3 (2004): 251-260.
#' \item Gibson, Garvin Jarvis, C. A. Glasbey, and D. A. Elston.
#' "Monte Carlo evaluation of multivariate normal integrals and sensitivity to variate ordering."
#' Advances in Numerical Methods and Applications (1994): 120-126.
#' }
#' @param q Thresholds for the calculation (upper bound on the variables for which the CDF is to be calculated).
#' 
#' Note that infinites are not supported.
#' @param Cov Covariance matrix. If passing `is_standardized=TRUE`, will assume that it is a correlation matrix.
#' Being a covariance or correlation matrix, should have a non-negative determinant.
#' @param mean Means (expected values) of each variable. If passing `NULL`, will assume that all means are zero.
#' Cannot pass it when using `is_standardized=TRUE`.
#' @param is_standardized Whether the parameters correspond to a standardized MVN distribution - that is, the
#' means for all variables are zero, and `Cov` has only ones on its diagonal.
#' @param log.p Whether to return the logarithm of the probability instead of the probability itself.
#' This is helpful when dealing with very small probabilities, since they
#' might get rounded down to exactly zero in their raw form or not be representable
#' very exactly as float64, while the logarithm is likely still well defined.
#' Note that it will use slightly different methods here (all calculations, even
#' for 2D and 3D will use uni/bi-variate screening methods) and thus exponentiating
#' this result might not match exactly with the result obtained with `log.p=FALSE`.
#' @returns The (approximate) probability that each variable drawn from the MVN distribution parameterized by
#' `Cov` and `mean` will be lower than each corresponding threshold in `q`. If for some reason the calculation
#' failed, will return NaN.
#' @examples 
#' library(approxcdf)
#' 
#' ### Example from Plackett's paper
#' b <- c(0, 0, 0, 0)
#' S <- matrix(
#' c( 1,  -0.60,  0.85,  0.75,
#' -0.60,    1,  -0.70, -0.80,
#'  0.85, -0.70,    1,   0.65,
#'  0.75, -0.80,  0.65,    1),
#' nrow=4, ncol=4, byrow=TRUE)
#' pmvn(b, S, is_standardized=TRUE)
#' ### (Plackett's estimate was 0.042323)
#' 
#' ### Compare against Genz's more exact method:
#' if (require(mvtnorm)) {
#'     set.seed(123)
#'     p_Genz <- pmvnorm(upper=b, corr=S)
#'     abs(p_Genz - pmvn(b, S, is_standardized=TRUE))
#' }
#' ### (Result should not be too far from Genz's or Plackett's,
#' ###  but should be around 50-100x faster to calculate, and this
#' ###  speed up should increase to around 1,000x for larger 'n')
pmvn <- function(q, Cov, mean = NULL, is_standardized = FALSE, log.p = FALSE) {
    if (!is.matrix(Cov)) {
        stop("'Cov' must be a numeric matrix.")
    }
    if (nrow(Cov) != ncol(Cov)) {
        stop("Covariance matrix must have square shape.")
    }
    if (length(q) != nrow(Cov)) {
        stop("Dimensions of 'q' and 'Cov' do not match.")
    }
    if (!is.null(mean)) {
        if (length(mean) != length(q)) {
            stop("'q' and 'mean' must have the same length.")
        }
        if (anyNA(mean)) {
            stop("Cannot pass missing values in parameters.")
        }
    }

    if (is_standardized && !is.null(mean)) {
        stop("Cannot pass 'mean' when using 'is_standardized=TRUE'.")
    }
    if (anyNA(q) || anyNA(Cov)) {
        stop("Cannot pass missing values in parameters.")
    }
    if (any(is.infinite(q)) || any(is.infinite(Cov))) {
        stop("Cannot pass infinite values in parameters.")
    }

    return(.Call(R_norm_cdf_tvbs, q, Cov, mean, is_standardized, log.p))
}

#' @export
#' @title Cumulative Distribution Function for Quadrivariate Normal Distribution
#' @description Calculates the CDF (cumulative distribution function) of a
#' 4D/quadrivariate standardized normal distribution using Plackett's or Plackett-Gassmann's
#' reduction, aided by the more exact 3D and 2D CDF methods from Genz (adapted from the TVPACK
#' library) for lower-dimensional subproblems.
#' 
#' In general, this method is both slower and less precise than Bhat's method as used by \link{pmvn},
#' and is provided for experimentation purposes only.
#' @details The implementation here differs from Gassmann's paper in a few ways:
#' \itemize{
#' \item It prefers the reference matrix number 2 in Gassmann's classification, unless some
#' correlation coefficients are zero or one, in which case it will prefer matrix number 5
#' (which was Gassmann's recommendation).
#' \item If the determinant of the correlation matrix is very low, it will prefer instead
#' Gassmann's matrix number 7 (Plackett's original choice).
#' \item When using reference matrices 2 or 5, it will make the probability corrections
#' in a recursive fashion, zeroing out one correlation coefficient at a time, instead of making
#' corrections for all correlations in aggregate. From some experiments, this turned out to result
#' in slower but more accurate calculations when correcting for multiple correlations.
#' \item The number of Gaussian-Legendre points used here is higher than in Plackett's, but
#' lower than in Gassmann's.
#' }
#' 
#' Although Gassmann's paper suggested that this method should have a maximum error of \eqn{5 \times 10^{-5}}
#' in a suite of typical test problems, some testing with random matrices (typically not representative
#' of problems of interest due to their structure) shows that the average error to expect is around
#' \eqn{10^{-3}} and the maximum error can be as bad as \eqn{10^{-1}}.
#' 
#' Note that this function will not perform any validation of the data that is passed - it is
#' the user's responsibility to ensure that the provided arguments are correct.
#' @references \itemize{
#' \item Plackett, Robin L. "A reduction formula for normal multivariate integrals." Biometrika 41.3/4 (1954): 351-360.
#' \item Gassmann, H. I.
#' "Multivariate normal probabilities: implementing an old idea of Plackett's."
#' Journal of Computational and Graphical Statistics 12.3 (2003): 731-752.
#' \item Genz, Alan.
#' "Numerical computation of rectangular bivariate and trivariate normal and t probabilities."
#' Statistics and Computing 14.3 (2004): 251-260.
#' }
#' @param q A 4-dimensional vector with the thresholds/upper bounds for the variables for which the CDF
#' will be calculated.
#' @param Rho The correlation matrix parameterizing the distribution.
#' @param prefer_original Whether to prefer Plackett's original reduction form (reference matrix number
#' 7 in Gassmann's classification) regardless of the determinant.
#' @return The CDF calculated through Plackett's recursive reduction for the quadrivariate normal
#' distribution and thresholds provided here.
#' @examples 
#' library(approxcdf)
#' 
#' ### Example from Plackett's paper
#' b <- c(0, 0, 0, 0)
#' S <- matrix(
#' c( 1,  -0.60,  0.85,  0.75,
#' -0.60,    1,  -0.70, -0.80,
#'  0.85, -0.70,    1,   0.65,
#'  0.75, -0.80,  0.65,    1),
#' nrow=4, ncol=4, byrow=TRUE)
#' 
#' ### Solution using Plackett's original reduction
#' pqvn(b, S, prefer_original=TRUE)
#' ### (Plackett's estimate was 0.042323)
pqvn <- function(q, Rho, prefer_original=FALSE) {
    if (nrow(Rho) != ncol(Rho)) {
        stop("Correlation matrix must have square shape.")
    }
    if (length(q) != nrow(Rho)) {
        stop("Dimensions of 'q' and 'Rho' do not match.")
    }
    if (!prefer_original) {
        return(.Call(R_norm_cdf_4d_pg, q, Rho))
    } else {
        return(.Call(R_norm_cdf_4d_pg7, q, Rho))
    }
}

#' @export
#' @title Cumulative Distribution Function for Bivariate Normal Distribution
#' @description Calculates the CDF (cumulative distribution function) of a
#' 2D/bivariate standardized normal distribution using a fast approximation
#' based on the 'erf' function.
#' 
#' This is faster than the more exact method from Drezner used for 2D problems
#' in \link{pmvn}, but it is much less precise (error can be as high as \eqn{10^{-3}}).
#' @details Note that this function will not perform any input validation. It is up to
#' the user to check that the data passed here is valid.
#' @references Tsay, Wen-Jen, and Peng-Hsuan Ke.
#' "A simple approximation for the bivariate normal integral."
#' Communications in Statistics-Simulation and Computation (2021): 1-14.
#' @param q1 Threshold or upper bound for the first variable.
#' @param q2 Threshold or upper bound for the second variable.
#' @param rho Correlation between the two variables (must be between -1 and +1).
#' @return The CDF (probability that both variables drawn from a standardized bivariate
#' normal distribution parameterized by `rho` will be lower than the corresponding values
#' in `q1` and `q2`).
#' @examples 
#' library(approxcdf)
#' 
#' ### Short example problem
#' q1 <- 0.25
#' q2 <- -0.25
#' rho <- 0.5
#' 
#' ### Calculate an approximate CDF for this distribution
#' pbvn(q1, q2, rho)
#' 
#' ### Compare against Genz's more precise method
#' ### (exact to more than 8 decimal places)
#' if (require(mvtnorm)) {
#'     pmvnorm(upper=c(q1,q2), corr=matrix(c(1,rho,rho,1), nrow=2), algorithm=TVPACK())
#' }
pbvn <- function(q1, q2, rho) {
    return(.Call(R_norm_cdf_2d_vfast, q1, q2, rho))
}
