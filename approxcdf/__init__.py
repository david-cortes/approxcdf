import numpy as np
from . import _cpp_wrapper
from typing import Optional

__all__ = ["mvn_cdf", "bvn_cdf", "qvn_cdf"]

def mvn_cdf(b: np.ndarray, Cov: np.ndarray, mean: Optional[np.ndarray] = None, is_standardized: bool = False, logp: bool = False):
    """
    Cumulative Distribution Function for Multivariate Normal Distribution

    Computes an approximation to the CDF (cumulative distribution function)
    of the multivariate normal (MVN) distribution, defined as the probability that each
    variable will take a value lower than a desired threshold - e.g. if there are three
    variables, then :math:`\\text{CDF}(q_1, q_2, q_3, \\mu, \\Sigma) = P(x_1 < q_1, x_2 < q_2, x_3 < q_3)` given
    that :math:`x_1, x_2, x_3` are random variables following a MVN distribution parameterized by
    :math:`\\mu` (variable means) and :math:`\\Sigma` (covariance).

    This could be seen as a much faster version of ``scipy.stats.multivariate_normal.cdf``, but
    less precise.

    Notes
    -----
    The method used for the calculation will depend on the dimensionality (number of variables)
    of the problem:
        - For :math:`n \\geq 5`, it will use the TVBS method (two-variate bivariate screening), which makes
          iterative approximations by calculating the CDF for two variables at a time and then conditioning the
          remaining variables after assuming that the earlier variables already took values below their thresholds.
          Note that the implementation here differs from the author's in a few ways: (a) instead of sorting the
          variables by the bounds/thresholds, it sorts them iteratively by applying univariate conditioning (this
          is referred to as the "GGE order"in the references), (b) it does not truncate the thresholds (author
          truncates thresholds to :math:`-6` if they are lower than that), (c) given that it does not apply
          truncation, it instead applies regularization to matrices with low determinants that are problematic to
          invert, (d) while the author applies the same approximation technique until reducing the problem down to
          a 1D CDF, the implementation here instead uses more exact methods for 2D and 3D CDF subproblems, adapted
          from Genz's TVPACK library.
        - For :math:`n = 4`, it will use Bhat's method, but given that there's less than 5 variables, it
          differs a bit from TVBS since it uses fewer variables at a time. If the determinant of the correlation
          matrix is too low, it will instead switch to Plackett's original method. Bhat's method as implemented
          here again differs from the original in the same ways as TVBS, and Plackett's method as implemented
          here also differs from the original in that (a) it applies regularization to correlation matrices with
          low determinant, (c) it uses more Gaussian-Legendre points for evaluation (author's number was meant
          for a paper-and-pen calculation).
        - For :math:`n = 3`, it will use Plackett-Drezner's method. The implementation here differs from the
          author's original in that it uses more Gauss-Legendre points.
        - For :math:`n = 2`, it will use Drezner's method, again with more Gaussian-Legendre points.

    Note
    ----
    Depending on the BLAS and LAPACK libraries being used by your setup, it is recommended for better
    speed to limit the number of BLAS threads to 1 if the library doesn't automatically determine when to
    use multi-threading (MKL does for example, while not all OpenBLAS versions will), which can be done
    through the package ``threadpoolctl``.

    Parameters
    ----------
    b : array(n,)
        Thresholds for the calculation (upper bound on the variables for which the CDF is to be calculated).
    Cov : array(n, n)
        Covariance matrix. If passing ``is_standardized=True``, will assume that it is a correlation matrix.
        Being a covariance or correlation matrix, should have a non-negative determinant.
    mean : None or array(n,)
        Means (expected values) of each variable. If passing ``None``, will assume that all means are zero.
        Cannot pass it when using ``is_standardized=True``.
    is_standardized : bool
        Whether the parameters correspond to a standardized MVN distribution - that is, the
        means for all variables are zero, and ``Cov`` has only ones on its diagonal.
    logp : bool
        Whether to return the logarithm of the probability instead of the probability itself.
        This is helpful when dealing with very small probabilities, since they
        might get rounded down to exactly zero in their raw form or not be representable
        very exactly as float64, while the logarithm is likely still well defined.
        Note that it will use slightly different methods here (all calculations, even
        for 2D and 3D will use uni/bi-variate screening methods) and thus exponentiating
        this result might not match exactly with the result obtained with ``logp=False``.

    Returns
    -------
    cdf : float
        The (approximate) probability that each variable drawn from the MVN distribution parameterized by
        ``Cov`` and ``mean`` will be lower than each corresponding threshold in ``b``. If for some reason the calculation
        failed, will return NaN.

    References
    ----------
    .. [1] Bhat, Chandra R.
           "New matrix-based methods for the analytic evaluation of the multivariate cumulative normal distribution function."
           Transportation Research Part B: Methodological 109 (2018): 238-256.
    .. [2] Plackett, Robin L. "A reduction formula for normal multivariate integrals." Biometrika 41.3/4 (1954): 351-360.
    .. [3] Gassmann, H. I.
           "Multivariate normal probabilities: implementing an old idea of Plackett's."
           Journal of Computational and Graphical Statistics 12.3 (2003): 731-752.
    .. [4] Drezner, Zvi, and George O. Wesolowsky.
           "On the computation of the bivariate normal integral."
           Journal of Statistical Computation and Simulation 35.1-2 (1990): 101-107.
    .. [5] Drezner, Zvi. "Computation of the trivariate normal integral." Mathematics of Computation 62.205 (1994): 289-294.
    .. [6] Genz, Alan.
           "Numerical computation of rectangular bivariate and trivariate normal and t probabilities."
           Statistics and Computing 14.3 (2004): 251-260.
    .. [7] Gibson, Garvin Jarvis, C. A. Glasbey, and D. A. Elston.
           "Monte Carlo evaluation of multivariate normal integrals and sensitivity to variate ordering."
           Advances in Numerical Methods and Applications (1994): 120-126.
    """
    if (len(Cov.shape) != 2) or (Cov.shape[0] != Cov.shape[1]):
        raise ValueError("Covariance matrix must have square shape.")
    if (len(b.shape) != 1) or (b.shape[0] != Cov.shape[1]):
        raise ValueError("Dimensions of 'q' and 'Cov' do not match.")

    if b.strides[0] != b.itemsize:
        b = np.ascontiguousarray(b)
    if mean is None:
        mean = np.array([], dtype=np.float64)
    else:
        if is_standardized:
            raise ValueError("Cannot pass 'mean' when using 'is_standardized=True'.")
        if (len(mean.shape) != 1) or mean.shape[0] != Cov.shape[0]:
            raise ValueError("Dimensions of 'mean' and 'Cov' do not match.")
        if mean.strides[0] != mean.itemsize:
            mean = np.ascontiguousarray(mean)
        if np.any(np.isnan(mean)):
            raise ValueError("Cannot pass missing values in parameters.")
    if Cov.strides[1] != Cov.itemsize:
        Cov = np.ascontiguousarray(Cov)
    ld_Cov = int(Cov.strides[0] / Cov.itemsize)

    if np.any(np.isnan(b)) or np.any(np.isnan(Cov)):
        raise ValueError("Cannot pass missing values in parameters.")

    return _cpp_wrapper.py_norm_cdf_tvbs(b, mean, Cov, ld_Cov, is_standardized, logp)


def bvn_cdf(b1: float, b2: float, rho: float):
    """
    Cumulative Distribution Function for Bivariate Normal Distribution

    Calculates the CDF (cumulative distribution function) of a
    2D/bivariate standardized normal distribution using a fast approximation
    based on the 'erf' function.

    This is faster than the more exact method from Drezner used for 2D problems
    in function ``mvn_cdf``, but it is much less precise (error can be as high as :math:`10^{-3}`).

    Note
    ----
    This function will not perform any input validation. It is up to
    the user to check that the data passed here is valid.

    Parameters
    ----------
    b1 : float
        Threshold or upper bound for the first variable.
    b2 : float
        Threshold or upper bound for the second variable.
    rho : float
        Correlation between the two variables (must be between -1 and +1).

    Returns
    -------
    cdf : float
        The CDF (probability that both variables drawn from a standardized bivariate
        normal distribution parameterized by `rho` will be lower than the corresponding values
        in ``b1`` and ``b2``).

    References
    ----------
    .. [1bv] Tsay, Wen-Jen, and Peng-Hsuan Ke.
             "A simple approximation for the bivariate normal integral."
             Communications in Statistics-Simulation and Computation (2021): 1-14.
    """
    return _cpp_wrapper.py_norm_cdf_2d_vfast(b1, b2, rho)

def qvn_cdf(b: np.ndarray, Rho: np.ndarray, prefer_original: bool = False):
    """
    Cumulative Distribution Function for Quadrivariate Normal Distribution

    Calculates the CDF (cumulative distribution function) of a
    4D/quadrivariate standardized normal distribution using Plackett's or Plackett-Gassmann's
    reduction, aided by the more exact 3D and 2D CDF methods from Genz (adapted from the TVPACK
    library) for lower-dimensional subproblems.
    
    In general, this method is both slower and less precise than Bhat's method as used by function ``mvn_cdf``,
    and is provided for experimentation purposes only.

    Note
    ----
    The implementation here differs from Gassmann's paper in a few ways:
        - It prefers the reference matrix number 2 in Gassmann's classification, unless some
          correlation coefficients are zero or one, in which case it will prefer matrix number 5
          (which was Gassmann's recommendation).
        - If the determinant of the correlation matrix is very low, it will prefer instead
          Gassmann's matrix number 7 (Plackett's original choice).
        - When using reference matrices 2 or 5, it will make the probability corrections
          in a recursive fashion, zeroing out one correlation coefficient at a time, instead of making
          corrections for all correlations in aggregate. From some experiments, this turned out to result
          in slower but more accurate calculations when correcting for multiple correlations.
        - The number of Gaussian-Legendre points used here is higher than in Plackett's, but
          than in Gassmann's.

    Although Gassmann's paper suggested that this method should have a maximum error of :math:`5 \\times 10^{-5}`
    in a suite of typical test problems, some testing with random matrices (typically not representative
    of problems of interest due to their structure) shows that the average error to expect is around
    :math:`10^{-3}` and the maximum error can be as bad as :math:`10^{-1}`.

    Note
    ----
    This function will not perform any validation of the data that is passed - it is
    the user's responsibility to ensure that the provided arguments are correct.

    Parameters
    ----------
    b : array(4,)
        A 4-dimensional vector with the thresholds/upper bounds for the variables for which the CDF
        will be calculated. Must have ``float64`` type.
    Rho : array(4, 4)
        The correlation matrix parameterizing the distribution. Must have ``float64`` type.
    prefer_original : bool
        Whether to prefer Plackett's original reduction form (reference matrix number
        7 in Gassmann's classification) regardless of the determinant.

    Returns
    -------
    cdf : float
        The CDF calculated through Plackett's recursive reduction for the quadrivariate normal
        distribution and thresholds provided here.

    References
    ----------
    .. [1qv] Plackett, Robin L. "A reduction formula for normal multivariate integrals." Biometrika 41.3/4 (1954): 351-360.
    .. [2qv] Gassmann, H. I.
             "Multivariate normal probabilities: implementing an old idea of Plackett's."
             Journal of Computational and Graphical Statistics 12.3 (2003): 731-752.
    .. [3qv] Genz, Alan.
             "Numerical computation of rectangular bivariate and trivariate normal and t probabilities."
             Statistics and Computing 14.3 (2004): 251-260.
    """
    if (len(Rho.shape) != 2) or (Rho.shape[0] != Rho.shape[1]):
        raise ValueError("Correlation matrix must have square shape.")
    if b.shape[0] != Rho.shape[0]:
        raise ValueError("Dimensions of 'q' and 'Rho' do not match.")
    if b.strides[0] != b.itemsize:
        b = np.ascontiguousarray(b)
    if not prefer_original:
        return _cpp_wrapper.py_norm_cdf_4d_pg(b, Rho)
    else:
        return _cpp_wrapper.py_norm_cdf_4d_pg7(b, Rho)
