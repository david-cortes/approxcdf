#if defined(_WIN32)
    #define APPROXCDF_EXPORTED __declspec(dllimport)
#else
    #define APPROXCDF_EXPORTED 
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Parameters

x : dim=n
    Array with the upper bounds for the CDF calculation (will calculate
    the probability that variables from a random draw will be lower than
    this threshold).
Sigma : dim=n*n
    Covariance or correlation matrix. If passing ld_Sigma larger than 'n',
    then it must be in row-major order.
ld_Sigma
    Leading dimension for Sigma.
    It is assumed that the row 'r' of 'Sigma' starts at position 'r*n' from
    the pointer passed to 'Sigma'.
    Typically this is equal to 'n', but can be larger (e.g. if 'Sigma' is
    a memoryview of a larger array).
mu : dim=n, optional (pass NULL when using is_standardized=true)
    Expected means for the variables. Typically, one works with standardized
    distributions in which the means are all zeros, in which case one should
    pass a NULL pointer here.
n
    Dimensionality of the problem (number of variables).
is_standardized
    Whether the disitribution parameterized here is standardized, meaning:
    - The expected mean for each variable is zero ('mu' should not be passed).
    - The 'Sigma' passed here is a correlation matrix (has values of one at each
      diagonal entry).
    (Value will be interpreted as boolean - i.e. a value of zero means "false",
     any value other than zero means "true").
logp
    Whether to return the logarithm of the probability instead of the probability
    itself. This is helpful when dealing with very small probabilities, since they
    might get rounded down to exactly zero in their raw form or not be representable
    very exactly as float64, while the logarithm is likely still well defined.
    Note that it will use slightly different methods here (all calculations, even
    for 2D and 3D will use uni/bi-variate screening methods) and thus exponentiating
    this result might not match exactly with the result obtained with logp=0.
    (Value will be interpreted as boolean - i.e. a value of zero means "false",
     any value other than zero means "true").
buffer : dim=6*n*n + 6*n - 8
    A temporary memory buffer that will be used for intermediate steps of the
    calculation. If not passed, this memory will be allocated and deallocated
    inside of the function call.

If an error occurs, it will return NAN as result.
*/
APPROXCDF_EXPORTED
double norm_cdf
(
    const double x[],
    const double Sigma[], const int ld_Sigma, /* typically ld_Sigma=n */
    const double mu[], /* ignored when is_standardized=false */
    const int n,
    const char is_standardized,
    const char logp,
    double buffer[] /* dim: 6*n^2 + 6*n - 8 */
);

#ifdef __cplusplus
}
#endif
