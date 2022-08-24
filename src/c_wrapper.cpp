#include "approxcdf.h" 

extern "C" {

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
)
{
    bool is_standardized_ = (bool)is_standardized;

    try {
        return norm_cdf_tvbs(
            x,
            Sigma, ld_Sigma,
            mu,
            n,
            is_standardized,
            logp,
            buffer
        );
    }
    catch (...) {
        return NAN;
    }
}

}
