#include "approxcdf.h"

/* https://bugs.r-project.org/show_bug.cgi?id=15620 */
double norm_pdf_1d(double x)
{
    if (likely(x < 5. && x > -5.)) {
        return inv_sqrt2_pi * std::exp(-0.5 * x * x);
    }
    else {
        double x1 = std::floor(x * 0x1.0p16 + 0.5) * 0x1.0p-16;
        double x2 = x - x1;
        return inv_sqrt2_pi * (std::exp(-0.5 * x1 * x1) * std::exp((-0.5 * x2 - x1) * x2));
    }
}

/* Adapted from cephes' 'ndtr' */
double norm_cdf_1d(double x)
{
    /* This is technically correct but very imprecise:  
    return 0.5 * std::erfc(neg_inv_sqrt2 * x);
    */
    double x_, y, z;
    x_ = x * inv_sqrt2;
    z = std::fabs(x_);

    /* if( z < SQRTH ) */
    // if (z < 1.) {
    if (z < inv_sqrt2) {
        y = .5 + .5*std::erf(x_);
    }
    else {
        y = .5*std::erfc(z);
        if(x_ > 0.) {
            y = 1. - y;
        }
    }
    return y;
}

double norm_lcdf_1d(double x)
{
    /* This is technically correct but very imprecise:
    return 0.5 * std::erfc(inv_sqrt2 * x);
    */
    return norm_cdf_1d(-x);
}
