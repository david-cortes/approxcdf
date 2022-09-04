#include "approxcdf.h"

/* https://bugs.r-project.org/show_bug.cgi?id=15620 */
double norm_pdf_1d(double x)
{
    if (std::isinf(x)) return 0.;
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
    if (std::isinf(x)) {
        return (x >= 0.)? 1. : 0.;
    }
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

/* Adapted from SciPy:
   https://github.com/scipy/scipy/blob/main/scipy/stats/_continuous_distns.py */
double norm_logpdf_1d(double x)
{
    if (std::isinf(x)) return -std::numeric_limits<double>::infinity();
    return -0.5 * (x*x) - log_sqrt_twoPI;
}

/* Adapted from SciPy:
   https://github.com/scipy/scipy/blob/8a64c938ddf1ae4c02a08d2c5e38daeb8d061d38/scipy/special/cephes/ndtr.c */
double norm_logcdf_1d(double a)
{
    if (std::isinf(a)) {
        return (a >= 0.)? 0. : -std::numeric_limits<double>::infinity();
    }
    const double a_sq = a * a;
    double log_LHS;              /* we compute the left hand side of the approx (LHS) in one shot */
    double last_total = 0;       /* variable used to check for convergence */
    double right_hand_side = 1;  /* includes first term from the RHS summation */
    double numerator = 1;        /* numerator for RHS summand */
    double denom_factor = 1;     /* use reciprocal for denominator to avoid division */
    double denom_cons = 1./a_sq; /* the precomputed division we use to adjust the denominator */
    long sign = 1;
    long i = 0;

    if (a > 6.) {
        return -norm_cdf_1d(-a);        /* log(1+x) \approx x */
    }
    if (a > -20.) {
        return std::log(norm_cdf_1d(a));
    }
    log_LHS = -0.5*a_sq - std::log(-a) - half_log_twoPI;

    while (std::fabs(last_total - right_hand_side) > std::numeric_limits<double>::epsilon()) {
        i++;
        last_total = right_hand_side;
        sign = -sign;
        denom_factor *= denom_cons;
        numerator *= 2 * i - 1;
        right_hand_side = std::fma(sign*numerator, denom_factor, right_hand_side);
    }
    
    return log_LHS + std::log(right_hand_side);
}
