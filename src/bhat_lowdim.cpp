#include "approxcdf.h"

double norm_logcdf_2d(double x1, double x2, double rho)
{
    double abs_rho = std::fabs(rho);
    if (unlikely(abs_rho <= LOW_RHO)) {
        return norm_logcdf_1d(x1) + norm_logcdf_1d(x2);
    }
    else if (unlikely(abs_rho >= HIGH_RHO)) {
        if (rho >= 0.) {
            return norm_logcdf_1d(std::fmin(x1, x2));
        }
        else {
            return norm_logcdf_1d(std::fmin(x1, -x2));
        }
    }
    if (x2 < x1) {
        std::swap(x1, x2);
    }
    double log_d1 = norm_logpdf_1d(x1);
    double log_p1 = norm_logcdf_1d(x1);
    double log_l1 = log_d1 - log_p1;
    double sign_l1 = -1.;
    double log_rho = std::log(std::fabs(rho));
    double sign_rho = (rho >= 0.)? 1. : -1.;
    double log_rl1 = log_rho + log_l1;
    double sign_rl1 = sign_rho * sign_l1;
    double log_x1 = std::log(std::fabs(x1));
    double sign_x1 = (x1 >= 0.)? 1. : -1.;

    double v2 = sign_rho * sign_x1 * sign_rl1 * std::exp(log_rho + log_x1 + log_rl1);
    double rl1 = sign_rl1 * std::exp(log_rl1);
    v2 += std::fma(-rl1, rl1, 1.);
    v2 = std::fmax(v2, std::numeric_limits<double>::min());

    return norm_logcdf_1d(x1) + norm_logcdf_1d((x2 - rl1) / std::sqrt(v2));
}

double norm_logcdf_3d(double x1, double x2, double x3, double rho12, double rho13, double rho23)
{
    if (unlikely(rho12*rho12 + rho13*rho13 <= EPS_BLOCK)) {
        return norm_logcdf_1d(x1) + norm_logcdf_2d(x2, x3, rho23);
    }
    else if (unlikely(rho12*rho12 + rho23*rho23 <= EPS_BLOCK)) {
        return norm_logcdf_1d(x2) + norm_logcdf_2d(x1, x3, rho13);
    }
    else if (unlikely(rho13*rho13 + rho23*rho23 <= EPS_BLOCK)) {
        return norm_logcdf_1d(x3) + norm_logcdf_2d(x1, x2, rho12);
    }

    if (x3 < x2) {
        std::swap(x2, x3);
        std::swap(rho12, rho13);
    }
    if (x2 < x1) {
        std::swap(x1, x2);
        std::swap(rho13, rho23);
    }
    
    double temp = norm_logpdf_1d(x1) - norm_logcdf_1d(x1);
    double mutilde = -std::exp(temp);
    double omega = 1. + (mutilde * (x1 - mutilde));

    double rho12_sq = rho12 * rho12;
    double rho13_sq = rho13 * rho13;
    double omega_m1 = omega - 1.;

    double t1 = std::fma(rho12_sq, omega_m1, 1.);
    t1 = std::fmax(t1, std::numeric_limits<double>::min());
    double t2 = std::fma(rho13_sq, omega_m1, 1.);
    t2 = std::fmax(t2, std::numeric_limits<double>::min());

    double s11 = std::sqrt(t1);
    double s22 = std::sqrt(t2);
    double v12 = rho23 + rho12 * rho13 * omega_m1;

    double p1 = norm_logcdf_2d(x1, x2, rho12);
    double p2 = norm_logcdf_2d(
        std::fma(-rho12, mutilde, x2) / s11,
        std::fma(-rho13, mutilde, x3) / s22,
        v12 / (s11 * s22)
    );
    double p3 = norm_logcdf_1d(std::fma(-rho12, mutilde, x2) / s11);
    return p1 + p2 - p3;
}
