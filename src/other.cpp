#include "approxcdf.h"

/* Tsay, Wen-Jen, and Peng-Hsuan Ke.
   "A simple approximation for the bivariate normal integral."
   Communications in Statistics-Simulation and Computation (2021): 1-14. */
constexpr const static double c1 = -1.0950081470333;
constexpr const static double c2 = -0.75651138383854;
double norm_cdf_2d_vfast(double x1, double x2, double rho)
{
    if (std::fabs(rho) <= std::numeric_limits<double>::epsilon()) {
        return norm_cdf_1d(x1) * norm_cdf_1d(x2);
    }

    double denom = std::sqrt(1 - rho * rho);
    double a = -rho / denom;
    double b = x1 / denom;
    double aq_plus_b = a*x2 + b;

    if (a > 0) {
        if (aq_plus_b >= 0) {
            double aa = a * a;
            double a_sq_c1 = aa*c1;
            double a_sq_c2 = aa*c2;
            double sqrt2b = sqrt2*b;
            double sqrt2x2 = sqrt2*x2;
            double sqrt_recpr_a_sq_c2 = std::sqrt(1. - a_sq_c2);
            double twicea_sqrt_recpr_a_sq_c2 = 2.*a*sqrt_recpr_a_sq_c2;
            double temp = 1. / (4. * sqrt_recpr_a_sq_c2);
            double t1 = a_sq_c1*c1 + 2.*b*b*c2;
            double t2 = 2.*sqrt2b*c1;
            double t3 = 4. - 4.*a_sq_c2;

            return
                0.5 * (std::erf(x2 / sqrt2) + std::erf(b / (sqrt2*a))) +
                temp
                    * std::exp((t1 - t2) / t3)
                    * (1. - std::erf((sqrt2b - a_sq_c1) / twicea_sqrt_recpr_a_sq_c2)) -
                temp
                    * std::exp((t1 + t2) / t3)
                    * (
                        std::erf((sqrt2x2 - sqrt2x2*a_sq_c2 - sqrt2b*a*c2 - a*c1) / (2.*sqrt_recpr_a_sq_c2)) +
                        std::erf((a_sq_c1 + sqrt2b) / twicea_sqrt_recpr_a_sq_c2)
                    );

        }
        else {
            double sqrt2b = sqrt2*b;
            double sqrt2x2 = sqrt2*x2;
            double a_sq_c2 = a*a*c2;
            double recpr_a_sq_c2 = 1. - a_sq_c2;
            double sqrt_recpr_a_sq_c2 = std::sqrt(recpr_a_sq_c2);
            double a_c1 = a*c1;

            return
                (1. / (4. * sqrt_recpr_a_sq_c2)) *
                std::exp((a_c1*a_c1 - 2.*sqrt2b*c1 + 2*b*b*c2) / (4.*recpr_a_sq_c2)) *
                (1. + std::erf((sqrt2x2 - sqrt2x2*a_sq_c2 - sqrt2b*a*c2 + a_c1) / (2.*sqrt_recpr_a_sq_c2)));
        }
    }
    else {
        if (aq_plus_b >= 0) {
            double sqrt2b = sqrt2*b;
            double a_sq_c2 = a*a*c2;
            double recpr_a_sq_c2 = 1. - a_sq_c2;
            double sqrt_recpr_a_sq_c2 = std::sqrt(recpr_a_sq_c2);
            double a_c1 = a*c1;
            double sqrt2_x2 = sqrt2*x2;

            return
                0.5 + 0.5 * std::erf(x2 / sqrt2) -
                (1. / (4. * sqrt_recpr_a_sq_c2)) *
                std::exp((a_c1*a_c1 + 2.*sqrt2b*c1 + 2.*b*b*c2) / (4.*recpr_a_sq_c2)) *
                (1. + std::erf((sqrt2_x2 - sqrt2_x2*a_sq_c2 - sqrt2b*a*c2 - a_c1) / (2.*sqrt_recpr_a_sq_c2)));
        }
        else {
            double sqrt2a = sqrt2*a;
            double sqrt2b = sqrt2*b;
            double a_sq_c2 = a*a*c2;
            double recpr_a_sq_c2 = 1. - a_sq_c2;
            double sqrt_recpr_a_sq_c2 = std::sqrt(recpr_a_sq_c2);
            double a_c1 = a*c1;
            double temp = 1. / (4. * sqrt_recpr_a_sq_c2);
            double t1 = a_c1*a_c1 + 2.*b*b*c2;
            double t2 = 2.*sqrt2b*c1;
            double t3 = 4.*recpr_a_sq_c2;
            double sqrt2_x2 = sqrt2*x2;

            return
                0.5 - 0.5 * std::erf(b / sqrt2a) -
                temp
                    * std::exp((t1 + t2) / t3)
                    * (1. - std::erf((sqrt2b + a*a_c1) / (2.*a*sqrt_recpr_a_sq_c2))) +
                temp
                    * std::exp((t1 - t2) / t3)
                    * (
                        std::erf((sqrt2_x2 - sqrt2_x2*a_sq_c2 - sqrt2b*a*c2 + a_c1) / (2.*sqrt_recpr_a_sq_c2)) +
                        std::erf((sqrt2b - a*a_c1) / (2.*a*sqrt_recpr_a_sq_c2))
                    );
        }
    }
} 
