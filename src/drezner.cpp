#include "approxcdf.h"

/* Drezner, Zvi, and George O. Wesolowsky.
   "On the computation of the bivariate normal integral."
   Journal of Statistical Computation and Simulation 35.1-2 (1990): 101-107. */
#ifndef _OPENMP
constexpr static const double GL5_w_div4pi[] = {
    GL5_w[0] / (4. * M_PI), GL5_w[1] / (4. * M_PI), GL5_w[2] / (4. * M_PI)
};
constexpr static const double GL5_xp[] = {
    0.5 * GL5_x[0] + 0.5, 0.5 * GL5_x[1] + 0.5, 0.5 * GL5_x[2] + 0.5
};
constexpr static const double GL5_xn[] = {
    -0.5 * GL5_x[0] + 0.5, -0.5 * GL5_x[1] + 0.5, -0.5 * GL5_x[2] + 0.5
};
#else
constexpr static const double GL8_w_div4pi[] = {
    GL8_w[0] / (4. * M_PI), GL8_w[1] / (4. * M_PI),
    GL8_w[2] / (4. * M_PI), GL8_w[3] / (4. * M_PI)
};
#endif
double norm_lcdf_2d_fast(double x1, double x2, double rho)
{
    double x12 = 0.5 * (x1*x1 + x2*x2);
    
    double out = 0;
    double r1, x3;
    if (std::fabs(rho) >= 0.7) {
        double r2 = 1. - rho*rho;
        double r3 = std::sqrt(r2);
        if (rho < 0) {
            x2 = -x2;
        }
        x3 = x1*x2;
        double x7 = std::exp(-0.5 * x3);
        if (r2) {
            double x6 = std::fabs(x1 - x2);
            double x5 = 0.5 * x6*x6;
            x6 /= r3;
            double aa = 0.5 - 0.125*x3;
            double ab = 3. - 2.*aa*x5;
            out = inv3sqrt2pi * (
                x6 * ab * norm_lcdf_1d(x6) -
                std::exp(-x5/r2) * std::fma(aa, r2, ab) * inv_sqrt2_pi
            );
            double rr;
            double nr1, nrr, nr2;
            #ifndef _OPENMP
            #pragma GCC unroll 2
            for (int ix = 0; ix < 2; ix++) {
                r1 = r3 * GL5_xp[ix];
                rr = r1*r1;
                r2 = std::sqrt(1. - rr);

                nr1 = r3 * GL5_xn[ix];
                nrr = nr1*nr1;
                nr2 = std::sqrt(1. - nrr);
                
                out -= GL5_w_div4pi[ix] * (
                    std::exp(-x5/rr) * (std::exp(-x3/(1. + r2))/r2/x7 - 1.- aa*rr) +
                    std::exp(-x5/nrr) * (std::exp(-x3/(1. + nr2))/nr2/x7 - 1.- aa*nrr)
                );
            }
            r1 = r3 * GL5_xp[2];
            rr = r1*r1;
            r2 = std::sqrt(1. - rr);
            out -= GL5_w_div4pi[2] * std::exp(-x5/rr) * (std::exp(-x3/(1. + r2))/r2/x7 - 1.- aa*rr);
            #else
            #ifndef _MSC_VER
            #pragma omp simd
            #endif
            for (int ix = 0; ix < 4; ix++) {
                r1 = r3 * GL8_xp[ix];
                rr = r1*r1;
                r2 = std::sqrt(1. - rr);

                nr1 = r3 * GL8_xn[ix];
                nrr = nr1*nr1;
                nr2 = std::sqrt(1. - nrr);
                
                out -= GL8_w_div4pi[ix] * (
                    std::exp(-x5/rr) * (std::exp(-x3/(1. + r2))/r2/x7 - 1.- aa*rr) +
                    std::exp(-x5/nrr) * (std::exp(-x3/(1. + nr2))/nr2/x7 - 1.- aa*nrr)
                );
            }
            #endif
        }
        if (rho > 0) {
            out = std::fma(out, r3*x7, norm_lcdf_1d(std::fmax(x1, x2)));
        }
        else {
            out = std::fmax(0., norm_lcdf_1d(x1) - norm_lcdf_1d(x2)) - out*r3*x7;
        }
        return out;
    }
    else {
        x3 = x1*x2;
        double rr2;
        double nr1, nrr2;
        #ifndef _OPENMP
        #pragma GCC unroll 2
        for (int ix = 0; ix < 2; ix++) {
            r1 = rho * GL5_xp[ix];
            rr2 = 1. - r1*r1;

            nr1 = rho * GL5_xn[ix];
            nrr2 = 1. - nr1*nr1;

            out += GL5_w_div4pi[ix] * (
                std::exp((r1*x3 - x12) / rr2) / std::sqrt(rr2) +
                std::exp((nr1*x3 - x12) / nrr2) / std::sqrt(nrr2)
            );
        }
        r1 = rho * GL5_xp[2];
        rr2 = 1. - r1*r1;
        out += GL5_w_div4pi[2] * std::exp((r1*x3 - x12) / rr2) / std::sqrt(rr2);
        #else
        #ifndef _MSC_VER
        #pragma omp simd
        #endif
        for (int ix = 0; ix < 4; ix++) {
            r1 = rho * GL8_xp[ix];
            rr2 = 1. - r1*r1;

            nr1 = rho * GL8_xn[ix];
            nrr2 = 1. - nr1*nr1;

            out += GL8_w_div4pi[ix] * (
                std::exp((r1*x3 - x12) / rr2) / std::sqrt(rr2) +
                std::exp((nr1*x3 - x12) / nrr2) / std::sqrt(nrr2)
            );
        }
        #endif
        return std::fma(out, rho, norm_lcdf_1d(x1) * norm_lcdf_1d(x2));
    }
}

double norm_cdf_2d_fast(double x1, double x2, double rho)
{
    return norm_lcdf_2d_fast(-x1, -x2, rho);
}

/* Drezner, Zvi.
   "Computation of the trivariate normal integral."
   Mathematics of Computation 62.205 (1994): 289-294. */
#define eps3d 1e-14
double norm_cdf_3d_fast(double x1, double x2, double x3, double rho12, double rho13, double rho23)
{
    if (std::fabs(rho12) > std::fabs(rho13)) {
        std::swap(x2, x3);
        std::swap(rho12, rho13);
    }
    if (std::fabs(rho13) > std::fabs(rho23)) {
        std::swap(x1, x2);
        std::swap(rho13, rho23);
    }

    if (std::fabs(x1) + std::fabs(x2) + std::fabs(x3) < eps3d) {
        return 0.125 * (1. + (std::asin(rho12) + std::asin(rho13) + std::asin(rho23)) / asin_one);
    }
    else if (std::fabs(rho12) + std::fabs(rho13) < eps3d) {
        return norm_cdf_1d(x1) * norm_lcdf_2d_fast(-x2, -x3, rho23);
    }
    else if (std::fabs(rho13) + std::fabs(rho23) < eps3d) {
        return norm_cdf_1d(x3) * norm_lcdf_2d_fast(-x1, -x2, rho12);
    }
    else if (std::fabs(rho12) + std::fabs(rho23) < eps3d) {
        return norm_cdf_1d(x2) * norm_lcdf_2d_fast(-x1, -x3, rho13);
    }
    else if (1. - rho23 < eps3d) {
        return norm_lcdf_2d_fast(-x1, -std::min(x2, x3), rho12);
    }
    else if (rho23 + 1. < eps3d) {
        if (x2 > -x3) {
            return norm_lcdf_2d_fast(-x1, -x2, rho12) - norm_lcdf_2d_fast(-x1, x3, rho12);
        }
        else {
            return 0;
        }
    }

    double x12 = x1 * x2;
    double x13 = x1 * x3;
    double x122 = 0.5 * (x1*x1 + x2*x2);
    double x132 = 0.5 * (x1*x1 + x3*x3);
    double rho23sq = rho23 * rho23;

    double rr12, rr13, fac;
    double rr122, rr133;
    double f1, f2, f3;
    double hp1, hp2;
    double sqrt_rr122, sqrt_rr133;

    double nrr12, nrr13, nfac;
    double nrr122, nrr133;
    double nf1, nf2, nf3;
    double sqrt_nrr122, sqrt_nrr133;
    double nhp1, nhp2;

    double correction = 0;
    #ifndef _MSC_VER
    #pragma omp simd
    #endif
    for (int ix = 0; ix < 16; ix++) {
        rr12 = rho12 * GL32_xp[ix];
        rr13 = rho13 * GL32_xp[ix];
        fac = std::sqrt(1. - rr12*rr12 - rr13*rr13 - rho23sq + 2.*rr12*rr13*rho23);
        rr122 = std::fma(-rr12, rr12, 1.);
        rr133 = std::fma(-rr13, rr13, 1.);
        f1 = std::fma(-rho23, rr12, rr13);
        f2 = std::fma(-rr12, rr13, rho23);
        f3 = std::fma(-rho23, rr13, rr12);
        sqrt_rr122 = std::sqrt(rr122);
        sqrt_rr133 = std::sqrt(rr133);
        hp1 = (x3*rr122 - x1*f1 - x2*f2) / fac / sqrt_rr122;
        hp2 = (x2*rr133 - x1*f3 - x3*f2) / fac / sqrt_rr133;

        nrr12 = rho12 * GL32_xn[ix];
        nrr13 = rho13 * GL32_xn[ix];
        nfac = std::sqrt(1. - nrr12*nrr12 - nrr13*nrr13 - rho23sq + 2.*nrr12*nrr13*rho23);
        nrr122 = std::fma(-nrr12, nrr12, 1.);
        nrr133 = std::fma(-nrr13, nrr13, 1.);
        nf1 = std::fma(-rho23, nrr12, nrr13);
        nf2 = std::fma(-nrr12, nrr13, rho23);
        nf3 = std::fma(-rho23, nrr13, nrr12);
        sqrt_nrr122 = std::sqrt(nrr122);
        sqrt_nrr133 = std::sqrt(nrr133);
        nhp1 = (x3*nrr122 - x1*nf1 - x2*nf2) / nfac / sqrt_nrr122;
        nhp2 = (x2*nrr133 - x1*nf3 - x3*nf2) / nfac / sqrt_nrr133;


        correction += GL32_w_div4pi[ix] * (
            rho12 * (std::exp(std::fma(rr12, x12, -x122) / rr122) / sqrt_rr122 * norm_cdf_1d(hp1) +
                     std::exp(std::fma(nrr12, x12, -x122) / nrr122) / sqrt_nrr122 * norm_cdf_1d(nhp1)) +
            rho13 * (std::exp(std::fma(rr13, x13, -x132) / rr133) / sqrt_rr133 * norm_cdf_1d(hp2) +
                     std::exp(std::fma(nrr13, x13, -x132) / nrr133) / sqrt_nrr133 * norm_cdf_1d(nhp2))
        );
    }
    
    return std::fma(norm_cdf_1d(x1), norm_cdf_2d(x2, x3, rho23), correction);   
}
