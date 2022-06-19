 #include "approxcdf.h"

/* This is an adaptation of Alan Genz's TVPACK:
https://www.math.wsu.edu/faculty/genz/software/fort77/tvpack.f

It adapts only the parts necessary for calculation of 2D and 3D MVN CDF,
with some modifications such as using math functions from the C/C++
standard library and changing the number of Gaussian-Legendre points used.


Copyright (C) 2013, Alan Genz,  All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided the following conditions are met:
1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in 
  the documentation and/or other materials provided with the 
  distribution.
3. The contributor name(s) may not be used to endorse or promote 
  products derived from this software without specific prior 
  written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS 
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* Drezner, Zvi, and George O. Wesolowsky.
   "On the computation of the bivariate normal integral."
   Journal of Statistical Computation and Simulation 35.1-2 (1990): 101-107.
   http://www.math.wsu.edu/faculty/genz/software/software.html
   This is an adaptation of TVPACK from Alan Genz.
   It additionally added this note below the reference:
   "with major modifications for double precision, and for |R| close to 1." */
[[gnu::flatten]]
double norm_lcdf_2d(double x1, double x2, double rho)
{
    double abs_rho = std::fabs(rho);

    double out = 0;
    double hk;
    if (abs_rho < 0.925) {
        if (abs_rho > std::numeric_limits<double>::epsilon()) {
            hk = x1 * x2;
            double hs = 0.5 * (x1*x1 + x2*x2);
            double asr = std::asin(rho);
            double asr_half = 0.5 * asr;
            double sn1, sn2;

            int gl_dim;
            #ifndef _OPENMP
            const double *restrict GL_wtable;
            const double *restrict GL_xtable;
            #endif
            if (abs_rho < 0.3) {
                gl_dim = (int)GL6;
                #ifndef _OPENMP
                GL_wtable = GL6_w;
                GL_xtable = GL6_x;
                #endif
            }
            else if (abs_rho < 0.5) {
                gl_dim = (int)GL12;
                #ifndef _OPENMP
                GL_wtable = GL12_w;
                GL_xtable = GL12_x;
                #endif
            }
            else {
                gl_dim = (int)GL20;
                #ifndef _OPENMP
                GL_wtable = GL20_w;
                GL_xtable = GL20_x;
                #endif
            }
            
            #ifndef _OPENMP
            #pragma GCC unroll 3
            for (int ix = 0; ix < gl_dim; ix++) {
                sn1 = std::sin(asr_half * (1. + GL_xtable[ix]));
                sn2 = std::sin(asr_half * (1. - GL_xtable[ix]));
                out += GL_wtable[ix] * (
                    std::exp(std::fma(sn1, hk, -hs) / std::fma(-sn1, sn1, 1.)) +
                    std::exp(std::fma(sn2, hk, -hs) / std::fma(-sn2, sn2, 1.))
                );
            }
            #else
            switch (gl_dim) {
                case GL6: {
                    #pragma omp simd
                    for (int ix = 0; ix < 4; ix++) {
                        sn1 = std::sin(asr_half * (1. + GL8_x[ix]));
                        sn2 = std::sin(asr_half * (1. - GL8_x[ix]));
                        out += GL8_w[ix] * (
                            std::exp(std::fma(sn1, hk, -hs) / std::fma(-sn1, sn1, 1.)) +
                            std::exp(std::fma(sn2, hk, -hs) / std::fma(-sn2, sn2, 1.))
                        );
                    }
                    break;
                }
                case GL12: {
                    #pragma omp simd
                    for (int ix = 0; ix < 8; ix++) {
                        sn1 = std::sin(asr_half * (1. + GL16_x[ix]));
                        sn2 = std::sin(asr_half * (1. - GL16_x[ix]));
                        out += GL16_w[ix] * (
                            std::exp(std::fma(sn1, hk, -hs) / std::fma(-sn1, sn1, 1.)) +
                            std::exp(std::fma(sn2, hk, -hs) / std::fma(-sn2, sn2, 1.))
                        );
                    }
                    break;
                }
                case GL20: {
                    #pragma omp simd
                    for (int ix = 0; ix < 12; ix++) {
                        sn1 = std::sin(asr_half * (1. + GL24_x[ix]));
                        sn2 = std::sin(asr_half * (1. - GL24_x[ix]));
                        out += GL24_w[ix] * (
                            std::exp(std::fma(sn1, hk, -hs) / std::fma(-sn1, sn1, 1.)) +
                            std::exp(std::fma(sn2, hk, -hs) / std::fma(-sn2, sn2, 1.))
                        );
                    }
                    break;
                }
            }
            #endif
            out *= asr / fourPI;
        }
        out = std::fma(norm_lcdf_1d(x1), norm_lcdf_1d(x2), out);
    }
    else {

        if (rho < 0) {
            x2 = -x2;
        }
        if (abs_rho < 1) {
            hk = x1 * x2;
            double as = std::fma(-rho, rho, 1.);
            double a = std::sqrt(as);
            double b;
            double bs = (x1 - x2) * (x1 - x2);
            double c = std::fma(-0.125, hk, 0.5);
            double d = std::fma(-0.0625, hk, 0.75);
            double asr = -0.5 * (hk + bs / as);
            double rfdbs = std::fma(-d, bs, 5.)*(1./15.);
            if (asr > -100.) {
                out = a * std::exp(asr) * (1. - c * (bs - as) * rfdbs + 0.2*c*d*as*as);
            }
            if (hk > -100.) {
                b = std::sqrt(bs);
                out -= std::exp(-0.5 * hk) * sqrt_twoPI * norm_lcdf_1d(b / a) * b * (1. - c*bs*rfdbs);
            }
            a *= 0.5;
            double xs;
            double rs;
            double temp;

            #ifndef _OPENMP
            #pragma GCC unroll 10
            for (int ix = 0; ix < 10; ix++) {
                temp = a * (1. + GL20_x[ix]);
                xs = temp * temp;
                rs = std::sqrt(1. - xs);
                asr = -0.5 * (hk + bs / xs);
                if (asr > -100.) {
                    out += a * GL20_w[ix] * std::exp(asr) *
                           (std::exp(-hk*xs/(2.*(1.+rs)*(1.+rs)))/rs - (1. + c*xs*std::fma(d, xs, 1.)));
                }

                temp = a * (1. - GL20_x[ix]);
                xs = temp * temp;
                rs = std::sqrt(1. - xs);
                asr = -0.5 * (hk + bs / xs);
                if (asr > -100.) {
                    out += a * GL20_w[ix] * std::exp(asr) *
                           (std::exp(-hk*xs/(2.*(1.+rs)*(1.+rs)))/rs - (1. + c*xs*std::fma(d, xs, 1.)));
                }
            }
            #else
            #pragma omp simd
            for (int ix = 0; ix < 12; ix++) {
                temp = a * (1. + GL24_x[ix]);
                xs = temp * temp;
                rs = std::sqrt(1. - xs);
                asr = -0.5 * (hk + bs / xs);
                out += a * GL24_w[ix] * std::exp(asr) *
                       (std::exp(-hk*xs/(2.*(1.+rs)*(1.+rs)))/rs - (1. + c*xs*std::fma(d, xs, 1.)));

                temp = a * (1. - GL24_x[ix]);
                xs = temp * temp;
                rs = std::sqrt(1. - xs);
                asr = -0.5 * (hk + bs / xs);
                out += a * GL24_w[ix] * std::exp(asr) *
                       (std::exp(-hk*xs/(2.*(1.+rs)*(1.+rs)))/rs - (1. + c*xs*std::fma(d, xs, 1.)));
            }
            #endif
            out *= minus_inv_twoPI;
        }
        if (rho > 0) {
            out += norm_lcdf_1d(std::fmax(x1, x2));
        }
        else {
            out = -out;
            if (x2 > x1) {
                if (x1 < 0) {
                    out += norm_cdf_1d(x2) - norm_cdf_1d(x1);
                }
                else {
                    out += norm_lcdf_1d(x1) - norm_lcdf_1d(x2);
                }
            }
        }
    }
    return out;
}

double norm_cdf_2d(double x1, double x2, double rho)
{
    return norm_lcdf_2d(-x1, -x2, rho);
}

double plackett_integrand(double ba, double bb, double bc, double ra, double rb, double r, double rr)
{
    double out = 0;
    double dt = rr * (std::fma(-ra + rb, ra - rb, rr) - 2.*ra*rb*(1. - r));
    if (dt > 0) {
        double bt = (bc*rr + ba * std::fma(r, rb, -ra) + bb * std::fma(r, ra, -rb)) / std::sqrt(dt);
        double temp = std::fma(-r, bb, ba);
        double ft = std::fma(bb, bb, temp * temp / rr);
        if (bt > -10. && ft < 100.) {
            out = std::exp(-0.5 * ft);
            if (bt < 10) {
                out *= norm_cdf_1d(bt);
            }
        }
    }
    return out;
}

[[gnu::flatten]]
double plackett_mvn_integrands(double x, double x1, double x2, double x3, double rho23, double rua, double rub)
{
    double rho12 = std::sin(rua * x);
    double rho13 = std::sin(rub * x);

    double out = 0;
    if (std::fabs(rua) > 0) {
        out = std::fma(rua, plackett_integrand(x1, x2, x3, rho13, rho23, rho12, std::fma(-rho12, rho12, 1.)), out);
    }
    if (std::fabs(rub) > 0) {
        out = std::fma(rub, plackett_integrand(x1, x3, x2, rho12, rho23, rho13, std::fma(-rho13, rho13, 1.)), out);
    }
    return out;
}

constexpr static const double wg[] = {
    0.2729250867779007E+00, 0.5566856711617449E-01, 0.1255803694649048E+00,
    0.1862902109277352E+00, 0.2331937645919914E+00, 0.2628045445102478E+00
};
constexpr static const double xgk[] = {
    0.9963696138895427E+00, 0.9782286581460570E+00,
    0.9416771085780681E+00, 0.8870625997680953E+00, 0.8160574566562211E+00,
    0.7301520055740492E+00, 0.6305995201619651E+00, 0.5190961292068118E+00,
    0.3979441409523776E+00, 0.2695431559523450E+00, 0.1361130007993617E+00
};
constexpr static const double wgk[] = {
    0.9765441045961290E-02, 0.2715655468210443E-01,
    0.4582937856442671E-01, 0.6309742475037484E-01, 0.7866457193222764E-01,
    0.9295309859690074E-01, 0.1058720744813894E+00, 0.1167395024610472E+00,
    0.1251587991003195E+00, 0.1312806842298057E+00, 0.1351935727998845E+00
};
double kronrod_mvn(double a, double b, double x1, double x2, double x3, double rho23, double rua, double rub,
                   double &err)
{

    double wid = 0.5 * (b - a);
    double cen = 0.5 * (b + a);
    double fc = plackett_mvn_integrands(cen, x1, x2, x3, rho23, rua, rub);
    double resg = fc * 0.2729250867779007E+00;
    double resk = fc * 0.1365777947111183E+00;

    for (int ix = 0; ix < 11; ix++) {
        fc = plackett_mvn_integrands(std::fma(-wid, xgk[ix], cen), x1, x2, x3, rho23, rua, rub) +
             plackett_mvn_integrands(std::fma( wid, xgk[ix], cen), x1, x2, x3, rho23, rua, rub);
        resk = std::fma(wgk[ix], fc, resk);
        if (ix & 1) {
            resg = std::fma(wg[(ix + 1) / 2], fc, resg);
        }
    }

    err = std::fabs(wid * (resk - resg));
    return wid * resk;
}


#define eps3d 1e-14
/* Note: in theory, should use 6.25e-30 for the tolerance criterion,
   but that can end up being very slow. Also, in order to match TVPACK,
   it would need to have a maximum of 100 iterations, but that can again
   result in very slow running times without much change in precision.  */
#ifndef IMPRECISE
    constexpr static const double tol_integration = 6.25e-30;
    #define MAX_ADAPTIVE_ITER 100
#else
    constexpr static const double tol_integration = 1e-20;
    #define MAX_ADAPTIVE_ITER 25
#endif
double adaptive_integration(double x1, double x2, double x3, double rho23, double rua, double rub)
{
    double ai[MAX_ADAPTIVE_ITER];
    double bi[MAX_ADAPTIVE_ITER];
    double fi[MAX_ADAPTIVE_ITER];
    double ei[MAX_ADAPTIVE_ITER];

    ai[0] = 0;
    bi[0] = 1;

    int ip = 0;
    double err;
    double out = 0;

    int im;
    for (im = 1; im < MAX_ADAPTIVE_ITER; im++) {
        bi[im] = bi[ip];
        ai[im] = 0.5 * (ai[ip] + bi[ip]);
        bi[ip] = ai[im];
        fi[ip] = kronrod_mvn(ai[ip], bi[ip], x1, x2, x3, rho23, rua, rub, ei[ip]);
        fi[im] = kronrod_mvn(ai[im], bi[im], x1, x2, x3, rho23, rua, rub, ei[im]);
        out = std::accumulate(fi, fi + im + 1, 0.);
        err = ddot1(im+1, ei, ei);
        if (err <= tol_integration) {
            break;
        }
        for (int ix = im; ix >= 0; ix--) {
            if (ei[ix] > ei[ip]) {
                ip = ix;
                break;
            }
        }
    }

    return out;
}

/* Genz, Alan.
   "Numerical computation of rectangular bivariate and trivariate normal and t probabilities."
   Statistics and Computing 14.3 (2004): 251-260. */
double norm_cdf_3d(double x1, double x2, double x3, double rho12, double rho13, double rho23)
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
        return norm_cdf_1d(x1) * norm_lcdf_2d(-x2, -x3, rho23);
    }
    else if (std::fabs(rho13) + std::fabs(rho23) < eps3d) {
        return norm_cdf_1d(x3) * norm_lcdf_2d(-x1, -x2, rho12);
    }
    else if (std::fabs(rho12) + std::fabs(rho23) < eps3d) {
        return norm_cdf_1d(x2) * norm_lcdf_2d(-x1, -x3, rho13);
    }
    else if (1. - rho23 < eps3d) {
        return norm_lcdf_2d(-x1, -std::min(x2, x3), rho12);
    }
    else if (rho23 + 1. < eps3d) {
        if (x2 > -x3) {
            return norm_lcdf_2d(-x1, -x2, rho12) - norm_lcdf_2d(-x1, x3, rho12);
        }
        else {
            return 0;
        }
    }
    else {
        double out = norm_lcdf_2d(-x2, -x3, rho23) * norm_cdf_1d(x1);
        double rua = std::asin(rho12);
        double rub = std::asin(rho13);
        out = std::fma(inv_four_asin_one, adaptive_integration(x1, x2, x3, rho23, rua, rub), out);

        out = std::fmax(out, 0.);
        out = std::fmin(out, 1.);
        return out;
    }
}
