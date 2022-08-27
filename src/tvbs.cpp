#include "approxcdf.h"

/*This one is an adaptation of function 'bivariatenormaltrunc' */
void truncate_bvn_2by2block(const double mu1, const double mu2,
                            const double v1, const double v2, const double cv,
                            const double t1, const double t2,
                            double &restrict mu1_out, double &restrict mu2_out,
                            double &restrict v1_out, double &restrict v2_out, double &restrict cv_out)
{
    double s1 = std::sqrt(v1);
    double s2 = std::sqrt(v2);
    #ifdef REGULARIZE_BHAT
    s1 = std::fmax(s1, 0.01);
    s2 = std::fmax(s2, 0.01);
    #endif
    double ntp1 = (t1 - mu1) / s1;
    double ntp2 = (t2 - mu2) / s2;
    double rho = cv / (s1 * s2);

    double p = norm_cdf_2d(ntp1, ntp2, rho);
    double rhotilde = std::sqrt(std::fma(-rho, rho, 1.));
    #ifdef REGULARIZE_BHAT
    rhotilde = std::fmax(rhotilde, 0.0075);
    #endif
    double tr1 = std::fma(-rho, ntp2, ntp1) / rhotilde;
    double tr2 = std::fma(-rho, ntp1, ntp2) / rhotilde;
    double pd1 = norm_pdf_1d(ntp1);
    double pd2 = norm_pdf_1d(ntp2);
    double cd1 = norm_cdf_1d(tr1);
    double cd2 = norm_cdf_1d(tr2);

    double pd1_cd2 = pd1 * cd2;
    double pd2_cd1 = pd2 * cd1;
    double rho_sq = rho * rho;
    double pdf_tr1 = norm_pdf_1d(tr1);
    double pdf_tr2 = norm_pdf_1d(tr2);

    double m1 = -std::fma(rho, pd2_cd1, pd1_cd2) / p;
    double m2 = -std::fma(rho, pd1_cd2, pd2_cd1) / p;
    double os1 = std::fma(-m1, m1,
        1. - (ntp1 * pd1_cd2 +
              ntp2 * rho_sq * pd2_cd1 -
              rhotilde * rho * pd2 * pdf_tr1) / p
    );
    double os2 = std::fma(-m2, m2,
        1. - (ntp2 * pd2_cd1 +
              ntp1 * rho_sq * pd1_cd2 -
              rhotilde * rho * pd1 * pdf_tr2) / p
    );
    /* TODO: check if this last part is correct.
       First try to come up with a case in which it gives
       different results when swapping x1 and x2.
       So far, all attempts have failed to find a bug though. */
    double orho = std::fma(-m1, m2,
        (
            rho * (p - ntp1 * pd1_cd2 - ntp2 * pd2_cd1) +
            rhotilde * pd1 * pdf_tr2
        ) / p
    );

    mu1_out = std::fma(m1, s1, mu1);
    mu2_out = std::fma(m2, s2, mu2);
    v1_out = v1 * os1;
    v2_out = v2 * os2;
    cv_out = s1 * s2 * orho;
}

void truncate_logbvn_2by2block(const double mu1, const double mu2,
                               const double v1, const double v2, const double cv,
                               const double t1, const double t2,
                               double &restrict mu1_out, double &restrict mu2_out,
                               double &restrict v1_out, double &restrict v2_out, double &restrict cv_out)
{
    double s1 = std::sqrt(v1);
    double s2 = std::sqrt(v2);
    s1 = std::fmax(s1, 1e-8);
    s2 = std::fmax(s2, 1e-8);
    double ntp1 = (t1 - mu1) / s1;
    double ntp2 = (t2 - mu2) / s2;
    double rho = cv / (s1 * s2);

    double logp = norm_logcdf_2d(ntp1, ntp2, rho);
    double rhotilde = std::sqrt(std::fma(-rho, rho, 1.));
    rhotilde = std::fmax(rhotilde, 1e-16);
    double tr1 = std::fma(-rho, ntp2, ntp1) / rhotilde;
    double tr2 = std::fma(-rho, ntp1, ntp2) / rhotilde;
    double logpd1 = norm_logpdf_1d(ntp1);
    double logpd2 = norm_logpdf_1d(ntp2);
    double logcd1 = norm_logcdf_1d(tr1);
    double logcd2 = norm_logcdf_1d(tr2);

    double log_pd1_cd2 = logpd1 + logcd2;
    double log_pd2_cd1 = logpd2 + logcd1;
    double log_pdf_tr1 = norm_logpdf_1d(tr1);
    double log_pdf_tr2 = norm_logpdf_1d(tr2);

    double log_rho = std::log(std::fabs(rho));
    double sign_rho = (rho >= 0.)? 1. : -1.;
    double log_rho_pd2_cd1 = log_rho + log_pd2_cd1;
    double log_rho_pd1_cd2 = log_rho + log_pd1_cd2;

    double temp1 = sign_rho * std::exp(log_rho_pd2_cd1 - log_pd1_cd2);
    double temp2 = sign_rho * std::exp(log_rho_pd1_cd2 - log_pd2_cd1);
    double log_m1, log_m2;
    if (temp1 > -1.) {
        log_m1 = log_pd1_cd2 + std::log1p(temp1) - logp;
    }
    else {
        log_m1 = log_pd1_cd2 - logp;
    }
    if (temp2 > -1.) {
        log_m2 = log_pd2_cd1 + std::log1p(temp2) - logp;
    }
    else {
        log_m2 = log_pd2_cd1 - logp;
    }

    double sign_ntp1 = (ntp1 >= 0.)? 1. : -1.;
    double sign_ntp2 = (ntp2 >= 0.)? 1. : -1.;
    double log_ntp1 = std::log(std::fabs(ntp1));
    double log_ntp2 = std::log(std::fabs(ntp2));
    double log_rhotilde = std::log(rhotilde);

    double os1 = 1. - (
        sign_ntp1 * std::exp(log_ntp1 + log_pd1_cd2 - logp)
        + sign_ntp2 * std::exp(log_ntp2 + 2. * log_rho + log_pd2_cd1 - logp)
        - sign_rho * std::exp(log_rhotilde + log_rho + logpd2 + log_pdf_tr1 - logp)
    ) - std::exp(2. * log_m1);
    double os2 = 1. - (
        sign_ntp2 * std::exp(log_ntp2 + log_pd2_cd1 - logp)
        + sign_ntp1 * std::exp(log_ntp1 + 2. * log_rho + log_pd1_cd2 - logp)
        - sign_rho * std::exp(log_rhotilde + log_rho + logpd1 + log_pdf_tr2 - logp)
    ) - std::exp(2. * log_m2);
    double orho = rho * (
        1.
        - sign_ntp1 * std::exp(log_ntp1 + log_pd1_cd2 - logp)
        - sign_ntp2 * std::exp(log_ntp2 + log_pd2_cd1 - logp)
    ) + std::exp(log_rhotilde + logpd1 + log_pdf_tr2 - logp)
     - std::exp(log_m1 + log_m2);

    mu1_out = std::fma(-std::exp(log_m1), s1, mu1);
    mu2_out = std::fma(-std::exp(log_m2), s2, mu2);
    v1_out = v1 * os1;
    v2_out = v2 * os2;
    cv_out = s1 * s2 * orho;

    v1_out = std::fmax(v1_out, std::numeric_limits<double>::min());
    v2_out = std::fmax(v1_out, std::numeric_limits<double>::min());
}

double norm_cdf_nd_tvbs_internal
(
    double *restrict x_reordered,
    double *restrict rho_reordered,
    const int n,
    double *restrict mu_trunc, /* buffer dim: n */
    double *restrict L, /* buffer dim: n^2 */
    double *restrict D, /* buffer dim: n^2 */
    double *restrict temp1, /* buffer dim: n^2 */
    double *restrict temp2, /* buffer dim: n^2 */
    double *restrict temp3, /* buffer dim: 2*(n-2) */
    double *restrict temp4, /* buffer dim: 2*(n-2) */
    double *restrict rho_copy, /* buffer dim: n^2 */
    const bool logp
)
{
    if (n <= 4) {
        switch (n) {
            case 1: {
                if (likely(!logp)) {
                    return norm_cdf_1d(x_reordered[0]);
                }
                else {
                    return norm_logcdf_1d(x_reordered[0]);
                }
            }
            case 2: {
                if (likely(!logp)) {
                    return norm_cdf_2d(x_reordered[0], x_reordered[1], rho_reordered[1]);
                }
                else {
                    return norm_logcdf_2d(x_reordered[0], x_reordered[1], rho_reordered[1]);
                }
            }
            case 3: {
                if (likely(!logp)) {
                    return norm_cdf_3d(x_reordered[0], x_reordered[1], x_reordered[2],
                                       rho_reordered[1], rho_reordered[2], rho_reordered[5]);
                }
                else {
                    return norm_logcdf_3d(x_reordered[0], x_reordered[1], x_reordered[2],
                                          rho_reordered[1], rho_reordered[2], rho_reordered[5]);
                }
            }
            case 4: {
                const double rho_flat[] = {
                    rho_reordered[1], rho_reordered[2], rho_reordered[3],
                    rho_reordered[6], rho_reordered[7], rho_reordered[11]
                };
                if (likely(!logp)) {
                    return norm_cdf_4d_bhat(x_reordered, rho_flat);
                }
                else {
                    return norm_logcdf_4d(x_reordered, rho_flat);
                }
            }
            default: {
                return NAN;
            }
        }
    }

    int size_block1, size_block2, pos_st;
    double p_independent;
    preprocess_rho(rho_reordered, n, n, x_reordered,
                   pos_st, p_independent,
                   size_block1, size_block2,
                   4, logp);
    if (likely(!logp)) {
        if (std::isnan(p_independent) || p_independent <= 0.) {
            return 0.;
        }
    }
    if (pos_st != 0 || size_block1 != n) {
        double p1 = logp? 0. : 1.;
        double p2 = logp? 0. : 1.;

        if (size_block1) {
            F77_CALL(dlacpy)(
                "A", &size_block1, &size_block1,
                rho_reordered + pos_st * (n + 1), &n,
                rho_copy, &size_block1 FCONE
            );
            p1 = norm_cdf_nd_tvbs_internal(
                x_reordered + pos_st,
                rho_copy,
                size_block1,
                mu_trunc,
                L,
                D,
                temp1,
                temp2,
                temp3,
                temp4,
                rho_reordered,
                logp
            );
        }

        if (size_block2) {
            if (size_block2) {
                F77_CALL(dlacpy)(
                    "A", &size_block2, &size_block2,
                    rho_reordered + (pos_st + size_block1) * (n + 1), &n,
                    rho_copy, &size_block2 FCONE
                );
            }
            p2 = norm_cdf_nd_tvbs_internal(
                x_reordered + pos_st + size_block1,
                rho_copy,
                size_block2,
                mu_trunc,
                L,
                D,
                temp1,
                temp2,
                temp3,
                temp4,
                rho_reordered,
                logp
            );
        }

        if (likely(!logp)) {
            return p1 * p2 * p_independent;
        }
        else {
            return p1 + p2 + p_independent;
        }
    }

    /* TODO: maybe it should also offer an option for simple reordering,
       or an option to order on-the-fly as it truncates distributions
       as in the TGE order referenced by the author. */
    gge_reorder(x_reordered, rho_reordered, n, temp1, temp2);

    factorize_ldl_2by2blocks(rho_reordered, n,
                             D, L,
                             temp1, temp2);

    
    temp1[0] = rho_reordered[1]; temp1[1] = rho_reordered[2]; temp1[2] = rho_reordered[3];
    temp1[3] = rho_reordered[2 + n]; temp1[4] = rho_reordered[3 + n]; temp1[5] = rho_reordered[3 + 2*n];
    double cumP;
    if (likely(!logp)) {
        cumP = norm_cdf_4d_bhat(x_reordered, temp1);
    }
    else {
        cumP = norm_logcdf_4d(x_reordered, temp1);
    }
    int n_steps = (n / 2) - !(n & 1) - 1;

    double bvn_trunc_mu[2];
    double bvn_trunc_cv[3];
    double bvn_trunc_cv_square[4];
    std::fill(mu_trunc, mu_trunc + n, 0.);
    double qvn_cv[16];
    double p2, p3, p4;

    for (int step = 0; step < n_steps; step++) {
        
        if (likely(!logp)) {
            if (std::isnan(cumP) || std::isinf(cumP) || cumP <= 0.) {
                return 0.;
            }
        }

        if (likely(!logp)) {
            truncate_bvn_2by2block(mu_trunc[2*step], mu_trunc[2*step + 1],
                                   D[2*step*(n+1)], D[(2*step+1)*(n+1)], D[2*step*(n+1) + 1],
                                   x_reordered[2*step], x_reordered[2*step + 1],
                                   bvn_trunc_mu[0], bvn_trunc_mu[1],
                                   bvn_trunc_cv[0], bvn_trunc_cv[1], bvn_trunc_cv[2]);
        } else {
            truncate_logbvn_2by2block(mu_trunc[2*step], mu_trunc[2*step + 1],
                                      D[2*step*(n+1)], D[(2*step+1)*(n+1)], D[2*step*(n+1) + 1],
                                      x_reordered[2*step], x_reordered[2*step + 1],
                                      bvn_trunc_mu[0], bvn_trunc_mu[1],
                                      bvn_trunc_cv[0], bvn_trunc_cv[1], bvn_trunc_cv[2]);
        }

        bvn_trunc_mu[0] -= mu_trunc[2*step];
        bvn_trunc_mu[1] -= mu_trunc[2*step + 1];

        cblas_dgemv(
            CblasRowMajor, CblasNoTrans,
            n - 2*(step+1), 2,
            1., L + 2*step*(n+1) + 2*n, n,
            bvn_trunc_mu, 1,
            1., mu_trunc + 2*(step+1), 1
        );


        bvn_trunc_cv_square[0] = bvn_trunc_cv[0];  bvn_trunc_cv_square[1] = bvn_trunc_cv[2];
        bvn_trunc_cv_square[2] = bvn_trunc_cv[2];  bvn_trunc_cv_square[3] = bvn_trunc_cv[1];
        update_ldl_rank2(L + (n+1)*2*(step), n,
                         D + (n+1)*2*(step), n,
                         bvn_trunc_cv_square, 2,
                         n - 2*(step),
                         temp1,
                         temp2,
                         temp3,
                         temp4);


        if (n >= 2*(step+1)+4) {
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                4, 4, 4,
                1., L + (n+1)*2*(step+1), n,
                D + (n+1)*2*(step+1), n,
                0., qvn_cv, 4
            );
            cblas_dtrmm(
                CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                4, 4,
                1., L + (n+1)*2*(step+1), n,
                qvn_cv, 4
            );


            if (likely(!logp)) {
                p4 = nonstd_cdf_4d(x_reordered + 2*(step+1), mu_trunc + 2*(step+1), qvn_cv, 4);
                p2 = nonstd_cdf_2d(x_reordered[2*(step+1)], x_reordered[2*(step+1)+1],
                                   mu_trunc[2*(step+1)], mu_trunc[2*(step+1)+1],
                                   qvn_cv[0], qvn_cv[5], qvn_cv[1]);
                if (p2 <= 0) {
                    return NAN;
                }

                cumP *= p4 / p2;
            }
            else {
                p4 = nonstd_logcdf_4d(x_reordered + 2*(step+1), mu_trunc + 2*(step+1), qvn_cv, 4);
                p2 = nonstd_logcdf_2d(x_reordered[2*(step+1)], x_reordered[2*(step+1)+1],
                                      mu_trunc[2*(step+1)], mu_trunc[2*(step+1)+1],
                                      qvn_cv[0], qvn_cv[5], qvn_cv[1]);

                cumP += p4 - p2;
            }
        }
        else {
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                3, 3, 3,
                1., L + (n+1)*2*(step+1), n,
                D + (n+1)*2*(step+1), n,
                0., qvn_cv, 3
            );
            cblas_dtrmm(
                CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                3, 3,
                1., L + (n+1)*2*(step+1), n,
                qvn_cv, 3
            );

            if (likely(!logp)) {
                p3 = nonstd_cdf_3d(x_reordered + 2*(step+1), mu_trunc + 2*(step+1), qvn_cv, 3);
                p2 = nonstd_cdf_2d(x_reordered[2*(step+1)], x_reordered[2*(step+1)+1],
                                   mu_trunc[2*(step+1)], mu_trunc[2*(step+1)+1],
                                   qvn_cv[0], qvn_cv[4], qvn_cv[1]);
                if (p2 <= 0) {
                    return NAN;
                }

                cumP *= p3 / p2;
            }
            else {
                p3 = nonstd_logcdf_3d(x_reordered + 2*(step+1), mu_trunc + 2*(step+1), qvn_cv, 3);
                p2 = nonstd_logcdf_2d(x_reordered[2*(step+1)], x_reordered[2*(step+1)+1],
                                      mu_trunc[2*(step+1)], mu_trunc[2*(step+1)+1],
                                      qvn_cv[0], qvn_cv[4], qvn_cv[1]);

                cumP += p3 - p2;
            }
        }

    }

    if (likely(!logp)) {
        cumP = std::fmax(cumP, 0.);
        cumP = std::fmin(cumP, 1.);
    }
    return cumP;
}


double norm_cdf_tvbs
(
    const double x[],
    const double Sigma[], const int ld_Sigma, /* typically ld_Sigma=n */
    const double mu[], /* ignored when is_standardized=false */
    const int n,
    const bool is_standardized,
    const bool logp,
    double *restrict buffer /* dim: 6*n^2 + 6*n - 8 */
)
{
    if (n <= 4) {
        double x_reordered[4];
        double rho_reordered[16];
        double buffer_sdtdev[4];
        copy_and_standardize(
            x,
            Sigma,
            ld_Sigma,
            mu,
            n,
            is_standardized,
            x_reordered,
            rho_reordered,
            buffer_sdtdev
        );

        return norm_cdf_nd_tvbs_internal(
            x_reordered,
            rho_reordered,
            n,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            logp
        );
    }

    std::unique_ptr<double[]> owned_buffer;
    if (!buffer) {
        owned_buffer = std::unique_ptr<double[]>(new double[
            2*n + 4*(n-2) + 6*n*n /* <- 6*n^2 + 6*n - 8 */
        ]);
        buffer = owned_buffer.get();
    }

    double *x_reordered = buffer; buffer += n;
    double *rho_reordered = buffer; buffer += n*n;
    double *mu_trunc = buffer; buffer += n;
    double *L = buffer; buffer += n*n;
    double *D = buffer; buffer += n*n;
    double *rho_copy = buffer; buffer += n*n;
    double *temp1 = buffer; buffer += n*n;
    double *temp2 = buffer; buffer += n*n;
    double *temp3 = buffer; buffer += 2*(n-2);
    double *temp4 = buffer; buffer += 2*(n-2);

    copy_and_standardize(
        x,
        Sigma,
        ld_Sigma,
        mu,
        n,
        is_standardized,
        x_reordered,
        rho_reordered,
        temp1
    );

    return norm_cdf_nd_tvbs_internal(
        x_reordered,
        rho_reordered,
        n,
        mu_trunc,
        L,
        D,
        temp1,
        temp2,
        temp3,
        temp4,
        rho_copy,
        logp
    );
}
