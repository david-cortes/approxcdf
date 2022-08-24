#include "approxcdf.h"

/* This one is a specialization of function 'multruncbivariate'
   'mu_out' has dim=4, 'Omega_out' has dim=10 (it's a 4x4 symmetric matrix)
   
   Mapped order for the output is as follows:
   [[o0, o1, o2, o3],
    [o1, o4, o5, o6],
    [o2, o5, o7, o8],
    [o3, o6, o8, o9]]

   Mapped order for the input is as follows:
   [[ 1, r0, r1, r2],
    [r0,  1, r3, r4],
    [r1, r3,  1, r5],
    [r2, r4, r5,  1]] */
static inline
void bv_trunc_std4d_loweronly(const double rho[6], const double tp[2],
                              double *restrict mu_out, double *restrict Omega_out)
{
    double detV11 = std::fma(-rho[0], rho[0], 1.);
    double invV11v = 1. / detV11;
    double invV11d = -rho[0] / detV11;
    #ifdef REGULARIZE_BHAT
    double reg = 1e-5;
    double d = 1.;
    while (detV11 <= 1e-2) {
        d += reg;
        detV11 = std::fma(-rho[0], rho[0], d*d);
        invV11v = d / detV11;
        invV11d = -rho[0] / detV11;
        reg *= 1.5;
    }
    #endif

    double Omega11[3];
    double mu_half[2];
    truncate_bvn_2by2block(0., 0., 1., 1., rho[0], tp[0], tp[1],
                           mu_half[0], mu_half[1],
                           Omega11[0], Omega11[1], Omega11[2]);

    mu_out[0] = (invV11v*rho[1] + invV11d*rho[3]) * (mu_half[0]) +
                (invV11d*rho[1] + invV11v*rho[3]) * (mu_half[1]);
    mu_out[1] = (invV11v*rho[2] + invV11d*rho[4]) * (mu_half[0]) +
                (invV11d*rho[2] + invV11v*rho[4]) * (mu_half[1]);

    double Omega11_invV11[] = {
        Omega11[0]*invV11v + Omega11[2]*invV11d, Omega11[0]*invV11d + Omega11[2]*invV11v,
        Omega11[2]*invV11v + Omega11[1]*invV11d, Omega11[2]*invV11d + Omega11[1]*invV11v
    };
    /* O12 */
    double O12[] = {
        Omega11_invV11[0]*rho[1] + Omega11_invV11[1]*rho[3], Omega11_invV11[0]*rho[2] + Omega11_invV11[1]*rho[4],
        Omega11_invV11[2]*rho[1] + Omega11_invV11[3]*rho[3], Omega11_invV11[2]*rho[2] + Omega11_invV11[3]*rho[4]
    };
    /* V12 - O12 */
    double temp1[] = {
        rho[1] - O12[0], rho[2] - O12[1],
        rho[3] - O12[2], rho[4] - O12[3]
    };
    /* iV11 * (V12 - O12) */
    double temp2[] = {
        invV11v*temp1[0] + invV11d*temp1[2], invV11v*temp1[1] + invV11d*temp1[3],
        invV11d*temp1[0] + invV11v*temp1[2], invV11d*temp1[1] + invV11v*temp1[3]
    };
    /* V22 - V21 * (iV11 * (V12 - O12)) */
    Omega_out[0] = 1. - rho[1]*temp2[0] - rho[3]*temp2[2];
    Omega_out[1] = 1. - rho[2]*temp2[1] - rho[4]*temp2[3];
    Omega_out[2] = rho[5] - rho[1]*temp2[1] - rho[3]*temp2[3];

    #ifdef REGULARIZE_BHAT
    Omega_out[0] = std::fmax(Omega_out[0], 0.005);
    Omega_out[1] = std::fmax(Omega_out[1], 0.005);
    #endif
}

/* This is an adaptation of 'cdfqvn'
   Note: "Bhat" is the author's surname.

   Note2: the author's code here clips all the variables at -6,
   but the code here doesn't clip anything. */
double norm_cdf_4d_bhat_internal(const double x[4], const double rho[6])
{
    double mu[2];
    double Sigma[3];
    bv_trunc_std4d_loweronly(rho, x, mu, Sigma);

    double p1 = norm_cdf_1d((x[2] - mu[0]) / std::sqrt(Sigma[0]));
    if (std::isnan(p1) || p1 <= 1e-3) {
        use_plackett:
        return norm_cdf_4d_pg(x, rho);
    }
    double p2 = nonstd_cdf_2d(x[2], x[3], mu[0], mu[1], Sigma[0], Sigma[1], Sigma[2]);
    if (std::isnan(p2)) {
        goto use_plackett;
    }
    if (p2 <= 0.) {
        return 0.;
    }
    double p12 = p2 / p1;
    if (std::isnan(p12) || p12 >= 1e3) {
        goto use_plackett;
    }
    if (p12 <= 0.) {
        return 0.;
    }
    double p3 = norm_cdf_3d(x[0], x[1], x[2], rho[0], rho[1], rho[3]);
    double out = p12 * p3;
    out = std::fmax(out, 0.);
    out = std::fmin(out, 1.);
    return out;
}


/* This is an adaptation of 'cdfqvn' */
double norm_cdf_4d_bhat(const double x[4], const double rho[6])
{
    const double rho_sq[] = {
        rho[0]*rho[0], rho[1]*rho[1], rho[2]*rho[2], rho[3]*rho[3], rho[4]*rho[4], rho[5]*rho[5]
    };
    if (unlikely(rho_sq[1] + rho_sq[2] + rho_sq[3] + rho_sq[4] < EPS_BLOCK)) {
        double p2;
        if ((x[0] < x[2] && x[0] < x[3]) || (x[1] < x[2] && x[1] < x[3])) {
            p2 = norm_cdf_2d(x[0], x[1], rho[0]);
            if (p2 <= 0) {
                return 0;
            }
            return p2 * norm_cdf_2d(x[2], x[3], rho[5]);
        }
        else {
            p2 = norm_cdf_2d(x[2], x[3], rho[5]);
            if (p2 <= 0) {
                return 0;
            }
            return p2 * norm_cdf_2d(x[0], x[1], rho[0]);
        }
    }
    else if (unlikely(rho_sq[0] + rho_sq[1] + rho_sq[4] + rho_sq[5] < EPS_BLOCK)) {
        double p2;
        if ((x[2] < x[0] && x[2] < x[3]) || (x[1] < x[0] && x[1] < x[3])) {
            p2 = norm_cdf_2d(x[2], x[1], rho[3]);
            if (p2 <= 0) {
                return 0;
            }
            return p2 * norm_cdf_2d(x[0], x[3], rho[2]);
        }
        else {
            p2 = norm_cdf_2d(x[0], x[3], rho[2]);
            if (p2 <= 0) {
                return 0;
            }
            return p2 * norm_cdf_2d(x[2], x[1], rho[3]);
        }
    }
    else if (unlikely(rho_sq[0] + rho_sq[2] + rho_sq[3] + rho_sq[5] < EPS_BLOCK)) {
        double p2;
        if ((x[3] < x[2] && x[3] < x[0]) || (x[1] < x[2] && x[1] < x[0])) {
            p2 = norm_cdf_2d(x[3], x[1], rho[4]);
            if (p2 <= 0) {
                return 0;
            }
            return p2 * norm_cdf_2d(x[2], x[0], rho[1]);
        }
        else {
            p2 = norm_cdf_2d(x[2], x[0], rho[1]);
            if (p2 <= 0) {
                return 0;
            }
            return p2 * norm_cdf_2d(x[3], x[1], rho[4]);
        }
    }
    else if (unlikely(rho_sq[0] + rho_sq[1] + rho_sq[2] < EPS_BLOCK)) {
        double p1 = norm_cdf_1d(x[0]);
        if (p1 <= 0) {
            return 0;
        }
        return p1 * norm_cdf_3d(x[1], x[2], x[3], rho[3], rho[4], rho[5]);
    }
    else if (unlikely(rho_sq[0] + rho_sq[3] + rho_sq[4] < EPS_BLOCK)) {
        double p1 = norm_cdf_1d(x[1]);
        if (p1 <= 0) {
            return 0;
        }
        return p1 * norm_cdf_3d(x[0], x[2], x[3], rho[1], rho[2], rho[5]);
    }
    else if (unlikely(rho_sq[1] + rho_sq[3] + rho_sq[5] < EPS_BLOCK)) {
        double p1 = norm_cdf_1d(x[2]);
        if (p1 <= 0) {
            return 0;
        }
        return p1 * norm_cdf_3d(x[0], x[1], x[3], rho[0], rho[2], rho[4]);
    }
    else if (unlikely(rho_sq[2] + rho_sq[4] + rho_sq[5] < EPS_BLOCK)) {
        double p1 = norm_cdf_1d(x[3]);
        if (p1 <= 0) {
            return 0;
        }
        return p1 * norm_cdf_3d(x[0], x[1], x[2], rho[0], rho[1], rho[3]);
    }
    /* If rho(1,2):+1 -> x1 ==  x2
       If rho(1,2):-1 -> x1 == -x2 */
    else if (unlikely(std::fabs(rho[0]) >= HIGH_RHO)) {
        if (x[0] <= rho[0] * x[1]) {
            return norm_cdf_3d(x[0], x[2], x[3], rho[1], rho[2], rho[5]);
        }
        else {
            return norm_cdf_3d(rho[0]*x[1], x[2], x[3], rho[3], rho[4], rho[5]);
        }
    }
    else if (unlikely(std::fabs(rho[1]) >= HIGH_RHO)) {
        if (x[0] <= rho[1] * x[2]) {
            return norm_cdf_3d(x[0], x[1], x[3], rho[0], rho[2], rho[4]);
        }
        else {
            return norm_cdf_3d(rho[1]*x[2], x[1], x[3], rho[3], rho[5], rho[4]);
        }
    }
    else if (unlikely(std::fabs(rho[2]) >= HIGH_RHO)) {
        if (x[0] <= rho[2] * x[3]) {
            return norm_cdf_3d(x[0], x[1], x[2], rho[0], rho[1], rho[3]);
        }
        else {
            return norm_cdf_3d(rho[2]*x[3], x[1], x[2], rho[4], rho[5], rho[3]);
        }
    }
    else if (unlikely(std::fabs(rho[3]) >= HIGH_RHO)) {
        if (x[1] <= rho[3] * x[2]) {
            return norm_cdf_3d(x[1], x[0], x[3], rho[0], rho[4], rho[2]);
        }
        else {
            return norm_cdf_3d(rho[3]*x[2], x[0], x[3], rho[1], rho[5], rho[2]);
        }
    }
    else if (unlikely(std::fabs(rho[4]) >= HIGH_RHO)) {
        if (x[1] <= rho[2] * x[3]) {
            return norm_cdf_3d(x[1], x[0], x[2], rho[0], rho[3], rho[1]);
        }
        else {
            return norm_cdf_3d(rho[2]*x[3], x[0], x[2], rho[2], rho[5], rho[1]);
        }
    }
    else if (unlikely(std::fabs(rho[5]) >= HIGH_RHO)) {
        if (x[2] <= rho[5] * x[3]) {
            return norm_cdf_3d(x[2], x[0], x[1], rho[1], rho[3], rho[0]);
        }
        else {
            return norm_cdf_3d(rho[5]*x[3], x[0], x[1], rho[2], rho[4], rho[0]);
        }
    }

    
    /* Note: for very low determinants, Plackett's original method tends to give better results */
    if (determinant4by4tri(rho) <= 1e-3) {
        return norm_cdf_4d_pg7(x, rho);
    }

    /* Note: the author recommends sorting variables in descending order of
       absicae, saying that it doesn't make much of a difference, but from
       some tests, it does end up making a large difference, particularly when
       the matrices have low determinants.

       The TG order doesn't turn out to provide any better results thant the
       GGE order, while incurring a larger speed penality. Thus, GGE order is
       the preferred choice. */
    #ifdef BHAT_TG_ORDER

    double bv_cdfs[] = {
        norm_cdf_2d_fast(x[0], x[1], rho[0]),
        norm_cdf_2d_fast(x[0], x[2], rho[1]),
        norm_cdf_2d_fast(x[0], x[3], rho[2]),
        norm_cdf_2d_fast(x[1], x[2], rho[3]),
        norm_cdf_2d_fast(x[1], x[3], rho[4]),
        norm_cdf_2d_fast(x[2], x[3], rho[5])
    };

    int lowest_cdf2 = std::distance(bv_cdfs, std::min_element(bv_cdfs, bv_cdfs + 6));
    double x_ordered[4];
    double rho_ordered[6];
    switch (lowest_cdf2) {
        case 0: {
            std::copy(x, x + 4, x_ordered);
            std::copy(rho, rho + 6, rho_ordered);
            break;
        }
        case 1: {
            x_ordered[0] = x[0];
            x_ordered[1] = x[2];
            x_ordered[2] = x[1];
            x_ordered[3] = x[3];
            rho_ordered[0] = rho[1];
            rho_ordered[1] = rho[0];
            rho_ordered[2] = rho[2];
            rho_ordered[3] = rho[3];
            rho_ordered[4] = rho[5];
            rho_ordered[5] = rho[4];
            break;
        }
        case 2: {
            x_ordered[0] = x[0];
            x_ordered[1] = x[3];
            x_ordered[2] = x[1];
            x_ordered[3] = x[2];
            rho_ordered[0] = rho[2];
            rho_ordered[1] = rho[0];
            rho_ordered[2] = rho[1];
            rho_ordered[3] = rho[4];
            rho_ordered[4] = rho[5];
            rho_ordered[5] = rho[3];
            break;
        }
        case 3: {
            x_ordered[0] = x[1];
            x_ordered[1] = x[2];
            x_ordered[2] = x[0];
            x_ordered[3] = x[3];
            rho_ordered[0] = rho[3];
            rho_ordered[1] = rho[0];
            rho_ordered[2] = rho[4];
            rho_ordered[3] = rho[1];
            rho_ordered[4] = rho[5];
            rho_ordered[5] = rho[2];
            break;
        }
        case 4: {
            x_ordered[0] = x[1];
            x_ordered[1] = x[3];
            x_ordered[2] = x[0];
            x_ordered[3] = x[2];
            rho_ordered[0] = rho[4];
            rho_ordered[1] = rho[0];
            rho_ordered[2] = rho[3];
            rho_ordered[3] = rho[2];
            rho_ordered[4] = rho[5];
            rho_ordered[5] = rho[1];
            break;
        }
        case 5: {
            x_ordered[0] = x[2];
            x_ordered[1] = x[3];
            x_ordered[2] = x[0];
            x_ordered[3] = x[1];
            rho_ordered[0] = rho[5];
            rho_ordered[1] = rho[1];
            rho_ordered[2] = rho[3];
            rho_ordered[3] = rho[2];
            rho_ordered[4] = rho[4];
            rho_ordered[5] = rho[0];
            break;
        }
    }

    double tmu[2];
    double trho[3];
    truncate_bvn_2by2block(0., 0.,
                           1., 1., rho_ordered[0],
                           x_ordered[0], x_ordered[1],
                           tmu[0], tmu[1],
                           trho[0], trho[1], trho[2]);
    double ex3 = (x_ordered[2] - tmu[0]) / std::sqrt(trho[0]);
    double ex4 = (x_ordered[3] - tmu[1]) / std::sqrt(trho[1]);
    if (ex3 > ex4) {
        std::swap(x_ordered[2], x_ordered[3]);
        std::swap(rho_ordered[1], rho_ordered[2]);
        std::swap(rho_ordered[3], rho_ordered[4]);
    }
    if (ex4 <= -5.) {
        return norm_cdf_4d_pg(x_ordered, rho_ordered);
    }
    return norm_cdf_4d_bhat_internal(x_ordered, rho_ordered);

    /* This is faster than TG and gives roughly the same results */
    #elif defined(BHAT_GGE_ORDER)
    
    double xpass[4];
    std::copy(x, x + 4, xpass);
    double R[] = {
        1.,     rho[0], rho[1], rho[2],
        rho[0],     1., rho[3], rho[4],
        rho[1], rho[3],     1., rho[5],
        rho[2], rho[4], rho[5],     1.
    };
    double Chol[16];
    double Emu[4];
    gge_reorder(xpass, R, 4, Chol, Emu);
    const double rhopass[] = {R[1], R[2], R[3], R[6], R[7], R[11]};
    return norm_cdf_4d_bhat_internal(xpass, rhopass);
    
    /* This is the author's original order */
    #elif defined(BHAT_SIMPLE_ORDER)
    
    if (x[0] > x[1] && x[0] > x[2] && x[0] > x[3]) {
        const double xpass[] = {x[1], x[2], x[3], x[0]};
        const double rhopass[] = {rho[3], rho[4], rho[0], rho[5], rho[1], rho[2]};
        return norm_cdf_4d_bhat_internal(xpass, rhopass);
    }
    else if (x[1] > x[0] && x[1] > x[2] && x[1] > x[3]) {
        const double xpass[] = {x[0], x[2], x[3], x[1]};
        const double rhopass[] = {rho[1], rho[2], rho[0], rho[5], rho[3], rho[4]};
        return norm_cdf_4d_bhat_internal(xpass, rhopass);
    }
    else if (x[2] > x[0] && x[2] > x[1] && x[2] > x[3]) {
        const double xpass[] = {x[0], x[1], x[3], x[2]};
        const double rhopass[] = {rho[0], rho[2], rho[1], rho[4], rho[3], rho[5]};
        return norm_cdf_4d_bhat_internal(xpass, rhopass);
    }
    else {
        return norm_cdf_4d_bhat_internal(x, rho);
    }

    /* This is another option, also possible in the author's code */
    #else
    return norm_cdf_4d_bhat_internal(x, rho);
    #endif
}

double norm_logcdf_4d_internal(const double x[4], const double rho[6])
{
    double mu[2];
    double Sigma[3];
    bv_trunc_std4d_loweronly(rho, x, mu, Sigma);

    double p1 = norm_logcdf_1d((x[2] - mu[0]) / std::sqrt(Sigma[0]));
    double p2 = norm_logcdf_2d(
        (x[2] - mu[0]) / std::sqrt(Sigma[0]),
        (x[3] - mu[1]) / std::sqrt(Sigma[1]),
        Sigma[2] / (std::sqrt(Sigma[0]) * std::sqrt(Sigma[1]))
    );
    double p3 = norm_logcdf_3d(x[0], x[1], x[2], rho[0], rho[1], rho[3]);
    return p2 - p1 + p3;
}

double norm_logcdf_4d(const double x[4], const double rho[6])
{
    const double rho_sq[] = {
        rho[0]*rho[0], rho[1]*rho[1], rho[2]*rho[2], rho[3]*rho[3], rho[4]*rho[4], rho[5]*rho[5]
    };
    if (unlikely(rho_sq[1] + rho_sq[2] + rho_sq[3] + rho_sq[4] < EPS_BLOCK)) {
        return norm_logcdf_2d(x[2], x[3], rho[5]) + norm_logcdf_2d(x[0], x[1], rho[0]);
    }
    else if (unlikely(rho_sq[0] + rho_sq[1] + rho_sq[4] + rho_sq[5] < EPS_BLOCK)) {
        return norm_logcdf_2d(x[0], x[3], rho[2]) + norm_logcdf_2d(x[2], x[1], rho[3]);
    }
    else if (unlikely(rho_sq[0] + rho_sq[2] + rho_sq[3] + rho_sq[5] < EPS_BLOCK)) {
        return norm_logcdf_2d(x[2], x[0], rho[1]) + norm_logcdf_2d(x[3], x[1], rho[4]);
    }
    else if (unlikely(rho_sq[0] + rho_sq[1] + rho_sq[2] < EPS_BLOCK)) {
        return norm_logcdf_1d(x[0]) + norm_logcdf_3d(x[1], x[2], x[3], rho[3], rho[4], rho[5]);
    }
    else if (unlikely(rho_sq[0] + rho_sq[3] + rho_sq[4] < EPS_BLOCK)) {
        return norm_logcdf_1d(x[1]) + norm_logcdf_3d(x[0], x[2], x[3], rho[1], rho[2], rho[5]);
    }
    else if (unlikely(rho_sq[1] + rho_sq[3] + rho_sq[5] < EPS_BLOCK)) {
        return norm_logcdf_1d(x[2]) + norm_logcdf_3d(x[0], x[1], x[3], rho[0], rho[2], rho[4]);
    }
    else if (unlikely(rho_sq[2] + rho_sq[4] + rho_sq[5] < EPS_BLOCK)) {
        return norm_logcdf_1d(x[3]) + norm_logcdf_3d(x[0], x[1], x[2], rho[0], rho[1], rho[3]);
    }
    /* If rho(1,2):+1 -> x1 ==  x2
       If rho(1,2):-1 -> x1 == -x2 */
    else if (unlikely(std::fabs(rho[0]) >= HIGH_RHO)) {
        if (x[0] <= rho[0] * x[1]) {
            return norm_logcdf_3d(x[0], x[2], x[3], rho[1], rho[2], rho[5]);
        }
        else {
            return norm_logcdf_3d(rho[0]*x[1], x[2], x[3], rho[3], rho[4], rho[5]);
        }
    }
    else if (unlikely(std::fabs(rho[1]) >= HIGH_RHO)) {
        if (x[0] <= rho[1] * x[2]) {
            return norm_logcdf_3d(x[0], x[1], x[3], rho[0], rho[2], rho[4]);
        }
        else {
            return norm_logcdf_3d(rho[1]*x[2], x[1], x[3], rho[3], rho[5], rho[4]);
        }
    }
    else if (unlikely(std::fabs(rho[2]) >= HIGH_RHO)) {
        if (x[0] <= rho[2] * x[3]) {
            return norm_logcdf_3d(x[0], x[1], x[2], rho[0], rho[1], rho[3]);
        }
        else {
            return norm_logcdf_3d(rho[2]*x[3], x[1], x[2], rho[4], rho[5], rho[3]);
        }
    }
    else if (unlikely(std::fabs(rho[3]) >= HIGH_RHO)) {
        if (x[1] <= rho[3] * x[2]) {
            return norm_logcdf_3d(x[1], x[0], x[3], rho[0], rho[4], rho[2]);
        }
        else {
            return norm_logcdf_3d(rho[3]*x[2], x[0], x[3], rho[1], rho[5], rho[2]);
        }
    }
    else if (unlikely(std::fabs(rho[4]) >= HIGH_RHO)) {
        if (x[1] <= rho[2] * x[3]) {
            return norm_logcdf_3d(x[1], x[0], x[2], rho[0], rho[3], rho[1]);
        }
        else {
            return norm_logcdf_3d(rho[2]*x[3], x[0], x[2], rho[2], rho[5], rho[1]);
        }
    }
    else if (unlikely(std::fabs(rho[5]) >= HIGH_RHO)) {
        if (x[2] <= rho[5] * x[3]) {
            return norm_logcdf_3d(x[2], x[0], x[1], rho[1], rho[3], rho[0]);
        }
        else {
            return norm_logcdf_3d(rho[5]*x[3], x[0], x[1], rho[2], rho[4], rho[0]);
        }
    }

    int argsorted[] = {0, 1, 2, 3};
    std::sort(argsorted, argsorted + 4, [&x](const int a, const int b){return x[a] < x[b];});
    const double xpass[] = {x[argsorted[0]], x[argsorted[1]], x[argsorted[2]], x[argsorted[3]]};
    double rhopass[6];
    std::copy(rho, rho + 6, rhopass);
    rearrange_tri(rhopass, argsorted);
    return norm_logcdf_4d_internal(x, rho);
}
