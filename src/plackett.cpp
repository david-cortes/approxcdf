#include "approxcdf.h"

/* Plackett, Robin L.
   "A reduction formula for normal multivariate integrals."
   Biometrika 41.3/4 (1954): 351-360.
   
   Gassmann, H. I.
   "Multivariate normal probabilities: implementing an old idea of Plackett's."
   Journal of Computational and Graphical Statistics 12.3 (2003): 731-752.

   (Particularly, matrices no. 2, 5, 7, with no.7 being Plackett's original)

   IMPORTANT: something's very unclear in both of the papers when there is
   more than one rho being corrected for:
       Should the matrix in the gradients replace only the rho for the entry
       that's being integrated for, keeping the rest as they were in the
       original matrix, or should it replace all of them with the linear
       combination for all the elements that are going to be corrected for?
   According to the original paper, each correction should replace only one
   entry at each matrix, but then the examples about PSD of the matrices mention
   all of them being replaced at once. The second paper is ambiguous about it.
   One can do the following thought experiment: set the reference matrix to be
   the original rho with one arbitrary element zeroed-out, then calculate the
   correction for it, and calculate the probability for this "reference"
   through the same plackett function recursively, until one or two variables
   become independent. In this way, it will end up integrating at each time
   a single element, but each matrix used for them will be neither the original
   with a different element, nor a linear combination of the corrections: it
   will at each time be a matrix with one less zeroed-out element, but with
   a remainder of elements that come before being the originals and the ones
   that come after being zeros, which does not match with either paper.

   The implementation was tested in the 3 possible ways:
   - Ascending zeroing out (as if it were called recursively).
   - Using the full linear combination as replacement.
   - Replacing a single element.

   Only the first one, even if it does not follow what's described in the papers
   to the letter (but can be obtained by the same papers result by just switching
   to a "bad" reference matrix which is evaluated through the same method), is
   able to produce results that are within the ballpark for full correlation matrices,
   while the other two can get *very* wrong.

   The second paper says that this method, when using matrices 2 and 5 (the ones
   implemented here) should give a maximum error of 5e-5, but from some tests
   the *average* error is around 1e-3 and it can get as bad as 1e-1. It is
   *particularly bad* when the matrices have a low determinant (probably
   because their inverses as calculated here are too imprecise) - for example,
   for a determinant of 1e-5, it might not be correct to even the first decimal.
   The paper also mentions that matrix 2 gives slightly lower errors than
   matrix 5, but some tests here show that matrix 5 has generally lower errors,
   and in general, the error grows very badly with the amount of corrections
   that need to be performed.

   A similar conclusion about the errors being unsatisfactory was reached in:
   Guillaume, Tristan.
   "Computation of the quadrivariate and pentavariate
   normal cumulative distribution functions."
   Communications in Statistics-Simulation and Computation 47.3 (2018): 839-851.

   Not recommended to use, unless there's 3 or 4 zero-valued correlations
   (meaning a single correction) and a relatively large determinant, in which
   case it ends up giving reasonable results with an error of around 1e-8,
   but still very slow. */

const int four = 4;

/* https://stackoverflow.com/questions/2937702/i-want-to-find-determinant-of-4x4-matrix-in-c-sharp */
double determinant4by4tri(const double x_tri[6])
{
    double x00 = x_tri[0] * x_tri[0];
    double x01 = x_tri[0] * x_tri[1];
    double x02 = x_tri[0] * x_tri[2];
    double x03 = x_tri[0] * x_tri[3];
    double x04 = x_tri[0] * x_tri[4];
    double x14 = x_tri[1] * x_tri[4];
    double x23 = x_tri[2] * x_tri[3];

    return
            x_tri[5] * (
                x_tri[1] * (x_tri[2] - x04) +
                x_tri[2] * (x_tri[1] - x03) +
                x_tri[3] * (x_tri[4] - x02) +
                x_tri[4] * (x_tri[3] - x01) +
                x_tri[5] * (x00 - 1.)
            ) +
            x_tri[1] * (x03 - x_tri[1]) +
            x_tri[2] * (x04 - x_tri[2]) +
            x_tri[3] * (x01 - x_tri[3]) +
            x_tri[4] * (x02 - x_tri[4]) +
            x14 * (x14 - x23) +
            x23 * (x23 - x14) +
            -x00 + 1.;
}

void regularized_4by4_inverse(double *restrict X, double *restrict Xinv)
{
    std::copy(X, X + 16, Xinv);
    int lapack_status;
    F77_CALL(dpotrf)("L", &four, Xinv, &four, &lapack_status FCONE);
    F77_CALL(dpotri)("L", &four, Xinv, &four, &lapack_status FCONE);
    #ifdef REGULARIZE_PLACKETT
    double reg = 1e-4;
    double sqroots[4];
    while (lapack_status > 0 || Xinv[10] >= 20. || Xinv[15] >= 20.) {
        X[0]  += reg;
        X[5]  += reg;
        X[10] += reg;
        X[15] += reg;
        std::copy(X, X + 16, Xinv);
        sqroots[0] = std::sqrt(X[0]);
        sqroots[1] = std::sqrt(X[5]);
        sqroots[2] = std::sqrt(X[10]);
        sqroots[3] = std::sqrt(X[15]);
        for (int row = 0; row < 3; row++) {
            for (int col = row+1; col < 4; col++) {
                Xinv[col + row*4] /= sqroots[row] * sqroots[col];
            }
        }
        Xinv[0]  = 1.;
        Xinv[5]  = 1.;
        Xinv[10] = 1.;
        Xinv[15] = 1.;
        F77_CALL(dpotrf)("L", &four, Xinv, &four, &lapack_status FCONE);
        F77_CALL(dpotri)("L", &four, Xinv, &four, &lapack_status FCONE);
        reg *= 2.;
    }
    #endif
}

/* Assuming a 4-by-4 symmetric matrix is partitioned in 2-by-2 blocks:
   X = [A, B]
       [B, C]
   This function calculates the "C" block of the inverse of X, which is
   itself symmetric.
   It assumes that the diagonal of 'X' is composed of ones, thus only the
   upper triangle of 'X' is referenced.

   If 'C' is partitioned as follows:
   C = [c1, c3]
       [c3, c2]
   Then this function outputs {c1, c2, c3}.

   Note that it's possible to hard-code the inversion formula for a
   matrix this small:
   https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix

   However, this hard-coded formula is rather imprecise, and if the
   determinant is small, a Cholesky-based inversion gives lower errors.
   What's more, if the determinant is too small, the result could have
   too big numbers. We don't want variances of 10^10 so in that case
   it's better to apply regularization, even if the result doesn't
   invert the original matrix.

   Plackett's choice of reference  matrix is particularly prone to
   giving problematic matrices to invert. */
void inv4by4tri_loweronly(const double x_tri[6], double invX[3])
{
    // double detX = determinant4by4tri(x_tri);
    double x00 = x_tri[0] * x_tri[0];
    double x01 = x_tri[0] * x_tri[1];
    double x02 = x_tri[0] * x_tri[2];
    double x03 = x_tri[0] * x_tri[3];
    double x04 = x_tri[0] * x_tri[4];
    double x14 = x_tri[1] * x_tri[4];
    double x23 = x_tri[2] * x_tri[3];

    double detX = 
            x_tri[5] * (
                x_tri[1] * (x_tri[2] - x04) +
                x_tri[2] * (x_tri[1] - x03) +
                x_tri[3] * (x_tri[4] - x02) +
                x_tri[4] * (x_tri[3] - x01) +
                x_tri[5] * (x00 - 1.)
            ) +
            x_tri[1] * (x03 - x_tri[1]) +
            x_tri[2] * (x04 - x_tri[2]) +
            x_tri[3] * (x01 - x_tri[3]) +
            x_tri[4] * (x02 - x_tri[4]) +
            x14 * (x14 - x23) +
            x23 * (x23 - x14) +
            -x00 + 1.;


    if (detX <= 1e-2) {
        use_cholesky:
        double X[] = {
            1., x_tri[0],  x_tri[1],  x_tri[2],
            0,        1.,  x_tri[3],  x_tri[4],
            0.,       0.,        1.,  x_tri[5],
            0.,       0.,        0.,         1.
        };
        double Xinv[16];
        regularized_4by4_inverse(X, Xinv);

        invX[0] = Xinv[10];
        invX[1] = Xinv[15];
        invX[2] = Xinv[11];
        return;
    }

    // double x00 = x_tri[0] * x_tri[0];
    // double x01 = x_tri[0] * x_tri[1];
    // double x02 = x_tri[0] * x_tri[2];

    double i10 = 1. + 
                 x_tri[4] * (
                    -x_tri[4] +
                    2. * x02
                 ) +
                 -x00 + 
                 -x_tri[2] * x_tri[2];
    double i11 = -x_tri[5] +
                  x_tri[4] * (
                      x_tri[3] +
                      -x01
                  ) +
                  x00 * x_tri[5] +
                  -x02 * x_tri[3] +
                  x_tri[1] * x_tri[2];
    double i15 = 1. +
                 x_tri[3] * (
                    -x_tri[3] +
                    2. * x01
                 ) +
                 -x00 +
                 -x_tri[1] * x_tri[1];


    invX[0] = i10 / detX;
    invX[1] = i15 / detX;
    invX[2] = i11 / detX;

    if (invX[0] >= 25. || invX[1] >= 25. || invX[0] <= 0. || invX[1] <= 0.) {
        goto use_cholesky;
    }
}

/* Assumes that rho_star takes the place of rho[5] */
void produce_placketts_singular_matrix_coefs(const double *restrict rho, double rho_star, double *restrict coefs)
{
    /* The system of equations can be solved in 6 possible orders,
       but not all of them provide the exact same solutions.
       This computation requires very high precision so it will
       try all possible combinations and take the best one.
       After that, if the solution is not good enough, it will
       refine it through conjugate gradient updates.
       Perhaps it should do these calculations in long double precision. */
    constexpr const int n = 4;
    constexpr const int n1 = n - 1;
    int permutation[] = {0, 1, 2};
    double best_gap = HUGE_VAL;
    double this_gap;
    int best_permutation[3];
    int ordering[4] = {0, 0, 0, 3};
    double coefs_this[3];
    double best_coefs[3];
    double rho_copy[6];
    do {
        
        std::copy(permutation, permutation + n1, ordering);
        std::copy(rho, rho + 5, rho_copy);
        rho_copy[5] = rho_star;
        rearrange_tri(rho_copy, ordering);

        coefs_this[0] = (
            - rho_copy[0]*rho_copy[3]*rho_copy[5]
            + rho_copy[0]*rho_copy[4]
            - rho_copy[1]*rho_copy[3]*rho_copy[4]
            + rho_copy[1]*rho_copy[5]
            + rho_copy[2]*rho_copy[3]*rho_copy[3]
            - rho_copy[2]
        ) / (
            rho_copy[0]*rho_copy[0]
            - 2.*rho_copy[0]*rho_copy[1]*rho_copy[3]
            + rho_copy[1]*rho_copy[1]
            + std::fma(rho_copy[3], rho_copy[3], -1.)
        );
        coefs_this[1] = (
            coefs_this[0]*rho_copy[0]
            - coefs_this[0]*rho_copy[1]*rho_copy[3]
            + rho_copy[3]*rho_copy[5]
            - rho_copy[4]
        ) / (
            std::fma(-rho_copy[3], rho_copy[3], 1.)
        );
        coefs_this[2] = rho_star - rho[1]*coefs_this[0] - rho[3]*coefs_this[1];
        if (
            std::isnan(coefs_this[0]) || std::isinf(coefs_this[0]) ||
            std::isnan(coefs_this[1]) || std::isinf(coefs_this[1]) ||
            std::isnan(coefs_this[2]) || std::isinf(coefs_this[2])
        ) {
            continue;
        }

        this_gap = rho_copy[2]*coefs_this[0] + rho_copy[4]*coefs_this[1] + rho_copy[5]*coefs_this[2];
        this_gap = std::fabs(this_gap - 1.);

        if (this_gap < best_gap) {
            best_gap = this_gap;
            std::copy(permutation, permutation + n1, best_permutation);
            std::copy(coefs_this, coefs_this + n1, best_coefs);
        }
    }
    while (std::next_permutation(permutation, permutation + n1));

    if (best_gap <= 0.1) {
        for (int ix = 0; ix < n1; ix++) {
            coefs[ix] = best_coefs[best_permutation[ix]];
        }
    }
    else {
        coefs[0] = 0; coefs[1] = 0, coefs[2] = 0;
    }

    if (best_gap >= 1e-4) {
        const double cg_tol = 1e-15;
        double reg = 1e-10;
        int cg_counter = 0;
        repeat_cg:
        double r[] = {
            std::fma(-rho[1], coefs[2], rho[2])   - std::fma(rho[0], coefs[1], coefs[0]) - reg*coefs[0],
            std::fma(-rho[3], coefs[2], rho[4])   - std::fma(rho[0], coefs[0], coefs[1]) - reg*coefs[1],
            std::fma(-rho[3], coefs[1], rho_star) - std::fma(rho[1], coefs[0], coefs[2]) - reg*coefs[2]
        };
        double rnorm = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
        double p[] = {r[0], r[1], r[2]};
        double Ap[3];
        double rnorm_new;
        double alpha;
        double beta;
        for (int cg_update = 0; cg_update < 5; cg_update++) {
            if (rnorm <= cg_tol) return;
            Ap[0] = std::fma(rho[0], p[1], p[0]) + rho[1]*p[2] + reg*p[0];
            Ap[1] = std::fma(rho[0], p[0], p[1]) + rho[3]*p[2] + reg*p[1];
            Ap[2] = rho[1]*p[0] + std::fma(rho[3], p[1], p[2]) + reg*p[2];
            alpha = rnorm / (p[0]*Ap[0] + p[1]*Ap[1] + p[2]*Ap[2]);
            coefs[0] = std::fma(alpha, p[0], coefs[0]);
            coefs[1] = std::fma(alpha, p[1], coefs[1]);
            coefs[2] = std::fma(alpha, p[2], coefs[2]);
            r[0] = std::fma(-rho[1], coefs[2], rho[2])   - std::fma(rho[0], coefs[1], coefs[0]) - reg*coefs[0];
            r[1] = std::fma(-rho[3], coefs[2], rho[4])   - std::fma(rho[0], coefs[0], coefs[1]) - reg*coefs[1];
            r[2] = std::fma(-rho[3], coefs[1], rho_star) - std::fma(rho[1], coefs[0], coefs[2]) - reg*coefs[2];

            rnorm_new = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
            beta = rnorm_new / rnorm;
            p[0] = std::fma(beta, p[0], r[0]);
            p[1] = std::fma(beta, p[1], r[1]);
            p[2] = std::fma(beta, p[2], r[2]);
            rnorm = rnorm_new;
        }
        cg_counter++;

        if (cg_counter < 15 && (std::isnan(rnorm) || rnorm > cg_tol)) {
            reg *= 2.5;
            if (std::fabs(coefs[0]) >= 25. || std::fabs(coefs[1]) >= 25. || std::fabs(coefs[2]) >= 25.) {
                coefs[0] = 0; coefs[1] = 0, coefs[2] = 0;
            }
            goto repeat_cg;
        }
    }
}

/* assumes rho{2,3} (index 5 in triangular arrray) is the one that gets changed */
double produce_placketts_singular_matrix_rho(const double *restrict rho)
{
    double one_mins_rho0sq = std::fma(-rho[0], rho[0], 1.);
    double root = std::sqrt(
        (one_mins_rho0sq - rho[1]*rho[1] - rho[3]*rho[3] + 2.*rho[0]*rho[1]*rho[3])
        *
        (one_mins_rho0sq - rho[2]*rho[2] - rho[4]*rho[4] + 2.*rho[0]*rho[2]*rho[4])
    );

    double c1 = rho[2] * std::fma(-rho[0], rho[3], rho[1]) +
                rho[4] * std::fma(-rho[0], rho[1], rho[3]);
    double sol1 = (c1 + root) / one_mins_rho0sq;
    double sol2 = (c1 - root) / one_mins_rho0sq;

    if (std::fabs(sol1) < 1 && std::fabs(sol2) > 1) {
        return sol1;
    }
    else if (std::fabs(sol1) > 1 && std::fabs(sol2) < 1) {
        return sol2;
    }
    else if (rho[5] >= 0) {
        return (sol1 >= 0)? sol1 : sol2;
    }
    else {
        return (sol1 <= 0)? sol1 : sol2;
    }
}

/* Checks which rho would result in the smallest integration interval
   and rearranges the data accordingly so as to put it in rho{2,3},
   but taking also into consideration that large rho values are problematic. */
void find_best_plackett_integrand(double *restrict x, double *restrict rho, double &restrict rho_star)
{
    double rho_ordered[6];
    double new_rhos[6];
    double diffs[6];

    /* (0,1) */
    rho_ordered[0] = rho[5]; rho_ordered[1] = rho[1]; rho_ordered[2] = rho[3];
                             rho_ordered[3] = rho[2]; rho_ordered[4] = rho[4];
                                                      rho_ordered[5] = rho[0];
    new_rhos[0] = produce_placketts_singular_matrix_rho(rho_ordered);
    diffs[0] = std::fabs(rho[0] - new_rhos[0]);

    /* (0,2) */
    rho_ordered[0] = rho[4]; rho_ordered[1] = rho[0]; rho_ordered[2] = rho[3];
                             rho_ordered[3] = rho[2]; rho_ordered[2] = rho[5];
                                                      rho_ordered[5] = rho[1];
    new_rhos[1] = produce_placketts_singular_matrix_rho(rho_ordered);
    diffs[1] = std::fabs(rho[1] - new_rhos[1]);

    /* (0,3) */
    rho_ordered[0] = rho[3]; rho_ordered[1] = rho[0]; rho_ordered[2] = rho[4];
                             rho_ordered[3] = rho[1]; rho_ordered[2] = rho[5];
                                                      rho_ordered[5] = rho[2];
    new_rhos[2] = produce_placketts_singular_matrix_rho(rho_ordered);
    diffs[2] = std::fabs(rho[2] - new_rhos[2]);


    /* (1,2) */
    rho_ordered[0] = rho[2]; rho_ordered[1] = rho[0]; rho_ordered[2] = rho[1];
                             rho_ordered[3] = rho[4]; rho_ordered[2] = rho[5];
                                                      rho_ordered[5] = rho[3];
    new_rhos[3] = produce_placketts_singular_matrix_rho(rho_ordered);
    diffs[3] = std::fabs(rho[3] - new_rhos[3]);

    /* (1,3) */
    rho_ordered[0] = rho[1]; rho_ordered[1] = rho[0]; rho_ordered[2] = rho[2];
                             rho_ordered[3] = rho[3]; rho_ordered[2] = rho[5];
                                                      rho_ordered[5] = rho[4];
    new_rhos[4] = produce_placketts_singular_matrix_rho(rho_ordered);
    diffs[4] = std::fabs(rho[4] - new_rhos[4]);

    /* (2,3) */
    new_rhos[5] = produce_placketts_singular_matrix_rho(rho_ordered);
    diffs[5] = std::fabs(rho[5] - new_rhos[5]);

    constexpr const double rho_too_high = 0.925;
    double best_gap = HUGE_VAL;
    int best_ix = 0;
    for (int ix = 0; ix < 6; ix++) {
        if (std::fmax(std::fabs(rho[ix]), std::fabs(new_rhos[ix])) < rho_too_high) {
            if (diffs[ix] < best_gap) {
                best_ix = ix;
                best_gap = diffs[ix];
            }
        }
    }
    double crit;
    if (best_gap > 1.) {
        for (int ix = 0; ix < 6; ix++) {
            crit = diffs[ix] +
                   std::fmax(
                    1. - rho_too_high,
                    1. - std::fmax(
                            std::fabs(rho[ix]),
                            std::fabs(new_rhos[ix])
                    )
                );
            if (crit < best_gap) {
                best_gap = crit;
                best_ix = ix;
            }
        }
    }

    double x_temp[4];
    switch (best_ix) {
        case 0: {
            x_temp[0] = x[2];
            x_temp[1] = x[3];
            x_temp[2] = x[0];
            x_temp[3] = x[1];
            std::copy(x_temp, x_temp + 4, x);
            rho_ordered[0] = rho[5]; rho_ordered[1] = rho[1]; rho_ordered[2] = rho[3];
                                     rho_ordered[3] = rho[2]; rho_ordered[4] = rho[4];
                                                              rho_ordered[5] = rho[0];
            std::copy(rho_ordered, rho_ordered + 6, rho);
            break;
        }
        case 1: {
            x_temp[0] = x[1];
            x_temp[1] = x[3];
            x_temp[2] = x[0];
            x_temp[3] = x[2];
            std::copy(x_temp, x_temp + 4, x);

            rho_ordered[0] = rho[4]; rho_ordered[1] = rho[0]; rho_ordered[2] = rho[3];
                                     rho_ordered[3] = rho[2]; rho_ordered[2] = rho[5];
                                                              rho_ordered[5] = rho[1];
            std::copy(rho_ordered, rho_ordered + 6, rho);
            break;
        }
        case 2: {
            x_temp[0] = x[1];
            x_temp[1] = x[2];
            x_temp[2] = x[0];
            x_temp[3] = x[3];
            std::copy(x_temp, x_temp + 4, x);

            rho_ordered[0] = rho[3]; rho_ordered[1] = rho[0]; rho_ordered[2] = rho[4];
                                     rho_ordered[3] = rho[1]; rho_ordered[2] = rho[5];
                                                              rho_ordered[5] = rho[2];
            std::copy(rho_ordered, rho_ordered + 6, rho);
            break;
        }
        case 3: {
            x_temp[0] = x[0];
            x_temp[1] = x[3];
            x_temp[2] = x[1];
            x_temp[3] = x[2];
            std::copy(x_temp, x_temp + 4, x);

            rho_ordered[0] = rho[2]; rho_ordered[1] = rho[0]; rho_ordered[2] = rho[1];
                                     rho_ordered[3] = rho[4]; rho_ordered[2] = rho[5];
                                                              rho_ordered[5] = rho[3];
            std::copy(rho_ordered, rho_ordered + 6, rho);
            break;
        }
        case 4: {
            x_temp[0] = x[0];
            x_temp[1] = x[2];
            x_temp[2] = x[1];
            x_temp[3] = x[3];
            std::copy(x_temp, x_temp + 4, x);

            rho_ordered[0] = rho[1]; rho_ordered[1] = rho[0]; rho_ordered[2] = rho[2];
                                     rho_ordered[3] = rho[3]; rho_ordered[2] = rho[5];
                                                              rho_ordered[5] = rho[4];
            std::copy(rho_ordered, rho_ordered + 6, rho);
            break;
        }
        case 5: {
            break;
        }
        default: {
            assert(0);
        }
    }

    rho_star = new_rhos[best_ix];
}

/* In Plackett's singular matrix, assume that x4 is expressed
as a linear combination of the other 3 variables, and assume
some inequality like the ones solved above would be implicit,
such as:
    x1>y1 & x2<y2 & x3<y4 -> x4<y4

This implies that:
    P(x1>y1 & x2<y2 & x3<y3 & x4>y4) = 0

Viewing it in terms of subspaces:
    s1 : (-Inf,y1]
    n1 : (y1, Inf)
    s0 : (-Inf, Inf)

We need to find the subspace:
    s1*s2*s3*s4
(Where s1*s2 is the intersection of the two subspaces)

We have from the inequality satisfaction that, if
e.g. x1>y1 & x2<y2 & x3<y3 -> x4<y4; then:
    n1*s2*s3*s4 = 0
    (s0-s1)*s2*s3*s4 = 0
    s2*s3*s4 - s1*s2*s3*s4 = 0
    s1*s2*s3*s4 = s2*s3*s4

Thus, assuming we sort the variables accordingly in
terms of their signs, one of the following will hold:

case 1:
n1*s2*s3*s4 = 0
    ->
s1*s2*s3*s4 = s2*s3*s4

case2:
n1*n2*s3*s4 = 0
    ->
s1*s2*s3*s4 = s1*s3*s4 + s2*s3*s4 - s3*s4

case3:
n1*n2*n3*s4 = 0
    ->
s1*s2*s3*s4 =
    s1*s2*s4 + s1*s3*s4 + s2*s3*s4
    - s1*s4 - s2*s4 - s3*s4
    + s4

case4:
n1*n2*n3*n4 = 0
    ->
s1*s2*s3*s4 = 
    s1*s2*s3 + s1*s2*s4 + s1*s3*s4 + s2*s3*s4
    - s1*s2 - s1*s3 - s1*s4 - s2*s3 - s2*s4 - s3*s4
    + s1 + s2 + s3 + s4
    - 1

Plackett's paper is extremely scant on details or examples,
and I wasn't sure about how the spaces are supposed to be found.

I thought of the following relationships:

Denote:
    x1 = y1 - s1*e1;
    e1 > 0
    and s1 is either +1 or -1
        if s1=+1, then x1<y1
        if s1=-1, then x1>y1

From the coefficients:
    x4 = c1*x1 + c2*x2 + c3*x3
    x4 = c1*(y1-s1*e1) + c2*(y2-s2*e2) + c3*(y3-s3*e3)
    x4 = c1*y1 + c2*y2 + c3*y3 - c1*s1*e1 - c2*s2*e2 - c3*s3*e3
    y4 - s4*e4 = c1*y1 + c2*y2 + c3*y3 - c1*s1*e1 - c2*s2*e2 - c3*s3*e3
    s4*e4 = c1*s1*e1 + c2*s2*e2 + c3*s3*e3 - c1*y1 - c2*y2 - c3*y3 + y4

Let:
    K = -(c1*y1 + c2*y2 + c3*y3) + y4
Then:
    s4*e4 = K + e1*(c1*s1) + e2*(c2*s2) + e3*(c3*s3)

    If:
        K > 0
        c1*s1 > 0
        c2*s2 > 0
        c3*s3 > 0
    Then it implies s4:+1, which means x4<y4

    If
        K < 0
        c1*s1 < 0
        c2*s2 < 0
        c3*s3 < 0
    Then it implies s4:-1, which means x4>y4

    (If K = 0 then either would do)

If:
    x4 < y4
    s1*x1 < s1*y1
    s2*x2 < s2*y2
    s3*x3 < s3*y3

Then:
    P(x4 < y4 | s1*x1 < s1*y1 & s2*x2 < s2*y2 & s3*x3 < s3*y3) = 1
    P(x4 > y4 & s1*x1 < s1*y1 & s2*x2 < s2*y2 & s3*x3 < s3*y3) = 0

Here we want to find the empty set, so we flip the last sign.
*/
double singular_cdf4(const double *restrict x, const double *restrict rho)
{
    double coefs[4];
    produce_placketts_singular_matrix_coefs(rho, rho[5], coefs);

    double K = -(coefs[0]*x[0] + coefs[1]*x[1] + coefs[2]*x[2]) + x[3];
    double signs[4];
    if (K >= 0.) {
        signs[0] = (coefs[0] >= 0)? 1. : -1.;
        signs[1] = (coefs[1] >= 0)? 1. : -1.;
        signs[2] = (coefs[2] >= 0)? 1. : -1.;
        signs[3] = -1.;
    }
    else {
        signs[0] = (coefs[0] <= 0)? 1. : -1.;
        signs[1] = (coefs[1] <= 0)? 1. : -1.;
        signs[2] = (coefs[2] <= 0)? 1. : -1.;
        signs[3] = +1.;
    }

    constexpr const int n = 4;
    int argsort_signs[] = {0, 1, 2, 3};
    std::sort(
        argsort_signs,
        argsort_signs + n,
        [&signs](const int a, const int b){return signs[a] < signs[b];}
    );
    double temp[4];
    for (int ix = 0; ix < n; ix++) {
        temp[ix] = signs[argsort_signs[ix]];
    }
    std::copy(temp, temp + n, signs);
    

    double x_ordered[4];
    double rho_ordered[6];
    for (int ix = 0; ix < n; ix++) {
        x_ordered[ix] = x[argsort_signs[ix]];
    }
    std::copy(rho, rho + 6, rho_ordered);
    rearrange_tri(rho_ordered, argsort_signs);

    double out;
    if (signs[0] > 0.) {
        return 0.;
    }
    else if (signs[1] > 0.) {
        return norm_cdf_3d(x_ordered[1], x_ordered[2], x_ordered[3], rho_ordered[3], rho_ordered[4], rho_ordered[5]);
    }
    else if (signs[2] > 0.) {
        out = 
            + norm_cdf_3d(x_ordered[0], x_ordered[2], x_ordered[3], rho_ordered[1], rho_ordered[2], rho_ordered[5])
            + norm_cdf_3d(x_ordered[1], x_ordered[2], x_ordered[3], rho_ordered[3], rho_ordered[4], rho_ordered[5])
            - norm_cdf_2d(x_ordered[2], x_ordered[3], rho_ordered[5]);
    }
    else if (signs[3] > 0.) {
        out = 
            + norm_cdf_3d(x_ordered[0], x_ordered[1], x_ordered[3], rho_ordered[0], rho_ordered[2], rho_ordered[4])
            + norm_cdf_3d(x_ordered[0], x_ordered[2], x_ordered[3], rho_ordered[1], rho_ordered[2], rho_ordered[5])
            + norm_cdf_3d(x_ordered[1], x_ordered[2], x_ordered[3], rho_ordered[3], rho_ordered[4], rho_ordered[5])
            - norm_cdf_2d(x_ordered[0], x_ordered[3], rho_ordered[2])
            - norm_cdf_2d(x_ordered[1], x_ordered[3], rho_ordered[4])
            - norm_cdf_2d(x_ordered[2], x_ordered[3], rho_ordered[5])
            + norm_cdf_1d(x_ordered[3]);
    }

    else {
        out = -1.
            + norm_cdf_3d(x_ordered[0], x_ordered[1], x_ordered[2], rho_ordered[0], rho_ordered[1], rho_ordered[3])
            + norm_cdf_3d(x_ordered[0], x_ordered[1], x_ordered[3], rho_ordered[0], rho_ordered[2], rho_ordered[4])
            + norm_cdf_3d(x_ordered[0], x_ordered[2], x_ordered[3], rho_ordered[1], rho_ordered[2], rho_ordered[5])
            + norm_cdf_3d(x_ordered[1], x_ordered[2], x_ordered[3], rho_ordered[3], rho_ordered[4], rho_ordered[5])
            - norm_cdf_2d(x_ordered[0], x_ordered[1], rho_ordered[0])
            - norm_cdf_2d(x_ordered[0], x_ordered[2], rho_ordered[1])
            - norm_cdf_2d(x_ordered[0], x_ordered[3], rho_ordered[2])
            - norm_cdf_2d(x_ordered[1], x_ordered[2], rho_ordered[3])
            - norm_cdf_2d(x_ordered[1], x_ordered[3], rho_ordered[4])
            - norm_cdf_2d(x_ordered[2], x_ordered[3], rho_ordered[5])
            + norm_cdf_1d(x_ordered[0])
            + norm_cdf_1d(x_ordered[1])
            + norm_cdf_1d(x_ordered[2])
            + norm_cdf_1d(x_ordered[3]);
    }

    out = std::fmax(out, 0.);
    out = std::fmin(out, 1.);
    return out;
}

/*  
[ 1, r0, r1, r2]
[r0,  1, r3, r4]
[r1, r3,  1, r5]
[r2, r4, r5,  1]

R11 = [ 1, r0]    R12 = [r1, r2]
      [r0,  1]          [r3, r4]

R21 = [r1, r3]    R22 = [ 1, r5]
      [r2, r4]          [r5,  1]
*/

/* diff(cdf4, rho{0,1}) * (2*pi) */
double grad_cdf4_rho0_mult2pi(const double x[4], const double rho[6])
{
    double expr1 = std::fma(-rho[0], rho[0], 1.);
    #ifdef REGULARIZE_PLACKETT
    double reg = 1e-5;
    double d = 1.;
    while (expr1 <= 0.025) {
        d += reg;
        expr1 = std::fma(-rho[0], rho[0], d*d);
        reg *= 1.5;
    }
    #endif
    expr1 = 1. / expr1;

    double f1 = std::sqrt(expr1);
    double f2 = std::exp(-0.5 * expr1 * (x[0]*x[0] + x[1]*x[1] + 2.*x[0]*x[1]*rho[0]));
    double b3 = expr1 * (
        x[0] * std::fma(-rho[0], rho[3], rho[1]) +
        x[1] * std::fma(-rho[0], rho[1], rho[3])
    );
    double b4 = expr1 * (
        x[0] * std::fma(-rho[0], rho[4], rho[2]) +
        x[1] * std::fma(-rho[0], rho[2], rho[4])
    );

    double C22[3];
    inv4by4tri_loweronly(rho, C22);
    double detC22 = C22[0]*C22[1] - C22[2]*C22[2];
    double var3 = std::sqrt(C22[1] / detC22);
    double var4 = std::sqrt(C22[0] / detC22);
    double cov = -C22[2] / detC22;

    double f3 = norm_cdf_2d(
        (x[2] - b3) / var3,
        (x[3] - b4) / var4,
        cov / (var3 * var4)
    );
    return f1 * f2 * f3;
}

/* This one assumes it is already given C22 (diag1, diag2, corner) */
double grad_cdf4_rho0_noinv_mult2pi(const double x[4], const double rho[6], const double C22[3])
{
    double expr1 = std::fma(-rho[0], rho[0], 1.);
    #ifdef REGULARIZE_PLACKETT
    double reg = 1e-5;
    double d = 1.;
    while (expr1 <= 0.025) {
        d += reg;
        expr1 = std::fma(-rho[0], rho[0], d*d);
        reg *= 1.5;
    }
    #endif
    expr1 = 1. / expr1;
    double f1 = std::sqrt(expr1);
    double f2 = std::exp(-0.5 * expr1 * (x[0]*x[0] + x[1]*x[1] + 2.*x[0]*x[1]*rho[0]));
    double b3 = expr1 * (
        x[0] * std::fma(-rho[0], rho[3], rho[1]) +
        x[1] * std::fma(-rho[0], rho[1], rho[3])
    );
    double b4 = expr1 * (
        x[0] * std::fma(-rho[0], rho[4], rho[2]) +
        x[1] * std::fma(-rho[0], rho[2], rho[4])
    );

    double detC22 = C22[0]*C22[1] - C22[2]*C22[2];
    double var3 = std::sqrt(C22[1] / detC22);
    double var4 = std::sqrt(C22[0] / detC22);
    double cov = -C22[2] / detC22;

    double f3 = norm_cdf_2d(
        (x[2] - b3) / var3,
        (x[3] - b4) / var4,
        cov / (var3 * var4)
    );
    return f1 * f2 * f3;
}

[[gnu::flatten]]
double plackett_correction_rho2_mult2pi(const double x[4], const double rho[6], const double rho_ref)
{
    const double x2[] = {x[0], x[3], x[1], x[2]};
    double rho2[] = {rho[2], rho[0], rho[1], rho[4], rho[5], rho[3]};

    /* Note: if we change only one entry, the inverse of the matrix R
       can be obtained using the Woodbury matrix idendity.

       Assuming we are correcting only for rho(1,2), if we define
       two column vectors as follows:
           u = [diff, 0, 0, 0]
           v = [   0, 1, 0, 0]
       Then R(t) is an SR2 update:
           R(t) = R(0) + u*t(v) + v*t(u)

       Define matrices:
           U =   [u, v]
           V = t([u, v])

       Then:
            R(t) = R(0) + U*V

       By the Woodbury matrix identity:
           inv(R(0) + U*V) = C(t) = C(0) - C(0)*U*(I + V*C(0)*U)*V*C(0)

       Thus, it's only necessary to invert the matrix once. Note however
       that the formula above is not always numerically stable, so it first
       needs to test whether it can use it, and whether the determinant is
       large enough that the loss of accuracy is tolerable.

       Alternatively, one might instead start with the inverse of the
       original matrix and then make updates by substracting instead.

       In theory, could also apply this same formula for getting the
       inverse of a full PG5 point (corrections for a full row):
           u = [d1, d2, d3, 0]
           v = [ 0,  0,  0, 1]

       But for a PG2 point it'd be more complex. */
    #ifdef PLACKETT_USE_WOODBURY
    double det_subst = determinant4by4tri(rho2);
    double det_orig = determinant4by4tri(rho);
    bool use_subst = det_subst >= det_orig;
    double detR = use_subst? det_subst : det_orig;
    constexpr const double det_threshold = 0.05;
    double R[] = {
        1., rho2[0], rho2[1], rho2[2],
        0,  1.,      rho2[3], rho2[4],
        0., 0.,          1.,  rho2[5],
        0., 0.,          0.,       1.
    };
    if (use_subst) {
        R[1] = rho_ref;
    }
    double C[16];
    if (detR > det_threshold) {
        regularized_4by4_inverse(R, C);
    }
    #endif

    double correction = 0;
    double gradp, gradn;
    const double rho_grad = rho2[0];

    #ifdef PLACKETT_USE_WOODBURY
    if (detR <= det_threshold) {
    #endif

        /* Note: here, a more precise estimate is needed, and one would ideally
           want to get more points. However, the gradients in this situation will
           not be well defined near the extreme points and whatever it outputs
           will be wrong, so better avoid points close to the extremes. */
        for (int ix = 0; ix < 4; ix++) {
            rho2[0] = rho_grad * GL8_xp[ix] + rho_ref * GL8_xn[ix];
            gradp = grad_cdf4_rho0_mult2pi(x2, rho2);
            rho2[0] = rho_grad * GL8_xn[ix] + rho_ref * GL8_xp[ix];
            gradn = grad_cdf4_rho0_mult2pi(x2, rho2);

            correction = std::fma(GL8_w[ix], gradp + gradn, correction);
        }

    #ifdef PLACKETT_USE_WOODBURY
    }
    else {

        
        double C26 = C[2] * C[6];
        double C27 = C[2] * C[7];
        double C36 = C[3] * C[6];
        double C37 = C[3] * C[7];

        double C06 = C[0] * C[6];
        double C35 = C[3] * C[5];
        double C066 = C06 * C[6];
        double C077 = C[0] * C[7] * C[7];
        double C225 = C[2] * C[2] * C[5];
        double C335 = C[3] * C35;
        double C067 = C06 * C[7];
        double C126 = C[1] * C26;
        double C127 = C[1] * C27;
        double C136 = C[1] * C36;
        double C137 = C[1] * C37;
        double C235 = C[2] * C35;

        double C126m2 = 2. * C126;
        double C137m2 = 2. * C137;
        double C26m2 = 2. * C26;
        double C37m2 = 2. * C37;
        double C27p36 =  C27 + C36;

        double C066pC126m2pC225 = C066 + C126m2 + C225;
        double C077pC137m2pC335 = C077 + C137m2 + C335;
        double C067p127p136p235 = C067 + C127 + C136 + C235;


        const double C22_base[] = {C[10], C[15], C[11]};
        double C22_corr[3];

        const double rbase = use_subst? rho_ref : rho[2];
        double rcorr;
        double diff, diff_sq, mult;
        double grad_this;
        for (int ix = 0; ix < 8; ix++) {
            for (int direction = 0; direction < 2; direction++) {
                if (direction == 0) {
                    rcorr = rho_grad * GL16_xp[ix] + rho_ref * GL16_xn[ix];
                }
                else {
                    rcorr = rho_grad * GL16_xn[ix] + rho_ref * GL16_xp[ix];
                }
                rho2[0] = rcorr;
                diff = rcorr - rbase;
                mult = std::fma(C[1], diff, 1.);
                if (std::fabs(mult) >= 40.) {
                    goto reinvert_R;
                }
                diff_sq = diff * diff;

                C22_corr[0] = std::fma(-diff_sq, C066pC126m2pC225 + C26m2  / diff, C22_base[0]);
                C22_corr[1] = std::fma(-diff_sq, C077pC137m2pC335 + C37m2  / diff, C22_base[1]);
                C22_corr[2] = std::fma(-diff_sq, C067p127p136p235 + C27p36 / diff, C22_base[2]);

                if (true) {
                    grad_this = grad_cdf4_rho0_noinv_mult2pi(x2, rho2, C22_corr);
                }
                else {
                    reinvert_R:
                    grad_this = grad_cdf4_rho0_mult2pi(x2, rho2);
                }

                if (direction == 0) {
                    gradp = grad_this;
                }
                else {
                    gradn = grad_this;
                }
            }

            correction = std::fma(GL16_w[ix], gradp + gradn, correction);
        }
    }
    #endif

    return correction * (rho[2] - rho_ref);
}

/* Plackett's original, decomposes as follows:
[ 1, r0, r1, r2]
[r0,  1, r3, r4]
[r1, r3,  1, r5]
[r2, r4, r5,  1]

    ->

Ref:
[ 1, r0, r1, r2]
[r0,  1, r3, r4]
[r1, r3,  1, r*]
[r2, r4, r*,  1]

Correction:
[r5 - r*]

With the reference matrix being singular (having a determinant of
zero), which is achieved by finding a rho that would make it singular.
As the matrix is singular, its CDF can be calculated in terms of
lower-dimensional CDFs without any corrections. */
double norm_cdf_4d_pg7(const double x[4], const double rho[6])
{
    double rho_star;
    double rho_ordered[6];
    double x_ordered[4];
    std::copy(rho, rho + 6, rho_ordered);
    std::copy(x, x + 4, x_ordered);
    find_best_plackett_integrand(x_ordered, rho_ordered, rho_star);
    double refP = singular_cdf4(x_ordered, rho_ordered);
    if (std::fabs(rho_star - rho_ordered[5]) <= 1e-2) {
        return refP;
    }

    double rho5to2[] = {
        rho_ordered[1], rho_ordered[3], rho_ordered[5],
        rho_ordered[0], rho_ordered[2], rho_ordered[4]
    };
    double x5to2[] = {x_ordered[2], x_ordered[0], x_ordered[1], x_ordered[3]};
    double correction = plackett_correction_rho2_mult2pi(x5to2, rho5to2, rho_star);
    double out = std::fma(correction, inv_fourPI, refP);
    out = std::fmax(out, 0.);
    out = std::fmin(out, 1.);
    return out;
}

/* This one decomposes as follows:
[ 1, r0, r1, r2]
[r0,  1, r3, r4]
[r1, r3,  1, r5]
[r2, r4, r5,  1]

    ->

Ref:
[ 1, r0, r1,  0]
[r0,  1, r3,  0]
[r1, r3,  1,  0]
[ 0,  0,  0,  1]

Corrections:
[r2, r4, r5]

Zeroes out the last variable, in descending order of its rhos.
The variable that is arranged in the last place is selected so
as to minimize the integration regions. */
double norm_cdf_4d_pg5_recursive(const double x[4], const double rho[6])
{
    const bool rho_nonzero[] {
        std::fabs(rho[0]) > LOW_RHO, std::fabs(rho[1]) > LOW_RHO, std::fabs(rho[2]) > LOW_RHO,
        std::fabs(rho[3]) > LOW_RHO, std::fabs(rho[4]) > LOW_RHO, std::fabs(rho[5]) > LOW_RHO
    };
    if (!rho_nonzero[0] && !rho_nonzero[1] && !rho_nonzero[2]) {
        return norm_cdf_1d(x[0]) * norm_cdf_3d(x[1], x[2], x[3], rho[3], rho[4], rho[5]);
    }
    else if (!rho_nonzero[0] && !rho_nonzero[3] && !rho_nonzero[4]) {
        return norm_cdf_1d(x[1]) * norm_cdf_3d(x[0], x[2], x[3], rho[1], rho[2], rho[5]);
    }
    else if (!rho_nonzero[1] && !rho_nonzero[3] && !rho_nonzero[5]) {
        return norm_cdf_1d(x[2]) * norm_cdf_3d(x[0], x[1], x[3], rho[0], rho[2], rho[4]);
    }
    else if (!rho_nonzero[2] && !rho_nonzero[4] && !rho_nonzero[5]) {
        return norm_cdf_1d(x[3]) * norm_cdf_3d(x[0], x[1], x[2], rho[0], rho[1], rho[3]);
    }

    /* Should maintain desired order: r{2,4,5} */
    if (!rho_nonzero[2]) {
        if (rho_nonzero[4]) {
            swap_rho4:
            const double xpass[] = {x[1], x[0], x[2], x[3]};
            const double rhopass[] = {rho[0], rho[3], rho[4], rho[1], rho[2], rho[5]};
            return norm_cdf_4d_pg5_recursive(xpass, rhopass);
        }
        else if (rho_nonzero[5]) {
            swap_rho5:
            const double xpass[] = {x[2], x[0], x[1], x[3]};
            const double rhopass[] = {rho[1], rho[3], rho[5], rho[0], rho[2], rho[4]};
            return norm_cdf_4d_pg5_recursive(xpass, rhopass);
        }
        else {
            assert(0);
            return NAN;
        }
    }

    /* Make the calculation in descending order of rhos */
    if (rho_nonzero[4] && std::fabs(rho[4]) < std::fabs(rho[2])) {
        goto swap_rho4;
    }
    else if (rho_nonzero[5] && std::fabs(rho[5]) < std::fabs(rho[2])) {
        goto swap_rho5;
    }

    /* If the matrix is near-singular, see if it can approximate it more easily */
    if (determinant4by4tri(rho) <= 1e-3) {
        double rho2to5[] = {rho[3], rho[5], rho[1], rho[4], rho[0], rho[2]};
        double singular_rho2 = produce_placketts_singular_matrix_rho(rho2to5);
        if (std::fabs(singular_rho2 - rho[2]) <= 1e-2) {
            double x_rho2to5[] = {x[2], x[1], x[3], x[0]};
            rho2to5[5] = singular_rho2;
            return singular_cdf4(x_rho2to5, rho2to5);
        }
    }
    
    /* Will zero-out r2 and correct for it */
    const double rhoref[] = {rho[0], rho[1], 0., rho[3], rho[4], rho[5]};
    double refP = norm_cdf_4d_pg5_recursive(x, rhoref);
    double correction = plackett_correction_rho2_mult2pi(x, rho, 0.);
    double out = std::fma(correction, inv_fourPI, refP);
    out = std::fmax(out, 0.);
    out = std::fmin(out, 1.);
    return out;
}

/* This one decomposes as follows:
[ 1, r0, r1, r2]
[r0,  1, r3, r4]
[r1, r3,  1, r5]
[r2, r4, r5,  1]

    ->

Ref:
[ 1, r0,  0,  0]
[r0,  1,  0,  0]
[ 0 , 0,  1, r5]
[ 0,  0, r5,  1]

Corrections:
[r1, r2]
[r3, r4]

Zeros out the upper-left corner, in descending order of its rhos.
The block that is placed there is selected so as to minimize
the integration regions. */
double norm_cdf_4d_pg2_recursive(const double x[4], const double rho[6])
{
    const bool rho_nonzero[] {
        std::fabs(rho[0]) > LOW_RHO, std::fabs(rho[1]) > LOW_RHO, std::fabs(rho[2]) > LOW_RHO,
        std::fabs(rho[3]) > LOW_RHO, std::fabs(rho[4]) > LOW_RHO, std::fabs(rho[5]) > LOW_RHO
    };
    if (!rho_nonzero[1] && !rho_nonzero[2] && !rho_nonzero[3] && !rho_nonzero[4]) {
        return norm_cdf_2d(x[0], x[1], rho[0]) * norm_cdf_2d(x[2], x[3], rho[5]);
    }
    else if (!rho_nonzero[0] && !rho_nonzero[1] &&!rho_nonzero[4] && !rho_nonzero[5]) {
        return norm_cdf_2d(x[2], x[1], rho[3]) * norm_cdf_2d(x[0], x[3], rho[2]);
    }
    else if (!rho_nonzero[0] && !rho_nonzero[2] && !rho_nonzero[3] && !rho_nonzero[5]) {
        return norm_cdf_2d(x[3], x[1], rho[4]) * norm_cdf_2d(x[2], x[0], rho[1]);
    }

    /* Should maintain desired order: r{2,4,3,1} */
    if (!rho_nonzero[2]) {
        if (rho_nonzero[4]) {
            swap_rho4:
            const double xpass[] = {x[1], x[0], x[2], x[3]};
            const double rhopass[] = {rho[0], rho[3], rho[4], rho[1], rho[2], rho[5]};
            return norm_cdf_4d_pg2_recursive(xpass, rhopass);
        }
        else if (rho_nonzero[3]) {
            swap_rho3:
            const double xpass[] = {x[1], x[0], x[3], x[2]};
            const double rhopass[] = {rho[0], rho[4], rho[3], rho[2], rho[1], rho[5]};
            return norm_cdf_4d_pg2_recursive(xpass, rhopass);
        }
        else if (rho_nonzero[1]) {
            swap_rho1:
            const double xpass[] = {x[0], x[1], x[3], x[2]};
            const double rhopass[] = {rho[0], rho[2], rho[1], rho[4], rho[3], rho[5]};
            return norm_cdf_4d_pg2_recursive(xpass, rhopass);
        }
        else {
            assert(0);
            return NAN;
        }
    }

    /* Make the calculation in descending order of rhos */
    if (rho_nonzero[4] && std::fabs(rho[4]) < std::fabs(rho[2])) {
        goto swap_rho4;
    }
    else if (rho_nonzero[3] && std::fabs(rho[3]) < std::fabs(rho[2])) {
        goto swap_rho3;
    }
    else if (rho_nonzero[1] && std::fabs(rho[1]) < std::fabs(rho[2])) {
        goto swap_rho1;
    }

    /* If the matrix is near-singular, see if it can approximate it more easily */
    if (determinant4by4tri(rho) <= 1e-3) {
        double rho2to5[] = {rho[3], rho[5], rho[1], rho[4], rho[0], rho[2]};
        double singular_rho2 = produce_placketts_singular_matrix_rho(rho2to5);
        if (std::fabs(singular_rho2 - rho[2]) <= 1e-2) {
            double x_rho2to5[] = {x[2], x[1], x[3], x[0]};
            rho2to5[5] = singular_rho2;
            return singular_cdf4(x_rho2to5, rho2to5);
        }
    }
    
    /* Will zero-out r2 and correct for it */
    const double rhoref[] = {rho[0], rho[1], 0., rho[3], rho[4], rho[5]};
    double refP = norm_cdf_4d_pg2_recursive(x, rhoref);
    double correction = plackett_correction_rho2_mult2pi(x, rho, 0.);
    double out = std::fma(correction, inv_fourPI, refP);
    out = std::fmax(out, 0.);
    out = std::fmin(out, 1.);
    return out;
}

double norm_cdf_4d_pg2or5(const double x[4], const double rho[6])
{
    const double rho_sq[] = {
        rho[0]*rho[0], rho[1]*rho[1], rho[2]*rho[2],
        rho[3]*rho[3], rho[4]*rho[4], rho[5]*rho[5]
    };

    double norm_b1 = rho_sq[1] + rho_sq[2] + rho_sq[3] + rho_sq[4];
    double norm_b2 = rho_sq[0] + rho_sq[1] + rho_sq[4] + rho_sq[5];
    double norm_b3 = rho_sq[0] + rho_sq[2] + rho_sq[3] + rho_sq[5];

    if (unlikely(norm_b1 < EPS_BLOCK)) {
        double p2;
        if ((x[0] < x[2] && x[0] < x[3]) || (x[1] < x[2] && x[1] < x[3])) {
            p2 = norm_cdf_2d(x[0], x[1], rho[0]);
            if (p2 <= 0.) {
                return 0.;
            }
            return p2 * norm_cdf_2d(x[2], x[3], rho[5]);
        }
        else {
            p2 = norm_cdf_2d(x[2], x[3], rho[5]);
            if (p2 <= 0.) {
                return 0.;
            }
            return p2 * norm_cdf_2d(x[0], x[1], rho[0]);
        }
    }
    else if (unlikely(norm_b2 < EPS_BLOCK)) {
        double p2;
        if ((x[2] < x[0] && x[2] < x[3]) || (x[1] < x[0] && x[1] < x[3])) {
            p2 = norm_cdf_2d(x[2], x[1], rho[3]);
            if (p2 <= 0.) {
                return 0.;
            }
            return p2 * norm_cdf_2d(x[0], x[3], rho[2]);
        }
        else {
            p2 = norm_cdf_2d(x[0], x[3], rho[2]);
            if (p2 <= 0.) {
                return 0.;
            }
            return p2 * norm_cdf_2d(x[2], x[1], rho[3]);
        }
    }
    else if (unlikely(norm_b3 < EPS_BLOCK)) {
        double p2;
        if ((x[3] < x[2] && x[3] < x[0]) || (x[1] < x[2] && x[1] < x[0])) {
            p2 = norm_cdf_2d(x[3], x[1], rho[4]);
            if (p2 <= 0.) {
                return 0.;
            }
            return p2 * norm_cdf_2d(x[2], x[0], rho[1]);
        }
        else {
            p2 = norm_cdf_2d(x[2], x[0], rho[1]);
            if (p2 <= 0.) {
                return 0.;
            }
            return p2 * norm_cdf_2d(x[3], x[1], rho[4]);
        }
    }

    double norm_r1 = rho_sq[0] + rho_sq[1] + rho_sq[2];
    double norm_r2 = rho_sq[0] + rho_sq[3] + rho_sq[4];
    double norm_r3 = rho_sq[1] + rho_sq[3] + rho_sq[5];
    double norm_r4 = rho_sq[2] + rho_sq[4] + rho_sq[5];

    if (unlikely(norm_r1 < EPS_BLOCK)) {
        double p1 = norm_cdf_1d(x[0]);
        if (p1 <= 0) {
            return 0;
        }
        return p1 * norm_cdf_3d(x[1], x[2], x[3], rho[3], rho[4], rho[5]);
    }
    else if (unlikely(norm_r2 < EPS_BLOCK)) {
        double p1 = norm_cdf_1d(x[1]);
        if (p1 <= 0) {
            return 0;
        }
        return p1 * norm_cdf_3d(x[0], x[2], x[3], rho[1], rho[2], rho[5]);
    }
    else if (unlikely(norm_r3 < EPS_BLOCK)) {
        double p1 = norm_cdf_1d(x[2]);
        if (p1 <= 0) {
            return 0;
        }
        return p1 * norm_cdf_3d(x[0], x[1], x[3], rho[0], rho[2], rho[4]);
    }
    else if (unlikely(norm_r4 < EPS_BLOCK)) {
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

    const bool rho_nonzero[] {
        std::fabs(rho[0]) > LOW_RHO, std::fabs(rho[1]) > LOW_RHO, std::fabs(rho[2]) > LOW_RHO,
        std::fabs(rho[3]) > LOW_RHO, std::fabs(rho[4]) > LOW_RHO, std::fabs(rho[5]) > LOW_RHO
    };

    const int zeros_b1 = !rho_nonzero[1] + !rho_nonzero[2] + !rho_nonzero[2] + !rho_nonzero[4];
    const int zeros_b2 = !rho_nonzero[0] + !rho_nonzero[1] + !rho_nonzero[4] + !rho_nonzero[5];
    const int zeros_b3 = !rho_nonzero[0] + !rho_nonzero[2] + !rho_nonzero[3] + !rho_nonzero[5];

    const int zeros_r1 = !rho_nonzero[0] + !rho_nonzero[1] + !rho_nonzero[2];
    const int zeros_r2 = !rho_nonzero[0] + !rho_nonzero[3] + !rho_nonzero[4];
    const int zeros_r3 = !rho_nonzero[1] + !rho_nonzero[3] + !rho_nonzero[5];
    const int zeros_r4 = !rho_nonzero[2] + !rho_nonzero[4] + !rho_nonzero[5];

    double smallest_from_rows = std::fmin(std::fmin(norm_r1, norm_r2), std::fmin(norm_r3, norm_r4));
    double smallest_from_blocks = std::fmin(std::fmin(norm_b1, norm_b1), norm_b3);

    if (unlikely(zeros_b1 >= 2 || zeros_b2 >= 2 || zeros_b3 >= 2)) {
        int ncorr_pg2 = 4 - std::max(std::max(zeros_r1, zeros_r2), std::max(zeros_r3, zeros_r4));
        int ncorr_pg5 = 3 - std::max(zeros_b1, std::max(zeros_b2, zeros_b3));

        if (ncorr_pg5 < ncorr_pg2) {
            goto use_pg5;
        }
        else {
            goto use_pg2;
        }
    }

    if (unlikely(zeros_r1 || zeros_r2 || zeros_r3)) {
        if (zeros_r4 > zeros_r1 && zeros_r4 > zeros_r2 && zeros_r4 > zeros_r3) {
            pg5_row4:
            return norm_cdf_4d_pg5_recursive(x, rho);
        }
        else if (zeros_r3 > zeros_r1 && zeros_r3 > zeros_r2 && zeros_r3 > zeros_r4) {
            pg5_row3:
            const double xpass[] = {x[0], x[1], x[3], x[2]};
            const double rhopass[] = {rho[0], rho[2], rho[1], rho[4], rho[3], rho[5]};
            return norm_cdf_4d_pg5_recursive(xpass, rhopass);
        }
        else if (zeros_r2 > zeros_r1 && zeros_r2 > zeros_r3 && zeros_r2 > zeros_r4) {
            pg5_row2:
            const double xpass[] = {x[0], x[2], x[3], x[1]};
            const double rhopass[] = {rho[1], rho[2], rho[0], rho[5], rho[3], rho[4]};
            return norm_cdf_4d_pg5_recursive(xpass, rhopass);
        }
        else {
            pg5_row1:
            const double xpass[] = {x[1], x[2], x[3], x[0]};
            const double rhopass[] = {rho[3], rho[4], rho[0], rho[5], rho[1], rho[2]};
            return norm_cdf_4d_pg5_recursive(xpass, rhopass);
        }
    }

    if (likely(smallest_from_rows < smallest_from_blocks)) {
        use_pg5:

        if (likely(zeros_r1 == 0 && zeros_r2 == 0 && zeros_r3 == 0 && zeros_r4 == 0)) {
            if (norm_r1 < norm_r2 && norm_r1 < norm_r3 && norm_r1 < norm_r4) {
                goto pg5_row1;
            }
            else if (norm_r2 < norm_r1 && norm_r2 < norm_r3 && norm_r2 < norm_r4) {
                goto pg5_row2;
            }
            else if (norm_r3 < norm_r1 && norm_r3 < norm_r2 && norm_r3 < norm_r4) {
                goto pg5_row3;
            }
            else {
                goto pg5_row4;
            }
        }

        if (zeros_r4 > zeros_r1 && zeros_r4 > zeros_r2 && zeros_r4 > zeros_r3) {
            goto pg5_row4;
        }
        else if (zeros_r3 > zeros_r1 && zeros_r3 > zeros_r2 && zeros_r3 > zeros_r4) {
            goto pg5_row3;
        }
        else if (zeros_r2 > zeros_r1 && zeros_r2 > zeros_r3 && zeros_r2 > zeros_r4) {
            goto pg5_row2;
        }
        else {
            goto pg5_row1;
        }
    }

    else {
        use_pg2:

        if (likely(zeros_b1 == 0 && zeros_b2 == 0 && zeros_b3)) {
            if (norm_b1 < norm_b2 && norm_b1 < norm_b3) {
                pg2_block1:
                return norm_cdf_4d_pg2_recursive(x, rho);
            }
            else if (norm_b2 < norm_b1 && norm_b2 < norm_b3) {
                pg2_block2:
                const double xpass[] = {x[0], x[3], x[1], x[2]};
                const double rhopass[] = {rho[2], rho[0], rho[1], rho[4], rho[5], rho[3]};
                return norm_cdf_4d_pg2_recursive(xpass, rhopass);
            }
            else {
                pg2_block3:
                const double xpass[] = {x[0], x[2], x[1], x[3]};
                const double rhopass[] = {rho[1], rho[0], rho[2], rho[3], rho[5], rho[4]};
                return norm_cdf_4d_pg2_recursive(xpass, rhopass);
            }
        }

        if (zeros_b3 > zeros_b1 && zeros_b3 > zeros_b2) {
            goto pg2_block3;
        }
        else if (zeros_b2 > zeros_b1 && zeros_b2 > zeros_b3) {
            goto pg2_block2;
        }
        else {
            goto pg2_block1;
        }
    }
}

double norm_cdf_4d_pg(const double x[4], const double rho[6])
{
    double max_rho = 0;
    double min_rho = HUGE_VAL;
    for (int ix = 0; ix < 6; ix++) {
        max_rho = std::fmax(max_rho, std::fabs(rho[ix]));
        min_rho = std::fmin(min_rho, std::fabs(rho[ix]));
    }
    if (min_rho <= LOW_RHO || max_rho >= HIGH_RHO) {
        return norm_cdf_4d_pg2or5(x, rho);
    }

    /* These conditions differ significantly from Gassmann's recommendations.
       They were obtained by generating random correlations and thresholds,
       computing the error, and fitting a decision tree model to predict
       which one would do better based on the info that's available up to
       this point (x, rho, determinant).

       In general, Plackett's original method performs really bad, but
       Gassmann's will performs even worse when the determinant is low.
       Nevertheless, Plackett's is much faster, and gives very reasonable
       results when the determinant is below 0.001 or so (since it ends
       up doing very small corrections). */
    double detR = determinant4by4tri(rho);
    if (detR >= 0.04) {
        return norm_cdf_4d_pg2or5(x, rho);
    }
    if (detR <= 0.01) {
        return norm_cdf_4d_pg7(x, rho);
    }
    if (max_rho >= 0.8) {
        return norm_cdf_4d_pg2or5(x, rho);
    }
    double min_x = *std::min_element(x, x + 4);
    if (min_x <= -0.9) {
        return norm_cdf_4d_pg7(x, rho);
    }
    return norm_cdf_4d_pg2or5(x, rho);
}
