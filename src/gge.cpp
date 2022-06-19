#include "approxcdf.h"

static inline
double expectation_truncated_normal(double trunc)
{
    double out = -norm_pdf_1d(trunc) / norm_cdf_1d(trunc);
    out = std::fmax(out, -8.);
    return out;
}

/* Inputs and outputs are as follows:
x[n]      <- output will be written here (gets modified in-place)
R[n,n]    <- output will be written here (gets modified in-place)
Chol[n,n] <- will output the Cholesky of permuted R
Emu[n]    <- conditional expected values of 'x', not used anywhere

Note that 'R' must be filled fully, not just the upper triangle.

Note2: this is a verbatim implementation of the method according
to the paper. In reality, even if run for the full amount of
iterations (it avoids the last one), the "Cholesky" doesn't
actually end up factorizing the resulting matrix. */
[[gnu::flatten]]
void gge_reorder(double *restrict x, double *restrict R, int n, double *restrict Chol, double *restrict Emu)
{
    int best_ix = 0;
    double best_cdf = HUGE_VAL;
    for (int ix = 0; ix < n; ix++) {
        if (x[ix] < best_cdf) {
            best_cdf = x[ix];
            best_ix = ix;
        }
    }

    swap_entries_sq_matrix(x, R, n, n, 0, best_ix);
    Chol[0] = 1.;
    for (int ii = 1; ii < n; ii++) {
        Chol[ii*n] = R[ii*n];
    }
    Emu[0] = expectation_truncated_normal(x[0]);

    double num, div, best_div, cthis;
    double *restrict Chol_i;
    double *restrict Chol_j;
    for (int jj = 1; jj < n-1; jj++) {

        best_cdf = HUGE_VAL;
        for (int ii = jj; ii < n; ii++) {
            num = 0;
            div = 0;
            Chol_i = Chol + ii*n;
            for (int mm = 0; mm < jj-1; mm++) {
                cthis = Chol_i[mm];
                num = std::fma(cthis, Emu[mm], num);
                div = std::fma(cthis, cthis, div);
            }
            num = x[ii] - num;
            #ifdef REGULARIZE_BHAT
            div = std::fmin(div, 1. - 0.005);
            #endif
            div = std::sqrt(1. - div);
            if ((num / div) < best_cdf) {
                best_cdf = (num / div);
                best_ix = ii;
                best_div = div;
            }
        }

        swap_entries_sq_matrix(x, R, n, n, jj, best_ix);
        swap_entries_sq_matrix(nullptr, Chol, n, n, jj, best_ix);

        Chol_j = Chol + jj*n;
        Chol_j[jj] = best_div;
        for (int ii = jj+1; ii < n; ii++) {
            num = 0;
            Chol_i = Chol + ii*n;
            for (int mm = 0; mm < jj-1; mm++) {
                num = std::fma(Chol_i[mm], Chol_j[mm], num);
            }
            Chol_i[jj] = (R[ii*n + jj] - num) / best_div;

            num = 0;
            for (int mm = 0; mm < jj-1; mm++) {
                num = std::fma(Chol_j[mm], Emu[mm], num);
            }
            Emu[jj] = expectation_truncated_normal((x[jj] - num) / best_div);
        }

    }
}
