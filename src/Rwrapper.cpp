#include "approxcdf.h"

extern "C" {

/* main function from this package */
SEXP R_norm_cdf_tvbs(SEXP x, SEXP Sigma, SEXP mu, SEXP is_standardized, SEXP logp)
{
    const int n = Rf_xlength(x);
    double *mu_ptr = NULL;
    if (!Rf_isNull(mu)) {
        mu_ptr = REAL(mu);
    }
    double out = norm_cdf_tvbs(
        REAL(x),
        REAL(Sigma), n,
        mu_ptr,
        n,
        (bool) Rf_asLogical(is_standardized),
        (bool) Rf_asLogical(logp),
        NULL
    );
    return Rf_ScalarReal(out);
}

/* other optional functions */
SEXP R_norm_cdf_2d_vfast(SEXP x1, SEXP x2, SEXP rho)
{
    return Rf_ScalarReal(norm_cdf_2d_vfast(Rf_asReal(x1), Rf_asReal(x2), Rf_asReal(rho)));
}

SEXP R_norm_cdf_4d_pg(SEXP x, SEXP R)
{
    double *xx = REAL(x);
    double *RR = REAL(R);
    double Rt[] = {
        RR[1], RR[2], RR[3],
        RR[6], RR[7], RR[11]
    };
    return Rf_ScalarReal(norm_cdf_4d_pg(xx, Rt));
}

SEXP R_norm_cdf_4d_pg7(SEXP x, SEXP R)
{
    double *xx = REAL(x);
    double *RR = REAL(R);
    double Rt[] = {
        RR[1], RR[2], RR[3],
        RR[6], RR[7], RR[11]
    };
    return Rf_ScalarReal(norm_cdf_4d_pg7(xx, Rt));
}

/* for running tests */
#ifdef RUN_TESTS
SEXP R_norm_pdf_1d(SEXP x)
{
    return Rf_ScalarReal(norm_pdf_1d(Rf_asReal(x)));
}

SEXP R_norm_cdf_1d(SEXP x)
{
    return Rf_ScalarReal(norm_cdf_1d(Rf_asReal(x)));
}

SEXP R_norm_cdf_2d_fast(SEXP x1, SEXP x2, SEXP rho)
{
    return Rf_ScalarReal(norm_cdf_2d_fast(Rf_asReal(x1), Rf_asReal(x2), Rf_asReal(rho)));
}

SEXP R_norm_cdf_2d(SEXP x1, SEXP x2, SEXP rho)
{
    return Rf_ScalarReal(norm_cdf_2d(Rf_asReal(x1), Rf_asReal(x2), Rf_asReal(rho)));
}

SEXP R_norm_cdf_3d(SEXP x1, SEXP x2, SEXP x3, SEXP rho12, SEXP rho13, SEXP rho23)
{
    return Rf_ScalarReal(norm_cdf_3d(
        Rf_asReal(x1), Rf_asReal(x2), Rf_asReal(x3),
        Rf_asReal(rho12), Rf_asReal(rho13), Rf_asReal(rho23)
    ));
}

SEXP R_norm_cdf_3d_fast(SEXP x1, SEXP x2, SEXP x3, SEXP rho12, SEXP rho13, SEXP rho23)
{
    return Rf_ScalarReal(norm_cdf_3d_fast(
        Rf_asReal(x1), Rf_asReal(x2), Rf_asReal(x3),
        Rf_asReal(rho12), Rf_asReal(rho13), Rf_asReal(rho23)
    ));
}

SEXP R_singular_cdf4(SEXP x, SEXP R)
{
    double *x_ = REAL(x);
    double *R_ = REAL(R);

    double xpass[] = {
        x_[0], x_[1], x_[2], x_[3]
    };
    double rho[] = {
        R_[1], R_[2], R_[3],
        R_[6], R_[7], R_[11]
    };
    return Rf_ScalarReal(singular_cdf4(xpass, rho));
}

SEXP R_norm_cdf_4d_pg2or5(SEXP x, SEXP R)
{
    double *xx = REAL(x);
    double *RR = REAL(R);
    double Rt[] = {
        RR[1], RR[2], RR[3],
        RR[6], RR[7], RR[11]
    };
    return Rf_ScalarReal(norm_cdf_4d_pg2or5(xx, Rt));
}

SEXP R_norm_cdf_4d_bhat(SEXP x, SEXP R)
{
    double *xx = REAL(x);
    double *RR = REAL(R);
    double Rt[] = {
        RR[1], RR[2], RR[3],
        RR[6], RR[7], RR[11]
    };
    return Rf_ScalarReal(norm_cdf_4d_bhat(xx, Rt));
}

SEXP R_factorize_ldl_2by2blocks(SEXP X)
{
    SEXP dim = getAttrib(X, R_DimSymbol);
    int n = INTEGER(dim)[0];
    SEXP L = PROTECT(Rf_allocMatrix(REALSXP, n, n));
    SEXP D = PROTECT(Rf_allocMatrix(REALSXP, n, n));
    SEXP T1 = PROTECT(Rf_allocMatrix(REALSXP, n, n));
    SEXP T2 = PROTECT(Rf_allocMatrix(REALSXP, n, n));

    factorize_ldl_2by2blocks(REAL(X), n,
                             REAL(D), REAL(L),
                             REAL(T1), REAL(T2));

    SEXP tfun = PROTECT(Rf_install("t"));
    SEXP LcolmajorC = PROTECT(Rf_lang2(tfun, L));
    SEXP DcolmajorC = PROTECT(Rf_lang2(tfun, D));
    SEXP Lcolmajor = PROTECT(Rf_eval(LcolmajorC, R_GlobalEnv));
    SEXP Dcolmajor = PROTECT(Rf_eval(DcolmajorC, R_GlobalEnv));

    SEXP out = PROTECT(allocVector(VECSXP, 2));
    SET_VECTOR_ELT(out, 0, Lcolmajor);
    SET_VECTOR_ELT(out, 1, Dcolmajor);
    UNPROTECT(10);
    return out;
}

SEXP R_determinant_4by4(SEXP x_)
{
    double *x = REAL(x_);
    const double xpass[] = {
        x[1], x[2], x[3],
        x[6], x[7], x[11]
    };
    return Rf_ScalarReal(determinant4by4tri(xpass));
}

#endif /* ifdef RUN_TESTS */

static const R_CallMethodDef callMethods [] = {
    {"R_norm_cdf_tvbs", (DL_FUNC) &R_norm_cdf_tvbs, 5},
    {"R_norm_cdf_2d_vfast", (DL_FUNC) &R_norm_cdf_2d_vfast, 3},
    {"R_norm_cdf_4d_pg", (DL_FUNC) &R_norm_cdf_4d_pg, 2},
    {"R_norm_cdf_4d_pg7", (DL_FUNC) &R_norm_cdf_4d_pg7, 2},
    #ifdef RUN_TESTS
    {"R_norm_pdf_1d", (DL_FUNC) &R_norm_pdf_1d, 1},
    {"R_norm_cdf_1d", (DL_FUNC) &R_norm_cdf_1d, 1},
    {"R_norm_cdf_2d_fast", (DL_FUNC) &R_norm_cdf_2d_fast, 3},
    {"R_norm_cdf_2d", (DL_FUNC) &R_norm_cdf_2d, 3},
    {"R_norm_cdf_3d", (DL_FUNC) &R_norm_cdf_3d, 6},
    {"R_norm_cdf_3d_fast", (DL_FUNC) &R_norm_cdf_3d_fast, 6},
    {"R_singular_cdf4", (DL_FUNC) &R_singular_cdf4, 2},
    {"R_norm_cdf_4d_pg2or5", (DL_FUNC) &R_norm_cdf_4d_pg2or5, 2},
    {"R_norm_cdf_4d_bhat", (DL_FUNC) &R_norm_cdf_4d_bhat, 2},
    {"R_factorize_ldl_2by2blocks", (DL_FUNC) &R_factorize_ldl_2by2blocks, 1},
    {"R_determinant_4by4", (DL_FUNC) &R_determinant_4by4, 1},
    #endif
    {NULL, NULL, 0}
}; 

void attribute_visible R_init_approxcdf(DllInfo *info)
{
    R_registerRoutines(info, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(info, TRUE);
}

} /* extern "C" */
