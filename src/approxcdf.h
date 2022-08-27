/* Standard headers */
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <memory>
#include <cassert>

/* For marking functions to export */
#if defined(FOR_R) || defined(FOR_PYTHON)
    #define APPROXCDF_EXPORTED 
#else
    #if defined(_WIN32)
        #ifdef APPROXCDF_COMPILE_TIME
            #define APPROXCDF_EXPORTED __declspec(dllexport)
        #else
            #define APPROXCDF_EXPORTED __declspec(dllimport)
        #endif
    #else
        #if defined(EXPLICITLTY_EXPORT_SYMBOLS) && defined(APPROXCDF_COMPILE_TIME)
            #define APPROXCDF_EXPORTED [[gnu::visibility("default")]]
        #else
            #define APPROXCDF_EXPORTED 
        #endif
    #endif
#endif

/* 'restrict' qualifier from C, if supported */
#if defined(__GNUG__) || defined(__GNUC__) || defined(_MSC_VER) || defined(__clang__) || \
    defined(__INTEL_COMPILER) || defined(__IBMCPP__) || defined(__ibmxl__) || defined(SUPPORTS_RESTRICT)
#   define restrict __restrict
#else
#   define restrict 
#endif

/* Functions from this library */
extern "C" {

/* stdnorm.cpp */
APPROXCDF_EXPORTED
double norm_pdf_1d(double x);
APPROXCDF_EXPORTED
double norm_cdf_1d(double x);
double norm_lcdf_1d(double x);
double norm_logpdf_1d(double x);
double norm_logcdf_1d(double a);

/* drezner.cpp */
APPROXCDF_EXPORTED
double norm_cdf_2d_fast(double x1, double x2, double rho);
APPROXCDF_EXPORTED
double norm_cdf_3d_fast(double x1, double x2, double x3, double rho12, double rho13, double rho23);

/* genz.cpp */
double norm_lcdf_2d(double x1, double x2, double rho);
APPROXCDF_EXPORTED
double norm_cdf_2d(double x1, double x2, double rho);
APPROXCDF_EXPORTED
double norm_cdf_3d(double x1, double x2, double x3, double rho12, double rho13, double rho23);

/* other.cpp */
APPROXCDF_EXPORTED
double norm_cdf_2d_vfast(double x1, double x2, double rho);

/* plackett.cpp */
double determinant4by4tri(const double x_tri[6]);
void inv4by4tri_loweronly(const double x_tri[6], double invX[3]);
double singular_cdf4(const double *restrict x, const double *restrict rho);
double norm_cdf_4d_pg2or5(const double x[4], const double rho[6]);
double norm_cdf_4d_pg7(const double x[4], const double rho[6]);
APPROXCDF_EXPORTED
double norm_cdf_4d_pg(const double x[4], const double rho[6]);

/* bhat.cpp */
double norm_cdf_4d_bhat(const double x[4], const double rho[6]);
double norm_logcdf_4d(const double x[4], const double rho[6]);

/* bhat_lowdim.cpp */
double norm_logcdf_2d(double x1, double x2, double rho);
double norm_logcdf_3d(double x1, double x2, double x3, double rho12, double rho13, double rho23);

/* gge.cpp */
void gge_reorder(double *restrict x, double *restrict R, int n, double *restrict Chol, double *restrict Emu);

/* preprocess_rho.cpp */
void preprocess_rho(double *restrict R, const int ld_R, const int n, double *restrict x,
                    int &restrict pos_st, double &restrict p_independent,
                    int &restrict size_block1, int &restrict size_block2,
                    const int min_n_check_to_check, const bool logp);
void copy_and_standardize
(
    const double *restrict x,
    const double *restrict Sigma,
    const int ld_Sigma,
    const double *restrict mu,
    const int n,
    const bool is_standardized,
    double *restrict x_out,
    double *restrict rho_out,
    double *restrict buffer_sdtdev /* dimension is 'n' */
);

/* ldl.cpp */
void factorize_ldl_2by2blocks(const double *restrict X, const int n,
                              double *restrict diag, double *restrict L,
                              double *restrict temp, double *restrict temp2);
void update_ldl_rank2(double *restrict L, const int ld_L,
                      double *restrict D, const int ld_D,
                      double *restrict O, const int ld_O, /* O is 2x2 */
                      const int n,
                      double *restrict newL, /* buffer dim (n-2)^2 */
                      double *restrict newD, /* buffer dim (n-2)^2 */
                      double *restrict b, /* buffer dim n-2 */
                      double *restrict Z /* buffer dim n-2 */
                      );

/* tvbs.cpp */
void truncate_bvn_2by2block(const double mu1, const double mu2,
                            const double v1, const double v2, const double cv,
                            const double t1, const double t2,
                            double &restrict mu1_out, double &restrict mu2_out,
                            double &restrict v1_out, double &restrict v2_out, double &restrict cv_out);
void truncate_logbvn_2by2block(const double mu1, const double mu2,
                               const double v1, const double v2, const double cv,
                               const double t1, const double t2,
                               double &restrict mu1_out, double &restrict mu2_out,
                               double &restrict v1_out, double &restrict v2_out, double &restrict cv_out);
APPROXCDF_EXPORTED
double norm_cdf_tvbs
(
    const double x[],
    const double Sigma[], const int ld_Sigma, /* typically ld_Sigma=n */
    const double mu[], /* ignored when is_standardized=false */
    const int n,
    const bool is_standardized,
    const bool logp,
    double *restrict buffer /* dim: 6*n^2 + 6*n - 8 */
);

} /* extern "C" */

/* Other necessary includes and useful defines */
#ifdef FOR_R
    #include <R.h>
    #include <Rinternals.h>
    #include <R_ext/Visibility.h>
    #include <R_ext/BLAS.h>
    #include <R_ext/Lapack.h>
#endif

#ifdef __GNUC__
#   define likely(x) __builtin_expect((bool)(x), true)
#   define unlikely(x) __builtin_expect((bool)(x), false)
#else
#   define likely(x) (x)
#   define unlikely(x) (x)
#endif

#ifndef FCONE
    #define FCONE 
#endif

#ifndef F77_CALL
    #define F77_CALL(fn) fn ## _
#endif

#include "cblas.h"
#include "constants.h"
#include "gauss_legendre.h"
#include "helpers.h"

/* configuration for the functions in this library */
#define REGULARIZE_PLACKETT /* default=defined */
#define REGULARIZE_BHAT /* default=defined */

#define PLACKETT_USE_WOODBURY /* default=defined */
#define BHAT_GGE_ORDER /* default=defined */
// #define BHAT_TG_ORDER /* default=NOT defined */
// #define BHAT_SIMPLE_ORDER /* default=NOT defined */

