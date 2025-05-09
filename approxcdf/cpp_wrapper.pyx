#cython: freethreading_compatible=True, language_level=3

### BLAS and LAPACK
from scipy.linalg.cython_blas cimport (
    ddot, daxpy, dgemv, dgemm, dtrmm, dtrsm
)
from scipy.linalg.cython_lapack cimport (
    dlacpy, dpotri, dpotrf
)

ctypedef double (*ddot__)(const int*, const double*, const int*, const double*, const int*) noexcept nogil
ctypedef void (*daxpy__)(const int*, const double*, const double*, const int*, double*, const int*) noexcept nogil
ctypedef void (*dgemv__)(const char*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*) noexcept nogil
ctypedef void (*dgemm__)(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*) noexcept nogil
ctypedef void (*dtrmm__)(const char*,const char*,const char*,const char*,const int*,const int*,const double*,const double*,const int*,double*,const int*) noexcept nogil
ctypedef void (*dtrsm__)(const char*,const char*,const char*,const char*,const int*,const int*,const double*,const double*,const int*,double*,const int*) noexcept nogil


ctypedef void (*dlacpy__)(const char*, const int*, const int*, const double*, const int*, double*, const int*) noexcept nogil
ctypedef void (*dpotri__)(const char*,const int*,double*,const int*,const int*) noexcept nogil
ctypedef void (*dpotrf__)(const char*, const int*, double*, const int*, const int*) noexcept nogil



cdef public double ddot_(const int* a1, const double* a2, const int* a3, const double* a4, const int* a5) noexcept nogil:
    return (<ddot__>ddot)(a1, a2, a3, a4, a5)

cdef public void daxpy_(const int* a1, const double* a2, const double* a3, const int* a4, double* a5, const int* a6) noexcept nogil:
    (<daxpy__>daxpy)(a1, a2, a3, a4, a5, a6)

cdef public void dgemv_(const char* a1, const int* a2, const int* a3, const double* a4, const double* a5, const int* a6, const double* a7, const int* a8, const double* a9, double* a10, const int* a11) noexcept nogil:
    (<dgemv__>dgemv)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)

cdef public void dgemm_(const char* a1, const char* a2, const int* a3, const int* a4, const int* a5, const double* a6, const double* a7, const int* a8, const double* a9, const int* a10, const double* a11, double* a12, const int* a13) noexcept nogil:
    (<dgemm__>dgemm)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13)

cdef public void dtrmm_(const char* a1,const char* a2,const char* a3,const char* a4,const int* a5,const int* a6,const double* a7,const double* a8,const int* a9,double* a10,const int* a11) noexcept nogil:
    (<dtrmm__>dtrmm)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)

cdef public void dtrsm_(const char* a1,const char* a2,const char* a3,const char* a4,const int* a5,const int* a6,const double* a7,const double* a8,const int* a9,double* a10,const int* a11) noexcept nogil:
    (<dtrsm__>dtrsm)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)

cdef public void dlacpy_(const char* a1, const int* a2, const int* a3, const double* a4, const int* a5, double* a6, const int* a7) noexcept nogil:
    (<dlacpy__>dlacpy)(a1, a2, a3, a4, a5, a6, a7)

cdef public void dpotri_(const char* a1,const int* a2,double* a3,const int* a4,const int* a5) noexcept nogil:
    (<dpotri__>dpotri)(a1, a2, a3, a4, a5)

cdef public void dpotrf_(const char* a1, const int* a2, double* a3, const int* a4, const int* a5) noexcept nogil:
    (<dpotrf__>dpotrf)(a1, a2, a3, a4, a5)


### The acutal library
import numpy as np
cimport numpy as np
from libcpp cimport bool as bool

cdef extern from "approxcdf.h":
    double norm_cdf_2d_vfast(double x1, double x2, double rho)
    double norm_cdf_4d_pg(const double x[4], const double rho[6])
    double norm_cdf_4d_pg7(const double x[4], const double rho[6])
    double norm_cdf_tvbs(
        const double x[],
        const double Sigma[], const int ld_Sigma,
        const double mu[],
        const int n,
        const bool is_standardized,
        const bool logp,
        double *buffer
    )

def py_norm_cdf_tvbs(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] mu, np.ndarray[double, ndim=2] S, int ld_S, bool is_standardized, bool logp):
    cdef int n = x.shape[0]
    cdef double *ptr_mu = NULL
    if mu.shape[0]:
        ptr_mu = &mu[0]
    return norm_cdf_tvbs(
        &x[0],
        &S[0,0], ld_S,
        ptr_mu,
        n,
        is_standardized,
        logp,
        NULL
    );

def py_norm_cdf_2d_vfast(double x1, double x2, double rho):
    return norm_cdf_2d_vfast(x1, x2, rho)

def py_norm_cdf_4d_pg(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=2] S):
    cdef np.ndarray[double, ndim=1] Stri = np.ascontiguousarray(S[np.triu_indices(4, 1)])
    return norm_cdf_4d_pg(&x[0], &Stri[0])

def py_norm_cdf_4d_pg7(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=2] S):
    cdef np.ndarray[double, ndim=1] Stri = np.ascontiguousarray(S[np.triu_indices(4, 1)])
    return norm_cdf_4d_pg7(&x[0], &Stri[0])
