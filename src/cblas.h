#ifndef CBLAS_H

#if !defined(BLAS_H) && !defined(FOR_R)
extern "C" {
double ddot_(int const *, double const *, int const *, double const *, int const *);
void daxpy_(int const *, double const *, double const *, int const *, double *, int const *);
void dgemv_(char const *, int const *, int const *, double const *, double const *, int const *, double const *, int const *, double const *, double *, int const *);
void dgemm_(char const *, char const *, int const *, int const *, int const *, double const *, double const *, int const *, double const *, int const *, double const *, double *, int const *);
void dtrmm_(char const *, char const *, char const *, char const *, int const *, int const *, double const *, double const *, int const *, double *, int const *);
void dtrsm_(char const *, char const *, char const *, char const *, int const *, int const *, double const *, double const *, int const *, double *, int const *);
void dlacpy_(char const *, int const *, int const *, double const *, int const *, double *, int const *);
void dpotri_(char const *, int const *, double *, int const *, int const *);
void dpotrf_(char const *, int const *, double *, int const *, int const *);
}
#endif

extern "C" {

typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
typedef CBLAS_ORDER CBLAS_LAYOUT;

static inline
void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy)
{
    F77_CALL(daxpy)(&n, &alpha, x, &incx, y, &incy);
}

static inline
void cblas_dgemv(const CBLAS_ORDER order,  const CBLAS_TRANSPOSE TransA,  const int m, const int n,
         const double alpha, const double  *a, const int lda,  const double  *x, const int incx,  const double beta,  double  *y, const int incy)
{
    char trans = '\0';
    if (order == CblasColMajor)
    {
        if (TransA == CblasNoTrans)
            trans = 'N';
        else if (TransA == CblasTrans)
            trans = 'T';
        else
            trans = 'C';

        F77_CALL(dgemv)(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy FCONE);
    }

    else
    {
        if (TransA == CblasNoTrans)
            trans = 'T';
        else if (TransA == CblasTrans)
            trans = 'N';
        else
            trans = 'N';

        F77_CALL(dgemv)(&trans, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy FCONE);
    }
}

static inline
void cblas_dgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
         const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc)
{
    char transA = '\0';
    char transB = '\0';
    if (Order == CblasColMajor)
    {
        if (TransA == CblasTrans)
            transA = 'T';
        else if (TransA == CblasConjTrans)
            transA = 'C';
        else
            transA = 'N';

        if (TransB == CblasTrans)
            transB = 'T';
        else if (TransB == CblasConjTrans)
            transB = 'C';
        else
            transB = 'N';

        F77_CALL(dgemm)(&transA, &transB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc FCONE FCONE);
    }

    else
    {
        if (TransA == CblasTrans)
            transB = 'T';
        else if (TransA == CblasConjTrans)
            transB = 'C';
        else
            transB = 'N';

        if (TransB == CblasTrans)
            transA = 'T';
        else if (TransB == CblasConjTrans)
            transA = 'C';
        else
            transA = 'N';

        F77_CALL(dgemm)(&transA, &transB, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc FCONE FCONE);
    }
}

static inline
void cblas_dtrmm(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const  CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 double *B, const int ldb)
{
    char UL = '\0';
    char TA = '\0';
    char SD = '\0';
    char DI = '\0';
  
    if (layout == CblasColMajor)
    {
        SD = (Side == CblasRight)? 'R' : 'L';
        UL = (Uplo == CblasUpper)? 'U' : 'L';
        DI = (Diag == CblasUnit)? 'U' : 'N';
        switch (TransA) {
            case CblasTrans: {
                TA = 'T'; break;
            }
            case CblasConjTrans: {
                TA = 'C'; break;
            }
            case CblasNoTrans: {
                TA = 'N'; break;
            }
            default: {
                assert(0);
            }
        }

        F77_CALL(dtrmm)(&SD, &UL, &TA, &DI, &M, &N, &alpha, A, &lda, B, &ldb FCONE FCONE FCONE FCONE);
    }

    else
    {
        SD = (Side != CblasRight)? 'R' : 'L';
        UL = (Uplo != CblasUpper)? 'U' : 'L';
        DI = (Diag == CblasUnit)? 'U' : 'N';
        switch (TransA) {
            case CblasTrans: {
                TA = 'T'; break;
            }
            case CblasConjTrans: {
                TA = 'C'; break;
            }
            case CblasNoTrans: {
                TA = 'N'; break;
            }
            default: {
                assert(0);
            }
        }

        F77_CALL(dtrmm)(&SD, &UL, &TA, &DI, &N, &M, &alpha, A, &lda, B, &ldb FCONE FCONE FCONE FCONE);
    }
}

static inline
void cblas_dtrsm(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 double *B, const int ldb)
  
{
    char UL = '\0';
    char TA = '\0';
    char SD = '\0';
    char DI = '\0';
  
    if (layout == CblasColMajor)
    {
        SD = (Side == CblasRight)? 'R' : 'L';
        UL = (Uplo == CblasUpper)? 'U' : 'L';
        DI = (Diag == CblasUnit)? 'U' : 'N';
        switch (TransA) {
            case CblasTrans: {
                TA = 'T'; break;
            }
            case CblasConjTrans: {
                TA = 'C'; break;
            }
            case CblasNoTrans: {
                TA = 'N'; break;
            }
            default: {
                assert(0);
            }
        }

        F77_CALL(dtrsm)(&SD, &UL, &TA, &DI, &M, &N, &alpha, A, &lda, B, &ldb FCONE FCONE FCONE FCONE);
    }
    else
    {
        SD = (Side != CblasRight)? 'R' : 'L';
        UL = (Uplo != CblasUpper)? 'U' : 'L';
        DI = (Diag == CblasUnit)? 'U' : 'N';
        switch (TransA) {
            case CblasTrans: {
                TA = 'T'; break;
            }
            case CblasConjTrans: {
                TA = 'C'; break;
            }
            case CblasNoTrans: {
                TA = 'N'; break;
            }
            default: {
                assert(0);
            }
        }

        F77_CALL(dtrsm)(&SD, &UL, &TA, &DI, &N, &M, &alpha, A, &lda, B, &ldb FCONE FCONE FCONE FCONE);
    }
 }

} /* extern "C" */

#endif /* ifndef CBLAS_H */

const int one = 1;
static inline double ddot1(int n, double x[], double y[])
{
    return F77_CALL(ddot)(&n, x, &one, y, &one);
}
