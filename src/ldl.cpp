#include "approxcdf.h"

/* Note: this is a highly non-standard factorization that
   that's not included in lapack.

   Factorizes X = L * D * t(L)

   Where D is not actually a diagonal, but it's a matrix with
   2-by-2 symmetric blocks on its main diagonal - example:
   [[v11, d1,    0,   0],
    [ d1, v12,   0,   0],
    [ 0,    0,  v21,  d2],
    [ 0,    0,   d2, v22]]
   And L is a lower triangular matrix, with the caveat that the
   main diagonal is equal to 1, and every second element in the
   first subdiagonal is zero - example:
   [[1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [L, L, 1, 0, 0],
    [L, L, 0, 1, 0],
    [L, L, L, L, 1]]

   Requires passing two temporary arrays of dimension n-by-n. */
void factorize_ldl_2by2blocks(const double *restrict X, const int n,
                              double *restrict diag, double *restrict L,
                              double *restrict temp, double *restrict temp2)
{
    std::copy(X, X + n*n, temp);
    std::fill(diag, diag + n*n, 0);
    std::fill(L, L + n*n, 0);
    for (int ix = 0; ix < n; ix++) {
        L[ix*(n+1)] = 1.;
    }

    #ifdef REGULARIZE_BHAT
    const double reg_base = 1e-8;
    double reg;
    #endif
    double detD;
    double invD[4];
    double blockD[4];
    int idx, idx2;
    int ix;
    for (ix = 0; ix < n-2; ix += 2) {
        
        blockD[0] = temp[ix*(n+1)];          blockD[1] = temp[ix*(n+1) + 1];
        blockD[2] = temp[(ix+1)*(n+1) - 1];  blockD[3] = temp[(ix+1)*(n+1)];
        detD = blockD[0] * blockD[3] - blockD[1] * blockD[2];
        
        #ifdef REGULARIZE_BHAT
        reg = reg_base;
        while (detD <= 1e-3) {
            blockD[0] += reg;
            blockD[3] += reg;
            detD = blockD[0] * blockD[3] - blockD[1] * blockD[2];
            reg *= 1.5;
        }
        #endif
        
        diag[ix*(n+1)] = blockD[0];          diag[ix*(n+1) + 1] = blockD[1];
        diag[(ix+1)*(n+1) - 1] = blockD[2];  diag[(ix+1)*(n+1)] = blockD[3];

        invD[0] =  blockD[3] / detD;   invD[1] = -blockD[1] / detD;
        invD[2] = -blockD[2] / detD;   invD[3] =  blockD[0] / detD;

        idx = n*(ix+2) + ix;
        idx2 = n - ix - 2;
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            idx2, 2, 2,
            1., temp + idx, n,
            invD, 2,
            0., L + idx, n
        );

        /* TODO: could it reuse space from 'temp' instead? */
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            idx2, 2, 2,
            1., L + idx, n,
            blockD, 2,
            0., temp2, 2
        );

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            idx2, idx2, 2,
            -1., temp2, 2,
            L + idx, n,
            1., temp + (ix+2)*(n+1), n
        );
    }

    int size_last = n - ix;
    assert(size_last < 2);
    if (size_last == 1) {
        diag[ix*(n+1)] = temp[ix*(n+1)];
    }
    else {
        diag[ix*(n+1)] = temp[ix*(n+1)];                   diag[ix*(n+1) + 1] = temp[ix*(n+1) + 1];
        diag[(ix+1)*(n+1) - 1] = temp[(ix+1)*(n+1) - 1];   diag[(ix+1)*(n+1)] = temp[(ix+1)*(n+1)];
    }
}

void update_ldl_rank2(double *restrict L, const int ld_L,
                      double *restrict D, const int ld_D,
                      double *restrict O, const int ld_O, /* O is 2x2 */
                      const int n,
                      double *restrict newL, /* buffer dim (n-2)^2 */
                      double *restrict newD, /* buffer dim (n-2)^2 */
                      double *restrict b, /* buffer dim n-2 */
                      double *restrict Z /* buffer dim n-2 */
                      )
{
    double M1[] = {O[0], O[1], O[ld_O], O[ld_O + 1]};
    

    const int n1 = n - 2;
    double *restrict L22 = L + 2 + 2*ld_L;
    double *restrict D2 = D + 2 + 2*ld_D;

    std::fill(b, b + n1*2, 0.);
    std::fill(Z, Z + n1*2, 0.);

    const int two = 2;
    F77_CALL(dlacpy)("A", &two, &n1, L + 2*ld_L, &ld_L, Z, &two FCONE);


    /* TODO: perhaps here it should add regularization if the determinant is too low */
    cblas_dtrsm(
        CblasRowMajor, CblasLeft,
        CblasLower, CblasNoTrans,
        CblasUnit, n1, 2,
        1., L22, ld_D,
        Z, 2
    );

    std::fill(newD, newD + n1*n1, 0.);
    std::fill(newL, newL + n1*n1, 0.);
    for (int ix = 0; ix < n1; ix++) {
        newL[ix*(n1+1)] = 1.;
    }

    if (n1 <= 2) {
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n1, 2, 2,
            1., Z, 2,
            M1, 2,
            0., b, 2
        );
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            n1, n1, 2,
            1., b, 2,
            Z, 2,
            0., newD, n1
        );
        for (int row = 0; row < n1; row++) {
            cblas_daxpy(n1, 1., D2 + row*ld_D, 1, newD + row*n1, 1);
        }
    }
    else {
        double T[4];
        double invD[4];
        double detD;
        #ifdef REGULARIZE_BHAT
        const double reg_base = 1e-8;
        double reg;
        #endif
        for (int ix = 0; ix < n1; ix += 2) {
            if (ix+2 <= n1) {
                T[0] = M1[0]*Z[ix*2]     + M1[1]*Z[ix*2 + 1];
                T[1] = M1[0]*Z[ix*2 + 2] + M1[1]*Z[ix*2 + 3];
                T[2] = M1[2]*Z[ix*2]     + M1[3]*Z[ix*2 + 1];
                T[3] = M1[2]*Z[ix*2 + 2] + M1[3]*Z[ix*2 + 3];

                newD[ix + 0 + (ix + 0)*n1] = D2[ix + 0 + (ix + 0)*ld_D] + Z[0 + (ix+0)*2]*T[0] + Z[1 + (ix+0)*2]*T[2];
                newD[ix + 1 + (ix + 0)*n1] = D2[ix + 1 + (ix + 0)*ld_D] + Z[0 + (ix+0)*2]*T[1] + Z[1 + (ix+0)*2]*T[3];
                newD[ix + 0 + (ix + 1)*n1] = D2[ix + 0 + (ix + 1)*ld_D] + Z[0 + (ix+1)*2]*T[0] + Z[1 + (ix+1)*2]*T[2];
                newD[ix + 1 + (ix + 1)*n1] = D2[ix + 1 + (ix + 1)*ld_D] + Z[0 + (ix+1)*2]*T[1] + Z[1 + (ix+1)*2]*T[3];

                detD = newD[ix + 0 + (ix + 0)*n1] * newD[ix + 1 + (ix + 1)*n1] - newD[ix + 1 + (ix + 0)*n1] * newD[ix + 0 + (ix + 1)*n1];
                #ifdef REGULARIZE_BHAT
                reg = reg_base;
                while (detD <= 1e-4) {
                    newD[0] += reg;
                    newD[3] += reg;
                    detD = newD[ix + 0 + (ix + 0)*n1] * newD[ix + 1 + (ix + 1)*n1] - newD[ix + 1 + (ix + 0)*n1] * newD[ix + 0 + (ix + 1)*n1];
                    reg *= 1.5;
                }
                #endif
                invD[0] =  newD[ix + 1 + (ix + 1)*n1] / detD;   invD[1] = -newD[ix + 1 + (ix + 0)*n1] / detD;
                invD[2] = -newD[ix + 0 + (ix + 1)*n1] / detD;   invD[3] =  newD[ix + 0 + (ix + 0)*n1] / detD;

                b[0 + (ix + 0)*2] = T[0]*invD[0] + T[1]*invD[2];
                b[1 + (ix + 0)*2] = T[2]*invD[0] + T[3]*invD[2];
                b[0 + (ix + 1)*2] = T[0]*invD[1] + T[1]*invD[3];
                b[1 + (ix + 1)*2] = T[2]*invD[1] + T[3]*invD[3];


                M1[0] -= T[0]*b[0 + (ix + 0)*2] + T[1]*b[0 + (ix + 1)*2];
                M1[1] -= T[0]*b[1 + (ix + 0)*2] + T[1]*b[1 + (ix + 1)*2];
                M1[2] -= T[2]*b[0 + (ix + 0)*2] + T[3]*b[0 + (ix + 1)*2];
                M1[3] -= T[2]*b[1 + (ix + 0)*2] + T[3]*b[1 + (ix + 1)*2];

                // TODO: maybe check if newL could be skipped by overwriting L instead
                if (ix < n1-2) {
                    if (ix+1 != n1-2) {
                        cblas_dgemm(
                            CblasRowMajor, CblasNoTrans, CblasTrans,
                            2, ix+2, 2,
                            1., Z + (ix+2)*2, 2,
                            b, 2,
                            0., newL + (ix+2)*n1, n1
                        );
                    }
                    else {
                        cblas_dgemv(
                            CblasRowMajor, CblasNoTrans,
                            ix+2, 2,
                            1., b, 2,
                            Z + (ix+2)*2, 1,
                            0., newL + (ix+2)*n1, 1
                        );
                    }
                }
            }
            else if (ix == n1 - 1) {
                T[0] = M1[0]*Z[0 + ix*2] + M1[1]*Z[1 + ix*2];
                T[1] = M1[2]*Z[0 + ix*2] + M1[3]*Z[1 + ix*2];
                
                newD[ix*(n1+1)] = D2[ix*(ld_D+1)] + Z[0 + ix*2]*T[0] + Z[1 + ix*2]*T[1];
            }
        }
    }


    if (n1 > 2) {
        cblas_dtrmm(
            CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit,
            n1, n1,
            1., newL, n1,
            L22, ld_L
        );
    }

    F77_CALL(dlacpy)("A", &n1, &n1, newD, &n1, D2, &ld_D FCONE);
} 
