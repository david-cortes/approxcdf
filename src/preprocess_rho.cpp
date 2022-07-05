#include "approxcdf.h"

/* This function pre-processes a correlation matrix by:
   - Eliminating perfectly correlated variables (only the one implying
     the lowest upper bound is required).
   - Eliminating independent variables (the final probability is a simple
     product of their probabilities and the probability of the remaining
     variables with have correlations among themselves).
   - Trying to split up variables into two independent blocks if possible
     (in which case the final probability is the product of the block
     probabilities).
  
  The output will be a potentially reduced R matrix, with rows/columns
  swapped accordingly as necessary, and the entries in 'x' also swapped
  in order to match the new correlation matrix.

  The output matrix will have the eliminated entries put as the first rows. */
void preprocess_rho(double *restrict R, const int ld_R, const int n, double *restrict x,
                    int &restrict pos_st, double &restrict p_independent,
                    int &restrict size_block1, int &restrict size_block2,
                    const int min_n_check_to_check)
{
    p_independent = 1.;
    size_block1 = n;
    size_block2 = 0;
    pos_st = 0;

    /* Step 1: look for perfect correlations */
    for (int row = pos_st; row < n-1; row++) {
        for (int col = row+1; col < n; col++) {
            if (rho_is_one(R[col + row*ld_R])) {
                if (x[row] < R[col + row*ld_R] * x[col]) {
                    swap_entries_sq_matrix(x, R, ld_R, n, col, row);
                }
                swap_entries_sq_matrix(x, R, ld_R, n, pos_st, row);
                pos_st++;
            }
        }
    }
    size_block1 = n - pos_st;
    if (size_block1 <= min_n_check_to_check) {
        return;
    }

    /* Step 2: look for independent variables. */
    int row_zeros;
    for (int row = pos_st; row < n; row++) {
        
        row_zeros = 0;
        for (int col = pos_st; col < n; col++) {
            row_zeros += rho_is_zero(R[col + row*ld_R]);
        }
        if (row_zeros >= n-pos_st-1) {
            p_independent *= norm_cdf_1d(x[row]);
            if (p_independent <= 0.) {
                return;
            }
            swap_entries_sq_matrix(x, R, ld_R, n, pos_st, row);
            pos_st++;
        }
    }
    size_block1 = n - pos_st;
    if (size_block1 <= min_n_check_to_check) {
        return;
    }

    /* Step 3: look for independent blocks.
    This one is more tricky to do. In broad terms, imagine that we want to put
    a block of zeros at the top-left corner that would imply independence, with
    the rows that have the largest numbers of zeros coming first.
    
    Examples:
    [1, r, 0, 0, 0]    <- Left is what we seek.            [1, r, r, 0, 0]
    [r, 1, 0, 0, 0]       Right one is an equivalent  ->   [r, 1, r, 0, 0]
    [0, 0, 1, r, r]       reordering of the matrix.        [r, r, 1, 0, 0]
    [0, 0, r, 1, r]                                        [0, 0, 0, 1, r]
    [0, 0, r, r, 1]                                        [0, 0, 0, r, 1]

    Note that diagonals cannot be zeros, so something like this would
    not be a valid correlation matrix (it's not possible by design),
    and thus such pattern we need not bother with:
    [r, r, 0, 0, 0, 0]
    [r, r, 0, 0, 0, 0]
    [r, r, 0, 0, 0, 0]   <- This is not a correlation matrix
    [0, 0, r, r, r, r]      (must always have ones at diagonals,
    [0, 0, r, r, r, r]       and must alawys be symmetric)
    [0, 0, r, r, r, r]
    
    The best possible scenario is something that splits the matrix into two
    equally-sized blocks - e.g.

    [1, r, r, 0, 0, 0]                      [1, r, 0, 0, 0, 0]
    [r, 1, r, 0, 0, 0]                      [r, 1, 0, 0, 0, 0]
    [r, r, 1, 0, 0, 0]   <- Left one is ->  [0, 0, 1, r, r, r]
    [0, 0, 0, 1, r, r]      preferrable     [0, 0, r, 1, r, r]
    [0, 0, 0, r, 1, r]                      [0, 0, r, r, 1, r]
    [0, 0, 0, r, r, 1]                      [0, 0, r, r, r, 1]

    Note in the left pattern that there must be at least n/2 rows with
    at least n/2 zeros each, while in the right pattern there must be at
    least (n/2 - 1) rows with at least (n/2 + 1) zeros each, and in each
    case, the columns with zeros must be the exact same ones among the
    rows that belong to the group.

    In the first group, there's a maximum of C(n, n/2) possible such
    blocks, while in the second, there's a maximum of C(n, n/2 - 1)
    possible such blocks, where C(n,k) = n!/((n-k)!*k!).

    Thus, finding such a block directly is hard (although it could also
    be done with graph-based methods, but that's a more complicated
    approach).

    Instead, we can approach it the other way around: first find two
    rows that are dependent (perhaps starting with the ones that have
    the largest amounts of non-zeros):
    [1, r]
    [r, 1]

    Then try to expand this set by adding another row that would be
    dependent on at least one of those two:
    [1, r, r]          [1, r, 0]          [1, r, r]
    [r, 1, r]   (or)   [r, 1, r]   (or)   [r, 1, 0]
    [r, r, 1]          [0, r, 1]          [r, 0, 1]

    And so on until being unable to add any further rows. This way, it's
    O(n^3) to check the full matrix for independent blocks (up to O(n^4)
    if we take into account the row swapping along the way).

    In any event, an independent block would imply a minimum number of
    zeros in the triangular part of the matrix of at least 2*(n-2) (this
    is the case in which two rows are independent of the rest, and any case
    of a single row being independent would have already been handled in the
    earlier conditions), so if there aren't enough zeros left, can avoid
    bothering to check at all.

    As a greedy trick to speed it up, depending on which exit is more likely
    to happen, could either:
      (a) Put the row with the highest number of non-zeros first and the
          row with the lowest number of non-zeros last, or perhaps do a
          full sort by descending number of non-zeros. This way, it will
          speed up the search for non-zero rows and reach the end of the
          full loop faster.
      (b) Sort the rows in ascending order of zeros. This way, the more
          sparse rows will be on top and it is more likely to encounter
          the exit condition early on.
    Since we expect that most cases will not have any independent blocks,
    strategy (a) is likely to result in a faster procedure.
    */
    int n_zeros = 0;
    for (int row = pos_st; row < n-1; row++) {
        for (int col = row+1; col < n; col++) {
            n_zeros += rho_is_zero(R[col + row*ld_R]);
        }
    }
    if (n_zeros < 2*(n-pos_st-2)) {
        return;
    }

    int max_zeros = 0;
    int min_zeros = n;
    int row_max_zeros = pos_st;
    int row_min_zeros = pos_st;
    int n_zeros_row;
    for (int row = pos_st; row < n; row++) {
        n_zeros_row = 0;
        for (int col = pos_st; col < n; col++) {
            n_zeros += rho_is_zero(R[col + row*ld_R]);
        }
        if (n_zeros_row > max_zeros) {
            max_zeros = n_zeros_row;
            row_max_zeros = row;
        }
        if (n_zeros_row < min_zeros) {
            min_zeros = n_zeros_row;
            row_min_zeros = row;
        }
    }
    if (max_zeros < 2) {
        return;
    }
    if (max_zeros - min_zeros >= 2) {
        swap_entries_sq_matrix(x, R, ld_R, n, row_max_zeros, pos_st);
        swap_entries_sq_matrix(x, R, ld_R, n, row_min_zeros, n-1);
    }

    int pos;
    for (pos = pos_st; pos < n-1; pos++) {
        
        for (int row = pos+1; row < n; row++) {
            for (int col = pos_st; col <= pos; col++) {
                if (!rho_is_zero(R[col + row*ld_R])) {
                    swap_entries_sq_matrix(x, R, ld_R, n, pos+1, row);
                    goto next_pos;
                }
            }
        }
        break;
        next_pos:
        {}
    }

    if (pos < n-2) {
        size_block1 = pos - pos_st + 1;
        size_block2 = (n - pos_st) - size_block1;
    }
}

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
)
{
    std::copy(x, x + n, x_out);
    if (ld_Sigma == n) {
        std::copy(Sigma, Sigma + n*n, rho_out);
    }
    else {
        F77_CALL(dlacpy)("A", &n, &n, Sigma, &ld_Sigma, rho_out, &n FCONE);
    }

    if (!is_standardized) {
        if (mu)
            cblas_daxpy(n, -1., mu, 1, x_out, 1);
        for (int ix = 0; ix < n; ix++) {
            buffer_sdtdev[ix] = std::sqrt(rho_out[ix*(n+1)]);
        }
        #ifndef _MSC_VER
        #pragma omp simd
        #endif
        for (int ix = 0; ix < n; ix++) {
            x_out[ix] /= buffer_sdtdev[ix];
        }
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                rho_out[col + row*n] /= buffer_sdtdev[row] * buffer_sdtdev[col];
            }
        }
    }
}
