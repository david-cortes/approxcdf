
static inline
double nonstd_cdf_2d(double x1, double x2, double m1, double m2, double v1, double v2, double cv)
{
    double s1 = std::sqrt(v1);
    double s2 = std::sqrt(v2);
    return norm_cdf_2d((x1 - m1) / s1, (x2 - m2) / s2, cv / (s1 * s2));
}

static inline
double nonstd_cdf_3d(const double x[3], const double mu[3], const double cov[], const int ld_cov)
{
    double stdevs[] = {
        std::sqrt(cov[0]),
        std::sqrt(cov[1 + ld_cov]),
        std::sqrt(cov[2 + 2*ld_cov])
    };
    return norm_cdf_3d((x[0] - mu[0]) / stdevs[0],
                       (x[1] - mu[1]) / stdevs[1],
                       (x[2] - mu[2]) / stdevs[2],
                       cov[1] / (stdevs[0] * stdevs[1]),
                       cov[2] / (stdevs[0] * stdevs[2]),
                       cov[2 + ld_cov] / (stdevs[1] * stdevs[2]));
}

static inline
double nonstd_cdf_4d(const double x[4], const double mu[4], const double cov[], const int ld_cov)
{
    double stdevs[] = {
        std::sqrt(cov[0]),
        std::sqrt(cov[1 + ld_cov]),
        std::sqrt(cov[2 + 2*ld_cov]),
        std::sqrt(cov[3 + 3*ld_cov])
    };
    double stdx[] = {
        (x[0] - mu[0]) / stdevs[0],
        (x[1] - mu[1]) / stdevs[1],
        (x[2] - mu[2]) / stdevs[2],
        (x[3] - mu[3]) / stdevs[3]
    };
    double rho[] = {
        cov[1] / (stdevs[0] * stdevs[1]),
        cov[2] / (stdevs[0] * stdevs[2]),
        cov[3] / (stdevs[0] * stdevs[3]),
        cov[2 + ld_cov] / (stdevs[1] * stdevs[2]),
        cov[3 + ld_cov] / (stdevs[1] * stdevs[3]),
        cov[3 + 2*ld_cov] / (stdevs[2] * stdevs[3])
    };
    return norm_cdf_4d_bhat(stdx, rho);
}

/* X[i,:] = X[j,:]; X[:,i] = X[:,j] */
static inline
void swap_entries_sq_matrix(double *restrict v, double *restrict X, const int ld_X,
                            const int n, const int pos1, const int pos2)
{
    if (pos1 == pos2) {
        return;
    }

    int row_st1 = pos1 * n;
    int row_st2 = pos2 * n;
    for (int ix = 0; ix < n; ix++) {
        std::swap(X[ix + row_st1], X[ix + row_st2]);
    }
    for (int ix = 0; ix < n; ix++) {
        std::swap(X[pos1 + ix*n], X[pos2 + ix*n]);
    }
    if (v) {
        std::swap(v[pos1], v[pos2]);
    }
}

[[gnu::always_inline]]
static inline
bool rho_is_zero(double x)
{
    return std::fabs(x) <= LOW_RHO;
}

[[gnu::always_inline]]
static inline
bool rho_is_one(double x)
{
    return std::fabs(x) >= HIGH_RHO;
}
