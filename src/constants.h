
#define EPS_BLOCK 1e-20
#define LOW_RHO 1e-9
constexpr const static double HIGH_RHO = 1. - 1e-3;

#ifndef M_PI
#   define M_PI 3.14159265358979323846
#endif

#ifdef CONSTEXPR_FOR_MATH /* not part of the c++ standards */
#   define CONSTEXPR_MATH constexpr
#else
#   define CONSTEXPR_MATH 
#endif


CONSTEXPR_MATH const static double inv_sqrt2_pi = 1. / std::sqrt(2. * M_PI);
CONSTEXPR_MATH const static double neg_inv_sqrt2 = -1. / std::sqrt(2.);
CONSTEXPR_MATH const static double inv_sqrt2 = 1. / std::sqrt(2.);
CONSTEXPR_MATH const static double sqrt2 = std::sqrt(2.);
CONSTEXPR_MATH static const double twoPI = 2. * M_PI;
CONSTEXPR_MATH static const double fourPI = 4. * M_PI;
CONSTEXPR_MATH static const double half_log_twoPI = 0.5 * std::log(twoPI);
CONSTEXPR_MATH static const double inv_fourPI = 1. / fourPI;
CONSTEXPR_MATH static const double sqrt_twoPI = std::sqrt(2. * M_PI);
CONSTEXPR_MATH static const double log_sqrt_twoPI = std::log(std::sqrt(2. * M_PI));
CONSTEXPR_MATH static const double minus_inv_twoPI = -1. / twoPI;
CONSTEXPR_MATH static const double inv3sqrt2pi = 1. / (3. * std::sqrt(2. * M_PI));
CONSTEXPR_MATH static const double asin_one = std::asin(1.);
CONSTEXPR_MATH static const double inv_four_asin_one = 1. / (4. * asin_one);
