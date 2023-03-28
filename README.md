# ApproxCDF

This is a library for fast approximate calculations of the CDF (cumulative distribution function) of multivariate normal distributions, based on the TVBS (two-variate bivariate screening) method. Written in C++ with interfaces for R, Python, and C.

Compared to the MVNDST library used by R's mvtnorm and Python's SciPy, calculations from this library are typically 2 to 3 orders of magnitue faster, while having results that typically match with MVNDST to around 3-5 decimal places.

Also implemented are Plackett's reduction for quadrivariate (4D) problems and Tsay's approximation for bivariate (2D) problems.

Can alternatively output log-probabilities which can be more exact when dealing with probabilities too close to zero.

# Installation

* R:

**Note:** This package benefits from extra optimizations that aren't enabled by default for R packages. See [this guide](https://github.com/david-cortes/installing-optimized-libraries) for instructions on how to enable them.

```r
remotes::install_github("david-cortes/approxcdf")
```

* Python

**Note:** requires a C compiler configured for Python. See [this guide](https://github.com/david-cortes/installing-optimized-libraries) for instructions.

```
pip install git+https://www.github.com/david-cortes/approxcdf.git
```

** *
**IMPORTANT:** the setup script will try to add compilation flag `-march=native`. This instructs the compiler to tune the package for the CPU in which it is being installed (by e.g. using AVX instructions if available), but the result might not be usable in other computers. If building a binary wheel of this package or putting it into a docker image which will be used in different machines, this can be overriden either by (a) defining an environment variable `DONT_SET_MARCH=1`, or by (b) manually supplying compilation `CFLAGS` as an environment variable with something related to architecture. For maximum compatibility (but slowest speed), it's possible to do something like this:

```
export DONT_SET_MARCH=1
pip install git+https://www.github.com/david-cortes/approxcdf.git
```

or, by specifying some compilation flag for architecture:
```
export CFLAGS="-march=x86-64"
export CXXFLAGS="-march=x86-64"
pip install git+https://www.github.com/david-cortes/approxcdf.git
```
** *

* C:

(Requires BLAS and LAPACK)

```
git clone https://www.github.com/david-cortes/approxcdf.git
cd approxcdf
mkdir build
cd build
cmake -DUSE_MARCH_NATIVE=1 ..
cmake --build .

### for a system-wide install in linux
sudo make install
sudo ldconfig
```
# Sample usage

* R:
```r
library(approxcdf)

### Example from Plackett's paper
b <- c(0, 0, 0, 0)
S <- matrix(
c( 1,  -0.60,  0.85,  0.75,
-0.60,    1,  -0.70, -0.80,
 0.85, -0.70,    1,   0.65,
 0.75, -0.80,  0.65,    1),
nrow=4, ncol=4, byrow=TRUE)
pmvn(b, S, is_standardized=TRUE)
### (Plackett's estimate was 0.042323)
```
```
[1] 0.04348699
```

* Python:
```python
import numpy as np
from approxcdf import mvn_cdf

### Example from Plackett's paper
b = np.array([0, 0, 0, 0], dtype=np.float64)
S = np.array([
    [   1,  -0.60,  0.85,  0.75,],
    [-0.60,    1,  -0.70, -0.80,],
    [ 0.85, -0.70,    1,   0.65,],
    [ 0.75, -0.80,  0.65,    1],
])
mvn_cdf(b, S, is_standardized=True)
```
```
0.043486989827893514
```

* C:
```c
#include "approxcdf.h"
#include <stdio.h>
#include <stdbool.h>

int main()
{
    /* Example from Plackett's paper */
    const int n = 4;
    const double b[] = {0., 0., 0., 0.};
    const double S[] = {
        1,  -0.60,  0.85,  0.75,
     -0.60,    1,  -0.70, -0.80,
      0.85, -0.70,    1,   0.65,
      0.75, -0.80,  0.65,    1
    };
    bool is_standardized = true;
    bool logp = false; /* log-probability */
    double prob = norm_cdf(
        b,
        S, n,
        NULL,
        n,
        is_standardized,
        logp,
        NULL
    );
    printf("Obtained estimate for Plackett's example: %.6f\n", prob);
    return 0;
}
```

# Documentation

* R: documentation is available internally (e.g. `help(pmvn)`).

* Python: documentation is available at [ReadTheDocs](http://approxcdf.readthedocs.io/en/latest/).

* C: documentation is available in the [public header](https://github.com/david-cortes/approxcdf/blob/master/include/approxcdf.h).

# References

* Bhat, Chandra R. "New matrix-based methods for the analytic evaluation of the multivariate cumulative normal distribution function." Transportation Research Part B: Methodological 109 (2018): 238-256.
* Plackett, Robin L. "A reduction formula for normal multivariate integrals." Biometrika 41.3/4 (1954): 351-360.
* Gassmann, H. I. "Multivariate normal probabilities: implementing an old idea of Plackett's." Journal of Computational and Graphical Statistics 12.3 (2003): 731-752.
* Drezner, Zvi, and George O. Wesolowsky. "On the computation of the bivariate normal integral." Journal of Statistical Computation and Simulation 35.1-2 (1990): 101-107.
* Drezner, Zvi. "Computation of the trivariate normal integral." Mathematics of Computation 62.205 (1994): 289-294.
* Genz, Alan. "Numerical computation of rectangular bivariate and trivariate normal and t probabilities." Statistics and Computing 14.3 (2004): 251-260.
* Gibson, Garvin Jarvis, C. A. Glasbey, and D. A. Elston. "Monte Carlo evaluation of multivariate normal integrals and sensitivity to variate ordering." Advances in Numerical Methods and Applications (1994): 120-126.
* Tsay, Wen-Jen, and Peng-Hsuan Ke. "A simple approximation for the bivariate normal integral." Communications in Statistics-Simulation and Computation (2021): 1-14.
