#include "approxcdf.h"
#include <stdio.h>
#include <stdbool.h>

/*
To run this example after having built through the cmake system but without
a system-wide install, assuming that this is being compiled from one directory
above from where this file lives:
    gcc c_example/c_example.c -o test -std=c99 -lapproxcdf -I./include -L./build -Wl,-rpath,./build
    */

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
    double prob = norm_cdf(
        b,
        S, n,
        NULL,
        n,
        true,
        NULL
    );
    printf("Obtained estimate for Plackett's example: %.6f\n", prob);
    return 0;
} 
