#ifndef GD_MCMC_DISTRIBUTIONS
#define GD_MCMC_DISTRIBUTIONS

#include <math.h>
#include <random>

double
gamma_pdf(double, double, double, double);

double
gamma_sum_pdf(double, double, double, double, double, double, int);

double
gaussian_pdf(double, double, double);

void
multinomial_1(std::mt19937 &, std::vector<double> &, std::vector<int> &);

void
dirichlet(std::mt19937 &, std::vector<double> &, std::vector<double> &);

#endif