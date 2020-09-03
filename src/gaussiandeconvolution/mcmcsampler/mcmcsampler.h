#ifndef GD_MCMC_SAMPLER
#define GD_MCMC_SAMPLER

#include <vector>
#include <map>
#include <string>
#include <thread>
#include <algorithm>
#include <iostream>
#include <exception>
#include <random>

double effective_gamma_not_normalized(double, std::vector<double>, std::vector<double>, std::vector<double>);
void sample_effective_gamma(std::mt19937 &, std::vector<std::vector<double>> &,
                             std::vector<std::vector<double>> &, 
                             std::vector<double> &, std::vector<double> &, std::vector<double> &,
                             double);
void Gibbs_convolved_step(std::mt19937 &, std::vector<double> &, std::vector<double>&,
                          std::vector<double> &, std::vector<double> &, std::vector<double> &,
                          std::vector<double> &, std::vector<double> &, std::vector<double> &, 
                          double,
                          std::vector<double> &, std::vector<double> &, std::vector<double> &,
                          std::vector<double> &, std::vector<double> &, std::vector<double> &,
                          double,
                          std::vector<std::vector<std::vector<double>>>,
                          double);
void chain(int, std::vector<std::vector<double>> &, std::vector<double> &, std::vector<double> &,                          
                                int, int, int,
                                int, int, double, double, double, bool, bool);
std::vector<std::vector<double>> fit(std::vector<double> &, std::vector<double>&,
                          int, int, int,
                          int, int, double, double, double, std::vector<std::vector<double>>, bool);

#endif