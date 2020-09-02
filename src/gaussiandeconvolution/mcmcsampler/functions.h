#ifndef GD_MCMC_AUXILIAR
#define GD_MCMC_AUXILIAR

#include <vector>
#include <math.h>
#include <random>
#include <iostream>

#include "probability_distributions.h"

//Define the EM algorithm
void EM_simple(std::mt19937 &, std::vector<double>&, std::vector<double>&, std::vector<double>&,
                 std::vector<double>&, const unsigned int &);
//Effective gamma
void sample_effective_gamma(std::mt19937 &r, std::vector<std::vector<double>> &n,
                             std::vector<std::vector<double>> &x2, 
                             std::vector<double> &sigma, std::vector<double> &sigmaold, std::vector<double> &sigmanew,
                             double width);
//Gibbs step
void Gibbs_convolved_step(std::mt19937 &, std::vector<double>&, std::vector<double>&,
                          std::vector<double>&, std::vector<double>&, std::vector<double>&,
                          std::vector<double>&, std::vector<double>&, std::vector<double>&, 
                          double,
                          std::vector<double>&, std::vector<double>&, std::vector<double>&,
                          std::vector<double>&, std::vector<double>&, std::vector<double>&,
                          double,
                          std::vector<std::vector<std::vector<double>>>,
                          double);

#endif