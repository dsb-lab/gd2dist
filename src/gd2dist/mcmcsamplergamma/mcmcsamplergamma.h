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

double logLikelihood(std::vector<double>, std::vector<double>, std::vector<double>,
                                    std::vector<double>, std::vector<double>, std::vector<double>,
                                    std::vector<double>, std::vector<double>);
void sample_theta(std::mt19937 &, std::vector<std::vector<double>> &,
                std::vector<std::vector<double>> &, 
                std::vector<double> &, std::vector<double> &, std::vector<double> &,
                double, double, double, double, double, double, double, double, double);
Gibbs_convolved_step(std::mt19937 &, std::vector<double> &, std::vector<double> &,
                    std::vector<double> &, std::vector<double> &, std::vector<double> &, 
                    std::vector<double> &, std::vector<double> &, std::vector<double> &, 
                    double, double, double, double, double,
                    std::vector<double> &, std::vector<double> &, std::vector<double> &, 
                    std::vector<double> &, std::vector<double> &, std::vector<double> &, 
                    double, double, double, double, double,
                    double &, double &,
                    double, double,
                    std::vector<std::vector<std::vector<double>>>, int);
void chain(int, std::vector<std::vector<double>> &, std::vector<double> &, std::vector<double> &,                          
                                int, int, int,
                                int, int, double, double,
                                double, double, double, double, double, double, double, double, double,
                                bool, bool, int, int);
std::vector<std::vector<double>> fit(std::vector<double> &, std::vector<double>&,
                          int, int, int,
                          int, int, double, double, 
                          double, double, double, double, double, double, double, double, double, int,
                          std::vector<std::vector<double>>, bool, int);

#endif