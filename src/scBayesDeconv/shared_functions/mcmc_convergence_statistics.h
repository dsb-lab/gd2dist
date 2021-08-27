#ifndef MCMC_CONVERGENCE_STATISTICS
#define MCMC_CONVERGENCE_STATISTICS

#include <vector>

//Effective number of 
double rstat(std::vector<double>&, int);
double rstat(std::vector<std::vector<double>>&);

//Effective number of degrees of freedom
double effnumber(std::vector<double>&, int);
double effnumber(std::vector<std::vector<double>>&);

#endif