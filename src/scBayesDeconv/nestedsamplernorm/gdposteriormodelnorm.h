#ifndef GD_POSTERIOR_MODEL
#define GD_POSTERIOR_MODEL

#include <vector>

class gdposteriormodelnorm{
    public:
        std::vector<double> dataNoise;
        std::vector<double> dataConvolution;
        int K;
        int Kc;
        std::vector<double> priors;
        gdposteriormodelnorm(std::vector<double>,std::vector<double>,int, int);
        double logLikelihood(std::vector<double>&);
        std::vector<double> prior(std::vector<double>&);
        std::vector<double> prior_uniform(std::vector<double>&);
        std::vector<double> x;
        std::vector<double> normcdf;
};

#endif