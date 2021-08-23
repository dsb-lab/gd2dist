#ifndef GD_POSTERIOR_MODEL
#define GD_POSTERIOR_MODEL

#include <vector>

class gdposteriormodelgamma{
    public:
        std::vector<double> dataNoise;
        std::vector<double> dataConvolution;
        
        int K;
        int Kc;

        double dataMin;
        double dataMax;

        double bias;

        std::vector<double> priors;

        double precission;

        gdposteriormodelgamma(std::vector<double>,std::vector<double>,int, int, double);
        double logLikelihood(std::vector<double>&);
        std::vector<double> prior(std::vector<double>&);
        std::vector<double> prior_uniform(std::vector<double>&);

};

#endif