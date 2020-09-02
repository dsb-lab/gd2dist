#ifndef GD_POSTERIOR_MODEL
#define GD_POSTERIOR_MODEL

#include <vector>

class gdposteriormodel{
    public:
        std::vector<double> dataNoise_;
        std::vector<double> dataConvolution_;
        int _K;
        int _Kc;
        gdposteriormodel(std::vector<double>,std::vector<double>,int, int);
        double logLikelihood(std::vector<double>&);
        std::vector<double> prior(std::vector<double>&);
        std::vector<double> x_;
        std::vector<double> normcdf_;
};

#endif