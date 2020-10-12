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

        double priortheta_theta;
        double priortheta_k;
        double priork_theta;
        double priork_k;

        double priortheta_thetac;
        double priortheta_kc;
        double priork_thetac;
        double priork_kc;

        double precission;

        gdposteriormodelgamma(std::vector<double>,std::vector<double>,int, int, double);
        double logLikelihood(std::vector<double>&);
        std::vector<double> prior(std::vector<double>&);
};

#endif