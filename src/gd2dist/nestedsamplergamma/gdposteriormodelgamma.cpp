#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <iostream>
#include "gdposteriormodelgamma.h"
#include "../shared_functions/probability_distributions.h"
#include "include/boost/math/special_functions/gamma.hpp"
#include "include/boost/math/special_functions/erf.hpp"
#include "include/boost/math/special_functions/detail/lgamma_small.hpp"
#include "pybind11/pybind11.h"

gdposteriormodelgamma::gdposteriormodelgamma(std::vector<double> datanoise, std::vector<double> dataconvolution, int k, int kc, double bias){
    dataNoise = datanoise;
    dataConvolution = dataconvolution;
    K = k;
    Kc = kc;
    bias = bias;
}

double gdposteriormodelgamma::logLikelihood(std::vector<double>& parameters){
    double likelihood =  0;
    double max = -INFINITY;
    std::vector<double> exponent(K*Kc,0);
    double total = 0;

    for(int i = 0; i < dataNoise.size(); i++){
        //Compute exponents and find the maximum
        max = -INFINITY;
//        pybind11::print(dataNoise.size());
        for(int j = 0; j < K; j++){
            exponent[j] = gamma_pdf(dataNoise[i]-bias,parameters[K+j],parameters[2*K+j],0);
            
            if (exponent[j] > max){
                max = exponent[j];
            }
        }
        //Compute the
        total = 0;
        for(int j = 0; j < K; j++){
            total += parameters[j]*std::exp(exponent[j]-max);
        }
        likelihood += std::log(total)+max;
    }

    for(int i = 0; i < dataConvolution.size(); i++){
        //Compute exponents and find the maximum
        max = -INFINITY;
        for(int j = 0; j < K; j++){
            for(int k = 0; k < Kc; k++){
                exponent[j*Kc+k] = gamma_sum_pdf(dataConvolution[i]-2*bias,parameters[K+j],parameters[2*K+j],parameters[3*K+Kc+k],parameters[3*K+2*Kc+k],0,precission);
                if (exponent[j*Kc+k] > max){
                    max = exponent[j*Kc+k];
                }
            }
        }
        //Compute the
        total = 0;
        for(int j = 0; j < K; j++){
            for(int k = 0; k < Kc; k++){
                total += parameters[j]*parameters[3*K+k]*std::exp(exponent[j*Kc+k]-max);
            }
        }
        likelihood += std::log(total)+max;
    }

    if(std::isnan(likelihood)){
        likelihood = -INFINITY;
    }

    return likelihood;
}

std::vector<double> gdposteriormodelgamma::prior(std::vector<double>& uniform){

    std::vector<double> transformed(3*K+3*Kc,0);

    double total = 0;
    //Uniform sphere
    for(int i = 0; i < K; i++){
        transformed[i] = boost::math::erf_inv(uniform[i]);
        total += transformed[i];
    }
    for(int i = 0; i < K; i++){
        transformed[i] /= total;
    }
    //Mean
    for(int i = 0; i < K; i++){
        transformed[K+i] = priors[0]*boost::math::gamma_p_inv(priors[1],uniform[K+i]);
    }
    //Std
    for(int i = 0; i < K; i++){
        transformed[2*K+i] = priors[2]*boost::math::gamma_p_inv(priors[3],uniform[2*K+i]);
    }

    //Uniform sphere
    total = 0;
    for(int i = 0; i < Kc; i++){
        transformed[3*K+i] = boost::math::erf_inv(uniform[3*K+i]);;
        total += transformed[3*K+i];
    }
    for(int i = 0; i < Kc; i++){
        transformed[3*K+i] /= total;
    }
    //Mean
    for(int i = 0; i < Kc; i++){
        transformed[3*K+Kc+i] = priors[4]*boost::math::gamma_p_inv(priors[5],uniform[3*K+Kc+i]);
    }
    //Std
    for(int i = 0; i < Kc; i++){
        transformed[3*K+2*Kc+i] = priors[6]*boost::math::gamma_p_inv(priors[7],uniform[3*K+2*Kc+i]);
    }

    return transformed;
}

std::vector<double> gdposteriormodelgamma::prior_uniform(std::vector<double>& uniform){

    std::vector<double> transformed(3*K+3*Kc,0);

    double total = 0;
    //Uniform sphere
    for(int i = 0; i < K; i++){
        transformed[i] = boost::math::erf_inv(uniform[i]);
        total += transformed[i];
    }
    for(int i = 0; i < K; i++){
        transformed[i] /= total;
    }
    //Mean
    for(int i = 0; i < K; i++){
        transformed[K+i] = (priors[1]-priors[0])*uniform[K+i]+priors[0];
    }
    //Std
    for(int i = 0; i < K; i++){
        transformed[2*K+i] = (priors[3]-priors[2])*uniform[2*K+i]+priors[2];
    }

    //Uniform sphere
    total = 0;
    for(int i = 0; i < Kc; i++){
        transformed[3*K+i] = boost::math::erf_inv(uniform[3*K+i]);;
        total += transformed[3*K+i];
    }
    for(int i = 0; i < Kc; i++){
        transformed[3*K+i] /= total;
    }
    //Mean
    for(int i = 0; i < Kc; i++){
        transformed[3*K+Kc+i] = (priors[5]-priors[4])*uniform[3*K+Kc+i]+priors[4];
    }
    //Std
    for(int i = 0; i < Kc; i++){
        transformed[3*K+2*Kc+i] = (priors[7]-priors[6])*uniform[3*K+2*Kc+i]+priors[6];
    }

    return transformed;
}
