#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <iostream>
#include "gdposteriormodelgamma.h"
#include "../shared_functions/probability_distributions.h"
#include "include/boost/math/special_functions/gamma.hpp"
#include "include/boost/math/special_functions/erf.hpp"
#include "pybind11/pybind11.h"

gdposteriormodelgamma::gdposteriormodelgamma(std::vector<double> datanoise, std::vector<double> dataconvolution, int k, int kc){
    dataNoise = datanoise;
    dataConvolution = dataconvolution;
    K = k;
    Kc = kc;

}

double gdposteriormodelgamma::logLikelihood(std::vector<double>& parameters, double precission = 5){
    double likelihood =  0;
    double max = -INFINITY;
    std::vector<double> exponent(K*Kc,0);
    double total = 0;
    int L = parameters.size()-1;

    for(int i = 0; i < dataNoise.size(); i++){
        //Compute exponents and find the maximum
        max = -INFINITY;
        for(int j = 0; j < K; j++){
            exponent[j] = gamma_pdf(dataNoise[i],parameters[K+j],parameters[2*K+j],parameters[L]);
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
                exponent[j*Kc+k] = -gamma_sum_pdf(dataConvolution[i],parameters[K+j],parameters[2*K+j],parameters[3*K+Kc+k],parameters[3*K+2*Kc+k],parameters[L],precission);
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
        transformed[K+i] = (dataMax-dataMin)*uniform[K+i]+dataMin;
    }
    //Std
    for(int i = 0; i < K; i++){
        transformed[2*K+i] = 3*(dataMax-dataMin)*uniform[2*K+i];
    }

    //Uniform sphere
    total = 0;
    for(int i = 0; i < Kc; i++){
        pos = 0;
        while(uniform[3*K+i] > normcdf[pos] && pos < 9998){
            pos++;
        }
        transformed[3*K+i] = abs(x[pos]+x[pos-1])/2;
        total += transformed[3*K+i];
    }
    for(int i = 0; i < Kc; i++){
        transformed[3*K+i] /= total;
    }
    //Mean
    for(int i = 0; i < Kc; i++){
        transformed[3*K+Kc+i] = (dataMax-dataMin)*uniform[3*K+Kc+i]+dataMin;
    }
    //Std
    for(int i = 0; i < Kc; i++){
        transformed[3*K+2*Kc+i] = 3*(dataMax-dataMin)*uniform[3*K+2*Kc+i];
    }

    return transformed;
}
