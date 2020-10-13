#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <iostream>
#include "gdposteriormodelnorm.h"
#include "../shared_functions/probability_distributions.h"
#include "include/boost/math/special_functions/gamma.hpp"
#include "include/boost/math/special_functions/erf.hpp"
#include "include/boost/math/special_functions/detail/lgamma_small.hpp"

#include "pybind11/pybind11.h"

gdposteriormodelnorm::gdposteriormodelnorm(std::vector<double> datanoise, std::vector<double> dataconvolution, int k, int kc){
    dataNoise = datanoise;
    dataConvolution = dataconvolution;
    K = k;
    Kc = kc;
    x.assign(10000,0);
    normcdf.assign(10000,0);
    for(int i = 0; i < 10000; i++){
        x[i] = i*0.01;
        normcdf[i] = 2*std::exp(-(std::pow(x[i],2)/2))*std::sqrt(1/M_PI/2)*0.01;
    }
    for(int i = 1; i < 10000; i++){
        normcdf[i] += normcdf[i-1];
    }
}

double gdposteriormodelnorm::logLikelihood(std::vector<double>& parameters){
    double likelihood =  0;
    double max = -INFINITY;
    std::vector<double> exponent(K*Kc,0);
    double total = 0;

    for(int i = 0; i < dataNoise.size(); i++){
        //Compute exponents and find the maximum
        max = -INFINITY;
        for(int j = 0; j < K; j++){
            exponent[j] = -std::pow(dataNoise[i]-parameters[K+j],2)/(2*std::pow(parameters[2*K+j],2));
            if (exponent[j] > max){
                max = exponent[j];
            }
        }
        //Compute the
        total = 0;
        for(int j = 0; j < K; j++){
            total += parameters[j]*std::exp(exponent[j]-max)*std::sqrt(1/(2*M_PI*std::pow(parameters[2*K+j],2)));
        }
        likelihood += std::log(total)+max;
    }

    for(int i = 0; i < dataConvolution.size(); i++){
        //Compute exponents and find the maximum
        max = -INFINITY;
        for(int j = 0; j < K; j++){
            for(int k = 0; k < Kc; k++){
                exponent[j*Kc+k] = -std::pow(dataConvolution[i]-parameters[K+j]-parameters[3*K+Kc+k],2)/(2*(std::pow(parameters[2*K+j],2)+std::pow(parameters[3*K+2*Kc+k],2)));
                if (exponent[j*Kc+k] > max){
                    max = exponent[j*Kc+k];
                }
            }
        }
        //Compute the
        total = 0;
        for(int j = 0; j < K; j++){
            for(int k = 0; k < Kc; k++){
                total += parameters[j]*parameters[3*K+k]*std::exp(exponent[j*Kc+k]-max)
                    *std::sqrt(1/(2*M_PI*(std::pow(parameters[2*K+j],2)+std::pow(parameters[3*K+2*Kc+k],2))));
            }
        }
        likelihood += std::log(total)+max;
    }

    return likelihood;
}

std::vector<double> gdposteriormodelnorm::prior(std::vector<double>& uniform){

    std::vector<double> transformed(3*K+3*Kc,0);

    int pos = 0;
    double total = 0;
    //Uniform sphere
    for(int i = 0; i < K; i++){
        pos = 0;
        while(uniform[i] > normcdf[pos] && pos < 9998){
            pos++;
        }
        transformed[i] = abs(x[pos]+x[pos-1])/2;
        total += transformed[i];
    }
    for(int i = 0; i < K; i++){
        transformed[i] /= total;
    }
    //Mean
    for(int i = 0; i < K; i++){
        if(uniform[K+i] < 0.5){
            transformed[K+i] = -priors[1]*boost::math::erf_inv(2*(0.5-uniform[K+i]))+priors[0];
        }else{
            transformed[K+i] = priors[1]*boost::math::erf_inv(2*(uniform[K+i]-0.5))+priors[0];
        }
    }
    //Std
    for(int i = 0; i < K; i++){
        transformed[2*K+i] = priors[2]*boost::math::gamma_p_inv(priors[3],uniform[2*K+i]);
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
        if(uniform[3*K+Kc+i] < 0.5){
            transformed[3*K+Kc+i] = -priors[5]*boost::math::erf_inv(2*(0.5-uniform[3*K+Kc+i]))+priors[4];
        }else{
            transformed[3*K+Kc+i] = priors[5]*boost::math::erf_inv(2*(uniform[3*K+Kc+i]-0.5))+priors[4];
        }
    }
    //Std
    for(int i = 0; i < Kc; i++){
        transformed[3*K+2*Kc+i] = priors[6]*boost::math::gamma_p_inv(priors[7],uniform[3*K+2*Kc+i]);
    }

    return transformed;
}

std::vector<double> gdposteriormodelnorm::prior_uniform(std::vector<double>& uniform){

    std::vector<double> transformed(3*K+3*Kc,0);

    int pos = 0;
    double total = 0;
    //Uniform sphere
    for(int i = 0; i < K; i++){
        pos = 0;
        while(uniform[i] > normcdf[pos] && pos < 9998){
            pos++;
        }
        transformed[i] = abs(x[pos]+x[pos-1])/2;
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
        transformed[3*K+Kc+i] = (priors[5]-priors[4])*uniform[3*K+Kc+i]+priors[4];
    }
    //Std
    for(int i = 0; i < Kc; i++){
        transformed[3*K+2*Kc+i] = 3*(priors[7]-priors[6])*uniform[3*K+2*Kc+i]+priors[6];
    }

    return transformed;
}
