#include <vector>
#include <math.h>
#include <iostream>
#include "gdposteriormodel.h"

gdposteriormodel::gdposteriormodel(std::vector<double> dataNoise, std::vector<double> dataConvolution, int K, int Kc){
    dataNoise_ = dataNoise;
    dataConvolution_ = dataConvolution;
    _K = K;
    _Kc = Kc;
    x_.assign(10000,0);
    normcdf_.assign(10000,0);
    for(int i = 0; i < 10000; i++){
        x_[i] = (i-5000)*0.01;
        normcdf_[i] = std::exp(-(std::pow(x_[i],2)/2))*std::sqrt(1/M_PI/2)*0.01;
    }
    for(int i = 1; i < 10000; i++){
        normcdf_[i] += normcdf_[i-1];
    }

}

double gdposteriormodel::logLikelihood(std::vector<double>& parameters){
    double likelihood =  0;
    double max = -INFINITY;
    std::vector<double> exponent(_K*_Kc,0);

    for(int i = 0; i < dataNoise_.size(); i++){
        //Compute exponents and find the maximum
        max = -INFINITY;
        for(int j = 0; j < _K; j++){
            exponent[j] = -std::pow(dataNoise_[i]-parameters[_K+j],2)/(2*std::pow(parameters[2*_K+j],2));
            if (exponent[j] > max){
                max = exponent[j];
            }
        }
        //Compute the
        double total = 0;
        for(int j = 0; j < _K; j++){
            total += parameters[j]*std::exp(exponent[j]-max)*std::sqrt(1/(2*M_PI*std::pow(parameters[2*_K+j],2)));
        }
        likelihood += std::log(total)+max;
    }

    for(int i = 0; i < dataConvolution_.size(); i++){
        //Compute exponents and find the maximum
        max = -INFINITY;
        for(int j = 0; j < _K; j++){
            for(int k = 0; k < _Kc; k++){
                exponent[j*_Kc+k] = -std::pow(dataConvolution_[i]-parameters[_K+j]-parameters[3*_K+_Kc+k],2)/(2*(std::pow(parameters[2*_K+j],2)+std::pow(parameters[3*_K+2*_Kc+k],2)));
                if (exponent[j*_Kc+k] > max){
                    max = exponent[j*_Kc+k];
                }
            }
        }
        //Compute the
        double total = 0;
        for(int j = 0; j < _K; j++){
            for(int k = 0; k < _Kc; k++){
                total += parameters[j]*parameters[3*_K+k]*std::exp(exponent[j*_Kc+k]-max)
                    *std::sqrt(1/(2*M_PI*(std::pow(parameters[2*_K+j],2)+std::pow(parameters[3*_K+2*_Kc+k],2))));
            }
        }
        likelihood += std::log(total)+max;
    }

    return likelihood;
}

std::vector<double> gdposteriormodel::prior(std::vector<double>& uniform){

    std::vector<double> transformed(3*_K+3*_Kc,0);

    int pos = 0;
    double total = 0;
    //Uniform sphere
    for(int i = 0; i < _K; i++){
        pos = 0;
        while(uniform[pos] > normcdf_[pos]){
            pos++;
        }
        transformed[i] = (x_[pos]+x_[pos-1])/2;
        total += transformed[i];
    }
    for(int i = 0; i < _K; i++){
        transformed[i] /= total;
    }
    //Mean
    for(int i = 0; i < _K; i++){
        transformed[_K+i] = 2*uniform[_K+i]-1;
    }
    //Std
    for(int i = 0; i < _K; i++){
        transformed[2*_K+i] = 2*uniform[2*_K+i];
    }

    //Uniform sphere
    total = 0;
    for(int i = 0; i < _Kc; i++){
        pos = 0;
        while(uniform[pos] > normcdf_[pos]){
            pos++;
        }
        transformed[3*_K+i] = (x_[pos]+x_[pos-1])/2;
        total += transformed[3*_K+i];
    }
    for(int i = 0; i < _Kc; i++){
        transformed[3*_K+i] /= total;
    }
    //Mean
    for(int i = 0; i < _Kc; i++){
        transformed[3*_K+_Kc+i] = 2*uniform[3*_K+_Kc+i]-1;
    }
    //Std
    for(int i = 0; i < _Kc; i++){
        transformed[3*_K+2*_Kc+i] = 2*uniform[3*_K+2*_Kc+i];
    }

    return transformed;
}
