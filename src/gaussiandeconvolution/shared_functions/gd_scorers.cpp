#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <string>

#include "global_random_generator.h"
#include "general_functions.h"
#include "gd_scorers.h"

//logpdf
double aut_norm_mixt_logpdf(double x, std::vector<double> parameters, int K, int Kc){

    std::vector<double> exponent(K,0);
    double max = -INFINITY;
    double value = 0;
    double st;
    double mean;
    //Find maximum to avoid underflows
    for(int i = 0; i < K; i++){
        mean = parameters[K+i];
        st = std::pow(parameters[2*K+i],2);
        exponent[i] = -std::pow(x-mean,2)/2/st;
        if(max < exponent[i]){
            max = exponent[i];
        }
    }
    //Compute the loglikelihood of the mixture
    for(int i = 0; i < K; i++){
        st = std::pow(parameters[2*K+i],2);
        value += parameters[i]*std::exp(exponent[i]-max)*std::sqrt(1/(2*M_PI*st));
    }
    value = std::log(value)+max;

    return value;
}

double deconv_norm_mixt_logpdf(double x, std::vector<double> parameters, int K, int Kc){

    std::vector<double> exponent(Kc,0);
    double max = -INFINITY;
    double value = 0;
    double st;
    double mean;
    //Find maximum to avoid underflows
    for(int j = 0; j < Kc; j++){
        mean = parameters[3*K+Kc+j];
        st = std::pow(parameters[3*K+2*Kc+j],2);
        exponent[j] = -std::pow(x-mean,2)/2/st;
        if(max < exponent[j]){
            max = exponent[j];
        }
    }
    //Compute the loglikelihood of the mixture

    for(int j = 0; j < Kc; j++){
        st = std::pow(parameters[3*K+2*Kc+j],2);
        value += parameters[3*K+j]*std::exp(exponent[j]-max)*std::sqrt(1/(2*M_PI*st));
    }

    value = std::log(value)+max;

    return value;
}

double conv_norm_mixt_logpdf(double x, std::vector<double> parameters, int K, int Kc){

    std::vector<double> exponent(K*Kc,0);
    double max = -INFINITY;
    double value = 0;
    double st;
    double mean;
    //Find maximum to avoid underflows
    for(int i = 0; i < K; i++){
        for(int j = 0; j < Kc; j++){
            mean = parameters[K+i]+parameters[3*K+Kc+j];
            st = std::pow(parameters[2*K+i],2)+std::pow(parameters[3*K+2*Kc+j],2);
            exponent[K*i+j] = -std::pow(x-mean,2)/2/st;
            if(max < exponent[K*i+j]){
                max = exponent[K*i+j];
            }
        }
    }
    //Compute the loglikelihood of the mixture
    for(int i = 0; i < K; i++){
        for(int j = 0; j < Kc; j++){
            st = std::pow(parameters[2*K+i],2)+std::pow(parameters[3*K+2*Kc+j],2);
            value += parameters[i]*parameters[3*K+j]*std::exp(exponent[K*i+j]-max)*std::sqrt(1/(2*M_PI*st));
        }
    }
    value = std::log(value)+max;

    return value;
}

//Scoring
std::vector<std::vector<double>> score_autofluorescence(std::vector<std::vector<double>>& sample, std::vector<double>& x, int K, int Kc, std::vector<double> percentiles, std::vector<double> weights, int size){

    if(sample.size() != weights.size()){
        throw std::invalid_argument("sample and weights must have the same length. Given lengths " + std::to_string(sample.size()) + " and " + std::to_string(weights.size())+ ", respectively.");                
    }

    int xsize = x.size();
    int psize = percentiles.size();
    std::vector<std::vector<double>> values(1+psize,std::vector<double>(xsize,0));
    std::vector<std::vector<double>> aux(xsize, std::vector<double>(size,0));

    std::vector<int> pos = choicepos(weights, size);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < xsize; j++){
            aux[j][i] = std::exp(aut_norm_mixt_logpdf(x[j],sample[pos[i]],K,Kc));
        }
    }

    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        values[0][i] = mean(aux[i]);
        per = percentile(aux[i], percentiles);
        for(int j = 0; j < psize; j++){
            values[1+j][i] = per[j];
        }
    }

    return values;
}

std::vector<std::vector<double>> score_autofluorescence(std::vector<std::vector<double>>& sample, std::vector<double>& x, int K, int Kc, std::vector<double> percentiles, int size){

    int xsize = x.size();
    int psize = percentiles.size();
    std::vector<std::vector<double>> values(1+psize,std::vector<double>(xsize,0));
    std::vector<std::vector<double>> aux(xsize, std::vector<double>(size,0));

    std::vector<int> pos = choicepos(sample.size(), size);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < xsize; j++){
            aux[j][i] = std::exp(aut_norm_mixt_logpdf(x[j],sample[pos[i]],K,Kc));
        }
    }

    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        values[0][i] = mean(aux[i]);
        per = percentile(aux[i], percentiles);
        for(int j = 0; j < psize; j++){
            values[1+j][i] = per[j];
        }
    }

    return values;
}

std::vector<std::vector<double>> score_deconvolution(std::vector<std::vector<double>>& sample, std::vector<double>& x, int K, int Kc, std::vector<double> percentiles, std::vector<double> weights, int size){
    
    if(sample.size() != weights.size()){
        throw std::invalid_argument("sample and weights must have the same length. Given lengths " + std::to_string(sample.size()) + " and " + std::to_string(weights.size())+ ", respectively.");                
    }

    int xsize = x.size();
    int psize = percentiles.size();
    std::vector<std::vector<double>> values(1+psize,std::vector<double>(xsize,0));
    std::vector<std::vector<double>> aux(xsize, std::vector<double>(size,0));

    std::vector<int> pos = choicepos(weights, size);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < xsize; j++){
            aux[j][i] = std::exp(deconv_norm_mixt_logpdf(x[j],sample[pos[i]],K,Kc));
        }
    }

    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        values[0][i] = mean(aux[i]);
        per = percentile(aux[i], percentiles);
        for(int j = 0; j < psize; j++){
            values[1+j][i] = per[j];
        }
    }

    return values;
}

std::vector<std::vector<double>> score_deconvolution(std::vector<std::vector<double>>& sample, std::vector<double>& x, int K, int Kc, std::vector<double> percentiles, int size){

    int xsize = x.size();
    int psize = percentiles.size();
    std::vector<std::vector<double>> values(1+psize,std::vector<double>(xsize,0));
    std::vector<std::vector<double>> aux(xsize, std::vector<double>(size,0));

    std::vector<int> pos = choicepos(sample.size(), size);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < xsize; j++){
            aux[j][i] = std::exp(deconv_norm_mixt_logpdf(x[j],sample[pos[i]],K,Kc));
        }
    }

    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        values[0][i] = mean(aux[i]);
        per = percentile(aux[i], percentiles);
        for(int j = 0; j < psize; j++){
            values[1+j][i] = per[j];
        }
    }

    return values;
}

std::vector<std::vector<double>> score_convolution(std::vector<std::vector<double>>& sample, std::vector<double>& x, int K, int Kc, std::vector<double> percentiles, std::vector<double> weights, int size){

    if(sample.size() != weights.size()){
        throw std::invalid_argument("sample and weights must have the same length. Given lengths " + std::to_string(sample.size()) + " and " + std::to_string(weights.size())+ ", respectively.");                
    }

    int xsize = x.size();
    int psize = percentiles.size();
    std::vector<std::vector<double>> values(1+psize,std::vector<double>(xsize,0));
    std::vector<std::vector<double>> aux(xsize, std::vector<double>(size,0));

    std::vector<int> pos = choicepos(weights, size);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < xsize; j++){
            aux[j][i] = std::exp(conv_norm_mixt_logpdf(x[j],sample[pos[i]],K,Kc));
        }
    }

    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        values[0][i] = mean(aux[i]);
        per = percentile(aux[i], percentiles);
        for(int j = 0; j < psize; j++){
            values[1+j][i] = per[j];
        }
    }

    return values;
}

std::vector<std::vector<double>> score_convolution(std::vector<std::vector<double>>& sample, std::vector<double>& x, int K, int Kc, std::vector<double> percentiles, int size){

    int xsize = x.size();
    int psize = percentiles.size();
    std::vector<std::vector<double>> values(1+psize,std::vector<double>(xsize,0));
    std::vector<std::vector<double>> aux(xsize, std::vector<double>(size,0));

    std::vector<int> pos = choicepos(sample.size(), size);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < xsize; j++){
            aux[j][i] = std::exp(conv_norm_mixt_logpdf(x[j],sample[pos[i]],K,Kc));
        }
    }

    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        values[0][i] = mean(aux[i]);
        per = percentile(aux[i], percentiles);
        for(int j = 0; j < psize; j++){
            values[1+j][i] = per[j];
        }
    }

    return values;
}
