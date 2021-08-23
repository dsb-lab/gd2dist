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
#include "probability_distributions.h"

#include "pybind11/pybind11.h"

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
    std::vector<double> aux(size);

    std::vector<int> pos = choicepos(weights, size);
    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        for(int j = 0; j < size; j++){
                aux[j] = std::exp(aut_norm_mixt_logpdf(x[i],sample[pos[j]],K,Kc));
        }
        values[0][i] = mean(aux);
        per = percentile(aux, percentiles);
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
    std::vector<double> aux(size);

    std::vector<int> pos = choicepos(sample.size(), size);
    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        for(int j = 0; j < size; j++){
                aux[j] = std::exp(aut_norm_mixt_logpdf(x[i],sample[pos[j]],K,Kc));
        }
        values[0][i] = mean(aux);
        per = percentile(aux, percentiles);
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
    std::vector<double> aux(size);

    std::vector<int> pos = choicepos(weights, size);
    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        for(int j = 0; j < size; j++){
                aux[j] = std::exp(deconv_norm_mixt_logpdf(x[i],sample[pos[j]],K,Kc));
        }
        values[0][i] = mean(aux);
        per = percentile(aux, percentiles);
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
    std::vector<double> aux(size);

    std::vector<int> pos = choicepos(sample.size(), size);
    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        for(int j = 0; j < size; j++){
            aux[j] = std::exp(deconv_norm_mixt_logpdf(x[i],sample[pos[j]],K,Kc));
        }
        values[0][i] = mean(aux);
        per = percentile(aux, percentiles);
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
    std::vector<double> aux(size);

    std::vector<int> pos = choicepos(weights, size);
    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        for(int j = 0; j < size; j++){
                aux[j] = std::exp(conv_norm_mixt_logpdf(x[i],sample[pos[j]],K,Kc));
        }
        values[0][i] = mean(aux);
        per = percentile(aux, percentiles);
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
    std::vector<double> aux(size);

    std::vector<int> pos = choicepos(sample.size(), size);
    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        for(int j = 0; j < size; j++){
                aux[j] = std::exp(conv_norm_mixt_logpdf(x[i],sample[pos[j]],K,Kc));
        }
        values[0][i] = mean(aux);
        per = percentile(aux, percentiles);
        for(int j = 0; j < psize; j++){
            values[1+j][i] = per[j];
        }
    }

    return values;
}

//logpdf
double aut_gamma_mixt_logpdf(double x, std::vector<double> parameters, int K, int Kc){

    std::vector<double> exponent(K,0);
    double max = -INFINITY;
    double value = 0;
    double st;
    double mean;
    //Find maximum to avoid underflows
    for(int i = 0; i < K; i++){
        exponent[i] = gamma_pdf(x,parameters[K+i],parameters[2*K+i],0);
        if(max < exponent[i]){
            max = exponent[i];
        }
    }
    //Compute the loglikelihood of the mixture
    for(int i = 0; i < K; i++){
        value += parameters[i]*std::exp(exponent[i]-max);
    }
    value = std::log(value)+max;

    return value;
}

double deconv_gamma_mixt_logpdf(double x, std::vector<double> parameters, int K, int Kc){

    std::vector<double> exponent(Kc,0);
    double max = -INFINITY;
    double value = 0;
    double st;
    double mean;
    //Find maximum to avoid underflows
    for(int j = 0; j < Kc; j++){
        exponent[j] = gamma_pdf(x,parameters[3*K+Kc+j],parameters[3*K+2*Kc+j],0);
        if(max < exponent[j]){
            max = exponent[j];
        }
    }
    //Compute the loglikelihood of the mixture

    for(int j = 0; j < Kc; j++){
        value += parameters[3*K+j]*std::exp(exponent[j]-max);
    }

    value = std::log(value)+max;

    //pybind11::print(value);

    return value;
}

double conv_gamma_mixt_logpdf(double x, std::vector<double> parameters, int K, int Kc){

    std::vector<double> exponent(K*Kc,0);
    double max = -INFINITY;
    double value = 0;
    double st;
    double mean;
    //Find maximum to avoid underflows
    for(int i = 0; i < K; i++){
        for(int j = 0; j < Kc; j++){
            exponent[K*i+j] = gamma_sum_pdf(x,parameters[K+i],parameters[2*K+i],parameters[3*K+Kc+j],parameters[3*K+2*Kc+j],0);
            if(max < exponent[K*i+j]){
                max = exponent[K*i+j];
            }
        }
    }
    //Compute the loglikelihood of the mixture
    for(int i = 0; i < K; i++){
        for(int j = 0; j < Kc; j++){
            value += parameters[i]*parameters[3*K+j]*std::exp(exponent[K*i+j]-max);
        }
    }
    value = std::log(value)+max;

    return value;
}

//Scoring
std::vector<std::vector<double>> score_autofluorescence_gamma(std::vector<std::vector<double>>& sample, std::vector<double>& x, int K, int Kc, std::vector<double> percentiles, std::vector<double> weights, int size, double bias){

    if(sample.size() != weights.size()){
        throw std::invalid_argument("sample and weights must have the same length. Given lengths " + std::to_string(sample.size()) + " and " + std::to_string(weights.size())+ ", respectively.");                
    }

    int xsize = x.size();
    int psize = percentiles.size();
    std::vector<std::vector<double>> values(1+psize,std::vector<double>(xsize,0));
    std::vector<double> aux(size);

    std::vector<int> pos = choicepos(weights, size);
    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        for(int j = 0; j < size; j++){
                aux[j] = std::exp(aut_gamma_mixt_logpdf(x[i],sample[pos[j]],K,Kc));
        }
        values[0][i] = mean(aux)+bias;
        per = percentile(aux, percentiles);
        for(int j = 0; j < psize; j++){
            values[1+j][i] = per[j]+bias;
        }
    }

    return values;
}

std::vector<std::vector<double>> score_autofluorescence_gamma(std::vector<std::vector<double>>& sample, std::vector<double>& x, int K, int Kc, std::vector<double> percentiles, int size, double bias){

    int xsize = x.size();
    int psize = percentiles.size();
    std::vector<std::vector<double>> values(1+psize,std::vector<double>(xsize,0));
    std::vector<double> aux(size);

    std::vector<int> pos = choicepos(sample.size(), size);
    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        for(int j = 0; j < size; j++){
                aux[j] = std::exp(aut_gamma_mixt_logpdf(x[i],sample[pos[j]],K,Kc));
        }
        values[0][i] = mean(aux)+bias;
        per = percentile(aux, percentiles);
        for(int j = 0; j < psize; j++){
            values[1+j][i] = per[j]+bias;
        }
    }

    return values;
}

std::vector<std::vector<double>> score_deconvolution_gamma(std::vector<std::vector<double>>& sample, std::vector<double>& x, int K, int Kc, std::vector<double> percentiles, std::vector<double> weights, int size, double bias){
    
    if(sample.size() != weights.size()){
        throw std::invalid_argument("sample and weights must have the same length. Given lengths " + std::to_string(sample.size()) + " and " + std::to_string(weights.size())+ ", respectively.");                
    }

    int xsize = x.size();
    int psize = percentiles.size();
    std::vector<std::vector<double>> values(1+psize,std::vector<double>(xsize,0));
    std::vector<double> aux(size);

    std::vector<int> pos = choicepos(weights, size);
    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        for(int j = 0; j < size; j++){
                aux[j] = std::exp(deconv_gamma_mixt_logpdf(x[i],sample[pos[j]],K,Kc));
        }
        values[0][i] = mean(aux)+bias;
        per = percentile(aux, percentiles);
        for(int j = 0; j < psize; j++){
            values[1+j][i] = per[j]+bias;
        }
    }

    return values;
}

std::vector<std::vector<double>> score_deconvolution_gamma(std::vector<std::vector<double>>& sample, std::vector<double>& x, int K, int Kc, std::vector<double> percentiles, int size, double bias){

    int xsize = x.size();
    int psize = percentiles.size();
    std::vector<std::vector<double>> values(1+psize,std::vector<double>(xsize,0));
    std::vector<double> aux(size);

    std::vector<int> pos = choicepos(sample.size(), size);
    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        for(int j = 0; j < size; j++){
            aux[j] = std::exp(deconv_gamma_mixt_logpdf(x[i],sample[pos[j]],K,Kc));
        }
        values[0][i] = mean(aux)+bias;
        per = percentile(aux, percentiles);
        for(int j = 0; j < psize; j++){
            values[1+j][i] = per[j]+bias;
        }
    }

    return values;
}

std::vector<std::vector<double>> score_convolution_gamma(std::vector<std::vector<double>>& sample, std::vector<double>& x, int K, int Kc, std::vector<double> percentiles, std::vector<double> weights, int size, double bias){

    if(sample.size() != weights.size()){
        throw std::invalid_argument("sample and weights must have the same length. Given lengths " + std::to_string(sample.size()) + " and " + std::to_string(weights.size())+ ", respectively.");                
    }

    int xsize = x.size();
    int psize = percentiles.size();
    std::vector<std::vector<double>> values(1+psize,std::vector<double>(xsize,0));
    std::vector<double> aux(size);

    std::vector<int> pos = choicepos(weights, size);
    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        for(int j = 0; j < size; j++){
                aux[j] = std::exp(conv_gamma_mixt_logpdf(x[i],sample[pos[j]],K,Kc));
        }
        values[0][i] = mean(aux)+bias;
        per = percentile(aux, percentiles);
        for(int j = 0; j < psize; j++){
            values[1+j][i] = per[j]+bias;
        }
    }

    return values;
}

std::vector<std::vector<double>> score_convolution_gamma(std::vector<std::vector<double>>& sample, std::vector<double>& x, int K, int Kc, std::vector<double> percentiles, int size, double bias){

    int xsize = x.size();
    int psize = percentiles.size();
    std::vector<std::vector<double>> values(1+psize,std::vector<double>(xsize,0));
    std::vector<double> aux(size);

    std::vector<int> pos = choicepos(sample.size(), size);
    std::vector<double> per(psize,0);
    for(int i = 0; i < xsize; i++){
        for(int j = 0; j < size; j++){
                aux[j] = std::exp(conv_gamma_mixt_logpdf(x[i],sample[pos[j]],K,Kc));
        }
        values[0][i] = mean(aux)+bias;
        per = percentile(aux, percentiles);
        for(int j = 0; j < psize; j++){
            values[1+j][i] = per[j]+bias;
        }
    }

    return values;
}
