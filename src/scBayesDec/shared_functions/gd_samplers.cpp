#include <cmath>
#include <vector>
#include <random>
#include "general_functions.h"
#include "gd_samplers.h"
#include "global_random_generator.h"

#include "pybind11/pybind11.h"

std::vector<double> sample_autofluorescence(std::vector<std::vector<double>>& postSamples, int K, int Kc, int nsamples){

    std::normal_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    std::vector<int> pos(choicepos(postSamples.size(), nsamples));
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin(),postSamples[pos[i]].begin()+K);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::normal_distribution<double>(postSamples[pos[i]][K+normpos],postSamples[pos[i]][2*K+normpos]);
        samples[i] = norm(AUX_R);
    }


    return samples;
}

std::vector<double> sample_autofluorescence(std::vector<std::vector<double>>& postSamples, int K, int Kc, std::vector<double>& weights, int nsamples){

    std::normal_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    std::vector<int> pos(choicepos(weights, nsamples));
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin(),postSamples[pos[i]].begin()+K);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::normal_distribution<double>(postSamples[pos[i]][K+normpos],postSamples[pos[i]][2*K+normpos]);
        samples[i] = norm(AUX_R);
    }


    return samples;
}

std::vector<double> sample_deconvolution(std::vector<std::vector<double>>& postSamples, int K, int Kc, int nsamples){

    std::normal_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    std::vector<int> pos(choicepos(postSamples.size(), nsamples));
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin()+3*K,postSamples[pos[i]].begin()+3*K+Kc);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::normal_distribution<double>(postSamples[pos[i]][3*K+Kc+normpos],postSamples[pos[i]][3*K+2*Kc+normpos]);
        samples[i] = norm(AUX_R);
    }


    return samples;
}

std::vector<double> sample_deconvolution(std::vector<std::vector<double>>& postSamples, int K, int Kc, std::vector<double>& weights, int nsamples){

    std::normal_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    std::vector<int> pos(choicepos(weights, nsamples));
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin()+3*K,postSamples[pos[i]].begin()+3*K+Kc);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::normal_distribution<double>(postSamples[pos[i]][3*K+Kc+normpos],postSamples[pos[i]][3*K+2*Kc+normpos]);
        samples[i] = norm(AUX_R);
    }


    return samples;
}

std::vector<double> sample_convolution(std::vector<std::vector<double>>& postSamples, int K, int Kc, int nsamples){

    std::normal_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    int normpos2;
    std::vector<double> aux;

    //Choose from the samples
    double st;
    double mean;
    std::vector<int> pos(choicepos(postSamples.size(), nsamples));
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin(),postSamples[pos[i]].begin()+K);
        normpos = choicepos(aux)[0];
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin()+3*K,postSamples[pos[i]].begin()+3*K+Kc);
        normpos2 = choicepos(aux)[0];
        //Sample from the gaussian
        mean = postSamples[pos[i]][K+normpos]+postSamples[pos[i]][3*K+Kc+normpos2];
        st = std::sqrt(std::pow(postSamples[pos[i]][2*K+normpos],2)+std::pow(postSamples[pos[i]][3*K+2*Kc+normpos2],2));
        norm = std::normal_distribution<double>(mean,st);
        samples[i] = norm(AUX_R);
    }


    return samples;
}

std::vector<double> sample_convolution(std::vector<std::vector<double>>& postSamples, int K, int Kc, std::vector<double>& weights, int nsamples){

    std::normal_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    int normpos2;
    std::vector<double> aux;

    //Choose from the samples
    double st;
    double mean;
    std::vector<int> pos(choicepos(weights, nsamples));
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin(),postSamples[pos[i]].begin()+K);
        normpos = choicepos(aux)[0];
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin()+3*K,postSamples[pos[i]].begin()+3*K+Kc);
        normpos2 = choicepos(aux)[0];
        //Sample from the gaussian
        mean = postSamples[pos[i]][K+normpos]+postSamples[pos[i]][3*K+Kc+normpos2];
        st = std::sqrt(std::pow(postSamples[pos[i]][2*K+normpos],2)+std::pow(postSamples[pos[i]][3*K+2*Kc+normpos2],2));
        norm = std::normal_distribution<double>(mean,st);
        samples[i] = norm(AUX_R);
    }

    return samples;
}

std::vector<double> sample_autofluorescence_single(std::vector<std::vector<double>>& postSamples, int K, int Kc, int nsamples, int pos){

    std::normal_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin(),postSamples[pos].begin()+K);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::normal_distribution<double>(postSamples[pos][K+normpos],postSamples[pos][2*K+normpos]);
        samples[i] = norm(AUX_R);
    }


    return samples;
}

std::vector<double> sample_autofluorescence_single(std::vector<std::vector<double>>& postSamples, int K, int Kc, std::vector<double>& weights, int nsamples, int pos){

    std::normal_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin(),postSamples[pos].begin()+K);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::normal_distribution<double>(postSamples[pos][K+normpos],postSamples[pos][2*K+normpos]);
        samples[i] = norm(AUX_R);
    }


    return samples;
}

std::vector<double> sample_deconvolution_single(std::vector<std::vector<double>>& postSamples, int K, int Kc, int nsamples, int pos){

    std::normal_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin()+3*K,postSamples[pos].begin()+3*K+Kc);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::normal_distribution<double>(postSamples[pos][3*K+Kc+normpos],postSamples[pos][3*K+2*Kc+normpos]);
        samples[i] = norm(AUX_R);
    }


    return samples;
}

std::vector<double> sample_deconvolution_single(std::vector<std::vector<double>>& postSamples, int K, int Kc, std::vector<double>& weights, int nsamples, int pos){

    std::normal_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin()+3*K,postSamples[pos].begin()+3*K+Kc);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::normal_distribution<double>(postSamples[pos][3*K+Kc+normpos],postSamples[pos][3*K+2*Kc+normpos]);
        samples[i] = norm(AUX_R);
    }


    return samples;
}

std::vector<double> sample_convolution_single(std::vector<std::vector<double>>& postSamples, int K, int Kc, int nsamples, int pos){

    std::normal_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    int normpos2;
    std::vector<double> aux;

    //Choose from the samples
    double st;
    double mean;
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin(),postSamples[pos].begin()+K);
        normpos = choicepos(aux)[0];
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin()+3*K,postSamples[pos].begin()+3*K+Kc);
        normpos2 = choicepos(aux)[0];
        //Sample from the gaussian
        mean = postSamples[pos][K+normpos]+postSamples[pos][3*K+Kc+normpos2];
        st = std::sqrt(std::pow(postSamples[pos][2*K+normpos],2)+std::pow(postSamples[pos][3*K+2*Kc+normpos2],2));
        norm = std::normal_distribution<double>(mean,st);
        samples[i] = norm(AUX_R);
    }


    return samples;
}

std::vector<double> sample_convolution_single(std::vector<std::vector<double>>& postSamples, int K, int Kc, std::vector<double>& weights, int nsamples, int pos){

    std::normal_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    int normpos2;
    std::vector<double> aux;

    //Choose from the samples
    double st;
    double mean;
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin(),postSamples[pos].begin()+K);
        normpos = choicepos(aux)[0];
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin()+3*K,postSamples[pos].begin()+3*K+Kc);
        normpos2 = choicepos(aux)[0];
        //Sample from the gaussian
        mean = postSamples[pos][K+normpos]+postSamples[pos][3*K+Kc+normpos2];
        st = std::sqrt(std::pow(postSamples[pos][2*K+normpos],2)+std::pow(postSamples[pos][3*K+2*Kc+normpos2],2));
        norm = std::normal_distribution<double>(mean,st);
        samples[i] = norm(AUX_R);
    }

    return samples;
}

std::vector<double> sample_autofluorescence_gamma(std::vector<std::vector<double>>& postSamples, int K, int Kc, int nsamples, double bias){

    std::gamma_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    std::vector<int> pos(choicepos(postSamples.size(), nsamples));
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin(),postSamples[pos[i]].begin()+K);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::gamma_distribution<double>(postSamples[pos[i]][2*K+normpos],postSamples[pos[i]][K+normpos]);
        samples[i] = norm(AUX_R)+bias;
        //pybind11::print(pos[i]," ",samples[i]," ",postSamples[pos[i]][K+normpos]," ",postSamples[pos[i]][2*K+normpos]);
    }


    return samples;
}

std::vector<double> sample_autofluorescence_gamma(std::vector<std::vector<double>>& postSamples, int K, int Kc, std::vector<double>& weights, int nsamples, double bias){

    std::gamma_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    std::vector<int> pos(choicepos(weights, nsamples));
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin(),postSamples[pos[i]].begin()+K);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::gamma_distribution<double>(postSamples[pos[i]][2*K+normpos],postSamples[pos[i]][K+normpos]);
        samples[i] = norm(AUX_R)+bias;
    }


    return samples;
}

std::vector<double> sample_deconvolution_gamma(std::vector<std::vector<double>>& postSamples, int K, int Kc, int nsamples, double bias){

    std::gamma_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    std::vector<int> pos(choicepos(postSamples.size(), nsamples));
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin()+3*K,postSamples[pos[i]].begin()+3*K+Kc);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::gamma_distribution<double>(postSamples[pos[i]][3*K+2*Kc+normpos],postSamples[pos[i]][3*K+Kc+normpos]);
        samples[i] = norm(AUX_R)+bias;
    }

    return samples;
}

std::vector<double> sample_deconvolution_gamma(std::vector<std::vector<double>>& postSamples, int K, int Kc, std::vector<double>& weights, int nsamples, double bias){

    std::gamma_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    std::vector<int> pos(choicepos(weights, nsamples));
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin()+3*K,postSamples[pos[i]].begin()+3*K+Kc);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::gamma_distribution<double>(postSamples[pos[i]][3*K+2*Kc+normpos],postSamples[pos[i]][3*K+Kc+normpos]);
        samples[i] = norm(AUX_R)+bias;
    }


    return samples;
}

std::vector<double> sample_convolution_gamma(std::vector<std::vector<double>>& postSamples, int K, int Kc, int nsamples, double bias){

    std::gamma_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    int normpos2;
    std::vector<double> aux;

    //Choose from the samples
    double st;
    double mean;
    std::vector<int> pos(choicepos(postSamples.size(), nsamples));
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin(),postSamples[pos[i]].begin()+K);
        normpos = choicepos(aux)[0];
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin()+3*K,postSamples[pos[i]].begin()+3*K+Kc);
        normpos2 = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::gamma_distribution<double>(postSamples[pos[i]][2*K+normpos],postSamples[pos[i]][K+normpos]);
        samples[i] = norm(AUX_R);
        norm = std::gamma_distribution<double>(postSamples[pos[i]][3*K+2*Kc+normpos2],postSamples[pos[i]][3*K+Kc+normpos2]);
        samples[i] += norm(AUX_R)+bias;
    }


    return samples;
}

std::vector<double> sample_convolution_gamma(std::vector<std::vector<double>>& postSamples, int K, int Kc, std::vector<double>& weights, int nsamples, double bias){

    std::gamma_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    int normpos2;
    std::vector<double> aux;

    //Choose from the samples
    double st;
    double mean;
    std::vector<int> pos(choicepos(weights, nsamples));
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin(),postSamples[pos[i]].begin()+K);
        normpos = choicepos(aux)[0];
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos[i]].begin()+3*K,postSamples[pos[i]].begin()+3*K+Kc);
        normpos2 = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::gamma_distribution<double>(postSamples[pos[i]][2*K+normpos],postSamples[pos[i]][K+normpos]);
        samples[i] = norm(AUX_R);
        norm = std::gamma_distribution<double>(postSamples[pos[i]][3*K+2*Kc+normpos2],postSamples[pos[i]][3*K+Kc+normpos2]);
        samples[i] += norm(AUX_R)+bias;
    }

    return samples;
}

std::vector<double> sample_autofluorescence_single_gamma(std::vector<std::vector<double>>& postSamples, int K, int Kc, int nsamples, int pos, double bias){

    std::gamma_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin(),postSamples[pos].begin()+K);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::gamma_distribution<double>(postSamples[pos][2*K+normpos],postSamples[pos][K+normpos]);
        samples[i] = norm(AUX_R)+bias;
    }


    return samples;
}

std::vector<double> sample_autofluorescence_single_gamma(std::vector<std::vector<double>>& postSamples, int K, int Kc, std::vector<double>& weights, int nsamples, int pos, double bias){

    std::gamma_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin(),postSamples[pos].begin()+K);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::gamma_distribution<double>(postSamples[pos][2*K+normpos],postSamples[pos][K+normpos]);
        samples[i] = norm(AUX_R)+bias;
    }


    return samples;
}

std::vector<double> sample_deconvolution_single_gamma(std::vector<std::vector<double>>& postSamples, int K, int Kc, int nsamples, int pos, double bias){

    std::gamma_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin()+3*K,postSamples[pos].begin()+3*K+Kc);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::gamma_distribution<double>(postSamples[pos][3*K+2*Kc+normpos],postSamples[pos][3*K+Kc+normpos]);
        samples[i] = norm(AUX_R)+bias;
    }


    return samples;
}

std::vector<double> sample_deconvolution_single_gamma(std::vector<std::vector<double>>& postSamples, int K, int Kc, std::vector<double>& weights, int nsamples, int pos, double bias){

    std::gamma_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    std::vector<double> aux;

    //Choose from the samples
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin()+3*K,postSamples[pos].begin()+3*K+Kc);
        normpos = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::gamma_distribution<double>(postSamples[pos][3*K+2*Kc+normpos],postSamples[pos][3*K+Kc+normpos]);
        samples[i] = norm(AUX_R)+bias;
    }

    return samples;
}

std::vector<double> sample_convolution_single_gamma(std::vector<std::vector<double>>& postSamples, int K, int Kc, int nsamples, int pos, double bias){

    std::gamma_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    int normpos2;
    std::vector<double> aux;

    //Choose from the samples
    double st;
    double mean;
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin(),postSamples[pos].begin()+K);
        normpos = choicepos(aux)[0];
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin()+3*K,postSamples[pos].begin()+3*K+Kc);
        normpos2 = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::gamma_distribution<double>(postSamples[pos][2*K+normpos],postSamples[pos][K+normpos]);
        samples[i] = norm(AUX_R);
        norm = std::gamma_distribution<double>(postSamples[pos][3*K+2*Kc+normpos2],postSamples[pos][3*K+Kc+normpos2]);
        samples[i] += norm(AUX_R)+bias;
    }

    return samples;
}

std::vector<double> sample_convolution_single_gamma(std::vector<std::vector<double>>& postSamples, int K, int Kc, std::vector<double>& weights, int nsamples, int pos, double bias){

    std::gamma_distribution<double> norm(0,1);
    std::vector<double> samples(nsamples, 0);

    int normpos;
    int normpos2;
    std::vector<double> aux;

    //Choose from the samples
    double st;
    double mean;
    for( int i = 0; i < nsamples; i++){
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin(),postSamples[pos].begin()+K);
        normpos = choicepos(aux)[0];
        //Choose from the gaussians of the sample
        aux = std::vector<double>(postSamples[pos].begin()+3*K,postSamples[pos].begin()+3*K+Kc);
        normpos2 = choicepos(aux)[0];
        //Sample from the gaussian
        norm = std::gamma_distribution<double>(postSamples[pos][2*K+normpos],postSamples[pos][K+normpos]);
        samples[i] = norm(AUX_R);
        norm = std::gamma_distribution<double>(postSamples[pos][3*K+2*Kc+normpos2],postSamples[pos][3*K+Kc+normpos2]);
        samples[i] += norm(AUX_R)+bias;
    }

    return samples;
}
