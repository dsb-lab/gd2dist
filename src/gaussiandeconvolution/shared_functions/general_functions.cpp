#include <vector>
#include <random>
#include <exception>
#include <iostream>
#include "global_random_generator.h"

//Cumsum definitions

std::vector<double> cumsum(std::vector<double>& x){
    std::vector<double> cumx(x.size(),0);

    cumx[0] = x[0];
    for(int i = 1; i < x.size(); i++){
        cumx[i] = cumx[i-1] + x[i];
    }

    return cumx;
}

//Choicepos definitions

std::vector<int> choicepos(int sup, int nsamples){

    if(0<sup){ //Check if sup is bigger than inf
        std::cout << "superior (sup) has to be bigger than 0\n";
    }

    std::uniform_int_distribution<int> unif(0,sup);
    std::vector<int> samples(nsamples, 0);

    for( int i = 0; i < nsamples; i++){
        //Sample weight
        samples[i] = unif(AUX_R);
    }

    return samples;
}

std::vector<int> choicepos(std::vector<double>& weights, int nsamples){

    int size = weights.size();
    std::vector<int> samples(nsamples, 0);
    double w;
    int count;
    std::uniform_real_distribution<double> unif(0,1);

    std::vector<double> cumweight = cumsum(weights);
    double total = cumweight[size-1];

    for( int i = 0; i < nsamples; i++){
        //Sample weight
        w = total*unif(AUX_R);
        count = 0;
        //Find 
        while(cumweight[count] < w and count < size){
            count++;
        }
        samples[i] = count;
    }

    return samples;
}