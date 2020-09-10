#include <vector>
#include <random>
#include <exception>
#include <stdexcept>
#include <iostream>

#include "general_functions.h"
#include "global_random_generator.h"

//Mean
double mean(std::vector<double>& x){

    double value = 0;
    int size = x.size();

    for(int i = 0; i < size; i++){
        value += x[i]/size;
    }

    return value;
}

//Percentiles
std::vector<double> percentile(std::vector<double>& x, std::vector<double>& percentiles){

    std::vector<double> aux = mergeSort(x);
    std::vector<double> value(percentiles.size(),0);

    int pos = 0;
    for(int i = 0; i < percentiles.size(); i++){
        pos = floor(percentiles[i]*x.size());
        if(pos < x.size()-2){
            value[i] = (aux[pos]+aux[pos+1])/2;
        }else{
            value[i] = aux[x.size()-1];
        }
    }

    return value;
}

//Cumsum definitions
std::vector<double> cumsum(std::vector<double>& x){
    std::vector<double> cumx(x.size(),0);

    cumx[0] = x[0];
    for(int i = 1; i < x.size(); i++){
        cumx[i] = cumx[i-1] + x[i];
    }

    return cumx;
}

//Mergesort
std::vector<double> mergeSort(std::vector<double>& toSort){

    std::vector<double> sorted(toSort);

    int pos;
    int pos2;

    int max;
    int max2;

    int length = toSort.size();

    int step = 1;

    int counter;
    int counterMax;

    int sortPosition;

    std::vector<double> aux;

    bool change;

    while(step < length){
        pos = 0;
        pos2 = pos + step;
        max2 = 0;

        sortPosition = 0;
        
        change = false;

        while(max2 < length){
            max = pos2;
            max2 = std::min(max+step,length);
            while((pos < max) && (pos2 < max2)){
                if(toSort[pos] < toSort[pos2]){
                    sorted[sortPosition] = toSort[pos];
                    pos++;
                    sortPosition++;
                }
                else{
                    sorted[sortPosition] = toSort[pos2];
                    pos2++;
                    sortPosition++;
                    change = true;
                }
            }
            while(pos < max){
                sorted[sortPosition] = toSort[pos];
                pos++;
                sortPosition++;            
            }
            while(pos2 < max2){
                sorted[sortPosition] = toSort[pos2];
                pos2++;
                sortPosition++;            
            }

            pos = max2;
            pos2 = pos + step;
            if(pos2 > length){
                max2 = length;
                change = true;
            }

        }

        aux = toSort;
        toSort = sorted;
        sorted = aux;

        if(!change){
            break;
        }

        step *= 2;
    }

    aux = toSort;
    toSort = sorted;
    sorted = aux;

    return sorted; 
}

//Choicepos definitions

std::vector<int> choicepos(int sup, int nsamples){

    if(0>sup){ //Check if sup is bigger than inf
        throw std::invalid_argument("superior (sup) has to be bigger than 0\n");
    }

    std::uniform_int_distribution<int> unif(0,sup-1);
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
        while((cumweight[count] < w) && (count < size)){
            count++;
        }
        samples[i] = count;
    }

    return samples;
}