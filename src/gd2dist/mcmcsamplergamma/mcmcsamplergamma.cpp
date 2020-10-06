#include <vector>
#include <map>
#include <string>
#include <thread>
#include <algorithm>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <random>
#include <cmath>

#include "../shared_functions/probability_distributions.h"
#include "mcmcsamplergamma.h"
#include "pybind11/pybind11.h"

double gamma_pdf_batch(double x, double xlog, double n, double theta, double kconst,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta){
    
    double loglikelihood = 0;
    loglikelihood = -x/theta+(kconst-1)*xlog-n*kconst*std::log(theta)-n*std::lgamma(kconst); 
    //Add priors
    loglikelihood += gamma_pdf(theta,priortheta_theta,priortheta_k,0); 
    loglikelihood += gamma_pdf(kconst,priork_theta,priork_k,0); 
    
    return loglikelihood;
}

double gamma_sum_pdf_batch(std::vector<double> &datac, double theta, double kconst, double thetac, double kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission,
                            std::vector<int> &id, int counter){
    
    double loglikelihood = 0;
    int loc;
    for(int i = 0; i < counter; i++){
        loc = id[i];
        //loglikelihood = gamma_sum_pdf(datac[loc],theta,kconst,thetac,kconstc,bias,precission);
        loglikelihood += gamma_sum_pdf(datac[loc],theta,kconst,thetac,kconstc,bias,precission);
    }
    //Add priors
    //loglikelihood += gamma_pdf(thetac,priortheta_thetac,priortheta_kc,0); 
    //loglikelihood += gamma_pdf(kconstc,priork_thetac,priork_kc,0); 
    
    return loglikelihood;
}

double gamma_pdf_full_batch(std::vector<double> &datac, double theta, double kconst, std::vector<double> thetac, std::vector<double> kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission,
                            std::vector<std::vector<int>> &id, std::vector<int> &counter,
                            double x, double xlog, double n,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta){

    double loglikelihood = 0;

    //Add fluorescence
    loglikelihood += gamma_pdf_batch(x, xlog, n, theta, kconst, priortheta_k, priortheta_theta, priork_k, priork_theta);
    for(int i = 0; i < thetac.size(); i++){
        loglikelihood += gamma_sum_pdf_batch(datac, theta, kconst, thetac[i], kconstc[i], bias, priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, id[i], counter[i]);
    }
    return loglikelihood;
}

double gamma_pdf_full_batch(std::vector<double> &datac, std::vector<double> theta, std::vector<double> kconst, double thetac, double kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission,
                            std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter, int pos){

    double loglikelihood = 0;

    //Add fluorescence
    for(int i = 0; i < theta.size(); i++){
        loglikelihood += gamma_sum_pdf_batch(datac, theta[i], kconst[i], thetac, kconstc, bias, priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, id[i][pos], counter[i][pos]);
    }
    return loglikelihood;
}

double gamma_pdf_full_batch_slow(std::vector<double> &data, std::vector<double> &datac, std::vector<double> theta, std::vector<double> kconst, std::vector<double> thetac, std::vector<double> kconstc,
                            double bias,
                            double precission,
                            std::vector<std::vector<int>> &id, std::vector<int> counter,
                            std::vector<std::vector<std::vector<int>>> &idc, std::vector<std::vector<int>> &counterc,
                            double priorbias_sigma){
    
    double loglikelihood = 0;
    int loc;
    //Autofluorescence
    for(int i =  0; i < theta.size(); i++){
        for(int j = 0; j < counter[i]; j++){
            loc = id[i][j];
            loglikelihood += gamma_pdf(data[loc],theta[i],kconst[i],bias);
        }
    }
    //Convolution
    for(int i =  0; i < theta.size(); i++){
        for(int j = 0; j < thetac.size(); j++){
            for(int k = 0; k < counterc[i][j]; k++){
                loc = idc[i][j][k];
                loglikelihood += gamma_sum_pdf(datac[loc],theta[i],kconst[i],theta[j],kconst[j],bias,precission);
            }
        }
    }
    //Prior
    loglikelihood += -std::pow(bias,2)/(2*std::pow(priorbias_sigma,2));

    return loglikelihood;
}

void slice_theta(std::mt19937 &r, std::vector<double> &n, std::vector<double> &x, std::vector<double> &xlog, 
                            std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                            std::vector<double> &thetanew, std::vector<double> &datac, std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                            double bias, double precission){

        int N = theta.size();
        std::uniform_real_distribution<double> uniform(0,1);
        double loss_old;
        double loss_new;
        double newkconst;
        double acceptance;
        double min;
        double max;
        double expansion = 0.5;
        int count;
        double old;

        //Slice sampling
        for (int i = 0; i < N; i++){

            old = theta[i];
            for (int j = 0; j < 10; j++){
                loss_old = gamma_pdf_full_batch(datac, old, kconst[i], thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, 
                            priork_k, priork_theta);
                //Chose new height
                loss_old += std::log(uniform(r));
                //Expand
                min = old-expansion;
                loss_new = gamma_pdf_full_batch(datac, min, kconst[i], thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    min -= expansion;
                    if(min <= 0){
                        min = 0.01;
                        break;
                    }
                    loss_new = gamma_pdf_full_batch(datac, min, kconst[i], thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                    count++;
                }
                max = old+expansion;
                loss_new = gamma_pdf_full_batch(datac, max, kconst[i], thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    max += expansion;
                    loss_new = gamma_pdf_full_batch(datac, max, kconst[i], thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                    count++;
                }

                //Sample
                count = 0;
                do{
                    newkconst = (max-min)*uniform(r)+min;
                    loss_new = gamma_pdf_full_batch(datac, newkconst, kconst[i], thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                    //Adapt boundaries
                    if(loss_new < loss_old){
                        if(newkconst < old){
                            min = newkconst;
                        }
                        else if(newkconst > old){
                            max = newkconst;
                        }
                    }
                    count++;
                }while(loss_new < loss_old && count < 200000);

                old = newkconst;
            }

            thetanew[i] = newkconst;
        }

    return;
}

void slice_k(std::mt19937 &r, std::vector<double> &n, std::vector<double> &x, std::vector<double> &xlog, 
                            std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                            std::vector<double> &kconstnew, std::vector<double> &datac, std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                            double bias, double precission){

        int N = theta.size();
        std::uniform_real_distribution<double> uniform(0,1);
        double loss_old;
        double loss_new;
        double newkconst;
        double acceptance;
        double min;
        double max;
        double expansion = 0.5;
        int count;
        double old;

        //Slice sampling
        for (int i = 0; i < N; i++){

            old = kconst[i];
            for (int j = 0; j < 10; j++){
                loss_old = gamma_pdf_full_batch(datac, theta[i], old, thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                //Chose new height
                loss_old += std::log(uniform(r));
                //Expand
                min = old-expansion;
                loss_new = gamma_pdf_full_batch(datac, theta[i], min, thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    min -= expansion;
                    if(min <= 0){
                        min = 0.01;
                        break;
                    }
                    loss_new = gamma_pdf_full_batch(datac, theta[i], min, thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                    count++;
                }
                max = old+expansion;
                loss_new = gamma_pdf_full_batch(datac, theta[i], max, thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    max += expansion;
                    loss_new = gamma_pdf_full_batch(datac, theta[i], max, thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                    count++;
                }

                //Sample
                count = 0;
                do{
                    newkconst = (max-min)*uniform(r)+min;
                    loss_new = gamma_pdf_full_batch(datac, theta[i], newkconst, thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                    //Adapt boundaries
                    if(loss_new < loss_old){
                        if(newkconst < old){
                            min = newkconst;
                        }
                        else if(newkconst > old){
                            max = newkconst;
                        }
                    }
                    count++;
                }while(loss_new < loss_old && count < 200000);

                old = newkconst;
            }

            kconstnew[i] = newkconst;
        }

    return;
}

void slice_thetac(std::mt19937 &r, 
                std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                std::vector<double> &thetanewc, std::vector<double> &datac, std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter,
                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                double bias, double precission){

        int N = thetac.size();
        std::uniform_real_distribution<double> uniform(0,1);
        double loss_old;
        double loss_new;
        double newkconst;
        double acceptance;
        double min;
        double max;
        double expansion = 0.5;
        int count;
        double old;

        //Slice sampling
        for (int i = 0; i < N; i++){

            old = thetac[i];
            for (int j = 0; j < 10; j++){
                loss_old = gamma_pdf_full_batch(datac, theta, kconst, old, kconstc[i],
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission,
                            id, counter, i);
                //Chose new height
                loss_old += std::log(uniform(r));
                //Expand
                min = old-expansion;
                loss_new = gamma_pdf_full_batch(datac, theta, kconst, min, kconstc[i],
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission,
                            id, counter, i);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    min -= expansion;
                    if(min <= 0){
                        min = 0.01;
                        break;
                    }
                    loss_new = gamma_pdf_full_batch(datac, theta, kconst, min, kconstc[i],
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission,
                            id, counter, i);
                    count++;
                }
                max = old+expansion;
                loss_new = gamma_pdf_full_batch(datac, theta, kconst, max, kconstc[i],
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission,
                            id, counter, i);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    max += expansion;
                    loss_new = gamma_pdf_full_batch(datac, theta, kconst, max, kconstc[i],
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission,
                            id, counter, i);
                    count++;
                }

                //Sample
                count = 0;
                do{
                    newkconst = (max-min)*uniform(r)+min;
                    loss_new = gamma_pdf_full_batch(datac, theta, kconst, newkconst, kconstc[i],
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission,
                            id, counter, i);
                    //Adapt boundaries
                    if(loss_new < loss_old){
                        if(newkconst < old){
                            min = newkconst;
                        }
                        else if(newkconst > old){
                            max = newkconst;
                        }
                    }
                    count++;
                }while(loss_new < loss_old && count < 200000);

                old = newkconst;
            }

            thetanewc[i] = newkconst;
        }

    return;
}

void slice_kc(std::mt19937 &r, 
                std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                std::vector<double> &kconstnewc, std::vector<double> &datac, std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter,
                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                double bias, double precission){

        int N = thetac.size();
        std::uniform_real_distribution<double> uniform(0,1);
        double loss_old;
        double loss_new;
        double newkconst;
        double acceptance;
        double min;
        double max;
        double expansion = 0.5;
        int count;
        double old;

        //Slice sampling
        for (int i = 0; i < N; i++){

            old = kconstc[i];
            for (int j = 0; j < 10; j++){
                loss_old = gamma_pdf_full_batch(datac, theta, kconst, thetac[i], old,
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission,
                            id, counter, i);
                //Chose new height
                loss_old += std::log(uniform(r));
                //Expand
                min = old-expansion;
                loss_new = gamma_pdf_full_batch(datac, theta, kconst, thetac[i], min,
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission,
                            id, counter, i);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    min -= expansion;
                    if(min <= 0){
                        min = 0.01;
                        break;
                    }
                    loss_new = gamma_pdf_full_batch(datac, theta, kconst, thetac[i], min,
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission,
                            id, counter, i);
                    count++;
                }
                max = old+expansion;
                loss_new = gamma_pdf_full_batch(datac, theta, kconst, thetac[i], max,
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission,
                            id, counter, i);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    max += expansion;
                    loss_new = gamma_pdf_full_batch(datac, theta, kconst, thetac[i], max,
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission,
                            id, counter, i);
                    count++;
                }

                //Sample
                count = 0;
                do{
                    newkconst = (max-min)*uniform(r)+min;
                    loss_new = gamma_pdf_full_batch(datac, theta, kconst, thetac[i], newkconst,
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission,
                            id, counter, i);
                    //Adapt boundaries
                    if(loss_new < loss_old){
                        if(newkconst < old){
                            min = newkconst;
                        }
                        else if(newkconst > old){
                            max = newkconst;
                        }
                    }
                    count++;
                }while(loss_new < loss_old && count < 200000);

                old = newkconst;
            }

            kconstnewc[i] = newkconst;
        }

    return;
}

void slice_bias(std::mt19937 &r, 
                std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                std::vector<double> &data, std::vector<double> &datac, 
                std::vector<std::vector<int>> &id, std::vector<int> &counter,
                std::vector<std::vector<std::vector<int>>> &idc, std::vector<std::vector<int>> &counterc,
                double bias, double & biasnew, double priorbias_sigma, double precission){

        int N = theta.size();
        std::uniform_real_distribution<double> uniform(0,1);
        double loss_old;
        double loss_new;
        double newkconst;
        double acceptance;
        double min;
        double max;
        double expansion = 0.5;
        int count;
        double old;

        //Slice sampling
        for (int i = 0; i < N; i++){

            old = bias;
            for (int j = 0; j < 10; j++){
                loss_old = gamma_pdf_full_batch_slow(data, datac, theta, kconst, thetac, kconstc,
                            old, precission,
                            id, counter,
                            idc, counterc,
                            priorbias_sigma);
                //Chose new height
                loss_old += std::log(uniform(r));
                //Expand
                min = old-expansion;
                loss_new = gamma_pdf_full_batch_slow(data, datac, theta, kconst, thetac, kconstc,
                            min, precission,
                            id, counter,
                            idc, counterc,
                            priorbias_sigma);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    min -= expansion;
                    if(min <= 0){
                        min = 0.01;
                        break;
                    }
                    loss_new = gamma_pdf_full_batch_slow(data, datac, theta, kconst, thetac, kconstc,
                            min, precission,
                            id, counter,
                            idc, counterc,
                            priorbias_sigma);
                    count++;
                }
                max = old+expansion;
                loss_new = gamma_pdf_full_batch_slow(data, datac, theta, kconst, thetac, kconstc,
                            max, precission,
                            id, counter,
                            idc, counterc,
                            priorbias_sigma);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    max += expansion;
                    loss_new = gamma_pdf_full_batch_slow(data, datac, theta, kconst, thetac, kconstc,
                            max, precission,
                            id, counter,
                            idc, counterc,
                            priorbias_sigma);
                    count++;
                }

                //Sample
                count = 0;
                do{
                    newkconst = (max-min)*uniform(r)+min;
                    loss_new = gamma_pdf_full_batch_slow(data, datac, theta, kconst, thetac, kconstc,
                            newkconst, precission,
                            id, counter,
                            idc, counterc,
                            priorbias_sigma);
                    //Adapt boundaries
                    if(loss_new < loss_old){
                        if(newkconst < old){
                            min = newkconst;
                        }
                        else if(newkconst > old){
                            max = newkconst;
                        }
                    }
                    count++;
                }while(loss_new < loss_old && count < 200000);

                old = newkconst;
            }

            biasnew = newkconst;
        }

    return;
}

void Gibbs_convolved_step(std::mt19937 & r, std::vector<double> & data, std::vector<double> & datac,
                    std::vector<double> & pi, std::vector<double> & theta, std::vector<double> & kconst, 
                    std::vector<double> & pinew, std::vector<double> & thetanew, std::vector<double> & kconstnew, 
                    double alpha, double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                    std::vector<double> & pic, std::vector<double> & thetac, std::vector<double> & kconstc, 
                    std::vector<double> & pinewc, std::vector<double> & thetanewc, std::vector<double> & kconstnewc, 
                    double alphac, double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                    double & bias, double & biasnew,
                    double priorbias_sigma, double priorbias_min,
                    std::vector<std::vector<int>> id, std::vector<std::vector<std::vector<int>>> idc,
                    int precission){

    //Step of the convolution
    unsigned int K = pi.size();
    unsigned int Kc = pic.size();
    int size = K*Kc;
    std::vector<double> probabilities(K,0);    //Auxiliar vector for the weights of z
    std::vector<double> probabilitiesc(size,0);   //Auxiliar vector for the weights of zc
    std::vector<int> choice(K,0);
    std::vector<int> choicec(size,0);
    std::vector<int> counter(K,0);
    std::vector<std::vector<int>> counterc(K,std::vector<int>(Kc,0));

    std::vector<double> n(K,0);   //Counts of the different convolved gaussians
    std::vector<double> x(K,0);   //Mean of the different convolved gaussians
    std::vector<double> xlog(K,0);   //Squared expression of the different convolved gaussians

    double thetaj = 0, phij = 0;
    std::vector<double> nminalpha(K,0);
    std::vector<double> nminalphac(Kc,0);
    double effmean;
    double effkconst;

    double max;

    //Evaluate the autofluorescence data
    for (unsigned int i = 0; i < data.size(); i++){
        //Compute the weights for each gaussian
        max = -INFINITY;
        for (unsigned int j = 0; j < K; j++){
            probabilities[j] = std::log(pi[j])
                                +gamma_pdf(data[i],theta[j],kconst[j], bias);
            if (probabilities[j] > max){
                max = probabilities[j];
            }
        }
        //Normalize
        for (unsigned int j = 0; j < K; j++){
            probabilities[j] -= max;
            probabilities[j] = std::exp(probabilities[j]);
        }
        //Assign a gaussian
        multinomial_1(r, probabilities, choice);
        //Compute the basic statistics
        //We compute all the statistics already since we are going to use them only for the autofluorescence sampling
        for (unsigned int j = 0; j < K; j++){
            n[j] += choice[j];
            x[j] += choice[j]*data[i]-bias;
            xlog[j] += std::log(choice[j]-bias);
            nminalpha[j] += choice[j];
            if(choice[j] == 1){
                id[j][counter[j]] = i;
                counter[j]++;
            }
        }
    }
    
    //Evaluate the convolved data
    for (unsigned int i = 0; i < datac.size(); i++){
        //Compute the weights for each gamma
        max = -INFINITY;
        for (unsigned int j = 0; j < K; j++){
            for (unsigned int k = 0; k < Kc; k++){
                probabilitiesc[K*k+j] = std::log(pic[k])+std::log(pi[j])
                                    +gamma_sum_pdf(datac[i],theta[j],kconst[j],thetac[k],kconstc[k],bias,precission);
                if (probabilitiesc[K*k+j]>max){
                    max = probabilitiesc[K*k+j];
                }
            }
        }
        //Normalize
        for (unsigned int j = 0; j < K; j++){
            for (unsigned int k = 0; k < Kc; k++){
                probabilitiesc[K*k+j] -= max;
                probabilitiesc[K*k+j] = std::exp(probabilitiesc[K*k+j]);
            }
        }
        //Assign a convolved gamma
        multinomial_1(r, probabilitiesc, choicec);
        //Save the identity
        //We do not compute the statistics here because they will have to be updated since this dataset is used for sampling twice
        for (unsigned int j = 0; j < K; j++){
            for (unsigned int k = 0; k < Kc; k++){
                if(choicec[K*k+j] == 1){
                    //Add to list of identities to sum
                    idc[j][k][counterc[j][k]] = i;
                    counterc[j][k]++;
                }
                //Add to list of contributions
                nminalpha[j] += choicec[K*k+j];
                nminalphac[k] += choicec[K*k+j];
            }
        }
    }
    //Add the priors
    for (unsigned int k = 0; k < K; k++){
        nminalpha[k] += alpha/K;
    }    
    for (unsigned int k = 0; k < Kc; k++){
        nminalphac[k] += alphac/Kc;
    }    

    //Sample the new mixtures
    dirichlet(r, nminalpha, pinew);
    dirichlet(r, nminalphac, pinewc);

    //Sample autofluorescence
    //Sample the thetas
    slice_theta(r, n, x, xlog, theta, kconst, thetac, kconstc, thetanew, datac, idc, counterc,
                priortheta_k, priortheta_theta, priork_k, priork_theta,
                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac,
                bias, precission);
    //Sample the kconst
    slice_k(r, n, x, xlog, theta, kconst, thetac, kconstc, thetanew, datac, idc, counterc,
                priortheta_k, priortheta_theta, priork_k, priork_theta,
                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac,
                bias, precission);

    //Sample the convolution
    //Sample the thetas
    slice_thetac(r, theta, kconst, thetac, kconstc, thetanewc, datac, idc, counterc, priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, bias, precission);
    //Sample the kconst
    slice_kc(r, theta, kconst, thetac, kconstc, kconstnewc, datac, idc, counterc, priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, bias, precission);

    //Sample the bias
    slice_bias(r,theta, kconst, thetac, kconstc, data, datac, id, counter, idc, counterc, bias, biasnew, priorbias_sigma, precission);

    return;
}

void chain(int pos0, std::vector<std::vector<double>> & posterior, std::vector<double> & data, std::vector<double> & datac,                          
                                int ignored_iterations, int iterations, int nChains,
                                int K, int Kc, double alpha, double alphac, 
                                double priortheta_k, double priortheta_theta, double priork_k, double priork_theta, 
                                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                                double priorbias_sigma, double priorbias_min, 
                                bool initialised, bool showProgress, int seed, int precission){
    //Variables for the random generation
    std::mt19937 r;
    r.seed(seed);

    std::vector<double> pi(K), theta(K), kconst(K), pinew(K), thetanew(K), kconstnew(K);

    std::vector<double> pic(Kc), thetac(Kc), kconstc(Kc), pinewc(Kc), thetanewc(Kc), kconstnewc(Kc);

    double bias, biasnew;

    std::vector<std::vector<std::vector<int>>> idc(K,std::vector<std::vector<int>>(Kc,std::vector<int>(datac.size(),0)));
    std::vector<std::vector<int>> id(K,std::vector<int>(datac.size(),0));

    //Initialise
    //Initialized sampling from the prior
    if (!initialised){
        std::gamma_distribution<double> dist(priortheta_k,priortheta_theta);
        std::gamma_distribution<double> dist2(priork_k,priork_theta);
        for (int i = 0; i < K; i++){
            pi[i] = 1;
            theta[i] = dist(r);
            kconst[i] = dist2(r);
        }

        dist = std::gamma_distribution<double>(priortheta_kc,priortheta_thetac);
        dist2 = std::gamma_distribution<double>(priork_kc,priork_thetac);
        for (int i = 0; i < K; i++){
            pic[i] = 1;
            thetac[i] = dist(r);
            kconstc[i] = dist2(r);
        }
        bias = priorbias_min;
    }else{
        for (int i = 0; i < K; i++){
            pi[i] = posterior[pos0][i];
            theta[i] = posterior[pos0][K+i];
            kconst[i] = posterior[pos0][2*K+i];
        }
        for (int i = 0; i < Kc; i++){
            pic[i] = posterior[pos0][3*K+i];
            thetac[i] = posterior[pos0][3*K+Kc+i];
            kconstc[i] = posterior[pos0][3*K+2*Kc+i];
        }
    }
    
    int progressStep = floor(ignored_iterations/10);
    int progressCounter = 0;
    int chainId = int(pos0/iterations);
    //Ignorable, steps
    for (int i = 0; i < ignored_iterations; i++){
        Gibbs_convolved_step(r, data, datac,
                         pi, theta, kconst, pinew, thetanew, kconstnew, alpha, priortheta_k, priortheta_theta, priork_k, priork_theta,
                         pic, thetac, kconstc, pinewc, thetanewc, kconstnewc, alphac, priortheta_kc, priortheta_thetac, priork_kc, priork_thetac,
                         bias, biasnew,
                         priorbias_sigma, priorbias_min, 
                         id, idc, precission);
        pi = pinew;
        theta = thetanew;
        kconst = kconstnew;
        pic = pinewc;
        thetac = thetanewc;
        kconstc = kconstnewc;
        bias = biasnew;

        if(showProgress){
            if(i % progressStep == 0){
                pybind11::gil_scoped_acquire acquire;
                pybind11::print("Chain", chainId, " Ignorable iterations: ", progressCounter * 10, "%");
                pybind11::gil_scoped_release release;
                progressCounter++;
            }
        }
    }
    if(showProgress){
        pybind11::gil_scoped_acquire acquire;
        pybind11::print("Chain", chainId, " Ignorable iterations: 100%");
        pybind11::gil_scoped_release release;
    }

    progressStep = floor(iterations/10);
    progressCounter = 0;
    //Recorded steps
    for (unsigned int i = 0; i < iterations; i++){
        Gibbs_convolved_step(r, data, datac,
                         pi, theta, kconst, pinew, thetanew, kconstnew, alpha, priortheta_k, priortheta_theta, priork_k, priork_theta,
                         pic, thetac, kconstc, pinewc, thetanewc, kconstnewc, alphac, priortheta_kc, priortheta_thetac, priork_kc, priork_thetac,
                         bias, biasnew, 
                         priorbias_sigma, priorbias_min,
                         id, idc, precission);
        pi = pinew;
        theta = thetanew;
        kconst = kconstnew;
        for (unsigned int j = 0; j < K; j++){
            posterior[pos0+i][j] = pinew[j];
            posterior[pos0+i][K+j] = thetanew[j];
            posterior[pos0+i][2*K+j] = kconstnew[j];
        }
        pic = pinewc;
        thetac = thetanewc;
        kconstc = kconstnewc;
        for (unsigned int j = 0; j < Kc; j++){
            posterior[pos0+i][3*K+j] = pinewc[j];
            posterior[pos0+i][3*K+Kc+j] = thetanewc[j];
            posterior[pos0+i][3*K+2*Kc+j] = kconstnewc[j];
        }
        bias = biasnew;
        posterior[pos0+i][3*K+3*Kc] = biasnew;


        if(showProgress){
            if(i % progressStep == 0){
                pybind11::gil_scoped_acquire acquire;
                pybind11::print("Chain",chainId," Recorded iterations: ",progressCounter * 10,"%");
                pybind11::gil_scoped_release release;
                progressCounter++;
            }
        }
    }
    if(showProgress){
        pybind11::gil_scoped_acquire acquire;
        pybind11::print("Chain",chainId," Recorded iterations: 100%");
        pybind11::gil_scoped_release release;
    }

    return;
}
std::vector<std::vector<double>> fit(std::vector<double> & data, std::vector<double>& datac,
                          int ignored_iterations, int iterations, int nChains,
                          int K, int Kc, 
                          double alpha, double alphac, 
                          double priortheta_k, double priortheta_theta, double priork_k, double priork_theta, 
                          double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                          double priorbias_sigma, double priorbias_min,
                          int precission,
                          std::vector<std::vector<double>> initial_conditions, bool showProgress, int seed){

    //Variable to check if initialised
    bool initialised = false;
    //Initialise posterior
    std::vector<std::vector<double>> posterior;
    //Check if initial conditions are given
    if(!initial_conditions.empty()){
        //Check correct 
        if(initial_conditions.size()!=nChains){
            throw std::length_error("initial_conditions must have as many initial conditions as chains in the model.");
        }
        if(initial_conditions[0].size()!=(3*K+3*Kc)){
            throw std::length_error("Each chain requires as initial conditions all the parameters of the model: \n" 
                                    + std::to_string(K) + " weights for the noise mixture \n"
                                    + std::to_string(K) + " means for the noise mixture \n"
                                    + std::to_string(K) + " std's for the noise mixture \n"
                                    + std::to_string(Kc) + " weights for the noise mixture \n"
                                    + std::to_string(Kc) + " means for the noise mixture \n"
                                    + std::to_string(Kc) + " std's for the noise mixture \n" );
        }
        //Create matrix
        posterior = std::vector<std::vector<double>>(iterations*nChains,std::vector<double>(3*K+3*Kc,0));
        //Assign initial conditions
        for(int i = 0; i < nChains; i++){
            posterior[iterations*i] = initial_conditions[i];
        }
        initialised = true;
    }else{
        //Create matrix
        posterior = std::vector<std::vector<double>>(iterations*nChains,std::vector<double>(3*K+3*Kc,0));
        initialised = false;
    }

    //Create threads
    std::vector<std::thread> chains;
    for(int i = 0; i < nChains; i++){
        int a = i*iterations;
        int seedchain = seed+i;
        chains.push_back(std::thread(chain, a, std::ref(posterior), std::ref(data), std::ref(datac),                          
                                ignored_iterations, iterations, nChains,
                                K, Kc, alpha, alphac, 
                                priortheta_k, priortheta_theta, priork_k, priork_theta, 
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                                priorbias_sigma, priorbias_min,
                                initialised, showProgress, seedchain, precission)); //Need explicit by reference std::refs
    }
    //Wait for rejoining
    for(int i = 0; i < nChains; i++){
        chains[i].join();
    }
    
    return posterior;
}