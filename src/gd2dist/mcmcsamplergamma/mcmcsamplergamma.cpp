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
    if(n == 0){
        loglikelihood = 0; 
    }else{
        loglikelihood = -x/theta+(kconst-1)*xlog-n*kconst*std::log(theta)-n*std::lgamma(kconst); 
        //Add priors
        loglikelihood += gamma_pdf(theta,priortheta_theta,priortheta_k,0); 
        loglikelihood += gamma_pdf(kconst,priork_theta,priork_k,0); 
    }
    
    return loglikelihood;
}

double gamma_sum_pdf_batch(std::vector<double> &datac, double theta, double kconst, double thetac, double kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission, std::string method,
                            std::vector<int> &id, int counter){
    
    double loglikelihood = 0;
    int loc;
    double x;
    double logx;
    for(int i = 0; i < counter; i++){
        loc = id[i];
        loglikelihood += gamma_sum_pdf(datac[loc],theta,kconst,thetac,kconstc,bias,precission,method);
    }
    //Add priors
    loglikelihood += gamma_pdf(thetac,priortheta_thetac,priortheta_kc,0); 
    loglikelihood += gamma_pdf(kconstc,priork_thetac,priork_kc,0); 
    
    return loglikelihood;
}

double gamma_sum_pdf_batch(double x, double xlog, double n, double theta, double kconst, double thetac, double kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission, std::string method){
    
    double loglikelihood = 0;
    //Add likelihood
    double mu = theta*kconst+thetac*kconstc;
    double s = theta*theta*kconst+thetac*thetac*kconstc;
    double thetastar = s/mu;
    double kconststar = mu*mu/s;
    if(n == 0){
        loglikelihood = 0;
    }else{
        loglikelihood = -x/thetastar+(kconststar-1)*xlog-n*kconststar*std::log(thetastar)-n*std::lgamma(kconststar); 
        //Add priors
        loglikelihood += gamma_pdf(thetac,priortheta_thetac,priortheta_kc,0); 
        loglikelihood += gamma_pdf(kconstc,priork_thetac,priork_kc,0); 
    }
    
    return loglikelihood;
}

double gamma_pdf_full_batch(std::vector<double> &datac, double theta, double kconst, std::vector<double> thetac, std::vector<double> kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission, std::string method,
                            std::vector<std::vector<int>> &id, std::vector<int> &counter,
                            double x, double xlog, double n,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta){

    double loglikelihood = 0;

    //Add fluorescence
    loglikelihood += gamma_pdf_batch(x, xlog, n, theta, kconst, priortheta_k, priortheta_theta, priork_k, priork_theta);
    for(int i = 0; i < thetac.size(); i++){
        loglikelihood += gamma_sum_pdf_batch(datac, theta, kconst, thetac[i], kconstc[i], bias, priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method, id[i], counter[i]);
    }
    return loglikelihood;
}

double gamma_pdf_full_batch(double theta, double kconst, std::vector<double> thetac, std::vector<double> kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission, std::string method,
                            double x, double xlog, double n, std::vector<double> xc, std::vector<double> xlogc, std::vector<double> nc,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta){

    double loglikelihood = 0;

    //Add fluorescence
    loglikelihood += gamma_pdf_batch(x, xlog, n, theta, kconst, priortheta_k, priortheta_theta, priork_k, priork_theta);
    for(int i = 0; i < thetac.size(); i++){
        loglikelihood += gamma_sum_pdf_batch(xc[i], xlogc[i], nc[i], theta, kconst, thetac[i], kconstc[i], bias, priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method);
    }
    return loglikelihood;
}

double gamma_pdf_full_batch(std::vector<double> &datac, std::vector<double> theta, std::vector<double> kconst, double thetac, double kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission, std::string method,
                            std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter, int pos){

    double loglikelihood = 0;

    //Add fluorescence
    for(int i = 0; i < theta.size(); i++){
        loglikelihood += gamma_sum_pdf_batch(datac, theta[i], kconst[i], thetac, kconstc, bias, priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method, id[i][pos], counter[i][pos]);
    }
    return loglikelihood;
}

double gamma_pdf_full_batch(std::vector<std::vector<double>> &xc, std::vector<std::vector<double>> &xlogc, std::vector<std::vector<double>> &nc,
                            std::vector<double> theta, std::vector<double> kconst, double thetac, double kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission, std::string method,
                            int pos){

    double loglikelihood = 0;

    //Add fluorescence
    for(int i = 0; i < theta.size(); i++){
        loglikelihood += gamma_sum_pdf_batch(xc[i][pos], xlogc[i][pos], nc[i][pos], theta[i], kconst[i], thetac, kconstc, bias, priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method);
    }
    return loglikelihood;
}

double gamma_pdf_full_batch_slow(std::vector<double> &data, std::vector<double> &datac, std::vector<double> theta, std::vector<double> kconst, std::vector<double> thetac, std::vector<double> kconstc,
                            double bias,
                            double precission, std::string method,
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
                loglikelihood += gamma_sum_pdf(datac[loc],theta[i],kconst[i],theta[j],kconst[j],bias,precission,method);
            }
        }
    }
    //Prior
    loglikelihood += -std::pow(bias,2)/(2*std::pow(priorbias_sigma,2));

    return loglikelihood;
}

void slice_theta(std::mt19937 & r, std::vector<double> &n, std::vector<double> &x, std::vector<double> &xlog, 
                            std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                            std::vector<double> &thetanew, std::vector<double> &datac, std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                            double bias, double precission, std::string method){

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
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, 
                            priork_k, priork_theta);
                //Chose new height
                loss_old += std::log(uniform(r));
                //Expand
                min = old-expansion;
                loss_new = gamma_pdf_full_batch(datac, min, kconst[i], thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    min -= expansion;
                    if(min <= 0){
                        min = 0;
                        break;
                    }
                    loss_new = gamma_pdf_full_batch(datac, min, kconst[i], thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                    count++;
                }
                max = old+expansion;
                loss_new = gamma_pdf_full_batch(datac, max, kconst[i], thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    max += expansion;
                    loss_new = gamma_pdf_full_batch(datac, max, kconst[i], thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                    count++;
                }

                //Sample
                count = 0;
                do{
                    newkconst = (max-min)*uniform(r)+min;
                    loss_new = gamma_pdf_full_batch(datac, newkconst, kconst[i], thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
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

void slice_theta(std::mt19937 & r, std::vector<double> &n, std::vector<double> &x, std::vector<double> &xlog, std::vector<std::vector<double>> &nc, std::vector<std::vector<double>> &xc, std::vector<std::vector<double>> &xlogc, 
                            std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                            std::vector<double> &thetanew,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                            double bias, double precission, std::string method, std::vector<double> & slice){

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
        int contribution;

        //Slice sampling
        for (int i = 0; i < N; i++){

            //Check if someone if voting
            contribution = 0;
            contribution = n[i];
            for (int j = 0; j < thetac.size(); j++){
                contribution += nc[i][j];
            }

            if (contribution > 0){
                old = theta[i];
                expansion = slice[i];
                for (int j = 0; j < 10; j++){
                    loss_old = gamma_pdf_full_batch(old, kconst[i], thetac, kconstc, bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                                x[i], xlog[i], n[i], xc[i], xlogc[i], nc[i], priortheta_k, priortheta_theta, 
                                priork_k, priork_theta);
                    //Chose new height
                    loss_old += std::log(uniform(r));
                    //Expand
                    min = old-expansion;
                    if(min <= 0){
                        min = 0;
                    }
                    loss_new = gamma_pdf_full_batch(min, kconst[i], thetac, kconstc, bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                                x[i], xlog[i], n[i], xc[i], xlogc[i], nc[i], priortheta_k, priortheta_theta, 
                                priork_k, priork_theta);
                    count = 0;
                    while(loss_new > loss_old && count < 200000){
                        min -= expansion;
                        if(min <= 0){
                            min = 0;
                            break;
                        }
                        loss_new = gamma_pdf_full_batch(min, kconst[i], thetac, kconstc, bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                                x[i], xlog[i], n[i], xc[i], xlogc[i], nc[i], priortheta_k, priortheta_theta, 
                                priork_k, priork_theta);
                        count++;
                    }
                    max = old+expansion;
                    loss_new = gamma_pdf_full_batch(max, kconst[i], thetac, kconstc, bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                                x[i], xlog[i], n[i], xc[i], xlogc[i], nc[i], priortheta_k, priortheta_theta, 
                                priork_k, priork_theta);
                    count = 0;
                    while(loss_new > loss_old && count < 200000){
                        max += expansion;
                        loss_new = gamma_pdf_full_batch(max, kconst[i], thetac, kconstc, bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                                x[i], xlog[i], n[i], xc[i], xlogc[i], nc[i], priortheta_k, priortheta_theta, 
                                priork_k, priork_theta);
                        count++;
                    }

                    slice[i] = (max-min)/10.0;

                    //Sample
                    count = 0;
                    do{
                        newkconst = (max-min)*uniform(r)+min;
                        loss_new = gamma_pdf_full_batch(newkconst, kconst[i], thetac, kconstc, bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                                x[i], xlog[i], n[i], xc[i], xlogc[i], nc[i], priortheta_k, priortheta_theta, 
                                priork_k, priork_theta);
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
            }else{
                thetanew[i] = theta[i];
            }
        }

        /*pybind11::gil_scoped_acquire acquire;
        pybind11::print("Theta ",thetanew[0]," ",thetanew[1]);
        pybind11::gil_scoped_release release;*/

    return;
}

void slice_k(std::mt19937 & r, std::vector<double> &n, std::vector<double> &x, std::vector<double> &xlog, 
                            std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                            std::vector<double> &kconstnew, std::vector<double> &datac, std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                            double bias, double precission, std::string method){

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
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                //Chose new height
                loss_old += std::log(uniform(r));
                //Expand
                min = old-expansion;
                loss_new = gamma_pdf_full_batch(datac, theta[i], min, thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    min -= expansion;
                    if(min <= 0){
                        min = 0.01;
                        break;
                    }
                    loss_new = gamma_pdf_full_batch(datac, theta[i], min, thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                    count++;
                }
                max = old+expansion;
                loss_new = gamma_pdf_full_batch(datac, theta[i], max, thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    max += expansion;
                    loss_new = gamma_pdf_full_batch(datac, theta[i], max, thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                            id[i], counter[i], x[i], xlog[i], n[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                    count++;
                }

                //Sample
                count = 0;
                do{
                    newkconst = (max-min)*uniform(r)+min;
                    loss_new = gamma_pdf_full_batch(datac, theta[i], newkconst, thetac, kconstc, bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
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

void slice_k(std::mt19937 & r, std::vector<double> &n, std::vector<double> &x, std::vector<double> &xlog, std::vector<std::vector<double>> &nc, std::vector<std::vector<double>> &xc, std::vector<std::vector<double>> &xlogc, 
                            std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                            std::vector<double> &kconstnew,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                            double bias, double precission, std::string method, std::vector<double> & slice){

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
        int contribution;

        //Slice sampling
        for (int i = 0; i < N; i++){

            //Check if someone if voting
            contribution = 0;
            contribution = n[i];
            for (int j = 0; j < thetac.size(); j++){
                contribution += nc[i][j];
            }

            if (contribution > 0){
                old = kconst[i];
                expansion = slice[i];
                for (int j = 0; j < 10; j++){
                    loss_old = gamma_pdf_full_batch(theta[i], old, thetac, kconstc, bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                                x[i], xlog[i], n[i], xc[i], xlogc[i], nc[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                    //Chose new height
                    loss_old += std::log(uniform(r));
                    //Expand
                    min = old-expansion;
                    if(min <= 0){
                        min = 0;
                    }
                    loss_new = gamma_pdf_full_batch(theta[i], min, thetac, kconstc, bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                                x[i], xlog[i], n[i], xc[i], xlogc[i], nc[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                    count = 0;
                    while(loss_new > loss_old && count < 200000){
                        min -= expansion;
                        if(min <= 0){
                            min = 0;
                            break;
                        }
                        loss_new = gamma_pdf_full_batch(theta[i], min, thetac, kconstc, bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                                x[i], xlog[i], n[i], xc[i], xlogc[i], nc[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                        count++;
                    }
                    max = old+expansion;
                    loss_new = gamma_pdf_full_batch(theta[i], max, thetac, kconstc, bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                                x[i], xlog[i], n[i], xc[i], xlogc[i], nc[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                    count = 0;
                    while(loss_new > loss_old && count < 200000){
                        max += expansion;
                        loss_new = gamma_pdf_full_batch(theta[i], max, thetac, kconstc, bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                                x[i], xlog[i], n[i], xc[i], xlogc[i], nc[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
                        count++;
                    }

                    slice[i] = (max-min)/10.0;
                    if(min < 0){
                        pybind11::gil_scoped_acquire acquire;
                        pybind11::print(min);
                        pybind11::gil_scoped_release release;
                    }
                    //Sample
                    count = 0;
                    do{
                        newkconst = (max-min)*uniform(r)+min;
                        loss_new = gamma_pdf_full_batch(theta[i], newkconst, thetac, kconstc, bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, precission, method,
                                x[i], xlog[i], n[i], xc[i], xlogc[i], nc[i], priortheta_k, priortheta_theta, priork_k, priork_theta);
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
            }else{
                kconstnew[i] = kconst[i];
            }
        }

        /*pybind11::gil_scoped_acquire acquire;
        pybind11::print("K ",kconstnew[0]," ",kconstnew[1]);
        pybind11::gil_scoped_release release;*/

    return;
}

void slice_thetac(std::mt19937 & r, 
                std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                std::vector<double> &thetanewc, std::vector<double> &datac, std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter,
                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                double bias, double precission, std::string method){

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
                            precission, method,
                            id, counter, i);
                //Chose new height
                loss_old += std::log(uniform(r));
                //Expand
                min = old-expansion;
                loss_new = gamma_pdf_full_batch(datac, theta, kconst, min, kconstc[i],
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission, method,
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
                            precission, method,
                            id, counter, i);
                    count++;
                }
                max = old+expansion;
                loss_new = gamma_pdf_full_batch(datac, theta, kconst, max, kconstc[i],
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission, method,
                            id, counter, i);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    max += expansion;
                    loss_new = gamma_pdf_full_batch(datac, theta, kconst, max, kconstc[i],
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission, method,
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
                            precission, method,
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

void slice_thetac(std::mt19937 & r, 
                std::vector<std::vector<double>> &xc, std::vector<std::vector<double>> &xlogc, std::vector<std::vector<double>> &nc,
                std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                std::vector<double> &thetanewc,
                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                double bias, double precission, std::string method, std::vector<double> & slice){

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
        int contribution;

        //Slice sampling
        for (int i = 0; i < N; i++){

            contribution = 0;
            for(int j = 0; j < theta.size(); j++){
                contribution += nc[j][i];
            }

            if(contribution > 0){
                old = thetac[i];
                expansion = slice[i];

                for (int j = 0; j < 10; j++){
                    loss_old = gamma_pdf_full_batch(xc, xlogc, nc, theta, kconst, old, kconstc[i],
                                bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                                precission, method,
                                i);
                    //Chose new height
                    loss_old += std::log(uniform(r));
                    //Expand
                    min = old-expansion;
                    if(min <= 0){
                        min = 0;
                    }
                    loss_new = gamma_pdf_full_batch(xc, xlogc, nc, theta, kconst, min, kconstc[i],
                                bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                                precission, method,
                                i);
                    count = 0;
                    while(loss_new > loss_old && count < 200000){
                        min -= expansion;
                        if(min <= 0){
                            min = 0;
                            break;
                        }
                        loss_new = gamma_pdf_full_batch(xc, xlogc, nc, theta, kconst, min, kconstc[i],
                                bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                                precission, method,
                                i);
                        count++;
                    }
                    max = old+expansion;
                    loss_new = gamma_pdf_full_batch(xc, xlogc, nc, theta, kconst, max, kconstc[i],
                                bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                                precission, method,
                                i);
                    count = 0;
                    while(loss_new > loss_old && count < 200000){
                        max += expansion;
                        loss_new = gamma_pdf_full_batch(xc, xlogc, nc, theta, kconst, max, kconstc[i],
                                bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                                precission, method,
                                i);
                        count++;
                    }

                    slice[i] = (max-min)/10.0;

                    //Sample
                    count = 0;
                    do{
                        newkconst = (max-min)*uniform(r)+min;
                        loss_new = gamma_pdf_full_batch(xc, xlogc, nc, theta, kconst, newkconst, kconstc[i],
                                bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                                precission, method,
                                i);
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
            }else{
                thetanewc[i] = thetac[i];
            }


        }

        /*pybind11::gil_scoped_acquire acquire;
        pybind11::print("Thetac ",thetanewc[0]," ",thetanewc[1]," ",thetanewc[2]," ");
        pybind11::gil_scoped_release release;*/

    return;
}

void slice_kc(std::mt19937 &r, 
                std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                std::vector<double> &kconstnewc, std::vector<double> &datac, std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter,
                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                double bias, double precission, std::string method){

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
                            precission, method,
                            id, counter, i);
                //Chose new height
                loss_old += std::log(uniform(r));
                //Expand
                min = old-expansion;
                loss_new = gamma_pdf_full_batch(datac, theta, kconst, thetac[i], min,
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission, method,
                            id, counter, i);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    min -= expansion;
                    if(min <= 0){
                        min = 0;
                        break;
                    }
                    loss_new = gamma_pdf_full_batch(datac, theta, kconst, thetac[i], min,
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission, method,
                            id, counter, i);
                    count++;
                }
                max = old+expansion;
                loss_new = gamma_pdf_full_batch(datac, theta, kconst, thetac[i], max,
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission, method,
                            id, counter, i);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    max += expansion;
                    loss_new = gamma_pdf_full_batch(datac, theta, kconst, thetac[i], max,
                            bias,
                            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                            precission, method,
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
                            precission, method,
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

void slice_kc(std::mt19937 &r, 
                std::vector<std::vector<double>> &xc, std::vector<std::vector<double>> &xlogc, std::vector<std::vector<double>> &nc,
                std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                std::vector<double> &kconstnewc,
                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                double bias, double precission, std::string method, std::vector<double> & slice){

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
        int contribution;

        //Slice sampling
        for (int i = 0; i < N; i++){

            contribution = 0;
            for(int j = 0; j < theta.size(); j++){
                contribution += nc[j][i];
            }

            if(contribution > 0){

                old = kconstc[i];
                expansion = slice[i];
                for (int j = 0; j < 10; j++){
                    loss_old = gamma_pdf_full_batch(xc, xlogc, nc, theta, kconst, thetac[i], old,
                                bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                                precission, method,
                                i);
                    //Chose new height
                    loss_old += std::log(uniform(r));
                    //Expand
                    min = old-expansion;
                    if(min <= 0){
                        min = 0;
                    }
                    loss_new = gamma_pdf_full_batch(xc, xlogc, nc, theta, kconst, thetac[i], min,
                                bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                                precission, method,
                                i);
                    count = 0;
                    while(loss_new > loss_old && count < 200000){
                        min -= expansion;
                        if(min <= 0){
                            min = 0;
                            break;
                        }
                        loss_new = gamma_pdf_full_batch(xc, xlogc, nc, theta, kconst, thetac[i], min,
                                bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                                precission, method,
                                i);
                        count++;
                    }
                    max = old+expansion;
                    loss_new = gamma_pdf_full_batch(xc, xlogc, nc, theta, kconst, thetac[i], max,
                                bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                                precission, method,
                                i);
                    count = 0;
                    while(loss_new > loss_old && count < 200000){
                        max += expansion;
                        loss_new = gamma_pdf_full_batch(xc, xlogc, nc, theta, kconst, thetac[i], max,
                                bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                                precission, method,
                                i);
                        count++;
                    }

                    slice[i] = (max-min)/10.0;

                    //Sample
                    count = 0;
                    do{
                        newkconst = (max-min)*uniform(r)+min;
                        loss_new = gamma_pdf_full_batch(xc, xlogc, nc, theta, kconst, thetac[i], newkconst,
                                bias,
                                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, 
                                precission, method,
                                i);
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

            }else{
                kconstnewc[i] = kconstc[i];
            }
        }

/*        pybind11::gil_scoped_acquire acquire;
        pybind11::print("Kc ",kconstnewc[0]," ",kconstnewc[1]," ",kconstnewc[2]," ", "contribution ", contribution);
        pybind11::print("Kold ",kconstc[0]," ",kconstc[1]," ",kconstc[2]," ", "contribution ", contribution);
        pybind11::print("Nc0 ",xc[0][0]," ",xc[0][1]," ",xc[0][2]," ");
        pybind11::print("Nc1 ",xc[1][0]," ",xc[1][1]," ",xc[1][2]," ");
        pybind11::gil_scoped_release release;*/

    return;
}

/*void slice_bias(std::mt19937 &r, 
                std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                std::vector<double> &data, std::vector<double> &datac, 
                std::vector<std::vector<int>> &id, std::vector<int> &counter,
                std::vector<std::vector<std::vector<int>>> &idc, std::vector<std::vector<int>> &counterc,
                double bias, double & biasnew, double priorbias_sigma, double precission, std::string method){

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
                            old, precission, method,
                            id, counter,
                            idc, counterc,
                            priorbias_sigma);
                //Chose new height
                loss_old += std::log(uniform(r));
                //Expand
                min = old-expansion;
                loss_new = gamma_pdf_full_batch_slow(data, datac, theta, kconst, thetac, kconstc,
                            min, precission, method,
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
                            min, precission, method,
                            id, counter,
                            idc, counterc,
                            priorbias_sigma);
                    count++;
                }
                max = old+expansion;
                loss_new = gamma_pdf_full_batch_slow(data, datac, theta, kconst, thetac, kconstc,
                            max, precission, method,
                            id, counter,
                            idc, counterc,
                            priorbias_sigma);
                count = 0;
                while(loss_new > loss_old && count < 200000){
                    max += expansion;
                    loss_new = gamma_pdf_full_batch_slow(data, datac, theta, kconst, thetac, kconstc,
                            max, precission, method,
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
                            newkconst, precission, method,
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
}*/

/*void Gibbs_convolved_step(std::mt19937 & r, std::vector<double> & data, std::vector<double> & datac,
                    std::vector<double> & pi, std::vector<double> & theta, std::vector<double> & kconst, 
                    std::vector<double> & pinew, std::vector<double> & thetanew, std::vector<double> & kconstnew, 
                    double alpha, double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                    std::vector<double> & pic, std::vector<double> & thetac, std::vector<double> & kconstc, 
                    std::vector<double> & pinewc, std::vector<double> & thetanewc, std::vector<double> & kconstnewc, 
                    double alphac, double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                    double bias,
                    std::vector<std::vector<int>> id, std::vector<std::vector<std::vector<int>>> idc,
                    double precission, std::string method){

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
                bias, precission, method);
    //Sample the kconst
    slice_k(r, n, x, xlog, theta, kconst, thetac, kconstc, kconstnew, datac, idc, counterc,
                priortheta_k, priortheta_theta, priork_k, priork_theta,
                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac,
                bias, precission, method);
    thetanew[0] = 3;
    //kconstnew[0] = 3;
    thetanewc[0] = 6;
    kconstnewc[0] = 6;

    //Sample the convolution
    //Sample the thetas
    slice_thetac(r, theta, kconst, thetac, kconstc, thetanewc, datac, idc, counterc, priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, bias, precission);
    //Sample the kconst
    slice_kc(r, theta, kconst, thetac, kconstc, kconstnewc, datac, idc, counterc, priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, bias, precission);

    //Sample the bias
    slice_bias(r,theta, kconst, thetac, kconstc, data, datac, id, counter, idc, counterc, bias, biasnew, priorbias_sigma, precission);
    
    return;
}*/

void Gibbs_convolved_step(std::mt19937 & r, std::vector<double> & data, std::vector<double> & datac,
                    std::vector<double> & pi, std::vector<double> & theta, std::vector<double> & kconst, 
                    std::vector<double> & pinew, std::vector<double> & thetanew, std::vector<double> & kconstnew, 
                    double alpha, double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                    std::vector<double> & pic, std::vector<double> & thetac, std::vector<double> & kconstc, 
                    std::vector<double> & pinewc, std::vector<double> & thetanewc, std::vector<double> & kconstnewc, 
                    double alphac, double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                    double bias,
                    double precission, std::string method, 
                    std::vector<double> & slice_step_theta, std::vector<double> & slice_step_k, 
                    std::vector<double> & slice_step_thetac, std::vector<double> & slice_step_kc){

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
    std::vector<std::vector<double>> nc(K,std::vector<double>(Kc,0));   //Counts of the different convolved gaussians
    std::vector<std::vector<double>> xc(K,std::vector<double>(Kc,0));   //Mean of the different convolved gaussians
    std::vector<std::vector<double>> xlogc(K,std::vector<double>(Kc,0));   //Squared expression of the different convolved gaussians

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
            xlog[j] += choice[j]*std::log(data[i]-bias);
            nminalpha[j] += choice[j];
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
        //Compute the statistics here because they will have to be updated since this dataset is used for sampling twice
        for (unsigned int j = 0; j < K; j++){
            for (unsigned int k = 0; k < Kc; k++){
                //Add to list of contributions
                nminalpha[j] += choicec[K*k+j];
                nminalphac[k] += choicec[K*k+j];
                nc[j][k] += choicec[K*k+j];
                xc[j][k] += choicec[K*k+j]*(datac[i]-bias);
                xlogc[j][k] += choicec[K*k+j]*std::log(datac[i]-bias);
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
    slice_theta(r, n, x, xlog, nc, xc, xlogc, theta, kconst, thetac, kconstc, thetanew,
                priortheta_k, priortheta_theta, priork_k, priork_theta,
                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac,
                bias, precission, method, slice_step_theta);

    //Sample the kconst
    slice_k(r, n, x, xlog, nc, xc, xlogc, thetanew, kconst, thetac, kconstc, kconstnew,
                priortheta_k, priortheta_theta, priork_k, priork_theta,
                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac,
                bias, precission, method, slice_step_k);

    //Sample the convolution
    //Sample the thetas
    slice_thetac(r, xc, xlogc, nc, thetanew, kconstnew, thetac, kconstc, thetanewc, 
                priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, bias, precission, method, 
                slice_step_thetac);

    //Sample the kconst
    slice_kc(r, xc, xlogc, nc, thetanew, kconstnew, thetanewc, kconstc, kconstnewc, 
            priortheta_kc, priortheta_thetac, priork_kc, priork_thetac, bias, precission, method,
            slice_step_kc);
    
    return;
}

void chain(int pos0, std::vector<std::vector<double>> & posterior, std::vector<double> & data, std::vector<double> & datac,                          
                                int ignored_iterations, int iterations, int nChains,
                                int K, int Kc, double alpha, double alphac, 
                                double priortheta_k, double priortheta_theta, double priork_k, double priork_theta, 
                                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                                double bias, 
                                bool initialised, bool showProgress, int seed, double precission, std::string method){
    //Variables for the random generation
    std::mt19937 r;
    r.seed(seed);

    std::vector<double> pi(K), theta(K), kconst(K), pinew(K), thetanew(K), kconstnew(K);

    std::vector<double> pic(Kc), thetac(Kc), kconstc(Kc), pinewc(Kc), thetanewc(Kc), kconstnewc(Kc);

    std::vector<double> slice_step_theta(K,0.5);
    std::vector<double> slice_step_k(K,0.5);
    std::vector<double> slice_step_thetac(Kc,0.5);
    std::vector<double> slice_step_kc(Kc,0.5);

    if(method == "exact"){
        /*std::vector<std::vector<std::vector<int>>> idc(K,std::vector<std::vector<int>>(Kc,std::vector<int>(datac.size(),0)));
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
            for (int i = 0; i < Kc; i++){
                pic[i] = 1;
                thetac[0] = dist(r);
                kconstc[0] = dist2(r);
                thetac[1] = dist(r);
                kconstc[1] = dist2(r);
            }
        }else{
            double mean = 0;
            double var = 0;
            int size = 0;

            size = data.size();
            for( int i = 0; i < size; i++){
                mean += data[i]/size;
                var += data[i]*data[i]/size;
            }
            var = var-mean*mean*size;
            for (int i = 0; i < K; i++){
                pi[i] = posterior[pos0][i];
                theta[i] = var/mean;
                kconst[i] = mean/theta[i];
            }

            size = datac.size();
            for( int i = 0; i < size; i++){
                mean += datac[i]/size;
                var += datac[i]*data[i]/size;
            }
            var = var-mean*mean*size;
            for (int i = 0; i < Kc; i++){
                pic[i] = posterior[pos0][3*K+i];
                thetac[i] = var/mean;
                kconstc[i] = mean/theta[i];
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
                            bias, 
                            id, idc, precission, method, slice_step_theta, slice_step_k, slice_step_thetac, slice_step_kc);
            pi = pinew;
            theta = thetanew;
            kconst = kconstnew;
            pic = pinewc;
            thetac = thetanewc;
            kconstc = kconstnewc;

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
                            bias,
                            id, idc, precission, method, slice_theta, slice_k, slice_thetac, slice_kc);
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
        }*/

    }else if(method == "moments"){

        //Initialise
        //Initialized sampling from the prior
        if(initialised){
            std::gamma_distribution<double> dist(priortheta_k,priortheta_theta);
            std::gamma_distribution<double> dist2(priork_k,priork_theta);
            for (int i = 0; i < K; i++){
                pi[i] = 1;
                theta[i] = 4;//dist(r);
                kconst[i] = 5;//dist2(r);
            }

            dist = std::gamma_distribution<double>(priortheta_kc,priortheta_thetac);
            dist2 = std::gamma_distribution<double>(priork_kc,priork_thetac);
            for (int i = 0; i < Kc; i++){
                pic[i] = 1;
                thetac[0] = 5;//dist(r);
                kconstc[0] = 3;//dist2(r);
                thetac[1] = 5;//dist(r);
                kconstc[1] = 3;//dist2(r);
            }
        }else{
        
            double mean = 0;
            double var = 0;
            int size = 0;

            size = data.size();
            for( int i = 0; i < size; i++){
                mean += data[i]/size;
                var += data[i]*data[i]/size;
            }
            var = var-mean*mean;
            std::uniform_real_distribution<double> dist(0.8*var/mean,var/mean);
            std::uniform_real_distribution<double> dist2(0.8*mean*mean/var,mean*mean/var);
            for (int i = 0; i < K; i++){
                pi[i] = 1;
                theta[i] = dist(r);
                kconst[i] = dist2(r);
            }

            size = datac.size();
            mean = 0;
            var = 0;
            for( int i = 0; i < size; i++){
                mean += datac[i]/size;
                var += datac[i]*datac[i]/size;
            }
            var = var-mean*mean;
            std::uniform_real_distribution<double> dist3(0.8*var/mean,var/mean);
            std::uniform_real_distribution<double> dist4(0.8*mean*mean/var,mean*mean/var);
            for (int i = 0; i < Kc; i++){
                pic[i] = 1;
                thetac[i] = dist3(r);
                kconstc[i] = dist4(r);
            }
        }

        /*pybind11::gil_scoped_acquire acquire;
        pybind11::print("Theta0 ",theta[0]," ",theta[1]);
        pybind11::print("K0 ",kconst[0]," ",kconst[1]);
        pybind11::print("Thetac0 ", thetac[0], " ",thetac[1], " ",thetac[2], " ");
        pybind11::print("Kc0 ", kconstc[0], " ",kconstc[1], " ",kconstc[2], " ");
        pybind11::gil_scoped_release release;*/

        int progressStep = floor(ignored_iterations/10);
        int progressCounter = 0;
        int chainId = int(pos0/iterations);
        //Ignorable, steps
        for (int i = 0; i < ignored_iterations; i++){
            Gibbs_convolved_step(r, data, datac,
                            pi, theta, kconst, pinew, thetanew, kconstnew, alpha, priortheta_k, priortheta_theta, priork_k, priork_theta,
                            pic, thetac, kconstc, pinewc, thetanewc, kconstnewc, alphac, priortheta_kc, priortheta_thetac, priork_kc, priork_thetac,
                            bias, 
                            precission,method, slice_step_theta, slice_step_k, slice_step_thetac, slice_step_kc);
            pi = pinew;
            theta = thetanew;
            kconst = kconstnew;
            pic = pinewc;
            thetac = thetanewc;
            kconstc = kconstnewc;

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
                            bias,
                            precission,method, slice_step_theta, slice_step_k, slice_step_thetac, slice_step_kc);
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

    }

    return;
}

std::vector<std::vector<double>> fit(std::vector<double> & data, std::vector<double>& datac,
                          int ignored_iterations, int iterations, int nChains,
                          int K, int Kc, 
                          double alpha, double alphac, 
                          double priortheta_k, double priortheta_theta, double priork_k, double priork_theta, 
                          double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                          double bias,
                          double precission, std::string method,
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
                                bias,
                                initialised, showProgress, seedchain, precission, method)); //Need explicit by reference std::refs
    }
    //Wait for rejoining
    for(int i = 0; i < nChains; i++){
        chains[i].join();
    }
    
    return posterior;
}