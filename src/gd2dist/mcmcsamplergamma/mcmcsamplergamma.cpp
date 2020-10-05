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

/*double logLikelihood(std::vector<double> & pi, std::vector<double> & theta, std::vector<double> & kconst,
                    std::vector<double> & pic, std::vector<double> & thetac, std::vector<double> & kconstc,
                    std::vector<double>  & data, std::vector<double>  & datac){

    double log = 0;
    double max;

    max = -INFINITY;
    for (int i = 0; i < data.size(); i++){
        for (unsigned int j = 0; j < pi.size(); j++){
            log += std::log(pi[j])
                +gaussian_pdf(data[i],theta[j],kconst[j]);
        }
    }

    for (int i = 0; i < datac.size(); i++){
        for (unsigned int j = 0; j < pi.size(); j++){
            for (unsigned int k = 0; k < pic.size(); k++){
                log += std::log(pic[k])+std::log(pi[j])
                                    +gaussian_pdf(datac[i],theta[j]+thetac[k],std::sqrt(std::pow(kconstc[k],2)+std::pow(kconst[j],2)));
            }
        }
    }

    return log;
}*/

double effective_gamma_not_normalized(double pos, std::vector<double> n, std::vector<double> x2, std::vector<double> kconst, double theta, double kconst){

    double aux = 0;
    int l = kconst.size();
    int l2 = x2.size();

    for (int i = 0; i < l; i++){
        aux += -n[i]*std::log(std::pow(pos,2)+std::pow(kconst[i],2))/2
                -x2[i]/(2*(std::pow(pos,2)+std::pow(kconst[i],2))); 
    }
    if (l == l2-1){
        aux += -n[l]*std::log(std::pow(pos,2))/2
                -x2[l]/(2*(std::pow(pos,2)));          
    }
    //Add prior
    aux += -std::pow(pos,2)/theta+(kconst-1)*std::log(std::pow(pos,2));

    return aux;
}

void slice_effective_gamma(std::mt19937 &r, std::vector<std::vector<double>> &n,
                             std::vector<std::vector<double>> &x2, 
                             std::vector<double> &kconst, std::vector<double> &kconstold, std::vector<double> &kconstnew,
                             double theta, double kconst){

        int N = kconstnew.size();
        std::uniform_real_distribution<double> uniform(0,1);
        double loss_old;
        double loss_new;
        double newkconst;
        double acceptance;
        double min;
        double max;
        double expansion = 0.5;
        int counter;
        double old;

        //Slice sampling
        for (int i = 0; i < N; i++){

            old = kconstold[i];
            for (int j = 0; j < 10; j++){
                loss_old = effective_gamma_not_normalized(old, n[i], x2[i], kconst, theta, kconst);
                //Chose new height
                loss_old += std::log(uniform(r));
                //Expand
                min = old-expansion;
                loss_new = effective_gamma_not_normalized(min, n[i], x2[i], kconst, theta, kconst);
                counter = 0;
                while(loss_new > loss_old && counter < 200000){
                    min -= expansion;
                    if(min <= 0){
                        min = 0.01;
                        break;
                    }
                    loss_new = effective_gamma_not_normalized(min, n[i], x2[i], kconst, theta, kconst);
                    counter++;
                }
                max = old+expansion;
                loss_new = effective_gamma_not_normalized(max, n[i], x2[i], kconst, theta, kconst);
                counter = 0;
                while(loss_new > loss_old && counter < 200000){
                    max += expansion;
                    loss_new = effective_gamma_not_normalized(max, n[i], x2[i], kconst, theta, kconst);
                    counter++;
                }

                //Sample
                counter = 0;
                do{
                    newkconst = (max-min)*uniform(r)+min;
                    loss_new = effective_gamma_not_normalized(newkconst, n[i], x2[i], kconst, theta, kconst);
                    //Adapt boundaries
                    if(loss_new < loss_old){
                        if(newkconst < old){
                            min = newkconst;
                        }
                        else if(newkconst > old){
                            max = newkconst;
                        }
                    }
                    counter++;
                }while(loss_new < loss_old && counter < 200000);

                old = newkconst;
            }

            kconstnew[i] = newkconst;
        }

    return;
}

Gibbs_convolved_step(std::mt19937 & r, std::vector<double> & data, std::vector<double> & datac,
                    std::vector<double> & pi, std::vector<double> & theta, std::vector<double> & kconst, 
                    std::vector<double> & pinew, std::vector<double> & thetanew, std::vector<double> & kconstnew, 
                    double alpha, double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                    std::vector<double> & pic, std::vector<double> & thetac, std::vector<double> & kconstc, 
                    std::vector<double> & pinewc, std::vector<double> & thetanewc, std::vector<double> & kconstnewc, 
                    double alphac, double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                    double & bias, double & biasnew,
                    double priorbias_sigma, double priorbias_min,
                    std::vector<std::vector<std::vector<double>>> id){

    //Step of the convolution
    unsigned int K = pi.size();
    unsigned int Kc = pic.size();
    int size = K*Kc;
    std::vector<double> probabilities(K,0);    //Auxiliar vector for the weights of z
    std::vector<double> probabilitiesc(size,0);   //Auxiliar vector for the weights of zc
    std::vector<int> choice(K,0);
    std::vector<int> choicec(size,0);

    std::vector<std::vector<double>> n(K,std::vector<double>(Kc+1,0));   //Counts of the different convolved gaussians
    std::vector<std::vector<double>> x(K,std::vector<double>(Kc+1,0));   //Mean of the different convolved gaussians
    std::vector<std::vector<double>> x2(K,std::vector<double>(Kc+1,0));   //Squared expression of the different convolved gaussians
    std::vector<double> nminalpha(K,0);

    std::vector<std::vector<double>> nc(Kc,std::vector<double>(K,0));   //Counts of the different convolved gaussians
    std::vector<std::vector<double>> xc(Kc,std::vector<double>(K,0));   //Mean of the different convolved gaussians
    std::vector<std::vector<double>> x2c(Kc,std::vector<double>(K,0));   //Squared expression of the different convolved gaussians
    double thetaj = 0, phij = 0;
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
            n[j][Kc] += choice[j];
            x[j][Kc] += choice[j]*data[i];
            x2[j][Kc] += choice[j]*std::pow(data[i]-theta[j],2);
            nminalpha[j] += choice[j];
        }
    }
    
    //Evaluate the convoluted data
    for (unsigned int i = 0; i < datac.size(); i++){
        //Compute the weights for each gaussian
        max = -INFINITY;
        for (unsigned int j = 0; j < K; j++){
            for (unsigned int k = 0; k < Kc; k++){
                probabilitiesc[K*k+j] = std::log(pic[k])+std::log(pi[j])
                                    +gaussian_pdf(datac[i],theta[j]+thetac[k],std::sqrt(std::pow(kconstc[k],2)+std::pow(kconst[j],2)));
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
        //Assign a convoluted gaussian
        thetaltinomial_1(r, probabilitiesc, choicec);
        //Save the identity
        //We do not compute the statistics because they will have to be updated since this dataset is used for sampling twice
        for (unsigned int j = 0; j < K; j++){
            for (unsigned int k = 0; k < Kc; k++){
                id[k][j][i] = choicec[K*k+j];
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

    //Sample the autofluorescence variance and mean
    //Compute the statistics
    for (unsigned int i = 0; i < datac.size(); i++){
        for (unsigned int j = 0; j < K; j++){
            for (unsigned int k = 0; k < Kc; k++){
                n[j][k] += id[k][j][i];
                x[j][k] += id[k][j][i]*(datac[i]-thetac[k]);
                x2[j][k] += id[k][j][i]*std::pow(datac[i]-theta[j]-thetac[k],2);
            }
        }
    }
    //Sample the variances
    //sample_effective_gamma(r, n, x2, kconstc, kconst, kconstnew, kconstWidth);
    slice_effective_gamma(r, n, x2, kconstc, kconst, kconstnew, theta, kconst);
    //Sample the means
    for (unsigned int j = 0; j < K; j++){
        //Convoluted terms
        for (unsigned int k = 0; k < Kc; k++){
            effkconst += n[j][k]/(std::pow(kconstc[k],2)+std::pow(kconstnew[j],2));
            effmean += x[j][k]/(std::pow(kconstc[k],2)+std::pow(kconstnew[j],2));
        }
        //Autofluorescence terms
        effkconst += n[j][Kc]/(std::pow(kconstnew[j],2));
        effmean += x[j][Kc]/(std::pow(kconstnew[j],2));
        
        if (effkconst == 0){
            thetanew[j] = theta[j];
        }else{
            effmean = effmean/effkconst;
            effkconst = 1/effkconst;

            if(std::isnan(effkconst)==false){
                std::normal_distribution<double> gaussian(effmean, effkconst);
                thetanew[j] = gaussian(r);
            }
        }

        //Clean variables
        effmean = 0;
        effkconst = 0;
    }    

    //Sample the convoluted variance and mean
    //Compute the statistics
    for (unsigned int i = 0; i < datac.size(); i++){
        for (unsigned int j = 0; j < K; j++){
            for (unsigned int k = 0; k < Kc; k++){
                nc[k][j] += id[k][j][i];
                xc[k][j] += id[k][j][i]*(datac[i]-thetanew[j]);
                x2c[k][j] += id[k][j][i]*std::pow(datac[i]-thetanew[j]-thetac[k],2);
            }
        }
    }
    //Sample the variances
    //sample_effective_gamma(r, nc, x2c, kconstnew, kconstc, kconstnewc, kconstWidth);
    slice_effective_gamma(r, nc, x2c, kconstnew, kconstc, kconstnewc, theta, kconst);
    //Sample the means
    for (unsigned int k = 0; k < Kc; k++){
        for (unsigned int j = 0; j < K; j++){
            //I have to solve the problem of the sampling
            effkconst += nc[k][j]/(std::pow(kconstnewc[k],2)+std::pow(kconstnew[j],2));
            effmean += xc[k][j]/(std::pow(kconstnewc[k],2)+std::pow(kconstnew[j],2));
        }
        if (effkconst == 0){
            thetanewc[k] = thetac[k];
        }else{
            effmean = effmean/effkconst;
            effkconst = 1/effkconst;

            if(std::isnan(effkconst)==false){
                std::normal_distribution<double> gaussian(effmean, effkconst);
                thetanewc[k] = gaussian(r);
            }
        }
        //Clean variables
        effmean = 0;
        effkconst = 0;
    }    

    return;
}

void chain(int pos0, std::vector<std::vector<double>> & posterior, std::vector<double> & data, std::vector<double> & datac,                          
                                int ignored_iterations, int iterations, int nChains,
                                int K, int Kc, double alpha, double alphac, 
                                double priortheta_k, double priortheta_theta, double priork_k, double priork_theta, 
                                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                                double priorbias_sigma, double priorbias_min, 
                                bool initialised, bool showProgress, int seed){
    //Variables for the random generation
    std::mt19937 r;
    r.seed(seed);

    std::vector<double> pi(K), theta(K), kconst(K), pinew(K), thetanew(K), kconstnew(K);

    std::vector<double> pic(Kc), thetac(Kc), kconstc(Kc), pinewc(Kc), thetanewc(Kc), kconstnewc(Kc);

    double bias, biasnew;

    std::vector<std::vector<std::vector<double>>> id(Kc,std::vector<std::vector<double>>(K,std::vector<double>(datac.size())));

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
                         id);
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
                         priorbias_sigma, priorbias_min,
                         id);
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
                                initialised, showProgress, seedchain)); //Need explicit by reference std::refs
    }
    //Wait for rejoining
    for(int i = 0; i < nChains; i++){
        chains[i].join();
    }
    
    return posterior;
}