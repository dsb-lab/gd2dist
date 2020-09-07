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
#include "mcmcsampler.h"
#include "pybind11/pybind11.h"

double effective_gamma_not_normalized(double pos, std::vector<double> n, std::vector<double> x2, std::vector<double> sigma){

    double aux = 0;
    int l = sigma.size();
    int l2 = x2.size();

    for (int i = 0; i < l; i++){
        aux += -n[i]*std::log(std::pow(pos,2)+std::pow(sigma[i],2))/2
                -x2[i]/(2*(std::pow(pos,2)+std::pow(sigma[i],2))); 
    }
    if (l == l2-1){
        aux += -n[l]*std::log(std::pow(pos,2))/2
                -x2[l]/(2*(std::pow(pos,2)));          
    }

    return aux;
}

void sample_effective_gamma(std::mt19937 &r, std::vector<std::vector<double>> &n,
                             std::vector<std::vector<double>> &x2, 
                             std::vector<double> &sigma, std::vector<double> &sigmaold, std::vector<double> &sigmanew,
                             double sigmaWidth){

        int N = sigmanew.size();
        std::normal_distribution<double> dist(0,sigmaWidth);
        std::uniform_real_distribution<double> uniform(0,1);
        double loss_old;
        double loss_new;
        double newsigma;
        double acceptance;


        //Metropolis acceptance algorithm
        for (int i = 0; i < N; i++){

            do{
                newsigma = dist(r)+sigmaold[i];
            }while(newsigma <= 0);

            loss_old = effective_gamma_not_normalized(sigmaold[i], n[i], x2[i], sigma); 
            loss_new = effective_gamma_not_normalized(newsigma, n[i], x2[i], sigma);

            acceptance = std::exp(loss_new-loss_old)-uniform(r);

            if(acceptance > 0 and std::isnan(acceptance)==false){
                sigmanew[i] = newsigma; 
            }else{
                sigmanew[i] = sigmaold[i];
            }
        }

    return;
}

void Gibbs_convolved_step(std::mt19937 & r, std::vector<double> & data, std::vector<double>& datac,
                          std::vector<double> & pi, std::vector<double> & mu, std::vector<double> & sigma,
                          std::vector<double> & pinew, std::vector<double> & munew, std::vector<double> & sigmanew, 
                          double alpha,
                          std::vector<double> & pic, std::vector<double> & muc, std::vector<double> & sigmac,
                          std::vector<double> & pinewc, std::vector<double> & munewc, std::vector<double> & sigmanewc,
                          double alphac,
                          std::vector<std::vector<std::vector<double>>> id,
                          double sigmaWidth){

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
    double muj = 0, phij = 0;
    std::vector<double> nminalphac(Kc,0);
    double effmean;
    double effsigma;

    double max;

    //Evaluate the autofluorescence data
    for (unsigned int i = 0; i < data.size(); i++){
        //Compute the weights for each gaussian
        max = -INFINITY;
        for (unsigned int j = 0; j < K; j++){
            probabilities[j] = std::log(pi[j])
                                +gaussian_pdf(data[i],mu[j],sigma[j]);
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
            x2[j][Kc] += choice[j]*std::pow(data[i]-mu[j],2);
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
                                    +gaussian_pdf(datac[i],mu[j]+muc[k],std::sqrt(std::pow(sigmac[k],2)+std::pow(sigma[j],2)));
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
        multinomial_1(r, probabilitiesc, choicec);
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
                x[j][k] += id[k][j][i]*(datac[i]-muc[k]);
                x2[j][k] += id[k][j][i]*std::pow(datac[i]-mu[j]-muc[k],2);
            }
        }
    }
    //Sample the variances
    sample_effective_gamma(r, n, x2, sigmac, sigma, sigmanew, sigmaWidth);
    //Sample the means
    for (unsigned int j = 0; j < K; j++){
        //Convoluted terms
        for (unsigned int k = 0; k < Kc; k++){
            effsigma += n[j][k]/(std::pow(sigmac[k],2)+std::pow(sigmanew[j],2));
            effmean += x[j][k]/(std::pow(sigmac[k],2)+std::pow(sigmanew[j],2));
        }
        //Autofluorescence terms
        effsigma += n[j][Kc]/(std::pow(sigmanew[j],2));
        effmean += x[j][Kc]/(std::pow(sigmanew[j],2));
        
        if (effsigma == 0){
            munew[j] = mu[j];
        }else{
            effmean = effmean/effsigma;
            effsigma = 1/effsigma;

            if(std::isnan(effsigma)==false){
                std::normal_distribution<double> gaussian(effmean, effsigma);
                munew[j] = gaussian(r);
            }
        }

        //Clean variables
        effmean = 0;
        effsigma = 0;
    }    

    //Sample the convoluted variance and mean
    //Compute the statistics
    for (unsigned int i = 0; i < datac.size(); i++){
        for (unsigned int j = 0; j < K; j++){
            for (unsigned int k = 0; k < Kc; k++){
                nc[k][j] += id[k][j][i];
                xc[k][j] += id[k][j][i]*(datac[i]-munew[j]);
                x2c[k][j] += id[k][j][i]*std::pow(datac[i]-munew[j]-muc[k],2);
            }
        }
    }
    //Sample the variances
    sample_effective_gamma(r, nc, x2c, sigmanew, sigmac, sigmanewc, sigmaWidth);
    //Sample the means
    for (unsigned int k = 0; k < Kc; k++){
        for (unsigned int j = 0; j < K; j++){
            //I have to solve the problem of the sampling
            effsigma += nc[k][j]/(std::pow(sigmanewc[k],2)+std::pow(sigmanew[j],2));
            effmean += xc[k][j]/(std::pow(sigmanewc[k],2)+std::pow(sigmanew[j],2));
        }
        if (effsigma == 0){
            munewc[k] = muc[k];
        }else{
            effmean = effmean/effsigma;
            effsigma = 1/effsigma;

            if(std::isnan(effsigma)==false){
                std::normal_distribution<double> gaussian(effmean, effsigma);
                munewc[k] = gaussian(r);
            }
        }
        //Clean variables
        effmean = 0;
        effsigma = 0;
    }    

    return;
}

void chain(int pos0, std::vector<std::vector<double>> & posterior, std::vector<double> & data, std::vector<double> & datac,                          
                                int ignored_iterations, int iterations, int nChains,
                                int K, int Kc, double alpha, double alphac, double sigmaWidth, bool initialised, bool showProgress){
    //Variables for the random generation
    std::mt19937 r;

    std::vector<double> pi(K), mu(K), sigma(K), pinew(K), munew(K), sigmanew(K);

    std::vector<double> pic(Kc), muc(Kc), sigmac(Kc), pinewc(Kc), munewc(Kc), sigmanewc(Kc);

    std::vector<std::vector<std::vector<double>>> id(Kc,std::vector<std::vector<double>>(K,std::vector<double>(datac.size())));

    double var = 0, varc = 0, mean = 0, meanc = 0;
    //Compute statistics
    for (int i = 0; i < data.size(); i++){
        mean += data[i]/data.size();
    }
    for (int i = 0; i < data.size(); i++){
        var += std::pow(data[i]-mean,2)/data.size();
    }
    for (int i = 0; i < datac.size(); i++){
        meanc += datac[i]/datac.size();
    }
    for (int i = 0; i < datac.size(); i++){
        varc += std::pow(datac[i]-meanc,2)/datac.size();
    }
    //Initialise
    if (!initialised){
        std::normal_distribution<double> gaussian(mean,std::sqrt(varc));
        for (int i = 0; i < K; i++){
            pi[i] = 1;
            mu[i] = gaussian(r);
            sigma[i] = std::sqrt(var);
        }

        std::normal_distribution<double> gaussianc(meanc-mean,std::sqrt(varc));
        for (int i = 0; i < Kc; i++){
            pic[i] = 1;
            muc[i] = gaussianc(r);
            sigmac[i] = std::sqrt(varc);
        }
    }else{
        for (int i = 0; i < K; i++){
            pi[i] = posterior[pos0][i];
            mu[i] = posterior[pos0][K+i];
            sigma[i] = posterior[pos0][2*K+i];
        }
        for (int i = 0; i < Kc; i++){
            pic[i] = posterior[pos0][3*K+i];
            muc[i] = posterior[pos0][3*K+Kc+i];
            sigmac[i] = posterior[pos0][3*K+2*Kc+i];
        }
    }
    
    int progressStep = floor(ignored_iterations/10);
    int progressCounter = 0;
    int chainId = int(pos0/iterations);
    //Ignorable, steps
    for (int i = 0; i < ignored_iterations; i++){
        Gibbs_convolved_step(r, data, datac,
                         pi, mu, sigma, pinew, munew, sigmanew, alpha,
                         pic, muc, sigmac, pinewc, munewc, sigmanewc, alphac,
                         id, sigmaWidth);
        pi = pinew;
        mu = munew;
        sigma = sigmanew;
        pic = pinewc;
        muc = munewc;
        sigmac = sigmanewc;

        if(showProgress){
            if(i % progressStep == 0){
                pybind11::print("Chain", chainId, " Ignorable iterations: ", progressCounter * 10, "%");
                progressCounter++;
            }
        }
    }
    if(showProgress){
        pybind11::print("Chain", chainId, " Ignorable iterations: 100%");
    }

    progressStep = floor(iterations/10);
    progressCounter = 0;
    //Recorded steps
    for (unsigned int i = 0; i < iterations; i++){
        Gibbs_convolved_step(r, data, datac,
                         pi, mu, sigma, pinew, munew, sigmanew, alpha,
                         pic, muc, sigmac, pinewc, munewc, sigmanewc, alphac,
                         id, sigmaWidth);
        pi = pinew;
        mu = munew;
        sigma = sigmanew;
        for (unsigned int j = 0; j < K; j++){
            posterior[pos0+i][j] = pinew[j];
            posterior[pos0+i][K+j] = munew[j];
            posterior[pos0+i][2*K+j] = sigmanew[j];
        }
        pic = pinewc;
        muc = munewc;
        sigmac = sigmanewc;
        for (unsigned int j = 0; j < Kc; j++){
            posterior[pos0+i][3*K+j] = pinewc[j];
            posterior[pos0+i][3*K+Kc+j] = munewc[j];
            posterior[pos0+i][3*K+2*Kc+j] = sigmanewc[j];
        }

        if(showProgress){
            if(i % progressStep == 0){
                pybind11::print("Chain",chainId," Recorded iterations: ",progressCounter * 10,"%");
                progressCounter++;
            }
        }
    }
    if(showProgress){
        pybind11::print("Chain",chainId," Recorded iterations: 100%");
    }

    return;
}
std::vector<std::vector<double>> fit(std::vector<double> & data, std::vector<double>& datac,
                          int ignored_iterations, int iterations, int nChains,
                          int K, int Kc, double alpha, double alphac, double sigmaWidth, std::vector<std::vector<double>> initial_conditions, bool showProgress){

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
    std::thread chains[nChains];
    for(unsigned int i = 0; i < nChains; i++){
        int a = i*iterations;
        chains[i] = std::thread(chain, a, std::ref(posterior), std::ref(data), std::ref(datac),                          
                                ignored_iterations, iterations, nChains,
                                K, Kc, alpha, alphac, sigmaWidth,
                                initialised, showProgress); //Need explicit by reference std::refs
    }
    //Wait for rejoining
    for(unsigned int i = 0; i < nChains; i++){
        chains[i].join();
    }
    
    return posterior;
}