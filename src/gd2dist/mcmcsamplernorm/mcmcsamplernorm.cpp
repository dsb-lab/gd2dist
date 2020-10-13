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
#include "mcmcsamplernorm.h"
#include "pybind11/pybind11.h"

/*double logLikelihood(std::vector<double> & pi, std::vector<double> & mu, std::vector<double> & sigma,
                    std::vector<double> & pic, std::vector<double> & muc, std::vector<double> & sigmac,
                    std::vector<double>  & data, std::vector<double>  & datac){

    double log = 0;
    double max;

    max = -INFINITY;
    for (int i = 0; i < data.size(); i++){
        for (int j = 0; j < pi.size(); j++){
            log += std::log(pi[j])
                +gaussian_pdf(data[i],mu[j],sigma[j]);
        }
    }

    for (int i = 0; i < datac.size(); i++){
        for (int j = 0; j < pi.size(); j++){
            for (int k = 0; k < pic.size(); k++){
                log += std::log(pic[k])+std::log(pi[j])
                                    +gaussian_pdf(datac[i],mu[j]+muc[k],std::sqrt(std::pow(sigmac[k],2)+std::pow(sigma[j],2)));
            }
        }
    }

    return log;
}*/

double effective_gamma_not_normalized(double pos, std::vector<double> n, std::vector<double> x2, std::vector<double> sigma, double theta, double kconst){

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
    //Add prior
    aux += -pos/theta+(kconst-1)*std::log(pos);

    return aux;
}

void slice_effective_gamma(std::mt19937 &r, std::vector<std::vector<double>> &n,
                             std::vector<std::vector<double>> &x2, 
                             std::vector<double> &sigma, std::vector<double> &sigmaold, std::vector<double> &sigmanew,
                             double theta, double kconst, std::vector<double> &slice_step){

        int N = sigmanew.size();
        std::uniform_real_distribution<double> uniform(0,1);
        double loss_old;
        double loss_new;
        double newsigma;
        double acceptance;
        double min;
        double max;
        double expansion = 0.5;
        int counter;
        double old;
        int contribution;

        //Slice sampling
        for (int i = 0; i < N; i++){

            contribution = 0;
            for(int j = 0; j < n[i].size(); j++){
                contribution += n[i][j];
            }
            if( contribution > 0){
                old = sigmaold[i];
                expansion = slice_step[i];

                for (int j = 0; j < 10; j++){
                    loss_old = effective_gamma_not_normalized(old, n[i], x2[i], sigma, theta, kconst);
                    //Chose new height
                    loss_old += std::log(uniform(r));
                    //Expand
                    min = old-expansion;
                    loss_new = effective_gamma_not_normalized(min, n[i], x2[i], sigma, theta, kconst);
                    counter = 0;
                    if(min <= 0){
                        min = 0;
                        break;
                    }
                    while(loss_new > loss_old && counter < 200000){
                        min -= expansion;
                        if(min <= 0){
                            min = 0;
                            break;
                        }
                        loss_new = effective_gamma_not_normalized(min, n[i], x2[i], sigma, theta, kconst);
                        counter++;
                    }
                    max = old+expansion;
                    loss_new = effective_gamma_not_normalized(max, n[i], x2[i], sigma, theta, kconst);
                    counter = 0;
                    while(loss_new > loss_old && counter < 200000){
                        max += expansion;
                        loss_new = effective_gamma_not_normalized(max, n[i], x2[i], sigma, theta, kconst);
                        counter++;
                    }
                    slice_step[i] = (max-min)/10;

                    //Sample
                    counter = 0;
                    do{
                        newsigma = (max-min)*uniform(r)+min;
                        loss_new = effective_gamma_not_normalized(newsigma, n[i], x2[i], sigma, theta, kconst);
                        //Adapt boundaries
                        if(loss_new < loss_old){
                            if(newsigma < old){
                                min = newsigma;
                            }
                            else if(newsigma > old){
                                max = newsigma;
                            }
                        }
                        counter++;
                    }while(loss_new < loss_old && counter < 200000);

                    old = newsigma;
                }

                sigmanew[i] = newsigma;
            }else{
                sigmanew[i] = sigma[i];
            }
        }

    return;
}

void Gibbs_convolved_step(std::mt19937 & r, std::vector<double> & data, std::vector<double>& datac,
                          std::vector<double> & pi, std::vector<double> & mu, std::vector<double> & sigma,
                          std::vector<double> & pinew, std::vector<double> & munew, std::vector<double> & sigmanew,
                          std::vector<double> & pic, std::vector<double> & muc, std::vector<double> & sigmac,
                          std::vector<double> & pinewc, std::vector<double> & munewc, std::vector<double> & sigmanewc,
                          std::vector<std::vector<std::vector<double>>>& id,
                          std::vector<double>& priors, std::vector<double>& slice_step_sigma, std::vector<double>& slice_step_sigmac){

    //Step of the convolution
    int K = pi.size();
    int Kc = pic.size();
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
    for (int i = 0; i < data.size(); i++){
        //Compute the weights for each gaussian
        max = -INFINITY;
        for (int j = 0; j < K; j++){
            probabilities[j] = std::log(pi[j])
                                +gaussian_pdf(data[i],mu[j],sigma[j]);
            if (probabilities[j] > max){
                max = probabilities[j];
            }
        }
        //Normalize
        for (int j = 0; j < K; j++){
            probabilities[j] -= max;
            probabilities[j] = std::exp(probabilities[j]);
        }
        //Assign a gaussian
        multinomial_1(r, probabilities, choice);
        //Compute the basic statistics
        //We compute all the statistics already since we are going to use them only for the autofluorescence sampling
        for (int j = 0; j < K; j++){
            n[j][Kc] += choice[j];
            x[j][Kc] += choice[j]*data[i];
            x2[j][Kc] += choice[j]*std::pow(data[i]-mu[j],2);
            nminalpha[j] += choice[j];
        }
    }
    
    //Evaluate the Colvolved data
    for (int i = 0; i < datac.size(); i++){
        //Compute the weights for each gaussian
        max = -INFINITY;
        for (int j = 0; j < K; j++){
            for (int k = 0; k < Kc; k++){
                probabilitiesc[K*k+j] = std::log(pic[k])+std::log(pi[j])
                                    +gaussian_pdf(datac[i],mu[j]+muc[k],std::sqrt(std::pow(sigmac[k],2)+std::pow(sigma[j],2)));
                if (probabilitiesc[K*k+j]>max){
                    max = probabilitiesc[K*k+j];
                }
            }
        }
        //Normalize
        for (int j = 0; j < K; j++){
            for (int k = 0; k < Kc; k++){
                probabilitiesc[K*k+j] -= max;
                probabilitiesc[K*k+j] = std::exp(probabilitiesc[K*k+j]);
            }
        }
        //Assign a Colvolved gaussian
        multinomial_1(r, probabilitiesc, choicec);
        //Save the identity
        //We do not compute the statistics because they will have to be updated since this dataset is used for sampling twice
        for (int j = 0; j < K; j++){
            for (int k = 0; k < Kc; k++){
                id[k][j][i] = choicec[K*k+j];
                nminalpha[j] += choicec[K*k+j];
                nminalphac[k] += choicec[K*k+j];
            }
        }
    }
    //Add the priors
    /*for (int k = 0; k < K; k++){
        nminalpha[k] += priors[0];
    }    
    for (int k = 0; k < Kc; k++){
        nminalphac[k] += priors[5];
    }    */

    //Sample the new mixtures
    dirichlet(r, nminalpha, pinew);
    dirichlet(r, nminalphac, pinewc);

    //Sample the autofluorescence variance and mean
    //Compute the statistics
    for (int i = 0; i < datac.size(); i++){
        for (int j = 0; j < K; j++){
            for (int k = 0; k < Kc; k++){
                n[j][k] += id[k][j][i];
                x[j][k] += id[k][j][i]*(datac[i]-muc[k]);
                x2[j][k] += id[k][j][i]*std::pow(datac[i]-mu[j]-muc[k],2);
            }
        }
    }
    //Sample the variances
    //sample_effective_gamma(r, n, x2, sigmac, sigma, sigmanew, sigmaWidth);
    slice_effective_gamma(r, n, x2, sigmac, sigma, sigmanew, priors[3], priors[4], slice_step_sigma);
    //Sample the means
    for (int j = 0; j < K; j++){
        //Clean variables
        effmean = 0;
        effsigma = 0;
        //Colvolved terms
        for (int k = 0; k < Kc; k++){
            effsigma += n[j][k]/(std::pow(sigmac[k],2)+std::pow(sigmanew[j],2));
            effmean += x[j][k]/(std::pow(sigmac[k],2)+std::pow(sigmanew[j],2));
        }
        //Autofluorescence terms
        effsigma += n[j][Kc]/(std::pow(sigmanew[j],2));
        effmean += x[j][Kc]/(std::pow(sigmanew[j],2));
        //Add priors
        effsigma += 1/std::pow(priors[2],2);
        effmean += priors[1]/std::pow(priors[2],2);
        
        if (effsigma == 0){
            munew[j] = mu[j];
        }else{
            effmean = effmean/effsigma;
            effsigma = 1/effsigma;

            if(std::isnan(effsigma)==false){
                std::normal_distribution<double> gaussian(effmean, std::pow(effsigma,0.5));
                munew[j] = gaussian(r);
            }
        }
    }    

    //Sample the Colvolved variance and mean
    //Compute the statistics
    for (int i = 0; i < datac.size(); i++){
        for (int j = 0; j < K; j++){
            for (int k = 0; k < Kc; k++){
                nc[k][j] += id[k][j][i];
                xc[k][j] += id[k][j][i]*(datac[i]-munew[j]);
                x2c[k][j] += id[k][j][i]*std::pow(datac[i]-munew[j]-muc[k],2);
            }
        }
    }
    //Sample the variances
    //sample_effective_gamma(r, nc, x2c, sigmanew, sigmac, sigmanewc, sigmaWidth);
    slice_effective_gamma(r, nc, x2c, sigmanew, sigmac, sigmanewc, priors[8], priors[9], slice_step_sigmac);
    //Sample the means
    for (int k = 0; k < Kc; k++){
        //Clean variables
        effmean = 0;
        effsigma = 0;
        for (int j = 0; j < K; j++){
            //I have to solve the problem of the sampling
            effsigma += nc[k][j]/(std::pow(sigmanewc[k],2)+std::pow(sigmanew[j],2));
            effmean += xc[k][j]/(std::pow(sigmanewc[k],2)+std::pow(sigmanew[j],2));
        }
        //Add priors
        effsigma += 1/std::pow(priors[7],2);
        effmean += priors[6]/std::pow(priors[7],2);

        if (effsigma == 0){
            munewc[k] = muc[k];
        }else{
            effmean = effmean/effsigma;
            effsigma = 1/effsigma;

            if(std::isnan(effsigma)==false){
                std::normal_distribution<double> gaussian(effmean, std::pow(effsigma,0.5));
                munewc[k] = gaussian(r);
            }
        }
    }    

    return;
}

void chain(int pos0, std::vector<std::vector<double>> & posterior, std::vector<double> & data, std::vector<double> & datac,                          
                                int ignored_iterations, int iterations, int nChains,
                                int K, int Kc, std::vector<double>& priors, bool initialised, bool showProgress, int seed){
    //Variables for the random generation
    std::mt19937 r;
    r.seed(seed);

    std::vector<double> pi(K), mu(K), sigma(K), pinew(K), munew(K), sigmanew(K);

    std::vector<double> pic(Kc), muc(Kc), sigmac(Kc), pinewc(Kc), munewc(Kc), sigmanewc(Kc);

    std::vector<std::vector<std::vector<double>>> id(Kc,std::vector<std::vector<double>>(K,std::vector<double>(datac.size())));

    std::vector<double> slice_step_sigma(K,0.5);
    std::vector<double> slice_step_sigmac(Kc,0.5);

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
        std::normal_distribution<double> gaussian(mean,0.1*std::sqrt(var));
        for (int i = 0; i < K; i++){
            pi[i] = 1;
            mu[i] = gaussian(r);
            sigma[i] = std::sqrt(var)/10;
        }

        std::normal_distribution<double> gaussianc(meanc-mean,0.1*std::sqrt(varc));
        for (int i = 0; i < Kc; i++){
            pic[i] = 1;
            muc[i] = gaussianc(r);
            sigmac[i] = std::sqrt(varc)/10;
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
                         pi, mu, sigma, pinew, munew, sigmanew,
                         pic, muc, sigmac, pinewc, munewc, sigmanewc,
                         id, priors, slice_step_sigma, slice_step_sigmac);
        pi = pinew;
        mu = munew;
        sigma = sigmanew;
        pic = pinewc;
        muc = munewc;
        sigmac = sigmanewc;

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
    int add = 0;
    if(ignored_iterations == 0){
        for (int j = 0; j < K; j++){
            pinew[j]=posterior[pos0][j];
            munew[j]=posterior[pos0][K+j];
            sigmanew[j]=posterior[pos0][2*K+j];
        }
        pi = pinew;
        mu = munew;
        sigma = sigmanew;
        for (int j = 0; j < Kc; j++){
            pinewc[j]=posterior[pos0][3*K+j];
            munewc[j]=posterior[pos0][3*K+Kc+j];
            sigmanewc[j]=posterior[pos0][3*K+2*Kc+j];
        }
        pic = pinewc;
        muc = munewc;
        sigmac = sigmanewc;

        add = 1;
    }
    //Recorded steps
    for (int i = add; i < iterations; i++){
        Gibbs_convolved_step(r, data, datac,
                         pi, mu, sigma, pinew, munew, sigmanew,
                         pic, muc, sigmac, pinewc, munewc, sigmanewc,
                         id, priors, slice_step_sigma, slice_step_sigmac);
        pi = pinew;
        mu = munew;
        sigma = sigmanew;
        for (int j = 0; j < K; j++){
            posterior[pos0+i][j] = pinew[j];
            posterior[pos0+i][K+j] = munew[j];
            posterior[pos0+i][2*K+j] = sigmanew[j];
        }
        pic = pinewc;
        muc = munewc;
        sigmac = sigmanewc;
        for (int j = 0; j < Kc; j++){
            posterior[pos0+i][3*K+j] = pinewc[j];
            posterior[pos0+i][3*K+Kc+j] = munewc[j];
            posterior[pos0+i][3*K+2*Kc+j] = sigmanewc[j];
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

    return;
}

std::vector<std::vector<double>> fit(std::vector<double> & data, std::vector<double>& datac,
                          int ignored_iterations, int iterations, int nChains,
                          int K, int Kc, std::vector<double>& priors, std::vector<std::vector<double>>& initial_conditions, bool showProgress, int seed){

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
                                K, Kc, std::ref(priors),
                                initialised, showProgress, seedchain)); //Need explicit by reference std::refs
    }
    //Wait for rejoining
    for(int i = 0; i < nChains; i++){
        chains[i].join();
    }
    
    return posterior;
}