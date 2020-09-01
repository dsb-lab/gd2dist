#include <vector>
#include <math.h>
#include <random>
#include "probability_distributions.h"
#include <iostream>

#include "functions.h"

//Define the EM algorithm
void EM_simple(std::mt19937 &r, std::vector<double>& data, std::vector<double>& pi, std::vector<double>& mu,
                 std::vector<double>& sigma, const unsigned int & max_steps){
    //Auxiliar variables of the EM algorithm
    //Lenght of data
    unsigned int N = data.size();
    unsigned int K = mu.size();
    //Identity of each observation
    double z[K][N];
    double max;
    double munew;
    double sigmanew;
    bool modified;

    //Initialise the variables
    double x = 0, x2 = 0;
    int l = data.size();
    for (int i = 0; i < N; i++){
        x += data[i]/N;
        x2 += std::pow(data[i],2)/N;
    }
    x2 = std::sqrt(x2-std::pow(x,2));
    std::normal_distribution<double> gaussian(x,x2);
    for (int i = 0; i < K; i++){
        pi[i] = 1;
        mu[i] = gaussian(r);
        sigma[i] = x2;
    }
    //Declare auxiliar variables
    double probabilities[K]; //Auxiliar vector for the weights of z
    double total = 0;        //Sum of the partial probabilities
    double weights[K]; //Weights of the gaussians
    for (int i = 0; i < K; i++){
        weights[i] = 0;
    }

    for (unsigned int k = 0; k < max_steps; k++){

        //Expectation step
        for (unsigned int i = 0; i < N; i++){
            max = -INFINITY;
            //Compute the weights for each gaussian
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
                total += probabilities[j]; //Keep the total mass in a vector
            }
            //Normalize by the total mass
            for (unsigned int j = 0; j < K; j++){
                z[j][i] = probabilities[j]/total;
                weights[j] += z[j][i];
            }
            total = 0; //Clear the total
        }

        //Optimization step
        for (unsigned int i = 0; i < K; i++){
            //New weights
            pi[i] = weights[i]/ N;
            munew = 0;
            sigmanew = 0;
            
            if (weights[i] > 0.001/K){ //Handle overfloats and nans
                modified = true;
                for (unsigned int j = 0; j < N; j++){
                    //New means
                    munew += z[i][j]*data[j]/weights[i];
                }
                for (unsigned int j = 0; j < N; j++){
                    //New standard deviations
                    sigmanew += z[i][j]*std::pow(data[j]-mu[i],2)/weights[i];
                }
                sigmanew = std::sqrt(sigmanew);
                weights[i] = 0;
            }
            if (modified == true){ //Only modify them if some change has happened
                sigma[i] = sigmanew;
                mu[i] = munew;
                modified = false; //Clean the access
            }
        }
    }

    return;
}
//Define the Gibbs sampler step for the simple model
void Gibbs_simple_step(std::mt19937 & r, std::vector<double> & data,
                 std::vector<double> & pi, std::vector<double> & mu, std::vector<double> & sigma,
                 std::vector<double> & pinew, std::vector<double> & munew, std::vector<double> & sigmanew, 
                 const double & alpha){

    //Auxiliar variables of the EM algorithm
    //Lenght of data
    unsigned int N = data.size();
    unsigned int K = pi.size();
    std::vector<int> z(K,0);
    //Suficient statistics
    double n[K]; //Total of observations
    double mean[K];    //Mean
    double std[K];     //Sandard deviation
    for (int i = 0; i < K; i++){
        n[i] = 0;
        mean[i] = 0;
        std[i] = 0;
    }

    //Declare auxiliar variables
    std::vector<double> probabilities(K,0); //Auxiliar vector for the weights of z
    for (unsigned int i = 0; i < N; i++){
        //Compute the weights for each gaussian
        for (unsigned int j = 0; j < K; j++){
            probabilities[j] = pi[j]*std::exp(gaussian_pdf(data[i], mu[j],sigma[j]));
        }
        //Assign values
        multinomial_1(r, probabilities, z);
        for (unsigned int j = 0; j < K; j++){
            n[j] += z[j];
            mean[j] += z[j] * data[i];
            std[j] += z[j] * std::pow(data[i], 2);
        }
    }
    //Normalize
    for (unsigned int j = 0; j < K; j++){
        if (n[j] != 0){
            mean[j] /= n[j];
            std[j] = std::sqrt(std[j]/n[j]-std::pow(mean[j],2));
        }
        else{
            mean[j] = mean[j];
            std[j] = std[j];
        }
    }

    //Other terms sampling step
    //Sampling for mixture
    std::vector<double> alpha0(K,0);
    for (unsigned int i = 0; i < K; i++){
        alpha0[i] += alpha/K+n[i];
    }
    dirichlet(r, alpha0, pinew);
    //Sampling the standard deviation
    for (unsigned int i = 0; i < K; i++){
        if (n[i] > 1){
            std::gamma_distribution<double> gamma(n[i]/2,2/(std::pow(std[i],2)*n[i]));
            sigmanew[i] = std::sqrt(1/gamma(r)); 
            std::normal_distribution<double> gaussian(mean[i],sigmanew[i]/std::sqrt(n[i]));
            munew[i] = gaussian(r);
        }
        else{
            sigmanew[i] = sigma[i];
            munew[i] = mu[i];
        }
    }

    return;
}
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
double max_effective_gamma(std::vector<double> & n,
                        std::vector<double> & x2, std::vector<double> & sigma){

    //Mean sure stepper
    double pos=1, dpos=0.01;
    double aux, aux1, aux2;
    double df, ddf;
    double precission = 0.000001;
    double c = 0;
    double step = 0;
    do{
        aux = effective_gamma_not_normalized(pos, n, x2, sigma);
        aux1 = effective_gamma_not_normalized(pos+dpos, n, x2, sigma);
        aux2 = effective_gamma_not_normalized(pos+2*dpos, n, x2, sigma);
        df = (aux1-aux)/dpos;
        ddf = (aux2-2*aux1+aux)/std::pow(dpos,2);
        step = -df/ddf;
        pos += step;
        if (pos < 0){
            pos = 0.00001;
        }
        c++;
    }while(c < 10000 && abs(df) > precission); //Some maximum number of steps


    return pos;
}
//Define the convolved EM algorithm
void EM_convolved(std::mt19937 &r, std::vector<double>& data, std::vector<double>& datac, 
                    std::vector<double>& pi, std::vector<double>& mu, std::vector<double>& sigma,
                    std::vector<double>& pic, std::vector<double>& muc, std::vector<double>& sigmac, 
                    const unsigned int & max_steps){
    //Auxiliar variables of the EM algorithm
    //Lenght of data
    unsigned int N = data.size();
    unsigned int Nc = datac.size();
    unsigned int K = pi.size();
    unsigned int Kc = pic.size();

    //Initialise the variables
    double xx = 0, xx2 = 0;
    for (int i = 0; i < N; i++){
        xx += data[i]/N;
        xx2 += std::pow(data[i],2)/N;
    }
    double sd = std::sqrt(xx2-std::pow(xx,2)); 

    xx = 0;
    xx2 = 0;
    for (int i = 0; i < Nc; i++){
        xx += datac[i]/Nc;
        xx2 += std::pow(datac[i],2)/Nc;
    }
    double sdc = std::sqrt(xx2-std::pow(xx,2)); 
    
    std::uniform_int_distribution<int> dist(0,N);
    for (unsigned int i = 0; i < K; i++){
        mu[i] = data[dist(r)];
        sigma[i] = sd;
        pi[i] = 1;

    }
    std::uniform_int_distribution<int> dist2(0,Nc);
    for (unsigned int i = 0; i < Kc; i++){
        muc[i] = datac[dist2(r)];
        sigmac[i] = sdc;
        pic[i] = 1;
    }
    EM_simple(r, datac, pic, muc, sigmac, max_steps);

    //Maximization of the noise distribution
    EM_simple(r, data, pi, mu, sigma, max_steps);

    //Maximization of the convolvd distribution
    //Declare auxiliar variables
    double probabilitiesc[Kc][K];   //Auxiliar vector for the weights of z
    double totalc = 0;              //Sum of the partial probabilities
    std::vector<std::vector<double>> n(Kc,std::vector<double>(K,0));   //Weights of the different convolved gaussians
    std::vector<std::vector<double>> x(Kc,std::vector<double>(K,0));   //Weigpihts of the different convolved gaussians
    std::vector<std::vector<double>> x2(Kc,std::vector<double>(K,0));   //Weights of the different convolved gaussians
    double muj = 0, phij = 0;
    std::vector<double> nmin(Kc,0);
    //Identity of each observation
    //double zc[K][Kc][Nc];

    for (unsigned int l = 0; l < max_steps; l++){
        //Expectation step convolution
        for (unsigned int i = 0; i < Nc; i++){
            //Compute the weights for each gaussian
            for (unsigned int j = 0; j < K; j++){
                for (unsigned int k = 0; k < Kc; k++){
                    probabilitiesc[k][j] = pic[k]*pi[j]
                                        *std::exp(gaussian_pdf(datac[i],mu[j]+muc[k],std::sqrt(std::pow(sigmac[k],2)+std::pow(sigma[j],2))));
                    totalc += probabilitiesc[k][j]; //Keep the total mass in a vector
                }
            }
            //Normalize by the total mass
            for (unsigned int j = 0; j < K; j++){            //New means
                for (unsigned int k = 0; k < Kc; k++){
                    //zc[j][k][i] = probabilitiesc[k][j]/totalc;
                    n[k][j] += probabilitiesc[k][j]/totalc;
                    x[k][j] += probabilitiesc[k][j]/totalc*(datac[i]-mu[j]);
                    x2[k][j] += probabilitiesc[k][j]/totalc*std::pow(datac[i]-mu[j]-muc[k],2);
                    nmin[k] += probabilitiesc[k][j]/totalc;
                }
            }
            totalc = 0; //Clear the total
        }

        //Optimization step
        for (unsigned int i = 0; i < Kc; i++){
            pic[i] = 0;
            for (unsigned int j = 0; j < K; j++){
                //New weights
                pic[i] += n[i][j]/Nc;
            }
            //New standard deviations
            if (nmin[i] > 1){
                sigmac[i] = max_effective_gamma(n[i], x2[i], sigma);
                //New mean    
                for (unsigned int j = 0; j < K; j++){
                    phij += n[i][j]/(std::pow(sigmac[i],2)+std::pow(sigma[j],2));
                    muj += x[i][j]/(std::pow(sigmac[i],2)+std::pow(sigma[j],2));
                }
                muc[i] = muj/phij;
            }
            //Reset parameters
            phij = 0;
            muj = 0;
            nmin[i] = 0;
            for (unsigned int j = 0; j < K; j++){
                n[i][j] = 0;
                x[i][j] = 0;
                x2[i][j] = 0;
            }
        }
    }

    return;
}
void sample_effective_gamma(std::mt19937 &r, std::vector<std::vector<double>> &n,
                             std::vector<std::vector<double>> &x2, 
                             std::vector<double> &sigma, std::vector<double> &sigmaold, std::vector<double> &sigmanew,
                             double width){

        int N = sigmanew.size();
        std::normal_distribution<double> dist(0,width);
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

            if(acceptance > 0 and isnan(acceptance)==false){
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
                          double width){

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
    sample_effective_gamma(r, n, x2, sigmac, sigma, sigmanew, width);
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

            if(isnan(effsigma)==false){
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
    sample_effective_gamma(r, nc, x2c, sigmanew, sigmac, sigmanewc, width);
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

            if(isnan(effsigma)==false){
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