#include <vector>
#include <map>
#include <string>
#include <thread>
#include <algorithm>
#include <iostream>
#include <exception>

#include "probability_distributions.h"
#include "functions.h"
#include "mcmcsampler.h"

mcmcsampler::mcmcsampler(unsigned int K, unsigned int Kc, double alpha, double alphac,
                            unsigned int iterations, unsigned int ignored_iterations, unsigned int chains, double sigmaWidth){
    K_ = K;
    Kc_ = Kc;
    alpha_ = alpha;
    alphac_ = alphac;
    iterations_ = iterations;//Define the auxiliar functions
    ignored_iterations_ = ignored_iterations;
    chains_ = chains;
    sigmaWidth_ = sigmaWidth;
    pi_.resize(K , std::vector<double> (iterations*chains));
    mu_.resize(K , std::vector<double> (iterations*chains));
    sigma_.resize(K , std::vector<double> (iterations*chains));
    pic_.resize(Kc , std::vector<double> (iterations*chains));
    muc_.resize(Kc , std::vector<double> (iterations*chains));
    sigmac_.resize(Kc , std::vector<double> (iterations*chains));
}
void mcmcsampler::set_parameters(unsigned int K = 429496729, unsigned int Kc = 429496729, double alpha = -1, double alphac = -1,
                            unsigned int iterations = 429496729, unsigned int ignored_iterations = 429496729, unsigned int chains = 429496729, double sigmaWidth = -1){
    // Update the parameters
    if (alpha != -1){
        alpha_ = alpha;
    }
    if (alphac != -1){
        alphac_ = alphac;
    }
    if (iterations != 429496729){
        iterations_ = iterations;
    }
    if (ignored_iterations != 429496729){
        ignored_iterations_ = ignored_iterations;
    }
    if (chains != 429496729){
        chains_ = chains;
        is_initial_condition_ = false;
    }
    if (K != 429496729){
        K_ = K;
        is_initial_condition_ = false;
    }
    if (Kc != 429496729){
        Kc_ = Kc;
        is_initial_condition_ = false;
    }
    if (K != 429496729 || iterations != 429496729 || chains != 429496729){    // Resize the vectors
        pi_ = std::vector<std::vector<double>> (K , std::vector<double> (iterations_*chains_,0));
        mu_ = std::vector<std::vector<double>> (K , std::vector<double> (iterations_*chains_,0));
        sigma_ = std::vector<std::vector<double>> (K , std::vector<double> (iterations_*chains_,0));
    }
    if (Kc != 429496729 || iterations != 429496729 || chains != 429496729){    // Resize the vectors
        pic_ = std::vector<std::vector<double>> (Kc , std::vector<double> (iterations_*chains_,0));
        muc_ = std::vector<std::vector<double>> (Kc , std::vector<double> (iterations_*chains_,0));
        sigmac_ = std::vector<std::vector<double>> (Kc , std::vector<double> (iterations_*chains_,0));
    }
    if (sigmaWidth != -1){
        sigmaWidth_ = sigmaWidth;
    }

    return;
}
void mcmcsampler::set_initial_condition(std::map<std::string, std::vector<std::vector<double>>> check){

    std::vector<std::string> names = {"pi","mu","sigma"};
    std::vector<std::string> namesc = {"pic","muc","sigmac"};
    std::vector<std::vector<double>> v;
    int c = 0;

    //Check autofluorescence
    for (int i = 0; i < names.size(); i++){
        try{
            v = check[names[i]];
            c += 1;
            if(v.size() != chains_){
                throw std::invalid_argument("ERROR: The function expects a dictionary with the keys {pi,mu,sigma,pic,muc,sigmac} with a matrix containing " 
                                            + std::to_string(chains_) + " chains and " + std::to_string(K_) + "/" + std::to_string(Kc_) + " conditions, depending if it is the mixture of autofluorescence or convolution.");                
            }
            if(v[0].size() != K_){
                throw std::invalid_argument("ERROR: The function expects a dictionary with the keys {pi,mu,sigma,pic,muc,sigmac} with a matrix containing " 
                                            + std::to_string(chains_) + " chains and " + std::to_string(K_) + "/" + std::to_string(Kc_) + " conditions, depending if it is the mixture of autofluorescence or convolution.");                
            }
        }
        catch(...){
                throw std::invalid_argument("ERROR: The function expects a dictionary with the keys {pi,mu,sigma,pic,muc,sigmac} with a matrix containing " 
                                            + std::to_string(chains_) + " chains and " + std::to_string(K_) + "/" + std::to_string(Kc_) + " conditions, depending if it is the mixture of autofluorescence or convolution.");                
        }
    }
    //Check autofluorescence
    for (int i = 0; i < namesc.size(); i++){
        try{
            v = check[namesc[i]];
            c += 1;
            if(v.size() != chains_){
                throw std::invalid_argument("ERROR: The function expects a dictionary with the keys {pi,mu,sigma,pic,muc,sigmac} with a matrix containing " 
                                            + std::to_string(chains_) + " chains and " + std::to_string(K_) + "/" + std::to_string(Kc_) + " conditions, depending if it is the mixture of autofluorescence or convolution.");                
            }
            if(v[0].size() != Kc_){
                throw std::invalid_argument("ERROR: The function expects a dictionary with the keys {pi,mu,sigma,pic,muc,sigmac} with a matrix containing " 
                                            + std::to_string(chains_) + " chains and " + std::to_string(K_) + "/" + std::to_string(Kc_) + " conditions, depending if it is the mixture of autofluorescence or convolution.");                
            }
        }
        catch(...){
                throw std::invalid_argument("ERROR: The function expects a dictionary with the keys {pi,mu,sigma,pic,muc,sigmac} with a matrix containing " 
                                            + std::to_string(chains_) + " chains and " + std::to_string(K_) + "/" + std::to_string(Kc_) + " conditions, depending if it is the mixture of autofluorescence or convolution.");                
        }
    }
    if (c < 6){
        throw std::invalid_argument("ERROR: The function expects a dictionary with the keys {pi,mu,sigma,pic,muc,sigmac} with a matrix containing " 
                                    + std::to_string(chains_) + " chains and " + std::to_string(K_) + "/" + std::to_string(Kc_) + " conditions, depending if it is the mixture of autofluorescence or convolution.");                
    }

    is_initial_condition_ = true;

    for (int i = 0; i < names.size(); i++){
        initial_condition_[names[i]] = check[names[i]];
    }
    for (int i = 0; i < namesc.size(); i++){
        initial_condition_[namesc[i]] = check[namesc[i]];
    }

    return;
}
std::map<std::string, double> mcmcsampler::get_parameters(){
    std::map<std::string, double> v;

    v["K"] = double(K_);
    v["Kc"] = double(Kc_);
    v["alpha"] = double(alpha_);
    v["alphac"] = double(alphac_);
    v["iterations"] = double(iterations_);
    v["ignored_iterations"] = double(ignored_iterations_);
    v["chains"] = double(chains_);
    v["sigmaWidth"] = double(sigmaWidth_);

    return v;
}
double mcmcsampler::get_parameter(std::string parameter){
    if (parameter == "K"){
        return double(K_);
    }else if (parameter == "Kc"){
        return double(Kc_);
    }else if (parameter == "alpha"){
        return double(alpha_);
    }else if (parameter == "alphac"){
        return double(alphac_);
    }else if (parameter == "iterations"){
        return double(iterations_);
    }else if (parameter == "ignored_iterations"){
        return double(ignored_iterations_);
    }else if (parameter == "chains"){
        return double(chains_);
    }else if (parameter == "sigmaWidth"){
        return double(sigmaWidth_);
    }
    
    return -1;
}
void mcmcsampler::chain(int pos0, std::vector<double> & data, std::vector<double> & datac){
    //Variables for the random generation
    std::mt19937 r{(long unsigned int)(time(0)+pos0)};

    std::vector<double> pi(K_), mu(K_), sigma(K_), pinew(K_), munew(K_), sigmanew(K_);

    std::vector<double> pic(Kc_), muc(Kc_), sigmac(Kc_), pinewc(Kc_), munewc(Kc_), sigmanewc(Kc_);

    std::vector<std::vector<std::vector<double>>> id(Kc_,std::vector<std::vector<double>>(K_,std::vector<double>(datac.size())));

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
    if (is_initial_condition_ == false){
        std::normal_distribution<double> gaussian(mean,std::sqrt(varc));
        for (int i = 0; i < K_; i++){
            pi[i] = 1;
            mu[i] = gaussian(r);
            sigma[i] = std::sqrt(var);
        }

        std::normal_distribution<double> gaussianc(meanc-mean,std::sqrt(varc));
        for (int i = 0; i < Kc_; i++){
            pic[i] = 1;
            muc[i] = gaussianc(r);
            sigmac[i] = std::sqrt(varc);
        }
    }else{
        for (int i = 0; i < K_; i++){
            pi[i] = pi_[i][pos0];
            mu[i] = mu_[i][pos0];
            sigma[i] = sigma_[i][pos0];
        }
        for (int i = 0; i < Kc_; i++){
            pic[i] = pic_[i][pos0];
            muc[i] = muc_[i][pos0];
            sigmac[i] = sigmac_[i][pos0];
        }
    }

    //Ignorable, steps
    for (unsigned int i = 0; i < ignored_iterations_; i++){
        Gibbs_convolved_step(r, data, datac,
                         pi, mu, sigma, pinew, munew, sigmanew, alpha_,
                         pic, muc, sigmac, pinewc, munewc, sigmanewc, alphac_,
                         id, sigmaWidth_);
        pi = pinew;
        mu = munew;
        sigma = sigmanew;
        pic = pinewc;
        muc = munewc;
        sigmac = sigmanewc;
    }
    //Recorded steps
    for (unsigned int i = 0; i < iterations_; i++){
        Gibbs_convolved_step(r, data, datac,
                         pi, mu, sigma, pinew, munew, sigmanew, alpha_,
                         pic, muc, sigmac, pinewc, munewc, sigmanewc, alphac_,
                         id, sigmaWidth_);
        pi = pinew;
        mu = munew;
        sigma = sigmanew;
        for (unsigned int j = 0; j < K_; j++){
            pi_[j][pos0+i] = pinew[j];
            mu_[j][pos0+i] = munew[j];
            sigma_[j][pos0+i] = sigmanew[j];
        }
        pic = pinewc;
        muc = munewc;
        sigmac = sigmanewc;
        for (unsigned int j = 0; j < Kc_; j++){
            pic_[j][pos0+i] = pinewc[j];
            muc_[j][pos0+i] = munewc[j];
            sigmac_[j][pos0+i] = sigmanewc[j];
        }
    }

    return;
}
void mcmcsampler::sort_chains(std::string flavour = "weights"){

    double aux;
    bool change;

    if (flavour == "means"){
        for( int i = 0; i < chains_*iterations_; i++){
            //Sort autofluorescence
            for( int j = 0; j < K_-1; j++){
                change = false;
                for( int k = 0; k < K_-1; k++){
                    if (mu_[k][i]>mu_[k+1][i]){
                        change = true;
                        //Interchange them
                        aux = mu_[k+1][i];
                        mu_[k+1][i] = mu_[k][i];
                        mu_[k][i] = aux;

                        aux = sigma_[k+1][i];
                        sigma_[k+1][i] = sigma_[k][i];
                        sigma_[k][i] = aux;

                        aux = pi_[k+1][i];
                        pi_[k+1][i] = pi_[k][i];
                        pi_[k][i] = aux;

                        change = true;
                    }
                }
                if (change = false){
                    break;
                }
            }
            //Sort autofluorescence
            for( int j = 0; j < Kc_-1; j++){
                change = false;
                for( int k = 0; k < Kc_-1; k++){
                    if (muc_[k][i]>muc_[k+1][i]){
                        change = true;
                        //Interchange them
                        aux = muc_[k+1][i];
                        muc_[k+1][i] = muc_[k][i];
                        muc_[k][i] = aux;

                        aux = sigmac_[k+1][i];
                        sigmac_[k+1][i] = sigmac_[k][i];
                        sigmac_[k][i] = aux;

                        aux = pic_[k+1][i];
                        pic_[k+1][i] = pic_[k][i];
                        pic_[k][i] = aux;

                        change = true;
                    }
                }
                if (change = false){
                    break;
                }
            }
        }
    }
    else if (flavour == "weights"){
        for( int i = 0; i < chains_*iterations_; i++){
            //Sort autofluorescence
            for( int j = 0; j < K_-1; j++){
                change = false;
                for( int k = 0; k < K_-1; k++){
                    if (pi_[k][i]>pi_[k+1][i]){
                        change = true;
                        //Interchange them
                        aux = mu_[k+1][i];
                        mu_[k+1][i] = mu_[k][i];
                        mu_[k][i] = aux;

                        aux = sigma_[k+1][i];
                        sigma_[k+1][i] = sigma_[k][i];
                        sigma_[k][i] = aux;

                        aux = pi_[k+1][i];
                        pi_[k+1][i] = pi_[k][i];
                        pi_[k][i] = aux;

                        change = true;
                    }
                }
                if (change = false){
                    break;
                }
            }
            //Sort autofluorescence
            for( int j = 0; j < Kc_-1; j++){
                change = false;
                for( int k = 0; k < Kc_-1; k++){
                    if (pic_[k][i]>pic_[k+1][i]){
                        change = true;
                        //Interchange them
                        aux = muc_[k+1][i];
                        muc_[k+1][i] = muc_[k][i];
                        muc_[k][i] = aux;

                        aux = sigmac_[k+1][i];
                        sigmac_[k+1][i] = sigmac_[k][i];
                        sigmac_[k][i] = aux;

                        aux = pic_[k+1][i];
                        pic_[k+1][i] = pic_[k][i];
                        pic_[k][i] = aux;

                        change = true;
                    }
                }
                if (change = false){
                    break;
                }
            }
        }
    }

    return;
}
void mcmcsampler::fit(std::vector<double> data, std::vector<double> datac){
    
    //Set initial conditions
    if (is_initial_condition_){
        for(unsigned int i = 0; i < chains_; i++){
            int a = i*iterations_;
            for (int j = 0; j < K_; j++){
                pi_[j][a] = initial_condition_["pi"][i][j]; 
                mu_[j][a] = initial_condition_["mu"][i][j]; 
                sigma_[j][a] = initial_condition_["sigma"][i][j]; 
            }
            for (int j = 0; j < Kc_; j++){
                pic_[j][a] = initial_condition_["pic"][i][j]; 
                muc_[j][a] = initial_condition_["muc"][i][j]; 
                sigmac_[j][a] = initial_condition_["sigmac"][i][j]; 
            }            
        }
    }
    //Create threads
    std::thread chains[chains_];
    for(unsigned int i = 0; i < chains_; i++){
        int a = i*iterations_;
        chains[i] = std::thread(&mcmcsampler::chain, this, a, std::ref(data), std::ref(datac)); //Need explicit by reference std::ref
    }
    //Wait for rejoining
    for(unsigned int i = 0; i < chains_; i++){
        chains[i].join();
    }

    return;
}
std::map<std::string, std::vector<std::vector<double>>> mcmcsampler::get_fit_parameters(){
    std::map<std::string, std::vector<std::vector<double>>> v;
    v["pi"] = pi_;
    v["mu"] = mu_;
    v["sigma"] = sigma_;
    v["pic"] = pic_;
    v["muc"] = muc_;
    v["sigmac"] = sigmac_;

    return v;
}
std::vector<std::vector<double>> mcmcsampler::score_deconvolution(std::vector<double> data){
    int N = data.size();
    int length = iterations_*chains_;
    std::vector<double> p(iterations_*chains_,0);
    std::vector<std::vector<double>> statistics(3,std::vector<double>(N,0));
    double aux;
    double pcum;
    
    for (int k = 0; k < N; k++){
        for (int i = 0; i < length; i++){
            for(unsigned int j = 0; j < Kc_; j++){
                aux = pic_[j][i]*std::exp(gaussian_pdf(data[k],muc_[j][i],sigmac_[j][i]));
                p[i] += aux;
                pcum += aux;
            }
        }
        //Compute the values
        std::sort(&(p[0]),&(p[length-1])); //Order
        //Statistics
        statistics[0][k] = pcum/length;
        statistics[1][k] = p[int(0.025*length)];
        statistics[2][k] = p[int(0.975*length)];
        //Cleean the array
        for (int i = 0; i < length; i++){
            p[i] = 0;
        }
        pcum = 0;
    }

    return statistics;
}
std::vector<std::vector<double>> mcmcsampler::score_autofluorescence(std::vector<double> data){
    int N = data.size();
    int length = iterations_*chains_;
    std::vector<double> p(iterations_*chains_,0);
    std::vector<std::vector<double>> statistics(3,std::vector<double>(N,0));
    double aux;
    double pcum;
    
    for (int k = 0; k < N; k++){
        for (int i = 0; i < length; i++){
            for(unsigned int j = 0; j < K_; j++){
                aux = pi_[j][i]*std::exp(gaussian_pdf(data[k],mu_[j][i],sigma_[j][i]));
                p[i] += aux;
                pcum += aux;
            }
        }
        //Compute the values
        std::sort(&(p[0]),&(p[length-1])); //Order
        //Statistics
        statistics[0][k] = pcum/length;
        statistics[1][k] = p[int(0.025*length)];
        statistics[2][k] = p[int(0.975*length)];
        //Cleean the array
        for (int i = 0; i < length; i++){
            p[i] = 0;
        }
        pcum = 0;
    }

    return statistics;
}
std::vector<std::vector<double>> mcmcsampler::score_convolution(std::vector<double> data){
    int N = data.size();
    int length = iterations_*chains_;
    std::vector<double> p(iterations_*chains_,0);
    std::vector<std::vector<double>> statistics(3,std::vector<double>(N,0));
    double aux;
    double pcum;
    
    for (int k = 0; k < N; k++){
        for (int i = 0; i < length; i++){
            for(unsigned int j = 0; j < Kc_; j++){
                for(unsigned int l = 0; l < K_; l++){
                    aux = pic_[j][i]*pi_[l][i]*std::exp(gaussian_pdf(data[k],muc_[j][i]+mu_[l][i],
                                                std::sqrt(std::pow(sigmac_[j][i],2)+std::pow(sigma_[l][i],2))));
                    p[i] += aux;
                    pcum += aux;
                }
            }
        }
        //Compute the values
        std::sort(&(p[0]),&(p[length-1])); //Order
        //Statistics
        statistics[0][k] = pcum/length;
        statistics[1][k] = p[int(0.025*length)];
        statistics[2][k] = p[int(0.975*length)];
        //Cleean the array
        for (int i = 0; i < length; i++){
            p[i] = 0;
        }
        pcum = 0;
    }

    return statistics;
}
std::vector<double> mcmcsampler::sample_deconvolution(int n_samples = 1, std::string method="all", int sample = 0){

    //Variables for the random generation
    std::mt19937 r{(long unsigned int)time(0)};
    //Choose for choosing from the samples
    std::uniform_int_distribution<int> distribution(0,iterations_*chains_);

    std::vector<double> v(n_samples,0);
    std::vector<int> choice(Kc_,0);
    std::vector<double> pi(Kc_,0);
    int pos;
    int gaussian1;
    
    if(method == "all"){
        for (int i = 0; i < n_samples; i++){
            pos = distribution(r);
            for (int j = 0; j < Kc_; j++){
                pi[j] = pic_[j][pos];
            }
            multinomial_1(r,pi,choice);
            for (int j = 0; j < Kc_; j++){
                if( 1 == choice[j]){
                    gaussian1 = j;
                    break;
                }
            }
            std::normal_distribution<double> gaussian(muc_[gaussian1][pos], sigmac_[gaussian1][pos]);
            v[i] = gaussian(r);
            std::cout << v[i] << std::endl;
        }
    } else if (method == "single"){
        pos = sample;
        for (int i = 0; i < n_samples; i++){
            for (int j = 0; j < Kc_; j++){
                pi[j] = pic_[j][pos];
            }
            multinomial_1(r,pi,choice);
            for (int j = 0; j < Kc_; j++){
                if( 1 == choice[j]){
                    gaussian1 = j;
                    break;
                }
            }
            std::normal_distribution<double> gaussian(muc_[gaussian1][pos], sigmac_[gaussian1][pos]);
            v[i] = gaussian(r);
            std::cout << v[i] << std::endl;
        }
    }
    else{
        throw std::invalid_argument("ERROR: The funtion expects the number of samples to be drawn (default 1), a string with the method (all or single) and a int indicating the sample to use in case of method being single.");                

    }

    return v;
}
std::vector<double> mcmcsampler::sample_autofluorescence(int n_samples = 1, std::string method="all", int sample = 0){

    //Variables for the random generation
    std::mt19937 r{(long unsigned int)time(0)};
    //Choose for choosing from the samples
    std::uniform_int_distribution<int> distribution(0,iterations_*chains_);

    std::vector<double> v(n_samples,0);
    std::vector<int> choice(pi_.size());
    std::vector<double> pi(pi_.size());
    int pos;
    int gaussian1;

    if (method == "all"){
        for (int i = 0; i < n_samples; i++){
            pos = distribution(r);
            for (int j = 0; j < pi_.size(); j++){
                pi[j] = pi_[j][pos];
            }
            multinomial_1(r,pi,choice);
            for (int j = 0; j < pi_.size(); j++){
                if( 1 == choice[j]){
                    gaussian1 = j;
                    break;
                }
            }
            std::normal_distribution<double> gaussian(mu_[gaussian1][pos], sigma_[gaussian1][pos]);
            v[i] = gaussian(r);
        }
    }else if (method == "single"){
        pos = sample;
        for (int i = 0; i < n_samples; i++){
            for (int j = 0; j < pi_.size(); j++){
                pi[j] = pi_[j][pos];
            }
            multinomial_1(r,pi,choice);
            for (int j = 0; j < pi_.size(); j++){
                if( 1 == choice[j]){
                    gaussian1 = j;
                    break;
                }
            }
            std::normal_distribution<double> gaussian(mu_[gaussian1][pos], sigma_[gaussian1][pos]);
            v[i] = gaussian(r);
        }
    }
    else{
        throw std::invalid_argument("ERROR: The funtion expects the number of samples to be drawn (default 1), a string with the method (all or single) and a int indicating the sample to use in case of method being single.");                

    }


    return v;
}
std::vector<double> mcmcsampler::sample_convolution(int n_samples = 1, std::string method="all", int sample = 0){

    //Variables for the random generation
    std::mt19937 r{(long unsigned int)time(0)};
    //Choose for choosing from the samples
    std::uniform_int_distribution<int> distribution(0,iterations_*chains_);

    std::vector<double> v(n_samples,0);
    std::vector<int> choice1(pic_.size());
    std::vector<int> choice2(pi_.size());
    std::vector<double> pi(pi_.size());
    std::vector<double> pic(pic_.size());
    int pos;
    int gaussian1;
    int gaussian2;
    if (method=="all"){
        for (int i = 0; i < n_samples; i++){
            pos = distribution(r);
            for (int j = 0; j < pi_.size(); j++){
                pi[j] = pi_[j][pos];
            }
            for (int j = 0; j < pic_.size(); j++){
                pic[j] = pic_[j][pos];
            }
            multinomial_1(r,pi,choice1);
            for (int j = 0; j < pic_.size(); j++){
                if( 1 == choice1[j]){
                    gaussian1 = j;
                    break;
                }
            }
            multinomial_1(r,pic,choice2);
            for (int j = 0; j < pi_.size(); j++){
                if( 1 == choice2[j]){
                    gaussian2 = j;
                    break;
                }
            }
            std::normal_distribution<double> gaussian(muc_[gaussian1][pos]+mu_[gaussian2][pos], 
                                                    std::sqrt(std::pow(sigmac_[gaussian1][pos],2)+std::pow(sigma_[gaussian2][pos],2)));
            v[i] = gaussian(r);
        }
    }
    else if (method=="single"){
        pos = sample;
        for (int i = 0; i < n_samples; i++){
            for (int j = 0; j < pi_.size(); j++){
                pi[j] = pi_[j][pos];
            }
            for (int j = 0; j < pic_.size(); j++){
                pic[j] = pic_[j][pos];
            }
            multinomial_1(r,pi,choice1);
            for (int j = 0; j < pic_.size(); j++){
                if( 1 == choice1[j]){
                    gaussian1 = j;
                    break;
                }
            }
            multinomial_1(r,pic,choice2);
            for (int j = 0; j < pi_.size(); j++){
                if( 1 == choice2[j]){
                    gaussian2 = j;
                    break;
                }
            }
            std::normal_distribution<double> gaussian(muc_[gaussian1][pos]+mu_[gaussian2][pos], 
                                                    std::sqrt(std::pow(sigmac_[gaussian1][pos],2)+std::pow(sigma_[gaussian2][pos],2)));
            v[i] = gaussian(r);
        }
    }
    else{
        throw std::invalid_argument("ERROR: The funtion expects the number of samples to be drawn (default 1), a string with the method (all or single) and a int indicating the sample to use in case of method being single.");                

    }


    return v;
}
double mcmcsampler::rstat(std::vector<double> vphi){
    int n = int(iterations_/2);
    int m = 2*chains_;
    std::vector<double> phij(m,0);
    double phi = 0;
    std::vector<double> sj(m,0);
    double B = 0;
    double W = 0;

    //Compute phij
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            phij[i] += vphi[i*n+j];
        }
        phij[i] /= n;
    }
    //Compute phi
    for (int i = 0; i < m; i++){
        phi += phij[i];
    }
    phi /= m;
    //Compute B
    for (int i = 0; i < m; i++){
        B += std::pow(phij[i]-phi,2);
    }
    B *= n/(m-1);
    //Compute sj
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            sj[i] += std::pow(vphi[i*n+j]-phij[i],2);
        }
        sj[i] /= (n-1);
    }
    //Compute W
    for (int i = 0; i < m; i++){
        W += sj[i];
    }
    W /= m;

    if(W == 0){
        return 1;
    }
    else{
        return std::sqrt(1-1/n+B/(W*n));
    }
}
double mcmcsampler::effnumber(std::vector<double> vphi){
    int n = int(iterations_/2);
    int m = 2*chains_;
    std::vector<double> phij(m,0);
    double phi = 0;
    std::vector<double> sj(m,0);
    double B = 0;
    double W = 0;
    double Var = 0;
    double rhoT = 0;
    double rho0 = 0;
    double rho1 = 0;
    int t = 1;
    double Vt = 0;

    //Compute phij
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            phij[i] += vphi[i*n+j];
        }
        phij[i] /= n;
    }
    //Compute phi
    for (int i = 0; i < m; i++){
        phi += phij[i];
    }
    phi /= m;
    //Compute B
    for (int i = 0; i < m; i++){
        B += std::pow(phij[i]-phi,2);
    }
    B *= n/(m-1);
    //Compute sj
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            sj[i] += std::pow(vphi[i*n+j]-phij[i],2);
        }
        sj[i] /= (n-1);
    }
    //Compute W
    for (int i = 0; i < m; i++){
        W += sj[i];
    }
    W /= m;
    //Compute Var+
    Var = (n-1)/n*W+B/n;

    //Compute neff
    do{
        Vt = 0;
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n-t; j++){
                Vt += std::pow(vphi[n*i+j]-vphi[n*i+j+t],2);
            }
        }
        Vt /= m*(n-t);
        rho1 = rho0;
        rho0 = 1-Vt/(2*Var);
        if (rho1+rho0 >= 0){
            rhoT += rho0;
        }
        else{
            break;
        }

        t += 1;
    }while(t < iterations_);

    return m*n/(1+2*rhoT);
}
std::map<std::string, std::map<std::string, double>> mcmcsampler::statistics(std::string flavour = "weights"){
    std::map<std::string, std::map<std::string, double>> v;

    sort_chains(flavour);

    for (int i = 0; i < K_; i++){
        //R
        v["pi"+std::to_string(i)]["R"]= rstat(pi_[i]); 
        v["mu"+std::to_string(i)]["R"]= rstat(mu_[i]) ;
        v["sigma"+std::to_string(i)]["R"]= rstat(sigma_[i]); 
        //neff
        v["pi"+std::to_string(i)]["neff"]= effnumber(pi_[i]); 
        v["mu"+std::to_string(i)]["neff"]= effnumber(mu_[i]) ;
        v["sigma"+std::to_string(i)]["neff"]= effnumber(sigma_[i]); 

    }
    for (int i = 0; i < Kc_; i++){
        //R
        v["pic"+std::to_string(i)]["R"]= rstat(pic_[i]); 
        v["muc"+std::to_string(i)]["R"]= rstat(muc_[i]); 
        v["sigmac"+std::to_string(i)]["R"]= rstat(sigmac_[i]); 
        //neff
        v["pic"+std::to_string(i)]["neff"]= effnumber(pic_[i]); 
        v["muc"+std::to_string(i)]["neff"]= effnumber(muc_[i]); 
        v["sigmac"+std::to_string(i)]["neff"]= effnumber(sigmac_[i]); 
    }

    return v;
}
