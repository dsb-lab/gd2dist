#ifndef GD_MCMC_SAMPLER
#define GD_MCMC_SAMPLER

#include <vector>
#include <map>
#include <string>
#include <thread>
#include <algorithm>
#include <iostream>
#include <exception>

#include "functions.h"

class mcmcsampler{
    private:
        unsigned int K_;
        unsigned int Kc_;
        double alpha_;
        double alphac_;
        //Parameters sampler
        unsigned int iterations_;
        unsigned int ignored_iterations_;
        unsigned int chains_;
        double sigmaWidth_;
        std::vector<std::vector<double>> pi_;
        std::vector<std::vector<double>> pic_;
        std::vector<std::vector<double>> mu_;
        std::vector<std::vector<double>> muc_;
        std::vector<std::vector<double>> sigma_;
        std::vector<std::vector<double>> sigmac_;
        //Parameters initial condition
        bool is_initial_condition_ = false;        
        std::map<std::string, std::vector<std::vector<double>>> initial_condition_;
        //Internal functions
        void chain(int, std::vector<double> &, std::vector<double> &);
        void sort_chains(std::string);
        double rstat(std::vector<double>);
        double effnumber(std::vector<double>);
    public:
        mcmcsampler();
        mcmcsampler(unsigned int, unsigned int, double, double,
                             unsigned int, unsigned int, unsigned int, double);
        void set_parameters(unsigned int, unsigned int, double, double,
                             unsigned int, unsigned int, unsigned int, double);
        void set_initial_condition(std::map<std::string, std::vector<std::vector<double>>>);
        std::map<std::string, double> get_parameters();
        double get_parameter(std::string);
        void fit(std::vector<double>, std::vector<double>);
        std::map<std::string, std::vector<std::vector<double>>> get_fit_parameters();
        std::vector<std::vector<double>> score_deconvolution(std::vector<double>);
        std::vector<std::vector<double>> score_autofluorescence(std::vector<double>);
        std::vector<std::vector<double>> score_convolution(std::vector<double>);
        std::vector<double> sample_deconvolution(int, std::string, int); 
        std::vector<double> sample_autofluorescence(int, std::string, int); 
        std::vector<double> sample_convolution(int, std::string, int); 
        std::map<std::string, std::map<std::string, double>> statistics(std::string);
};   

#endif