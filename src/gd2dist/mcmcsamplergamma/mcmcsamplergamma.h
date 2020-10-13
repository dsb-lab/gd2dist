#ifndef GD_MCMC_SAMPLER
#define GD_MCMC_SAMPLER

#include <vector>
#include <map>
#include <string>
#include <thread>
#include <algorithm>
#include <iostream>
#include <exception>
#include <random>

double gamma_pdf_batch(double x, double logx, double n, double theta, double kconst,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta);

double gamma_sum_pdf_batch(std::vector<double> &datac, double theta, double kconst, double thetac, double kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission, std::string method,
                            std::vector<int> &id, int counter);

double gamma_sum_pdf_batch(double x, double xlog, double n, double theta, double kconst, double thetac, double kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission, std::string method);

double gamma_pdf_full_batch(std::vector<double> &datac, double theta, double kconst, std::vector<double> thetac, std::vector<double> kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission, std::string method,
                            std::vector<std::vector<int>> &id, std::vector<int> &counter,
                            double x, double logx, double n,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta);

double gamma_pdf_full_batch(double theta, double kconst, std::vector<double> thetac, std::vector<double> kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission, std::string method,
                            double x, double xlog, double n, std::vector<double> xc, std::vector<double> xlogc, std::vector<double> nc,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta);

double gamma_pdf_full_batch(std::vector<double> &datac, std::vector<double> theta, std::vector<double> kconst, double thetac, double kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission, std::string method,
                            std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter, int pos);

double gamma_pdf_full_batch(std::vector<std::vector<double>> &xc, std::vector<std::vector<double>> &xlogc, std::vector<std::vector<double>> &nc,
                            std::vector<double> theta, std::vector<double> kconst, double thetac, double kconstc,
                            double bias,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                            double precission, std::string method,
                            int pos);

double gamma_pdf_full_batch_slow(std::vector<double> &data, std::vector<double> &datac, std::vector<double> theta, std::vector<double> kconst, std::vector<double> thetac, std::vector<double> kconstc,
                            double bias,
                            double precission, std::string method,
                            std::vector<std::vector<int>> &id, std::vector<int> counter,
                            std::vector<std::vector<std::vector<int>>> &idc, std::vector<std::vector<int>> &counterc,
                            double priorbias_sigma);

void slice_theta(std::mt19937 &r, std::vector<double> &n, std::vector<double> &x, std::vector<double> &xlog, 
                            std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                            std::vector<double> &kconstnew, std::vector<double> &datac, std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                            double bias, double precission, std::string method);

void slice_theta(std::mt19937 & r, std::vector<double> &n, std::vector<double> &x, std::vector<std::vector<double>> &xlog, std::vector<std::vector<double>> &nc, std::vector<std::vector<double>> &xc, std::vector<double> &xlogc, 
                            std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                            std::vector<double> &thetanew, std::vector<double> &datac,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                            double bias, double precission, std::string method, std::vector<double> & slice);

void slice_k(std::mt19937 &r, std::vector<double> &n, std::vector<double> &x, std::vector<double> &xlog, 
                            std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                            std::vector<double> &kconstnew, std::vector<double> &datac, std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                            double bias, double precission, std::string method);

void slice_k(std::mt19937 & r, std::vector<double> &n, std::vector<double> &x, std::vector<double> &xlog, std::vector<std::vector<double>> &nc, std::vector<std::vector<double>> &xc, std::vector<std::vector<double>> &xlogc, 
                            std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                            std::vector<double> &kconstnew,
                            double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                            double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                            double bias, double precission, std::string method, std::vector<double> & slice);

void slice_thetac(std::mt19937 &r, 
                std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                std::vector<double> &thetanewc, std::vector<double> &datac, std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter,
                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                double bias, double precission, std::string method);

void slice_thetac(std::mt19937 & r, 
                std::vector<std::vector<double>> &xc, std::vector<std::vector<double>> &xlogc, std::vector<std::vector<double>> &nc,
                std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                std::vector<double> &thetanewc,
                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                double bias, double precission, std::string method, std::vector<double> & slice);

void slice_kc(std::mt19937 &r, 
                std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                std::vector<double> &kconstnewc, std::vector<double> &datac, std::vector<std::vector<std::vector<int>>> &id, std::vector<std::vector<int>> &counter,
                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                double bias, double precission, std::string method);

void slice_kc(std::mt19937 &r, 
                std::vector<std::vector<double>> &xc, std::vector<std::vector<double>> &xlogc, std::vector<std::vector<double>> &nc,
                std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                std::vector<double> &kconstnewc,
                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                double bias, double precission, std::string method, std::vector<double> & slice);

void slice_bias(std::mt19937 &r, 
                std::vector<double> &theta, std::vector<double> &kconst, std::vector<double> &thetac, std::vector<double> &kconstc, 
                std::vector<double> &data, std::vector<double> &datac, 
                std::vector<std::vector<int>> &id, std::vector<int> &counter,
                std::vector<std::vector<std::vector<int>>> &idc, std::vector<std::vector<int>> &counterc,
                double bias, double & biasnew, double priorbias_sigma, double precission, std::string method);

/*void Gibbs_convolved_step(std::mt19937 & r, std::vector<double> & data, std::vector<double> & datac,
                    std::vector<double> & pi, std::vector<double> & theta, std::vector<double> & kconst, 
                    std::vector<double> & pinew, std::vector<double> & thetanew, std::vector<double> & kconstnew, 
                    double alpha, double priortheta_k, double priortheta_theta, double priork_k, double priork_theta,
                    std::vector<double> & pic, std::vector<double> & thetac, std::vector<double> & kconstc, 
                    std::vector<double> & pinewc, std::vector<double> & thetanewc, std::vector<double> & kconstnewc, 
                    double alphac, double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac,
                    double bias,
                    std::vector<std::vector<int>> id, std::vector<std::vector<std::vector<int>>> idc,
                    double precission, std::string method,
                    std::vector<double> & slice_step_theta, std::vector<double> & slice_step_k, 
                    std::vector<double> & slice_step_thetac, std::vector<double> & slice_step_kc);*/

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
                    std::vector<double> & slice_step_thetac, std::vector<double> & slice_step_kc);

void chain(int pos0, std::vector<std::vector<double>> & posterior, std::vector<double> & data, std::vector<double> & datac,                          
                                int ignored_iterations, int iterations, int nChains,
                                int K, int Kc, double alpha, double alphac, 
                                double priortheta_k, double priortheta_theta, double priork_k, double priork_theta, 
                                double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                                double bias, 
                                bool initialised, bool showProgress, int seed, double precission, std::string method);

std::vector<std::vector<double>> fit(std::vector<double> & data, std::vector<double>& datac,
                          int ignored_iterations, int iterations, int nChains,
                          int K, int Kc, 
                          double alpha, double alphac, 
                          double priortheta_k, double priortheta_theta, double priork_k, double priork_theta, 
                          double priortheta_kc, double priortheta_thetac, double priork_kc, double priork_thetac, 
                          double bias,
                          double precission, std::string method,
                          std::vector<std::vector<double>> initial_conditions, bool showProgress, int seed);

#endif