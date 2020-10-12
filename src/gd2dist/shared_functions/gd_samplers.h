#ifndef GD_SAMPLERS
#define GD_SAMPLERS

#include <vector>
#include <random>

//Samplers autofluorescence
std::vector<double> sample_autofluorescence(std::vector<std::vector<double>>&, int, int, int);
std::vector<double> sample_autofluorescence(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int);

//Samplers deconvolution
std::vector<double> sample_deconvolution(std::vector<std::vector<double>>&, int, int, int);
std::vector<double> sample_deconvolution(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int);

//Samplers convolution
std::vector<double> sample_convolution(std::vector<std::vector<double>>&, int, int, int);
std::vector<double> sample_convolution(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int);

//Samplers autofluorescence
std::vector<double> sample_autofluorescence_single(std::vector<std::vector<double>>&, int, int, int, int);
std::vector<double> sample_autofluorescence_single(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, int);

//Samplers deconvolution
std::vector<double> sample_deconvolution_single(std::vector<std::vector<double>>&, int, int, int, int);
std::vector<double> sample_deconvolution_single(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, int);

//Samplers convolution
std::vector<double> sample_convolution_single(std::vector<std::vector<double>>&, int, int, int, int);
std::vector<double> sample_convolution_single(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, int);

//Samplers autofluorescence
std::vector<double> sample_autofluorescence_gamma(std::vector<std::vector<double>>&, int, int, int, double);
std::vector<double> sample_autofluorescence_gamma(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, double);

//Samplers deconvolution
std::vector<double> sample_deconvolution_gamma(std::vector<std::vector<double>>&, int, int, int, double);
std::vector<double> sample_deconvolution_gamma(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, double);

//Samplers convolution
std::vector<double> sample_convolution_gamma(std::vector<std::vector<double>>&, int, int, int, double);
std::vector<double> sample_convolution_gamma(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, double);

//Samplers autofluorescence
std::vector<double> sample_autofluorescence_single_gamma(std::vector<std::vector<double>>&, int, int, int, int, double);
std::vector<double> sample_autofluorescence_single_gamma(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, int, double);

//Samplers deconvolution
std::vector<double> sample_deconvolution_single_gamma(std::vector<std::vector<double>>&, int, int, int, int, double);
std::vector<double> sample_deconvolution_single_gamma(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, int, double);

//Samplers convolution
std::vector<double> sample_convolution_single_gamma(std::vector<std::vector<double>>&, int, int, int, int, double);
std::vector<double> sample_convolution_single_gamma(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, int, double);


#endif