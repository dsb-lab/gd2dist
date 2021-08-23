//Load the wrapper headers
#include <pybind11/pybind11.h> //General
 //Printing cout
#include <pybind11/stl.h>   //For std:: containers (vectors, arrays...)
#include <pybind11/numpy.h> //For vectorizing functions 

#include <vector>
#include "gd_samplers.h"
#include "gd_scorers.h"
#include "mcmc_convergence_statistics.h"

namespace py = pybind11;

PYBIND11_MODULE(shared_functions, m) {
    m.doc() = "Between class shared functions"; // optional module docstring

    //Wrappers of the sampling functions
    m.def("sample_autofluorescence", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int)) &sample_autofluorescence,
     "Sample the autofluorecence from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"));
    
    m.def("sample_autofluorescence", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int)) &sample_autofluorescence,
     "Sample the autofluorecence from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"));

    m.def("sample_autofluorescence", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int, int)) &sample_autofluorescence_single,
     "Sample the autofluorecence from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"), py::arg("pos"));
    
    m.def("sample_autofluorescence", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, int)) &sample_autofluorescence_single,
     "Sample the autofluorecence from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"), py::arg("pos"));


    m.def("sample_deconvolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int)) &sample_deconvolution,
     "Sample the target from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"));
    
    m.def("sample_deconvolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int)) &sample_deconvolution,
     "Sample the target from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"));

    m.def("sample_deconvolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int, int)) &sample_deconvolution_single,
     "Sample the target from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"), py::arg("pos"));
    
    m.def("sample_deconvolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, int)) &sample_deconvolution_single,
     "Sample the target from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"), py::arg("pos"));


    m.def("sample_convolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int)) &sample_convolution,
     "Sample the convolution from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"));
    
    m.def("sample_convolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int)) &sample_convolution,
     "Sample the convolution from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"));

    m.def("sample_convolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int, int)) &sample_convolution_single,
     "Sample the convolution from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"), py::arg("pos"));
    
    m.def("sample_convolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, int)) &sample_convolution_single,
     "Sample the convolution from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"), py::arg("pos"));

    m.def("sample_autofluorescence_gamma", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int, double)) &sample_autofluorescence_gamma,
     "Sample the autofluorecence from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"), py::arg("bias"));
    
    m.def("sample_autofluorescence_gamma", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, double)) &sample_autofluorescence_gamma,
     "Sample the autofluorecence from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"), py::arg("bias"));

    m.def("sample_autofluorescence_gamma", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int, int, double)) &sample_autofluorescence_single_gamma,
     "Sample the autofluorecence from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"), py::arg("pos"), py::arg("bias"));
    
    m.def("sample_autofluorescence_gamma", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, int, double)) &sample_autofluorescence_single_gamma,
     "Sample the autofluorecence from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"), py::arg("pos"), py::arg("bias"));


    m.def("sample_deconvolution_gamma", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int, double)) &sample_deconvolution_gamma,
     "Sample the target from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"), py::arg("bias"));
    
    m.def("sample_deconvolution_gamma", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, double)) &sample_deconvolution_gamma,
     "Sample the target from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"), py::arg("bias"));

    m.def("sample_deconvolution_gamma", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int, int, double)) &sample_deconvolution_single_gamma,
     "Sample the target from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"), py::arg("pos"), py::arg("bias"));
    
    m.def("sample_deconvolution_gamma", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, int, double)) &sample_deconvolution_single_gamma,
     "Sample the target from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"), py::arg("pos"), py::arg("bias"));


    m.def("sample_convolution_gamma", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int, double)) &sample_convolution_gamma,
     "Sample the convolution from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"), py::arg("bias"));
    
    m.def("sample_convolution_gamma", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, double)) &sample_convolution_gamma,
     "Sample the convolution from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"), py::arg("bias"));

    m.def("sample_convolution_gamma", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int, int, double)) &sample_convolution_single_gamma,
     "Sample the convolution from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"), py::arg("pos"), py::arg("bias"));
    
    m.def("sample_convolution_gamma", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int, int, double)) &sample_convolution_single_gamma,
     "Sample the convolution from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"), py::arg("pos"), py::arg("bias"));

    //Wrappers of the scoring functions
    m.def("score_autofluorescence", 
    (std::vector<std::vector<double>> (*)(std::vector<std::vector<double>>&, std::vector<double>&, int, int, std::vector<double>, std::vector<double>, int)) &score_autofluorescence,
     "score the autofluorecence from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("percentiles"), py::arg("weights"), py::arg("size"));
    
    m.def("score_autofluorescence", 
    (std::vector<std::vector<double>> (*)(std::vector<std::vector<double>>&, std::vector<double>&, int, int, std::vector<double>, int)) &score_autofluorescence,
     "score the autofluorecence from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("percentiles"), py::arg("size"));


    /*m.def("score_deconvolution", 
    (std::vector<std::vector<double>> (*)(std::vector<std::vector<double>>&, std::vector<double>&, int, int, std::vector<double>, std::vector<double>, int)) &score_deconvolution,
     "score the target from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("percentiles"), py::arg("weights"), py::arg("size"));
    
    m.def("score_deconvolution", 
    (std::vector<std::vector<double>> (*)(std::vector<std::vector<double>>&, std::vector<double>&, int, int, std::vector<double>, int)) &score_deconvolution,
     "score the target from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("percentiles"), py::arg("size"));


    m.def("score_convolution", 
    (std::vector<std::vector<double>> (*)(std::vector<std::vector<double>>&, std::vector<double>&, int, int, std::vector<double>, std::vector<double>, int)) &score_convolution,
     "score the convolution from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("percentiles"), py::arg("weights"), py::arg("size"));
    
    m.def("score_convolution", 
    (std::vector<std::vector<double>> (*)(std::vector<std::vector<double>>&, std::vector<double>&, int, int, std::vector<double>, int)) &score_convolution,
     "score the convolution from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("percentiles"), py::arg("size"));

    m.def("score_autofluorescence_gamma", 
    (std::vector<std::vector<double>> (*)(std::vector<std::vector<double>>&, std::vector<double>&, int, int, std::vector<double>, std::vector<double>, int, double)) &score_autofluorescence_gamma,
     "score the autofluorecence from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("percentiles"), py::arg("weights"), py::arg("size"), py::arg("bias"));
    
    m.def("score_autofluorescence_gamma", 
    (std::vector<std::vector<double>> (*)(std::vector<std::vector<double>>&, std::vector<double>&, int, int, std::vector<double>, int, double)) &score_autofluorescence_gamma,
     "score the autofluorecence from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("percentiles"), py::arg("size"), py::arg("bias"));


    m.def("score_deconvolution_gamma", 
    (std::vector<std::vector<double>> (*)(std::vector<std::vector<double>>&, std::vector<double>&, int, int, std::vector<double>, std::vector<double>, int, double)) &score_deconvolution_gamma,
     "score the target from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("percentiles"), py::arg("weights"), py::arg("size"), py::arg("bias"));
    
    m.def("score_deconvolution_gamma", 
    (std::vector<std::vector<double>> (*)(std::vector<std::vector<double>>&, std::vector<double>&, int, int, std::vector<double>, int, double)) &score_deconvolution_gamma,
     "score the target from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("percentiles"), py::arg("size"), py::arg("bias"));


    m.def("score_convolution_gamma", 
    (std::vector<std::vector<double>> (*)(std::vector<std::vector<double>>&, std::vector<double>&, int, int, std::vector<double>, std::vector<double>, int, double)) &score_convolution_gamma,
     "score the convolution from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("percentiles"), py::arg("weights"), py::arg("size"), py::arg("bias"));
    
    m.def("score_convolution_gamma", 
    (std::vector<std::vector<double>> (*)(std::vector<std::vector<double>>&, std::vector<double>&, int, int, std::vector<double>, int, double)) &score_convolution_gamma,
     "score the convolution from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("percentiles"), py::arg("size"), py::arg("bias"));*/


    //Wrappers mcmc statistics
    m.def("rstat",
    (double (*)(std::vector<double>&, int)) &rstat, py::arg("samples"), py::arg("nChains"));
    m.def("rstat",
    (double (*)(std::vector<std::vector<double>>&)) &rstat, py::arg("samples"));

    m.def("effnumber",
    (double (*)(std::vector<double>&, int)) &effnumber, py::arg("samples"), py::arg("nChains"));
    m.def("effnumber",
    (double (*)(std::vector<std::vector<double>>&)) &effnumber, py::arg("samples"));
}