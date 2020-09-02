//Load the wrapper headers
#include <pybind11/pybind11.h> //General
 //Printing cout
#include <pybind11/stl.h>   //For std:: containers (vectors, arrays...)
#include <pybind11/numpy.h> //For vectorizing functions 

#include <vector>
#include "gd_samplers.h"
#include "gd_scorers.h"

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


    m.def("sample_deconvolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int)) &sample_deconvolution,
     "Sample the target from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"));
    
    m.def("sample_deconvolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int)) &sample_deconvolution,
     "Sample the target from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"));


    m.def("sample_convolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, int)) &sample_convolution,
     "Sample the convolution from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("size"));
    
    m.def("sample_convolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, int, int, std::vector<double>&, int)) &sample_convolution,
     "Sample the convolution from posterior",
     py::arg("posterior"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"));


    //Wrappers of the scoring functions
    m.def("score_autofluorescence", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, std::vector<double>, int, int, int)) &score_autofluorescence,
     "score the autofluorecence from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("size"));
    
    m.def("score_autofluorescence", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, std::vector<double>, int, int, std::vector<double>&, int)) &score_autofluorescence,
     "score the autofluorecence from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"));


    m.def("score_deconvolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, std::vector<double>, int, int, int)) &score_deconvolution,
     "score the target from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("size"));
    
    m.def("score_deconvolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, std::vector<double>, int, int, std::vector<double>&, int)) &score_deconvolution,
     "score the target from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"));


    m.def("score_convolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, std::vector<double>, int, int, int)) &score_convolution,
     "score the convolution from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("size"));
    
    m.def("score_convolution", 
    (std::vector<double> (*)(std::vector<std::vector<double>>&, std::vector<double>, int, int, std::vector<double>&, int)) &score_convolution,
     "score the convolution from posterior",
     py::arg("posterior"), py::arg("x"), py::arg("K"), py::arg("Kc"), py::arg("weights"), py::arg("size"));


}