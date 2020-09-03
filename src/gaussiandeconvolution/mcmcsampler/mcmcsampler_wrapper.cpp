//Load the wrapper headers
#include <pybind11/pybind11.h> //General
 //Printing cout
#include <pybind11/stl.h>   //For std:: containers (vectors, arrays...)
#include <pybind11/numpy.h> //For vectorizing functions 
//Load the fucntion headers
#include "mcmcsampler.h"
#include <vector>


namespace py = pybind11;

PYBIND11_MODULE(mcmcsampler, m) {
    m.doc() = "Gaussian deconvolution library"; // optional module docstring

    m.def("fit", &fit, "Function for the fit process of the mcmc model",
        py::arg("data"), py::arg("datac"), py::arg("ignored_iterations"), py::arg("iterations"), py::arg("nChains"),
        py::arg("K"), py::arg("Kc"), py::arg("alpha"), py::arg("alphac"), py::arg("sigmaWidth"),
        py::arg("initial_conditions") = std::vector<double>{}, py::arg("showProgress") = true);
}
