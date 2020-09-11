//Load the wrapper headers
#include <pybind11/pybind11.h> //General
 //Printing cout
#include "pybind11/iostream.h" 
#include <pybind11/stl.h>   //For std:: containers (vectors, arrays...)
#include <pybind11/numpy.h> //For vectorizing functions 
//Load the fucntion headers
#include "mcmcsampler.h"
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(mcmcposteriorsampler, m) {
    m.doc() = "Gaussian deconvolution library"; // optional module docstring

    m.def("fit", &fit, py::call_guard<py::gil_scoped_release>(), "Function for the fit process of the mcmc model",
        py::arg("data"), py::arg("datac"), py::arg("ignored_iterations"), py::arg("iterations"), py::arg("nChains"),
        py::arg("K"), py::arg("Kc"), py::arg("alpha"), py::arg("alphac"), py::arg("sigmaWidth"),
        py::arg("initial_conditions") = std::vector<double>{}, py::arg("showProgress"), py::arg("seed"));

    /*m.def("fit", [](std::vector<double> & data, std::vector<double>& datac,
                    int ignored_iterations, int iterations, int nChains,
                    int K, int Kc, double alpha, double alphac, double sigmaWidth,
                    std::vector<std::vector<double>> initial_conditions, bool showProgress){
                py::scoped_ostream_redirect stream(
                std::cout,
                py::module::import("sys").attr("stdout"));
                fit(data, datac, ignored_iterations, iterations, nChains, K, Kc, alpha, alphac, sigmaWidth, initial_conditions, showProgress);
                return;}, "Function for the fit process of the mcmc model",
            py::arg("data"), py::arg("datac"), py::arg("ignored_iterations"), py::arg("iterations"), py::arg("nChains"),
            py::arg("K"), py::arg("Kc"), py::arg("alpha"), py::arg("alphac"), py::arg("sigmaWidth"),
            py::arg("initial_conditions") = std::vector<double>{}, py::arg("showProgress") = true);*/
}
