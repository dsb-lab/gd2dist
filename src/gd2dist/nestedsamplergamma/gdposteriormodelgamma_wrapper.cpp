//Load the wrapper headers
#include <pybind11/pybind11.h> //General
 //Printing cout
#include <pybind11/stl.h>   //For std:: containers (vectors, arrays...)
#include <pybind11/numpy.h> //For vectorizing functions 

#include <vector>
#include "gdposteriormodelgamma.h"

namespace py = pybind11;

PYBIND11_MODULE(gdposteriormodelgamma, m) {
    m.doc() = "Gaussian deconvolution library"; // optional module docstring

    //Declare the simple density estimator class
    py::class_<gdposteriormodelgamma>(m, "gdposteriormodelgamma")
        //Show contructor
        .def(py::init<std::vector<double>, std::vector<double>, int, int, double>())
        .def("logLikelihood", &gdposteriormodelgamma::logLikelihood)
        .def("prior", &gdposteriormodelgamma::prior)
        .def("prior_uniform", &gdposteriormodelgamma::prior_uniform)
        .def_readwrite("K", &gdposteriormodelgamma::K)
        .def_readwrite("Kc", &gdposteriormodelgamma::Kc)
        .def_readwrite("data", &gdposteriormodelgamma::dataNoise)
        .def_readwrite("datac", &gdposteriormodelgamma::dataConvolution)
        .def_readwrite("bias", &gdposteriormodelgamma::bias)
        .def_readwrite("priors", &gdposteriormodelgamma::priors)
        .def_readwrite("precission", &gdposteriormodelgamma::precission)
        ;
}