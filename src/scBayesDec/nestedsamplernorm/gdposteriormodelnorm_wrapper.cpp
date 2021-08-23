//Load the wrapper headers
#include <pybind11/pybind11.h> //General
 //Printing cout
#include <pybind11/stl.h>   //For std:: containers (vectors, arrays...)
#include <pybind11/numpy.h> //For vectorizing functions 

#include <vector>
#include "gdposteriormodelnorm.h"

namespace py = pybind11;

PYBIND11_MODULE(gdposteriormodelnorm, m) {
    m.doc() = "Gaussian deconvolution library"; // optional module docstring

    //Declare the simple density estimator class
    py::class_<gdposteriormodelnorm>(m, "gdposteriormodelnorm")
        //Show contructor
        .def(py::init<std::vector<double>, std::vector<double>, int, int>())
        .def("logLikelihood", &gdposteriormodelnorm::logLikelihood)
        .def("prior", &gdposteriormodelnorm::prior)
        .def("prior_uniform", &gdposteriormodelnorm::prior_uniform)
        .def_readwrite("K", &gdposteriormodelnorm::K)
        .def_readwrite("Kc", &gdposteriormodelnorm::Kc)
        .def_readwrite("data", &gdposteriormodelnorm::dataNoise)
        .def_readwrite("datac", &gdposteriormodelnorm::dataConvolution)
        .def_readwrite("priors", &gdposteriormodelnorm::priors)
        ;
}