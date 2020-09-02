//Load the wrapper headers
#include <pybind11/pybind11.h> //General
 //Printing cout
#include <pybind11/stl.h>   //For std:: containers (vectors, arrays...)
#include <pybind11/numpy.h> //For vectorizing functions 

#include <vector>
#include "gdposteriormodel.h"

namespace py = pybind11;

PYBIND11_MODULE(gdposteriormodel, m) {
    m.doc() = "Gaussian deconvolution library"; // optional module docstring

    //Declare the simple density estimator class
    py::class_<gdposteriormodel>(m, "gdposteriormodel")
        //Show contructor
        .def(py::init<std::vector<double>, std::vector<double>, int, int>())
        .def("logLikelihood", &gdposteriormodel::logLikelihood)
        .def("prior", &gdposteriormodel::prior)
        .def_readwrite("_K", &gdposteriormodel::_K)
        .def_readwrite("_Kc", &gdposteriormodel::_Kc)
        ;
}