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
        .def(py::init<std::vector<double>, std::vector<double>, int, int>())
        .def("logLikelihood", &gdposteriormodelgamma::logLikelihood)
        .def("prior", &gdposteriormodelgamma::prior)
        .def_readwrite("K", &gdposteriormodelgamma::K)
        .def_readwrite("Kc", &gdposteriormodelgamma::Kc)
        .def_readwrite("data", &gdposteriormodelgamma::dataNoise)
        .def_readwrite("datac", &gdposteriormodelgamma::dataConvolution)
        .def_readwrite("priorbias_sigma", &gdposteriormodelgamma::priorbias_sigma)
        .def_readwrite("priortheta_theta", &gdposteriormodelgamma::priortheta_theta)
        .def_readwrite("priortheta_k", &gdposteriormodelgamma::priortheta_k)
        .def_readwrite("priork_theta", &gdposteriormodelgamma::priork_theta)
        .def_readwrite("priork_k", &gdposteriormodelgamma::priork_k)
        .def_readwrite("priortheta_thetac", &gdposteriormodelgamma::priortheta_thetac)
        .def_readwrite("priortheta_kc", &gdposteriormodelgamma::priortheta_kc)
        .def_readwrite("priork_thetac", &gdposteriormodelgamma::priork_thetac)
        .def_readwrite("priork_kc", &gdposteriormodelgamma::priork_kc)
        .def_readwrite("precission", &gdposteriormodelgamma::precission)
        ;
}