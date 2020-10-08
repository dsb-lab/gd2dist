//Load the wrapper headers
#include <pybind11/pybind11.h> //General
 //Printing cout
#include "pybind11/iostream.h" 
#include <pybind11/stl.h>   //For std:: containers (vectors, arrays...)
#include <pybind11/numpy.h> //For vectorizing functions 
//Load the fucntion headers
#include "mcmcsamplergamma.h"
#include "../shared_functions/probability_distributions.h"
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(mcmcposteriorsamplergamma, m) {
    m.doc() = "Gaussian deconvolution library"; // optional module docstring

    m.def("fit", &fit, py::call_guard<py::gil_scoped_release>(), "Function for the fit process of the mcmc model",
        py::arg("data"), py::arg("datac"), py::arg("ignored_iterations"), py::arg("iterations"), py::arg("nChains"),
        py::arg("K"), py::arg("Kc"), py::arg("alpha"), py::arg("alphac"), 
        py::arg("priortheta_k"), py::arg("priortheta_theta"), py::arg("priork_k"), py::arg("priork_theta"),
        py::arg("priortheta_kc"), py::arg("priortheta_thetac"), py::arg("priork_kc"), py::arg("priork_thetac"),
        py::arg("bias"),
        py::arg("precission"), py::arg("method"),
        py::arg("initial_conditions") = std::vector<double>{}, py::arg("showProgress"), py::arg("seed"));

    /*m.def("logLikelihood", &logLikelihood, "Function for the fit process of the mcmc model",
        py::arg("pi"), py::arg("mu"), py::arg("sigma"),
        py::arg("pic"), py::arg("muc"), py::arg("sigmac"),
        py::arg("data"), py::arg("datac"));*/

    m.def("checkfunction", &gamma_pdf, py::call_guard<py::gil_scoped_release>());
    
}
