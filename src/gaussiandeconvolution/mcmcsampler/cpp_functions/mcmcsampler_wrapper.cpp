//Load the wrapper headers
#include <pybind11/pybind11.h> //General
 //Printing cout
#include <pybind11/stl.h>   //For std:: containers (vectors, arrays...)
#include <pybind11/numpy.h> //For vectorizing functions 
//Load the fucntion headers
#include "mcmcsampler.h"

namespace py = pybind11;

PYBIND11_MODULE(mcmcsampler, m) {
    m.doc() = "Gaussian deconvolution library"; // optional module docstring

    //Declare the convolved Gibbs sampler class
    py::class_<mcmcsampler>(m, "mcmcsampler")
        //Show contructor
        .def(py::init<>())
        .def(py::init<unsigned int, unsigned int, double, double, unsigned int, unsigned int, unsigned int, double>(), 
            py::arg("K") = 1, py::arg("Kc") = 1, py::arg("alpha") = 1, py::arg("alphac") = 1, 
            py::arg("iterations") = 1000, py::arg("ignored_iterations") = 1000, py::arg("chains") = 4, py::arg("sigmaWidth") = 1)
        //Show set_parameters
        .def("set_parameters", &mcmcsampler::set_parameters, "Set parameters of the density estimation",
            py::arg("K") = 429496729, py::arg("Kc") = 429496729, py::arg("alpha") = -1, py::arg("alphac") = -1, 
            py::arg("iterations") = 429496729, py::arg("ignored_iterations") = 429496729, py::arg("chains") = 429496729, py::arg("sigmaWidth") = -1)
        //Show set_initial_condition
        .def("set_initial_condition", &mcmcsampler::set_initial_condition, "Set initial conditions od the chains")
        //Show get_parameters
        .def("get_parameters", &mcmcsampler::get_parameters, "Get parameters of the density estimation")
        //Show get_parameter
        .def("get_parameter", &mcmcsampler::get_parameter, "Get single parameter of the density estimation.")
        //Show fit
        .def("fit", &mcmcsampler::fit, "Fit the model",
            py::arg("data"), py::arg("datac"))
        //Show get_fit_parameters
        .def("get_fit_parameters", &mcmcsampler::get_fit_parameters, "Get parameters from the fit")
        //Show score
        .def("score_deconvolution", &mcmcsampler::score_deconvolution, "Score the probability at this point")
        .def("score_autofluorescence", &mcmcsampler::score_autofluorescence, "Score the probability at this point")
        .def("score_convolution", &mcmcsampler::score_convolution, "Score the probability at this point")
        //Sample from model
        .def("sample_deconvolution", &mcmcsampler::sample_deconvolution, "Sample from the model",
             py::arg("n_samples") = 1, py::arg("method") = "all", py::arg("sample") = 0)
        .def("sample_autofluorescence", &mcmcsampler::sample_autofluorescence, "Sample from the model", 
             py::arg("n_samples") = 1, py::arg("method") = "all", py::arg("sample") = 0)
        .def("sample_convolution", &mcmcsampler::sample_convolution, "Sample from the model",
             py::arg("n_samples") = 1, py::arg("method") = "all", py::arg("sample") = 0)
        //Statistics of sampler
        .def("statistics", &mcmcsampler::statistics, "Statistics", py::arg("flavour") = "weights")
        //Define object to show
        .def("__repr__", 
            [](mcmcsampler &a){
                return "<gaussian_deconvolution.convolved_EM_class with " + std::to_string(int(a.get_parameter("K"))) + " and " 
                + std::to_string(int(a.get_parameter("Kc"))) + " gaussians>";
                })
        ;
}
