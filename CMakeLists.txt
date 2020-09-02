cmake_minimum_required(VERSION 3.14...3.17)

project(gaussiandeconvolution VERSION "0.1")

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

# Fetch pybind11
include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(pybind11)

set(python_module_name1 gdposteriormodel)
pybind11_add_module(${python_module_name1} MODULE 
  src/gaussiandeconvolution/nestedsampler/gdposteriormodel_wrapper.cpp
  src/gaussiandeconvolution/nestedsampler/gdposteriormodel.cpp
)
install(TARGETS ${python_module_name1} DESTINATION ./nestedsampler)

set(python_module_name2 mcmcsampler)
pybind11_add_module(${python_module_name2} MODULE 
  src/gaussiandeconvolution/mcmcsampler/mcmcsampler_wrapper.cpp
  src/gaussiandeconvolution/mcmcsampler/mcmcsampler.cpp
  src/gaussiandeconvolution/mcmcsampler/functions.cpp
  src/gaussiandeconvolution/mcmcsampler/probability_distributions.cpp
)
install(TARGETS ${python_module_name2} DESTINATION ./mcmcsampler)

set(python_module_name3 shared_functions)
pybind11_add_module(${python_module_name3} MODULE 
  src/gaussiandeconvolution/shared_functions/shared_functions_wrapper.cpp
  src/gaussiandeconvolution/shared_functions/general_functions.cpp
  src/gaussiandeconvolution/shared_functions/gd_samplers.cpp
  src/gaussiandeconvolution/shared_functions/gd_scorers.cpp
  src/gaussiandeconvolution/shared_functions/global_random_generator.cpp
)
install(TARGETS ${python_module_name3} DESTINATION ./shared_functions)