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
  src/gaussiandeconvolution/nestedsampler/cpp_functions/gdposteriormodel_wrapper.cpp
  src/gaussiandeconvolution/nestedsampler/cpp_functions/gdposteriormodel.cpp
)
install(TARGETS ${python_module_name1} DESTINATION ./nestedsampler)

set(python_module_name2 mcmcsampler)
pybind11_add_module(${python_module_name2} MODULE 
  src/gaussiandeconvolution/mcmcsampler/cpp_functions/mcmcsampler_wrapper.cpp
  src/gaussiandeconvolution/mcmcsampler/cpp_functions/mcmcsampler.cpp
  src/gaussiandeconvolution/mcmcsampler/cpp_functions/functions.cpp
  src/gaussiandeconvolution/mcmcsampler/cpp_functions/probability_distributions.cpp
)
install(TARGETS ${python_module_name2} DESTINATION ./mcmcsampler)

# Quiet a warning, since this project is only valid with SKBUILD
set(ignoreMe "${SKBUILD}")
